"""Kalos portal — the science: run the engine on an arbitrary uploaded run sheet.

Outcome-like columns are candidate TARGETS and are excluded from input features
so the model never "predicts" titer from another measured output (leakage).
"""
from __future__ import annotations

import functools
import gc
import os
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
import pandas as pd

from kalos.core.drivers import benjamini_hochberg, bootstrap_spearman, spearman_driver_matrix
from kalos import __version__ as ENGINE_VERSION
from kalos.core.conformal import q_from_residuals
from kalos.core.gp_shape import gp_shape_report
from kalos.core.replicates import (
    aggregate_replicates,
    heteroscedasticity_report,
    noise_report,
)
from kalos.core.splits import row_hash_groups
from kalos.data.anonymizer import _hash
from kalos.domains import (
    BIOPROCESS_PROFILE,
    ColumnRoles,
    DesignSpace,
    Dimension,
    DomainProfile,
)
from kalos.portal.uploads import MAX_FIT_ROWS, UploadRejected, _ERR_TOO_MANY_FIT_ROWS
from kalos.portal.validate import column_provenance, provenance_dicts
from kalos.validation import apply_unit_conversions, report_dict, validate_frame
from kalos.validation.bounds import DIMENSION_BOUNDS, infer_dimension
from kalos.validation.checks import _RUN_ID_RE

# NOTE: `kalos.core.evaluation` (which imports `kalos.core.surrogate`),
# `kalos.core.optimize`, and `torch` itself are intentionally NOT imported at
# module level. This module is imported by `kalos.runner.singleton` (the
# `--watch` poller) and `kalos.portal.app` (the portal), so a top-level torch
# import here would tax the idle poller and portal boot with the whole
# torch/botorch/gpytorch stack (~220 MB) before any analysis ever runs. They
# are imported lazily inside `_analyze`/`_seed_everything`, the only places
# that actually need them. `kalos.domains` above is torch-free by contract.
#
# This is load-bearing and easy to break by accident: it was broken, by a single
# top-level `from kalos.core.evaluation import producer_only_spearman` added for
# one call inside `_analyze`. `evaluation` imports `surrogate`, so importing this
# module pulled the whole stack (measured: 1.2s and `torch in sys.modules` right
# after `import kalos.portal.analysis`) and the poller paid the tax it was
# written to avoid. `tests/test_portal.py` now asserts the property instead of
# trusting this comment. Every name from `kalos.core.evaluation` belongs in the
# deferred block inside `_analyze`.

# The default (bioprocess) column-role hints, kept as module-level names because
# `_anonymize_result` and the provenance defaults reference them. Sourced from
# BIOPROCESS_PROFILE so a run with no declared roles behaves exactly as before.
_ERR_VALIDATION_FAILED = (
    "The uploaded run sheet failed data validation. Fix the errors listed in "
    "`validation.findings` and upload again, or set KALOS_VALIDATION_MODE=warn "
    "to analyze it anyway and receive the findings as warnings."
)


def validation_mode() -> str:
    """Policy for what an error-severity validation finding does to an upload.

    - "warn" (the DEFAULT): analyze the sheet anyway and return the findings in
      the response. Chosen as the default deliberately, so adding this gate
      cannot start rejecting data that clients push through today - a validation
      gate that silently changes what the API accepts is its own outage.
    - "strict": refuse the upload with `UploadRejected` when the report's status
      is "fail". This is what a regulated workflow should run, and what a client
      who wants the sheet vetted before it reaches a model should ask for.

    Note that "warn" does NOT mean impossible values can reach a recommendation:
    the design box is built from physically valid observations regardless of
    mode (see `_physical_range`). Mode governs reporting, not that safety
    property. Read at call time, not import time, so a deployment can change it
    without a restart and tests can monkeypatch it.
    """
    mode = os.environ.get("KALOS_VALIDATION_MODE", "warn").strip().lower()
    return mode if mode in {"warn", "strict"} else "warn"


_OUTCOME_HINT = BIOPROCESS_PROFILE.outcome_hint
_TARGET_PREF = BIOPROCESS_PROFILE.target_pref
_ID_HINT = BIOPROCESS_PROFILE.id_hint
_GROUP_HINT = BIOPROCESS_PROFILE.group_hint


def _annotate(
    batch: np.ndarray,
    mean,
    std,
    best: float,
    cols=None,
    design: DesignSpace | None = None,
    *,
    p_feasible: np.ndarray | None = None,
    constraint: tuple[str, np.ndarray, np.ndarray] | None = None,
) -> list:
    """Attach predicted value, uncertainty, and an explore/exploit rationale to
    each proposed experiment. Explore = high model uncertainty (chosen to learn);
    exploit = high predicted value (chosen to win). Current human-in-the-loop BO
    research says a recommendation must carry exactly this.

    When a `design` is given, each row also carries `recipe`: the full proposed
    experiment decoded to `{feature: value}`, with categorical dimensions decoded
    back to their labels instead of raw integer codes.

    `p_feasible`, when given, is a length-`len(batch)` array and every row gets a
    `p_feasible` key - `None` for a NaN entry, a rounded float otherwise. `None`
    (the default, unchanged for the `kalos.portal.app` call site) omits the key
    entirely rather than adding it with a placeholder value, so a caller that
    never computed a feasibility probability is not told one exists.

    `constraint`, when given, is `(column_name, pred_mean, pred_std)` - three
    length-`len(batch)` arrays/name - and every row gets a
    `pred_<column_name>` key of `{"mean": ..., "std": ...}`. `None` (the
    default) omits the key entirely: it is only ever passed when PART 2's
    constraint was actually applied."""
    mean = np.asarray(mean, float).reshape(-1)
    std = np.asarray(std, float).reshape(-1)
    thr = float(np.quantile(std, 2 / 3)) if len(std) > 2 else float(std.max() if len(std) else 0.0)
    eps = 0.02 * max(abs(best), 1e-9)
    rows = []
    for i in range(len(batch)):
        m, sd = float(mean[i]), float(std[i])
        gain = m - best
        if gain > eps:
            mode, reason = "exploit", f"predicted high (+{gain:.3g} vs best)"
        elif sd >= thr:
            mode, reason = "explore", f"reduce model uncertainty here (±{sd:.3g})"
        else:
            mode, reason = "explore", f"diversifies the batch (predicted {m:.3g}, ±{sd:.3g})"
        row = {"pred": round(m, 3), "std": round(sd, 3), "mode": mode, "reason": reason}
        row["vals"] = (np.round(batch[i], 3).tolist() if cols is None
                       else [round(float(batch[i][j]), 3) for j in cols])
        if design is not None:
            row["recipe"] = dict(zip(design.names, design.decode_row(batch[i])))
        if p_feasible is not None:
            pf = float(p_feasible[i])
            row["p_feasible"] = None if pf != pf else round(pf, 3)  # NaN -> null (ungated)
        if constraint is not None:
            cname, cmean, cstd = constraint
            row[f"pred_{cname}"] = {
                "mean": round(float(cmean[i]), 3),
                "std": round(float(cstd[i]), 3),
            }
        rows.append(row)
    return rows


def _numeric_cols(df: pd.DataFrame) -> list:
    # Numeric if >=80% of NON-BLANK cells parse as numbers. Blanks are treated as
    # "absent" (filled with 0 later), so a sparse component column still counts.
    out = []
    for c in df.columns:
        s = df[c]
        nonblank = s.notna() & (s.astype(str).str.strip() != "")
        if nonblank.sum() < 3:
            continue
        if pd.to_numeric(s[nonblank], errors="coerce").notna().mean() >= 0.8:
            out.append(c)
    return out


# Deterministic seed for the analyze path. The same upload -> the same GP fit and
# the same proposed batch, which matters for client reproducibility and audit.
ANALYZE_SEED = 1234

# Driver-panel policy. Named here so the numbers can be argued with in one place.
DRIVER_FDR_Q = 0.05
# Benjamini-Hochberg false-discovery rate for the driver panel. 0.05 keeps the
# expected proportion of false findings among reported drivers at 1 in 20, which
# is the right error rate when a scientist will act on several of them.

DRIVER_TOP_K = 8
# How many drivers are shipped after ranking by |rho|. The cut is REPORTED
# (`driver_selection` below) rather than silent, because selecting the strongest
# of many tested features is itself a statistical act the client needs to know
# happened.

CONFORMAL_ALPHA = 0.1
# Miscoverage for the conformal band, so the response can state its own coverage
# (1 - alpha = 90%) instead of leaving every consumer to hardcode a number. The
# frontend had drifted to labelling this same band "95%" on one surface and
# "90%" on another; a stated coverage that is wrong on screen is an overclaim,
# not a hedge, so the engine now ships the number it actually computed.

CV_N_REPEATS = 2
# How many partitions `grouped_cv_report` draws for the reliability CI. On a
# continuous target the grouped CV takes the unshuffled GroupKFold branch, so
# a SINGLE partition's group-bootstrap CI only ever covers group-resampling
# variance, not how much the estimate moves under a different, equally valid
# partition - on the real media DoE the pooled Spearman moved 0.44 to 0.71
# across n_splits 3 to 8 alone. Repeated CV over shuffled partitions is what
# makes the CI honest about that.
#
# Repeat 0 is always the free unshuffled partition (the point estimate's
# anchor, already paid for today), so each ADDITIONAL repeat costs one more
# round of n_splits GP fits - not cheap at this module's per-fit cost. Timed
# end-to-end on a 60-row, 13-column sheet on the dev machine this was tuned
# on: n_repeats=1 (today's cost) ~5.3-6.1s, n_repeats=2 ~7.3-8.6s,
# n_repeats=3 ~10.3-10.6s - already over the ~10s budget before counting any
# slower CI hardware or a larger upload. 2 is the largest value that stayed
# comfortably under budget with margin to spare: one extra shuffled partition
# beyond today's, which is enough to turn "the CI is conditional on one fixed
# split" into "the CI has seen the estimate move at least once," without
# betting the request's latency on it.

RELIABILITY_SPEARMAN_FLOOR = 0.20
# The one out-of-fold Spearman floor this module uses, for the reliability verdict
# AND for gating the GP shape report. It was previously a bare 0.20 literal in two
# places here; a single name means the verdict and the shapes can never be held to
# different bars, and a future change to the bar cannot move one without the other.

LOGO_MAX_GROUPS = 12
# Leave-one-group-out (`kalos.core.evaluation.logo_report`) costs one Surrogate GP
# fit PER GROUP - unlike `grouped_cv_report`'s fixed `n_splits` folds, a LOGO run
# grows with however many groups the sheet declares, so an unbounded sheet could
# turn one request into dozens of GP fits. Timed end-to-end on this dev machine,
# a 60-row, 2-continuous-feature sheet (12 recipes x 5 reps): without a declared
# group column (`cv_logo` skipped) `_analyze` ran 1.46-2.05s across 4 runs
# (min 1.46s); with a declared 12-group column - LOGO_MAX_GROUPS exactly, the
# most expensive case this cap still allows - it ran 2.02-2.59s (min 2.02s), a
# delta of roughly 0.5-0.6s for 12 extra GP fits on this machine, each fit on a
# fold nearly as large as the full sheet (leave-ONE-group-out trains on
# n_groups-1 of n_groups groups, a much bigger training split per fit than the
# 5-fold CV's 4/5). That stays comfortably under a ~12s budget even with
# slower CI hardware or a larger upload; a sheet with more groups than this cap
# skips LOGO instead of letting the cost scale unbounded with group count, and
# says so via `cv_logo`'s `reason` field below rather than staying silent about
# why the number is missing.

CV_TOPK_K = 5
# How many of the top-ranked recipes `cv_topk` (kalos.core.evaluation.top_k_overlap)
# scores. 5 mirrors `q=5`, the size of the batch this module actually proposes
# below (`propose(..., q=5, ...)`) - the top-k question this metric answers
# ("of the ones I'd actually advance, how many does the model get right") is
# most honest when k matches the real decision size, not an arbitrary round number.

GROUP_MEAN_BASELINE_MIN_GROUPS = 5
# `cv_group_mean_baseline` (kalos.core.evaluation.group_mean_baseline_spearman)
# needs enough REPLICATED groups for the leave-one-row-out group mean to be a
# real prediction rather than noise from 1-2 groups' worth of pairs. 5 is a
# convention, chosen for the same reason `MIN_REPLICATED_FOR_HETERO` (see
# kalos.core.replicates) picks a small integer over a formula: below it, a
# Spearman computed against so few independent group means would read as
# confident nonsense, so the baseline declines rather than reports one.

GATE_MIN_N_INFEASIBLE = 5
# PART 1 (feasibility-gated proposals) POLICY constant: the sheet must have at
# least this many non-producer rows before the acquisition is gated by
# P(feasible) at all - "zero-inflated enough" to be the pathology BENCHMARK.md
# documents, not a producer-only sheet with a couple of stray zeros.
# `FeasibilityClassifier.fit` already refuses to fit sklearn below 3
# minority-class examples (its own cold-start guard), but 3-4 examples give an
# unstable decision boundary with no room to spare; 5 is the smallest count
# that lets a 5-fold CV split (the fold count `feasibility_cv_report` below
# uses) reserve at least one non-producer per held-out fold, which is the
# regime the classifier and its CV AUC below both implicitly assume.

# No separate GATE_MIN_FEASIBILITY_AUC constant is defined here. PART 1's gate
# policy reuses `kalos.core.gates.GatesConfig.min_feasibility_auc` (0.65)
# directly at the point it is used below, rather than duplicating that number
# under a second name: it is already the bar this codebase uses to decide "is
# this classifier's feasibility ranking trustworthy" for the promotion
# verdict, and a second, separately-tunable ~0.6 floor living next to it would
# fracture the single-number-single-meaning discipline
# `RELIABILITY_SPEARMAN_FLOOR`'s comment above states for the reliability
# verdict. A classifier too weak to trust for promotion is, by construction,
# too weak to trust for gating acquisition - not by coincidence.

CONSTRAINT_MIN_ROWS = 6
# PART 2 (constrained proposals) POLICY constant: the constraint column needs
# at least this many present values before a second `Surrogate` is fit on it.
# Mirrors the >=6 row floor `_analyze` already requires to fit the TARGET
# surrogate (see the `len(y) < 6` check above) - a GP fit on fewer points than
# that is not a floor this module considers meaningful for the target, and the
# constraint model is held to the identical bar rather than a laxer one.


def _seed_everything(seed: int = ANALYZE_SEED) -> None:
    """Seed torch + numpy so one upload yields one deterministic set of proposals."""
    import torch  # local: deferred, see module-level note above

    torch.manual_seed(seed)
    np.random.seed(seed)


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Suffix duplicate column labels (`X`, `X.1`, ...) so every column is a Series.

    A run sheet with a repeated header would otherwise make `df[label]` return a
    2-D frame and break the analysis. pandas already does this on CSV read; we do
    it here too so a directly-built frame (or an xlsx with duplicate headers) is
    handled identically, and the duplicate stays visible in the provenance report.
    """
    if not df.columns.duplicated().any():
        return df
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for col in df.columns:
        key = str(col)
        if key in seen:
            seen[key] += 1
            new_cols.append(f"{key}.{seen[key]}")
        else:
            seen[key] = 0
            new_cols.append(key)
    out = df.copy()
    out.columns = new_cols
    return out


def _resolve_columns(
    df: pd.DataFrame,
    num: list,
    target: str | None,
    roles: ColumnRoles | None,
    profile: DomainProfile,
) -> tuple[str, list, list, str | None, set]:
    """Decide the column roles for one run sheet.

    Returns `(target, cont_feats, cat_feats, gcol, declared)`, where the two
    feature lists are in sheet-column order and `declared` is the set of columns
    whose role came from an explicit `ColumnRoles` (empty in inference mode).

    Two modes:
      - `roles` given: use the declared target, features, ids, group, and
        categoricals. When `roles.features` is empty the continuous features are
        still inferred from the `profile`, honoring the declared ids/categoricals.
      - `roles` None (default): infer everything from `profile` (with the
        bioprocess profile this reproduces the original behavior exactly).
    """
    num_set = set(num)
    if roles is not None:
        if roles.target not in df.columns:
            raise ValueError(f"declared target {roles.target!r} is not a column")
        # A declared feature / categorical / group / id whose name does not match
        # a header is a silent data-loss trap (typo, case mismatch): the real
        # column would never be excluded/used and provenance would not flag it.
        # Fail loudly so the caller is never told a declared column was honored
        # when it was not - ids included (a typo'd id leaves the real id in as a
        # feature otherwise).
        declared_names = [*roles.features, *roles.categoricals, *roles.ids]
        if roles.groups:
            declared_names.append(roles.groups)
        unknown = sorted({c for c in declared_names if c not in df.columns})
        if unknown:
            raise ValueError(f"declared columns not found in the sheet: {unknown}")
        # A column cannot be both the target and an id/categorical - a
        # self-contradictory schema is a caller error, not something to resolve
        # silently by dropping one role.
        if roles.target in set(roles.ids) or roles.target in set(roles.categoricals):
            raise ValueError(
                f"declared target {roles.target!r} cannot also be declared an id or categorical"
            )
        target = roles.target
        cat_feats = [c for c in roles.categoricals if c in df.columns and c != target]
        cat_set = set(cat_feats)
        ids = set(roles.ids)
        # `declared` drives the provenance `source` field, so declared ids belong
        # in it too (their exclusion was the caller's instruction, not an inference).
        declared: set = {target, *cat_feats, *ids}
        if roles.features:
            declared.update(roles.features)
            # Declared continuous features must clear the same >=80% numeric-parse
            # gate inference mode uses (`num`). A text column declared as a feature
            # (the caller likely meant categorical) is left out here and reported
            # as `dropped_non_numeric` by provenance, not silently zero-filled in.
            cont_feats = [
                c for c in roles.features
                if c != target and c not in ids and c not in cat_set and c in num_set
            ]
        else:
            cont_feats = [
                c for c in num
                if c != target and c not in cat_set and c not in ids
                and not identifier_pattern(profile.id_hint.pattern).match(str(c).strip())
                and not profile.outcome_hint.search(str(c))
            ]
        gcol = roles.groups if (roles.groups and roles.groups in df.columns) else None
        if gcol is None:
            gcol = next((c for c in df.columns if profile.group_hint.search(str(c))), None)
        # normalize both feature lists to sheet-column order
        cont_feats = [c for c in df.columns if c in set(cont_feats)]
        cat_feats = [c for c in df.columns if c in cat_set]
        return target, cont_feats, cat_feats, gcol, declared

    # inference mode (default profile = bioprocess): unchanged legacy behavior
    outcomes = [c for c in num if profile.outcome_hint.search(str(c))]
    if target is None or target not in df.columns:
        target = next(
            (c for c in outcomes if profile.target_pref.search(str(c))),
            outcomes[0] if outcomes else num[-1],
        )
    cont_feats = [
        c for c in num
        if c != target
        and not identifier_pattern(profile.id_hint.pattern).match(str(c).strip())
        and not profile.outcome_hint.search(str(c))
    ]
    gcol = next((c for c in df.columns if profile.group_hint.search(str(c))), None)
    return target, cont_feats, [], gcol, set()


def _num(x: object, nd: int = 4) -> float | None:
    """Round to `nd` places, or None for anything that is not a finite number.
    NaN and inf both become None so the response stays strict-JSON safe.

    Module-level (not local to `_noise_block`) so `_alternative_scale_block`
    can round `icc_log`/`log_offset` through the exact same function `_noise_block`
    uses for the raw `icc_raw`/`icc_log`/`log_offset` in `noise.scale` - the two
    blocks report overlapping numbers from the same `heteroscedasticity_report`
    call and must never disagree because of a rounding difference between two
    copies of the same helper."""
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return None
    if not np.isfinite(float(x)):
        return None
    return round(float(x), nd)


def _noise_block(
    nr: dict,
    best_single: float,
    best_reproducible: float | None,
    replicate_aware: bool,
    het: dict | None = None,
) -> dict:
    """The `noise` report for the analysis result: replicate structure + the
    honest signal-to-noise picture (BENCHMARK.md, "the real lever is assay noise").

    `icc` is the intraclass correlation - the fraction of titer variance that is
    real recipe-to-recipe signal rather than assay noise; a low ICC means most of
    the spread is noise and single-measurement "bests" are largely luck.
    `best_single` is the best individual measurement (rewards noise spikes);
    `best_reproducible` is the best replicate-averaged recipe - the titer a client
    would actually ship. `replicate_aware` is True when the proposed batch was fit
    on that reproducible objective with the measured noise floor fed to the GP.
    NaN stats (too few recipes / no replicates to estimate them) serialize as null.
    """
    noise_var = nr["noise_var"]
    signal_var = nr["signal_var"]
    return {
        "n_recipes": int(nr["n_recipes"]),
        "n_replicated": int(nr["n_replicated"]),
        "replicate_aware": bool(replicate_aware),
        "icc": _num(nr["icc"], 3),
        "noise_sd": _num(np.sqrt(noise_var) if np.isfinite(noise_var) else None),
        "signal_sd": _num(np.sqrt(signal_var) if np.isfinite(signal_var) else None),
        "best_single": _num(best_single),
        "best_reproducible": _num(best_reproducible),
        # Whether the homoscedastic assumption behind `noise_sd` actually holds on
        # this sheet. A DIAGNOSTIC: nothing above is transformed, because changing
        # the target's scale would change every number in this response. When
        # `suggests_transform` is true, `icc` here is understating real signal.
        "scale": None
        if het is None
        else {
            "variance_mean_rho": _num(het.get("variance_mean_rho"), 3),
            "homoscedastic": het.get("homoscedastic"),
            "icc_raw": _num(het.get("icc_raw"), 3),
            "icc_log": _num(het.get("icc_log"), 3),
            "icc_gain": _num(het.get("icc_gain"), 3),
            "log_offset": _num(het.get("log_offset"), 6),
            "suggests_transform": bool(het.get("suggests_transform")),
            "reason": het.get("reason"),
        },
    }


# `alternative_scale` is a NULL-with-reason dict when not suggested (same
# pattern as `cv_logo` below: every key present, values `None`, `reason` states
# why), so a consumer never has to special-case "the key is missing."
_ALT_SCALE_NULL_KEYS = (
    "scale", "offset", "cv_spearman", "cv_ci95", "calibration", "icc", "interpretation",
)

_ALT_SCALE_INTERPRETATION = (
    "The log scale fits materially better here, which means the raw-scale assay "
    "noise floor is partly a scale artifact rather than uniform measurement "
    "noise, not that the underlying process changed. Proposals and every other "
    "number in this response remain on the raw scale; this block is "
    "evaluation-only."
)


def _alternative_scale_block(
    het: dict,
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups,
    bounds,
    cat_dims,
    grouped_cv_report: Callable[..., dict],
    interval_calibration: Callable[..., dict],
) -> dict:
    """The `alternative_scale` report: an EXPLICIT, LABELED second evaluation
    pass on the log scale, run only when `heteroscedasticity_report` (`het`,
    already computed by the caller and reported in `noise.scale`) says a log
    transform would materially raise the ICC (`suggests_transform`).

    WHY THIS EXISTS. `heteroscedasticity_report` diagnoses the scale problem
    every analysis, `_noise_block` reports the diagnosis, and the engine then
    went on to fit the PROPOSAL surrogate on the raw scale regardless - the
    standing design decision (see `heteroscedasticity_report`'s docstring) is
    that the target's scale is never changed SILENTLY, because every reported
    number lives on it. Diagnosing and then quietly modeling on the wrong scale
    anyway is worse than either alone: it is the gap this block closes, by
    running the alternative openly instead of switching to it in secret.

    OFFSET SEMANTICS - READ BEFORE CHANGING. `het["log_offset"]` is the exact
    `c` `heteroscedasticity_report` used to compute `het["icc_log"]` there, via
    `np.log(y + c)` (see that function's docstring). It is NOT the argument to
    literal `np.log1p`, i.e. `log(1 + y + c)` - that transform was the first
    thing tried in `heteroscedasticity_report`'s own history and was WRONG at
    titer's scale (~0.005-0.02): the `+1` dominates the tiny offset, the
    transform is nearly the identity, and it moved the ICC by only 0.002 on a
    sheet with a real variance-mean coupling. `"log1p"` in this module's (and
    that function's) naming is an informal label for the transform FAMILY - log
    of the value plus a small, data-scaled offset so zeros survive - not a
    literal call to `np.log1p`. Fitting on `np.log1p(y + c)` here would
    silently reproduce that exact historical bug AND desynchronize this
    block's `icc` from `het["icc_log"]`, which defeats the consistency this
    block exists to guarantee. So the transform below is `np.log(y + c)`,
    byte-for-byte the same call `heteroscedasticity_report` made.

    WHY REFIT RATHER THAN RE-RANK. `cv_spearman` is rank-based: ranking the
    SAME held-out predictions after a log transform would return an identical
    number, which would be free but would prove nothing about the log scale
    specifically. The point of this block is that a NEW GP is fit and
    cross-validated on `log(y + c)` - different lengthscales, different
    inferred noise, a different posterior - so `cv_spearman` here can
    genuinely differ from the raw `cv_spearman`, and `calibration` here answers
    "are the log-scale bands honest" independently of whether the raw-scale
    bands were. Both are real information about which scale the process lives
    on; do not "optimize away" this second fit as redundant with the first.

    COST. `n_repeats=1` (not `CV_N_REPEATS`): this is a diagnostic comparison,
    not the headline reliability number, so the partition-variance treatment
    the headline `cv_spearman`/`cv_ci95` pays double for is not duplicated here.
    Runs at all only when `het["suggests_transform"]` is True, so an
    unaffected sheet pays nothing extra for this block.

    SCOPE. Evaluation-only: this function never touches `X`/`y`, the proposal
    path, the feasibility gate, or the constraint path, and returns a plain
    dict, not a fitted model. `calibration`'s `ece`/`z_std` are computed on
    LOG-SCALE residuals - comparable to the raw block's calibration in
    STRUCTURE (is coverage honest at each nominal level) but not in absolute
    magnitude, since the two live in different units.

    Returns the shape above when suggested; the same keys with `None` values
    plus a stated `reason` (the `cv_logo`-style null pattern) otherwise."""
    null_block: dict[str, object] = dict.fromkeys(_ALT_SCALE_NULL_KEYS)
    null_block["reason"] = het.get("reason") or "log transform not suggested on this sheet"
    if not het.get("suggests_transform") or het.get("log_offset") is None:
        return null_block

    c = float(het["log_offset"])
    # Byte-for-byte the same transform `heteroscedasticity_report` used for
    # `icc_log` - see the offset-semantics note above for why this must not be
    # `np.log1p(y + c)`.
    y_log = np.log(np.asarray(y, dtype=float) + c)
    rep_log = grouped_cv_report(
        X, y_log, groups=groups, n_splits=5, bounds=bounds, cat_dims=cat_dims, n_repeats=1,
    )
    rho_log = rep_log["spearman"]
    ci95_log = (
        None
        if rho_log != rho_log
        else [round(rep_log["ci95"][0], 3), round(rep_log["ci95"][1], 3)]
    )
    calibration_log = interval_calibration(rep_log["oof_actual"], rep_log["oof_pred"], rep_log["oof_std"])
    return {
        "scale": "log1p",
        "offset": _num(c, 6),
        "cv_spearman": None if rho_log != rho_log else round(float(rho_log), 3),
        "cv_ci95": ci95_log,
        # LOG-SCALE OOF: `ece`/`z_std` inside this dict are in log units, NOT
        # comparable in magnitude to the raw block's `reliability.calibration` -
        # only to its structure (is coverage honest at each nominal level).
        "calibration": calibration_log,
        "icc": _num(het.get("icc_log"), 3),
        "reason": het.get("reason"),
        "interpretation": _ALT_SCALE_INTERPRETATION,
    }


def _physical_range(name: str, col: np.ndarray) -> tuple[float, float, int]:
    """Observed range of `col`, computed over PHYSICALLY POSSIBLE values only.

    The design box handed to the optimizer is the observed [min, max] of each
    feature, so a single impossible cell silently widens the search space to
    include impossible recipes. This is not hypothetical: one `-999` "sensor
    offline" sentinel in a temperature column stretched the box to [-999, 37]
    and the optimizer duly proposed a bioreactor run at -422 C, below absolute
    zero, with a confidence interval attached. A recommendation engine must not
    be able to emit that, in any mode, whatever the validation report says.

    So values outside the column's HARD physical bounds
    (`kalos.validation.bounds`) are excluded from the min/max, and the count of
    excluded cells is returned so the caller can report the narrowing rather
    than perform it silently. Note this only shapes the SEARCH SPACE - the rows
    themselves still reach the surrogate, and the validation gate is what tells
    the client their data has impossible values in it.

    A column whose header maps to no known dimension is returned unchanged
    (`infer_dimension` is deliberately silent rather than guessing), as is a
    column with no physically valid values at all - clamping the latter would
    invent a range from nothing, which is a worse lie than reporting the real
    one alongside an error-severity finding.
    """
    lo_obs, hi_obs = float(col.min()), float(col.max())
    dim = infer_dimension(name)
    if dim is None:
        return lo_obs, hi_obs, 0
    b = DIMENSION_BOUNDS[dim]
    valid = col[(col >= b.hard_lo) & (col <= b.hard_hi)]
    n_excluded = int(col.size - valid.size)
    if n_excluded == 0 or valid.size == 0:
        return lo_obs, hi_obs, 0 if valid.size else n_excluded
    lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        # The valid cells are all one value: a zero-width box would collapse the
        # dimension. Keep the real observed span rather than emit a degenerate
        # design; the validation report carries the impossible-value error.
        return lo_obs, hi_obs, n_excluded
    return lo, hi, n_excluded


def _analyze(
    df: pd.DataFrame,
    target: str | None = None,
    *,
    anonymize: bool = False,
    roles: ColumnRoles | None = None,
    profile: DomainProfile = BIOPROCESS_PROFILE,
    pending: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
    constraint: Mapping[str, Any] | None = None,
) -> dict:
    """Run the engine on an arbitrary run sheet: pick the target (the value to
    maximize), use the process INPUTS as features (other measured outputs are
    excluded to avoid leakage), then honest grouped-CV, signed drivers, and a
    proposed next batch.

    Column roles come from an explicit `roles` schema when given, else are
    inferred from `profile` (defaulting to the bioprocess profile, which
    reproduces the original behavior). `roles.categoricals` marks feature columns
    whose values are unordered labels; the engine fits a mixed GP over them and
    proposals decode back to labels.

    `pending` carries the recipes that are already RUNNING but not yet measured
    (the campaign's awaiting runs). They have no target value, so they cannot
    join the fit; they are encoded into the same design and handed to the
    acquisition as in-flight points, so the proposed batch does not spend budget
    re-running an experiment currently in the incubator. Omit it (the default)
    and the proposal path is unchanged.

    `constraint` requests a constrained single-objective proposal: maximize
    `target` subject to a SECOND outcome column clearing a floor, e.g.
    `{"column": "purity_pct", "floor": 95.0}`. This is per-run intent, not a
    declared schema role (`ColumnRoles` is deliberately not the place - a run
    can ask for a constraint on one upload and not the next, on the same
    sheet). `None` (the default) leaves the proposal unconstrained. Whether
    the constraint could actually be applied (column present, enough rows,
    the constraint model fit) is reported in the response's `constraint`
    block regardless of the outcome - a constraint that could not be honored
    on this sheet falls back to an unconstrained proposal rather than
    raising; see PART 2 below for the policy. Wiring this parameter through
    the portal's HTTP surface is out of scope here.

    Deterministic: seeds torch + numpy up front so the same sheet gives the same
    proposals. Returns a per-column `provenance` report (what was kept/dropped and
    why) plus `seed`, `timestamp`, and `engine_version` for audit. When
    `anonymize` is True, identifier-type column names are replaced with stable
    pseudonyms in the response (the owner UI keeps real names when False)."""
    # Deferred: these transitively import torch/botorch/gpytorch (see the
    # module-level note above). This is the one place in this module that
    # actually needs them, so this is where the torch tax is paid.
    # `kalos.core.gates` and `kalos.core.feasibility` do not themselves import
    # torch, but they are only ever used from inside this function, so they are
    # kept in this same deferred block rather than opening a second import site.
    from kalos.core.evaluation import (
        group_mean_baseline_spearman,
        grouped_cv_report,
        interval_calibration,
        logo_report,
        producer_only_spearman,
        top_k_overlap,
    )
    from kalos.core.feasibility import FeasibilityClassifier, feasibility_cv_report, feasible_labels
    from kalos.core.gates import GatesConfig, check_gates
    from kalos.core.optimize import MAX_MIXED_COMBOS, propose
    from kalos.core.surrogate import FitError, Surrogate

    _seed_everything()
    df = _dedupe_columns(df.dropna(axis=1, how="all"))

    # The validation gate runs FIRST, on the sheet as uploaded, before any
    # column is typed or dropped - it has to see the raw cells to catch a mixed
    # g/L-and-mg/mL column or a "34.6 C" string, both of which are invisible
    # once `_numeric_cols` has already discarded them as non-numeric.
    mode = validation_mode()
    validation = validate_frame(df, target=target, profile=profile, mode=mode)
    if mode == "strict" and validation.status == "fail":
        raise UploadRejected(_ERR_VALIDATION_FAILED)
    # Convert every single-unit column to its base unit before feature
    # selection. This is not merely cosmetic: a temperature column written
    # "34.6 C" fails the >=80% numeric-parse test and is silently DROPPED as
    # non-numeric today, so converting first recovers real process features
    # that were being thrown away. Only units the registry actually knows are
    # converted (see `check_units_consistency`).
    if validation.conversions:
        df = apply_unit_conversions(df, list(validation.conversions))

    num = _numeric_cols(df)
    if not num:
        raise ValueError("no numeric columns found")
    target, cont_feats, cat_feats, gcol, declared = _resolve_columns(
        df, num, target, roles, profile
    )
    # PART 2 leakage guard: the constraint column (e.g. "purity_pct") is
    # normally already excluded from the feature set by `profile.outcome_hint`
    # (the same mechanism that excludes the target itself), same as the
    # comment at `_resolve_columns` describes. But an explicit `roles.features`
    # list bypasses that hint entirely (see `_resolve_columns`'s declared-mode
    # branch), so a caller who both declares the constraint column as a
    # feature AND asks to constrain on it would otherwise leak that outcome
    # into the model it is meant to be held out from. Stripped here
    # unconditionally so the leakage rule holds regardless of how the column
    # got into the feature list.
    if constraint is not None:
        _constraint_col_name = str(constraint["column"])
        cont_feats = [c for c in cont_feats if str(c) != _constraint_col_name]
        cat_feats = [c for c in cat_feats if str(c) != _constraint_col_name]
    if roles is None:
        outcomes = [c for c in num if profile.outcome_hint.search(str(c))]
        candidate_targets = outcomes or [target]
    else:
        candidate_targets = [target]

    # Keep only continuous features that actually vary (a constant column carries
    # no signal and would collapse its normalization range). Matches the legacy
    # full-column variance filter.
    varying = []
    for c in cont_feats:
        col = pd.to_numeric(df[c], errors="coerce")
        if col.std(skipna=True) and col.std() > 1e-9:
            varying.append(c)
    cont_feats = varying
    if not cont_feats and not cat_feats:
        raise ValueError("no varying process-input columns found (only outputs/ids?)")

    y_all = pd.to_numeric(df[target], errors="coerce")
    keep = y_all.notna()
    Xc_raw = df.loc[keep, cont_feats].apply(pd.to_numeric, errors="coerce")  # NaN preserved (for grouping)
    Xc_zf = Xc_raw.fillna(0.0)                                               # zero-filled (for the GP + box)
    y = y_all[keep].to_numpy(float)
    if len(y) < 6:
        raise ValueError(f"need at least 6 rows with a numeric {target!r}; got {len(y)}")
    # Cap the rows the O(n^2) exact GP is fit on, separately from the raw-upload
    # cap. Reject rather than subsample: silent subsampling would be invisible,
    # non-deterministic data loss and contradict the reproducibility contract.
    if len(y) > MAX_FIT_ROWS:
        raise UploadRejected(_ERR_TOO_MANY_FIT_ROWS)

    # The design box is built from the target-present (fitted) rows, so the honest
    # varying-feature check must be recomputed on THOSE rows, not the full column. A
    # feature that varies over the whole sheet but is constant on the fitted rows
    # would otherwise collapse its bound to zero width silently. Drop such features
    # and record them so provenance flags them instead of misleading the client.
    constant_on_fitted: list = []
    if cont_feats:
        fitted_range = Xc_zf.max(axis=0) - Xc_zf.min(axis=0)
        constant_on_fitted = [c for c in cont_feats if float(fitted_range[c]) <= 1e-9]
        if constant_on_fitted:
            cont_feats = [c for c in cont_feats if c not in set(constant_on_fitted)]
            Xc_raw = Xc_raw.drop(columns=constant_on_fitted)
            Xc_zf = Xc_zf.drop(columns=constant_on_fitted)

    # Categorical features: derive ordered levels from the fitted rows. A blank
    # cell is NOT a proposable level - it means the categorical is unknown for that
    # row - so blanks are excluded from the levels here, and rows with a blank in
    # any kept categorical are dropped from the fit below (you cannot model, or
    # recommend, a recipe whose categorical component is unknown). Drop any
    # categorical that carries a single level too (no choice to optimize).
    kept_cats: list = []
    constant_cats: list = []
    cat_dims_map: dict = {}
    for c in cat_feats:
        labels = df.loc[keep, c].fillna("").astype(str).str.strip()
        levels = tuple(sorted({v for v in labels.tolist() if v != ""}))
        if len(levels) < 2:
            constant_cats.append(c)
            continue
        kept_cats.append(c)
        cat_dims_map[c] = levels

    if not cont_feats and not kept_cats:
        raise ValueError("no varying process-input columns on the target-present rows")

    # Drop fitted rows whose value for any kept categorical is blank/unknown, so a
    # missing categorical can never poison the GP (as a NaN code) or surface as a
    # proposed recipe with an empty categorical. `keep_index` is the original-frame
    # index of the surviving rows, used for every per-row lookup below.
    keep_index = df.index[keep.to_numpy()]
    complete = np.ones(len(y), dtype=bool)
    for c in kept_cats:
        labels = df.loc[keep_index, c].fillna("").astype(str).str.strip()
        complete &= labels.isin(cat_dims_map[c]).to_numpy()
    n_dropped_incomplete = int((~complete).sum())
    if n_dropped_incomplete:
        if int(complete.sum()) < 6:
            raise ValueError(
                f"only {int(complete.sum())} rows have all declared categoricals present; "
                "need at least 6 to fit"
            )
        pos = np.flatnonzero(complete)
        y = y[complete]
        Xc_raw = Xc_raw.iloc[pos]
        Xc_zf = Xc_zf.iloc[pos]
        keep_index = keep_index[pos]

    # Assemble the design space with continuous dims first, categorical dims last.
    # BoTorch's mixed models accept arbitrary cat_dims indices, so this order is not
    # required; it just keeps X and the continuous driver slice `X[:, :n_cont]` simple.
    dims: list[Dimension] = []
    cont_arr = Xc_zf.to_numpy(float) if cont_feats else np.empty((len(y), 0))
    box_exclusions: list[dict] = []
    for j, c in enumerate(cont_feats):
        col = cont_arr[:, j]
        lower, upper, n_excluded = _physical_range(str(c), col)
        if n_excluded:
            box_exclusions.append(
                {"column": str(c), "n_excluded": n_excluded, "lower": lower, "upper": upper}
            )
        dims.append(Dimension(str(c), "continuous", lower=lower, upper=upper))
    for c in kept_cats:
        dims.append(Dimension(str(c), "categorical", levels=cat_dims_map[c]))
    design = DesignSpace(tuple(dims))

    code_maps = {c: {lvl: i for i, lvl in enumerate(cat_dims_map[c])} for c in kept_cats}
    if kept_cats:
        cat_cols = [
            df.loc[keep_index, c].fillna("").astype(str).str.strip().map(code_maps[c]).to_numpy(float)
            for c in kept_cats
        ]
        X = np.column_stack([cont_arr, *cat_cols]) if cont_feats else np.column_stack(cat_cols)
    else:
        X = cont_arr
    bounds = design.bounds()
    cat_dims = design.cat_dims or None
    cat_cardinalities = design.cat_cardinalities or None

    # PART 2: constrained single-objective proposals. Fit a SECOND `Surrogate`
    # over the constraint column, on the SAME design/bounds/cat_dims the target
    # surrogate uses, so `propose` can hand both to a `ModelListGP` (see
    # `kalos.core.optimize.propose`'s `constraint_surrogate` argument). This
    # never crashes and never blocks the (unconstrained) proposal: a missing
    # column, too few rows, or a `FitError` from an ill-conditioned constraint
    # fit all fall back to reporting why, not raising.
    #
    # Fit on rows where the constraint value is PRESENT, which is not
    # necessarily every row `X`/`y` cover - the constraint column can be
    # sparser than the target (e.g. purity was only measured on a subset of
    # runs). `X`'s rows are 1:1 with `keep_index` at this point (both were
    # finalized before `X` was assembled above), so a boolean mask read off
    # `df.loc[keep_index, constraint_col]` lines up with `X`'s rows directly.
    constraint_surrogate: "Surrogate | None" = None
    constraint_col: str | None = None
    constraint_floor_val: float | None = None
    constraint_applied = False
    if constraint is None:
        constraint_report: dict[str, Any] = {
            "column": None, "floor": None, "n_rows_with_value": 0,
            "applied": False, "reason": "no constraint requested",
        }
    else:
        constraint_col = str(constraint["column"])
        constraint_floor_val = float(constraint["floor"])
        if constraint_col not in df.columns:
            constraint_report = {
                "column": constraint_col, "floor": constraint_floor_val,
                "n_rows_with_value": 0, "applied": False,
                "reason": f"constraint column {constraint_col!r} not found in the uploaded sheet",
            }
        else:
            cy_all = pd.to_numeric(df.loc[keep_index, constraint_col], errors="coerce")
            c_mask = cy_all.notna().to_numpy()
            n_rows_with_value = int(c_mask.sum())
            if n_rows_with_value < CONSTRAINT_MIN_ROWS:
                constraint_report = {
                    "column": constraint_col, "floor": constraint_floor_val,
                    "n_rows_with_value": n_rows_with_value, "applied": False,
                    "reason": (
                        f"only {n_rows_with_value} rows have a numeric "
                        f"{constraint_col!r} value; need at least {CONSTRAINT_MIN_ROWS} "
                        "to fit a constraint model"
                    ),
                }
            else:
                Xc = X[c_mask]
                yc = cy_all.to_numpy(float)[c_mask]
                try:
                    constraint_surrogate = Surrogate().fit(Xc, yc, bounds=bounds, cat_dims=cat_dims)
                    constraint_applied = True
                    constraint_report = {
                        "column": constraint_col, "floor": constraint_floor_val,
                        "n_rows_with_value": n_rows_with_value, "applied": True, "reason": None,
                    }
                except FitError as exc:
                    constraint_report = {
                        "column": constraint_col, "floor": constraint_floor_val,
                        "n_rows_with_value": n_rows_with_value, "applied": False,
                        "reason": f"constraint model could not be fit: {exc}",
                    }

    # RECIPE identity, on the RAW values (NaN preserved) so rows missing
    # different components are never merged by the zero-fill. Built from the one
    # leakage-checked, deterministic grouper; categorical labels join the key so
    # identical recipes stay one replicate group.
    #
    # This is computed ALWAYS and kept SEPARATE from the CV group below, because
    # the two answer different questions and conflating them is a bug in either
    # direction. A declared group column is a LEAKAGE BARRIER and is deliberately
    # coarser than a recipe - every run sharing a medium lot goes in one fold, but
    # those runs are not replicates of each other. Using it as a recipe key would
    # collapse genuinely different recipes into one and fabricate within-recipe
    # variance; using a recipe key as the CV group would let a lot straddle folds.
    recipe_gf = Xc_raw.copy()
    for c in kept_cats:
        recipe_gf[c] = df.loc[keep_index, c].fillna("").astype(str)
    recipe_key = row_hash_groups(recipe_gf)

    if gcol:
        # fillna("") before astype(str), matching every other cast of this
        # column family in this file (recipe_gf just above, and the label
        # casts below) - NOT redundant here. Under pandas 3.x's default
        # string-backed dtype, .astype(str) on a Series that mixes real
        # strings with NaN no longer stringifies the NaN (legacy object-dtype
        # behavior); it leaves it as a bare float, so `groups` silently
        # carries one float among strings. That is invisible until
        # np.unique's sort at line ~1200 (LOGO group counting) raises
        # `TypeError: '<' not supported between instances of 'float' and
        # 'str'` - found live via a campaign reanalyze, since a folded
        # campaign row has no value for a declared group column like
        # `medium_base` (it is not a proposable numeric feature).
        groups = df.loc[keep_index, gcol].fillna("").astype(str).tolist()
    else:
        # No declared barrier, so the recipe itself is the safest CV group: two
        # replicates of one recipe must never straddle a fold.
        groups = recipe_key

    # honest grouped cross-validation: pooled out-of-fold predictions + a
    # group-level bootstrap CI, all through the single leakage-checked splitter.
    rep = grouped_cv_report(
        X, y, groups=groups, n_splits=5, bounds=bounds, cat_dims=cat_dims, n_repeats=CV_N_REPEATS,
    )
    rho = rep["spearman"]
    # Repeat-0-only: the point estimate and the OOF pools below stay anchored to
    # the SAME unshuffled partition `grouped_cv_report` always uses for repeat 0,
    # so calibration/conformal/producer-only ranking all read off one consistent
    # set of predictions (see `grouped_cv_report`'s docstring for why pooling
    # across repeats here would double-count rows with correlated errors instead
    # of adding independent evidence). Only `cv_ci95` below draws on every
    # repeat's partition.
    oof_a, oof_p, oof_s = rep["oof_actual"], rep["oof_pred"], rep["oof_std"]

    # distribution-free +/- band from the pooled out-of-fold residuals (approximate
    # coverage under grouped CV). Honest alternative to the surrogate's own std,
    # which is often overconfident on small bioprocess datasets.
    resid = np.asarray(oof_a, float) - np.asarray(oof_p, float)
    conformal_q = round(q_from_residuals(resid, alpha=CONFORMAL_ALPHA), 4) if len(resid) else None

    # Honest reliability verdict: only what this path can actually assess. The
    # spearman floor mirrors GatesConfig.min_spearman (kalos/core/gates.py); we do
    # NOT assert the feasibility gate, which is not measured here.
    ci95 = None if rho != rho else [round(rep["ci95"][0], 3), round(rep["ci95"][1], 3)]

    # CALIBRATION. A model can rank held-out runs correctly and still state every
    # uncertainty at half its true width, and the scientist reading "4.2 +/- 0.3"
    # is acting on the 0.3. The held-out posterior sd needed to check that was
    # already computed inside the CV and then thrown away, so calibration was
    # reported as unmeasured for want of a value the engine had already paid for.
    # It is now kept and scored: `ece` is the mean gap between nominal and
    # empirical coverage across four central intervals, on the same 0-to-1 scale
    # `GatesConfig.max_ece` is written against, and `z_std` says which way a
    # miscalibrated model errs (above 1.0 = bands too narrow).
    #
    # Reported, NOT folded into `clears_floor`. Tightening the verdict changes
    # which uploads the API accepts, which is a product decision - the same line
    # `producer_clears_floor` sits on.
    calibration = interval_calibration(oof_a, oof_p, oof_s)

    # PRODUCER-ONLY ranking. The pooled `rho` above is taken over every held-out
    # row, producers and non-producers together, so a model can score well on it
    # by separating zeros from non-zeros - a feasibility classifier, not a ranking
    # of recipes. The client's question is "which of my producing recipes is
    # best", and BENCHMARK.md shows the two can diverge badly: on the real media
    # DoE the pooled score looked like 0.37-0.52 while feasibility was never the
    # bottleneck, so most of that agreement was the easy half of the problem.
    #
    # The threshold is 0.0, meaning "any non-zero measurement", which is a PROXY.
    # The assay LOD is the correct value - below it a reading is censored rather
    # than zero - and is not available until an SOP supplies it.
    prod = producer_only_spearman(oof_a, oof_p)
    reliability = {
        "spearman": None if rho != rho else round(rho, 3),
        "ci95": ci95,
        "spearman_floor": RELIABILITY_SPEARMAN_FLOOR,
        "clears_floor": bool(rho == rho and rho >= RELIABILITY_SPEARMAN_FLOOR),
        # Reported alongside, not folded into `clears_floor`: making the gate
        # stricter changes which uploads the API accepts, which is a product
        # decision rather than a bug fix. Surfaced so the divergence is visible
        # and the gate can be tightened deliberately.
        "producer_spearman": None if prod["spearman"] != prod["spearman"] else round(float(prod["spearman"]), 3),
        "producer_clears_floor": bool(
            prod["evaluable"] and float(prod["spearman"]) >= RELIABILITY_SPEARMAN_FLOOR
        ),
        "n_producers": int(prod["n_producers"]),
        "producer_threshold": float(prod["threshold"]),
        "ci_excludes_zero": bool(ci95 is not None and ci95[0] > 0),
        # Held-out coverage of the GP's own Gaussian bands. `available` is false
        # on a sheet with too few out-of-fold points for a coverage rate to mean
        # anything, and the matching "calibration" line stays in `unmodeled`.
        "calibration": calibration,
        "unmodeled": [
            "feasibility probability",
            *(
                []
                if calibration["available"]
                else ["calibration (ECE): " + str(calibration["reason"])]
            ),
            "scale-up transfer",
            *(
                []
                if prod["evaluable"]
                else ["producer ranking (too few producing runs to score it separately)"]
            ),
        ],
    }

    # PROMOTION VERDICT. `check_gates` (kalos/core/gates.py) has existed since
    # the lean-engine port but had ZERO callers on this path, and
    # `feasibility_cv_auc` had zero callers outside its own tests - the
    # fail-closed gate the codebase already built was never actually asked
    # anything. This wires it up, WITHOUT changing what the API accepts: the
    # verdict is REPORTED, never used to reject an upload. Gating uploads on a
    # promotion verdict is a product decision (same line `producer_clears_floor`
    # and `clears_floor` already sit on above) that this change does not make.
    #
    # feasibility_cv_report is cheap - a handful of sklearn LogisticRegression
    # fits under CV, no GP involved - unlike the grouped surrogate CV above, so
    # computing it here costs essentially nothing next to the GP fits this
    # function already pays for.
    feas = feasibility_cv_report(X, y, groups=recipe_key)

    # PART 1: feasibility-gated production proposals. `feas` above already
    # cross-validates the classifier for the promotion verdict below; this
    # section decides, from that same CV measurement plus a fresh full-data
    # fit, whether the PROPOSAL acquisition itself should be gated by
    # P(feasible) - a separate question from "is this classifier fit to
    # promote" (the two verdicts can legitimately disagree: a promotable
    # classifier on a producer-only sheet still should not gate, because there
    # is nothing zero-inflated to gate against).
    #
    # GATE POLICY - all three required, each independently reported (never
    # silent about why a run was not gated):
    #
    # (a) the classifier actually FIT on this sheet, not the cold-start
    #     fallback (`FeasibilityClassifier.fitted`, a clean public accessor -
    #     see its docstring) - gating on the fallback (a uniform P(feasible)=1)
    #     would multiply every candidate's EI by 1 and change nothing while
    #     still claiming to gate.
    # (b) the sheet is zero-inflated enough: `n_infeasible >=
    #     GATE_MIN_N_INFEASIBLE` (see that constant's comment above for the
    #     5-row justification).
    # (c) the CV feasibility AUC (`feas["auc"]`, reused rather than a second CV
    #     run) clears `GatesConfig().min_feasibility_auc` (0.65) - the SAME bar
    #     already used for the promotion verdict below (see
    #     `GATE_MIN_N_INFEASIBLE`'s comment above for why no second number is
    #     defined for this).
    #
    # When any condition fails, `gated=False` and `propose()` below is called
    # exactly as it always was (no `feasibility_classifier` argument at all) -
    # this is the no-regression guarantee: a producer-only or too-small sheet's
    # proposed batch is byte-identical to before this change.
    feas_clf = FeasibilityClassifier().fit(X, feasible_labels(y))
    n_infeasible = int((feasible_labels(y) == 0).sum())
    feas_auc = feas["auc"]
    _gate_auc_floor = GatesConfig().min_feasibility_auc
    _gate_fail_reasons: list[str] = []
    if not feas_clf.fitted:
        _gate_fail_reasons.append(
            "feasibility classifier did not fit (cold-start fallback: too few "
            "non-producer examples to train on)"
        )
    if n_infeasible < GATE_MIN_N_INFEASIBLE:
        _gate_fail_reasons.append(
            f"not zero-inflated enough: n_infeasible={n_infeasible} < {GATE_MIN_N_INFEASIBLE}"
        )
    _auc_ok = feas_auc == feas_auc and feas_auc >= _gate_auc_floor  # NaN-safe
    if not _auc_ok:
        _gate_fail_reasons.append(
            "feasibility CV AUC not measurable on this sheet"
            if feas_auc != feas_auc
            else f"feasibility CV AUC too low to trust: auc={feas_auc:.3f} < {_gate_auc_floor}"
        )
    gated = feas_clf.fitted and n_infeasible >= GATE_MIN_N_INFEASIBLE and _auc_ok
    gate_reason = (
        "; ".join(_gate_fail_reasons)
        if _gate_fail_reasons
        else (
            f"zero-inflated (n_infeasible={n_infeasible}) with a trustworthy "
            f"feasibility classifier (auc={feas_auc:.3f} >= {_gate_auc_floor}); "
            "gating acquisition by P(feasible)"
        )
    )

    # gate_stats assembles the four keys `check_gates` reads by name
    # (kalos/core/gates.py: surrogate_spearman, feasibility_auc, ece, brier).
    #
    # DELIBERATE CHOICE, stated here because it is easy to get backwards: the
    # `ece` fed to the gate is `feas["ece"]` - the FEASIBILITY CLASSIFIER's
    # expected calibration error (is the stated P(feasible) itself honest) -
    # NOT `calibration["ece"]` from `reliability` above, which is the
    # REGRESSION interval calibration ECE (are the titer error bars honest).
    # Both are legitimately named "ece", both live on the same 0-to-1 scale,
    # and they answer different questions. `GatesConfig.max_ece` was written
    # in the lean engine against the classifier metric - promotion asks "is
    # this model's feasibility judgment trustworthy enough to gate
    # acquisition on", not "are this model's regression error bars the right
    # width" - so the classifier's `ece` is what belongs here. Naming both in
    # this comment is the whole point: nothing downstream should ever swap
    # them without knowing it changed the question being asked.
    gate_stats = {
        "surrogate_spearman": rho,  # the pooled CV rho already computed above
        "feasibility_auc": feas["auc"],
        "ece": feas["ece"],
        "brier": feas["brier"],
    }
    promotion_result = check_gates(gate_stats)
    promotion = {
        "passed": promotion_result.passed,
        "failures": promotion_result.failures,
        # `check_gates` only ever inserts a value into its `metrics` dict when
        # `_finite` accepts it (isinstance float/int, not bool, math.isfinite),
        # so this is strict-JSON safe by construction - no NaN can reach here.
        "metrics": promotion_result.metrics,
        "summary": promotion_result.summary,
        # FAIL-CLOSED IS CORRECT, BUT ILLEGIBLE WITHOUT THIS. On an all-producer
        # sheet (no non-producers to build a feasibility classifier from) or
        # any sheet too small to measure one of the four gates,
        # `feasibility_auc`/`ece`/`brier` come back `nan`, `check_gates` fails
        # closed with "missing/NaN (unmeasured)" failures, and `passed` reads
        # `False` - which is the RIGHT answer (an unmeasured gate must block,
        # not silently pass), but reads exactly like a rejection if nothing
        # explains it. This fixed string is that explanation. It does not
        # soften the verdict - `passed` and `failures` are unchanged - it only
        # states what a `False` here does and does not mean.
        "meaning": (
            'passed=false means "not yet shown fit to promote", which '
            "includes the case where a required metric is unmeasurable on "
            "this data (e.g. an all-producer sheet with no non-producers to "
            "score feasibility against). It is not a rejection of the analysis."
        ),
    }

    # LEAVE-ONE-GROUP-OUT. Grouped K-fold above tests generalization to a FEW
    # unseen groups among many familiar ones; LOGO is the harder question -
    # generalization to the NEXT group the model has seen nothing like - and
    # the two can diverge sharply (see `logo_report`'s docstring: -0.12 vs 0.91
    # on the owner's real clone-selection data). Only run when there is a
    # DECLARED/detected group column (`gcol` is not None): the recipe-hash
    # fallback groups by feature identity, so leaving one recipe-hash group out
    # would just be a costlier copy of the CV already computed above, not a
    # harder question. Also capped at `LOGO_MAX_GROUPS` groups, since LOGO
    # costs one GP fit per group (see that constant's comment for the measured
    # timing this cap is tuned against) - absence is always explained via
    # `reason`, never silent.
    cv_logo: dict[str, Any]
    if gcol is None:
        cv_logo = {
            "spearman": None,
            "n_groups": None,
            "n_oof": None,
            "reason": (
                "no declared/detected group column; leave-one-group-out over "
                "the recipe-hash fallback would just be a costlier copy of "
                "the grouped CV already reported as cv_spearman"
            ),
        }
    else:
        n_logo_groups = int(len(np.unique(np.asarray(pd.Series(groups).astype(str).values))))
        if n_logo_groups > LOGO_MAX_GROUPS:
            cv_logo = {
                "spearman": None,
                "n_groups": n_logo_groups,
                "n_oof": None,
                "reason": (
                    f"{n_logo_groups} groups exceeds the leave-one-group-out cap "
                    f"of {LOGO_MAX_GROUPS} groups (one GP fit per held-out group)"
                ),
            }
        else:
            logo = logo_report(X, y, groups, bounds=bounds, cat_dims=cat_dims)
            logo_rho = logo["spearman"]
            cv_logo = {
                "spearman": None if logo_rho != logo_rho else round(float(logo_rho), 3),
                "n_groups": logo["n_groups"],
                "n_oof": logo["n_oof"],
                "reason": None,
            }

    # TOP-K OVERLAP. The client's actual decision is "which k recipes do I
    # advance", not "how well-ranked is the whole held-out set" - Spearman can
    # look fine while the model gets exactly the recipes that matter out of
    # order (see `top_k_overlap`'s docstring). Computed on the same repeat-0
    # pooled OOF everything else in this function reads off.
    topk = top_k_overlap(oof_a, oof_p, CV_TOPK_K)
    cv_topk: dict[str, Any] = {
        "overlap": None if not topk["evaluable"] else round(float(topk["overlap"]), 3),
        "k": topk["k"],
        "n": topk["n"],
        "evaluable": bool(topk["evaluable"]),
        "reason": topk["reason"],
    }

    # GROUP-MEAN BASELINE FLOOR. How much of the apparent CV signal is just the
    # model recognizing which recipe a row belongs to, rather than modeling the
    # process? (See `group_mean_baseline_spearman`'s docstring: on the owner's
    # real data this floor alone captured ~0.75 of a trained model's ~0.86
    # headline.) Computed over `recipe_key` - the same replicate grouping the
    # noise floor and driver panel already use - only when enough groups are
    # actually replicated to make the leave-one-row-out means a real signal
    # rather than noise from a couple of pairs.
    n_groups_with_reps = int((pd.Series(recipe_key).value_counts() >= 2).sum())
    cv_group_mean_baseline: dict[str, Any]
    if n_groups_with_reps < GROUP_MEAN_BASELINE_MIN_GROUPS:
        cv_group_mean_baseline = {
            "spearman": None,
            "n_evaluable": 0,
            "n_groups_used": 0,
            "interpretation": None,
            "reason": (
                f"only {n_groups_with_reps} recipes have 2+ rows; need at "
                f"least {GROUP_MEAN_BASELINE_MIN_GROUPS} to compute a "
                "group-mean floor"
            ),
        }
    else:
        gmb = group_mean_baseline_spearman(y, recipe_key)
        if gmb["evaluable"]:
            cv_group_mean_baseline = {
                "spearman": round(float(gmb["spearman"]), 3),
                "n_evaluable": gmb["n_evaluable"],
                "n_groups_used": gmb["n_groups_used"],
                "interpretation": (
                    "A model whose CV Spearman does not clearly beat this "
                    "floor may be recognizing recipes, not modeling the "
                    "process."
                ),
                "reason": None,
            }
        else:
            cv_group_mean_baseline = {
                "spearman": None,
                "n_evaluable": gmb["n_evaluable"],
                "n_groups_used": gmb["n_groups_used"],
                "interpretation": None,
                "reason": gmb["reason"],
            }

    # signed drivers, each with a bootstrap 95% CI so the client can tell a real
    # driver from noise. A driver whose CI straddles zero is NOT distinguishable
    # from no-correlation at this sample size; the UI must not present it as a
    # finding. (Same honesty contract as `reliability.ci_excludes_zero` above.)
    # `rho` is the sample Spearman POINT estimate (consistent with the point
    # estimate `reliability.spearman` reports); the bootstrap supplies only the CI.
    # Drivers are computed over the CONTINUOUS features only: a signed Spearman
    # rank correlation on an integer-coded nominal category is not a meaningful
    # "driver", so categorical dims are deliberately excluded here. Their
    # continuous indices (0..len(cont_feats)-1) align with the leading columns of X.
    # MULTIPLICITY. Every continuous feature is tested, then the strongest are
    # selected and shipped. With an uncorrected per-feature 95% CI that is a
    # machine for printing false process insights: over 400 simulated reports on
    # 30 pure-noise features against an independent target, 78.8% contained at
    # least one "significant" driver. Under Benjamini-Hochberg at q=0.05 that
    # falls to 3.5%. Selection on the same data makes it worse than the raw rate
    # suggests, because ranking by |rho| preferentially surfaces exactly the
    # flukes. So `significant` now requires BOTH tests to agree:
    #   - the bootstrap CI excludes zero (is it distinguishable from no-effect?)
    #   - the feature survives BH across ALL tested features (is it still a
    #     finding once we account for how many features we looked at?)
    # BH runs over every tested feature BEFORE the top-k cut, never after -
    # correcting for 8 tests when 30 were performed would understate the very
    # multiplicity it exists to control.
    #
    # THE UNIT OF ANALYSIS IS THE RECIPE, NOT THE ROW. Both tests above ask "how
    # surprising is this association, given how much independent evidence there
    # is?", and on a replicated sheet a row is not independent evidence: three
    # wells of one recipe carry one recipe's worth of information about the
    # process, plus three draws of assay noise. Testing rows counts them as three,
    # so the p-value that feeds BH is computed against an `n` the sheet does not
    # have, and BH stops correcting anything.
    #
    # That is not a rounding error. Re-running the 30-noise-feature simulation on
    # 20 recipes x 3 replicates (300 reports, the replication depth a media DoE
    # actually ships with) put a "significant" driver in 87.0% of reports, against
    # 7.3% for the same 60 rows drawn independently. Collapsing replicates to
    # recipe means first brings it back to 8.0%; a cluster bootstrap alone does
    # not (82.5%), because the row-level p-values are what BH is reading.
    #
    # So the panel is computed on replicate-averaged rows, keyed by the same
    # `recipe_key` the CV grouping and the noise floor already use. On a sheet
    # with no replicates every group is a singleton and `aggregate_replicates` is
    # an exact no-op, so unreplicated uploads are unchanged.
    drv: list[dict[str, Any]] = []
    n_tested = len(cont_feats)
    n_driver_units = int(len(y))
    if cont_feats:
        names = [str(c) for c in cont_feats]
        Xr, yr, _rv, _rn = aggregate_replicates(X, y, groups=recipe_key)
        n_driver_units = int(Xr.shape[0])
        Xcont = Xr[:, : len(cont_feats)]
        matrix = spearman_driver_matrix(Xcont, yr, feature_names=names)
        point = np.asarray(matrix["rho"]).astype(float)
        pvals = np.asarray(matrix["pvals"]).astype(float)
        bh_survives = benjamini_hochberg(pvals, q=DRIVER_FDR_Q)
        boot = bootstrap_spearman(Xcont, yr, feature_names=names)
        boot_lo = np.asarray(boot["lo"], dtype=float)
        boot_hi = np.asarray(boot["hi"], dtype=float)
        for j, c in enumerate(cont_feats):
            lo, hi = round(float(boot_lo[j]), 3), round(float(boot_hi[j]), 3)
            ci_excludes_zero = bool(lo > 0 or hi < 0)
            drv.append(
                {
                    "_idx": j,
                    "name": str(c),
                    "rho": round(float(point[j]), 3),
                    "ci95": [lo, hi],
                    "p": round(float(pvals[j]), 6),
                    # Both components are reported, not just the verdict, so a
                    # reviewer can see WHICH test a borderline feature failed.
                    "ci_excludes_zero": ci_excludes_zero,
                    "survives_fdr": bool(bh_survives[j]),
                    # significance uses the SAME rounded bounds the client sees, so
                    # the flag never disagrees with the displayed CI. Two-sided: a
                    # strong negative driver is a finding too.
                    "significant": bool(ci_excludes_zero and bh_survives[j]),
                }
            )
    drv.sort(key=lambda d: -abs(float(d["rho"])))
    drv = drv[:DRIVER_TOP_K]

    # Replicate structure + assay noise floor (the "SNR lever", BENCHMARK.md).
    # Media DoE sheets are frequently heavily replicated because the ASSAY is
    # noisy, not the process. Scoring/optimizing single measurements rewards
    # lucky noise spikes - which is why undirected search can out-score BO on
    # such data. When replicates are present we instead fit the PROPOSAL
    # surrogate on the reproducible (replicate-averaged) titer and hand the GP
    # the measured assay noise floor as fixed per-recipe observation variance
    # (sigma^2 / n_reps, the variance of each recipe mean), so it stops chasing
    # spikes. This is the configuration the benchmark shows actually beats random
    # on the real data. Diagnostics (grouped-CV reliability, drivers) stay on the
    # raw rows - they are already replicate-grouped for leakage and describe the
    # as-measured signal; only the proposed batch switches to the reproducible
    # objective.
    nr = noise_report(X, y, groups=recipe_key)
    # Is the homoscedastic assumption behind that noise floor actually true? A
    # diagnostic only: nothing is transformed here, because silently changing the
    # target's scale would change every number the engine reports.
    het = heteroscedasticity_report(X, y, groups=recipe_key)
    best_single = float(y.max())
    best_reproducible: float | None = None
    replicate_aware = False
    incumbent = best_single
    # The rows the PROPOSAL surrogate is fit on. They stay the raw rows unless the
    # replicate-aware swap below happens, and they are what the shape report has
    # to be handed - see the `gp_shape_report` call for why that matters.
    X_fit, y_fit = X, y
    if nr["n_replicated"] >= 1:
        Xf, yf, _yvar_g, n_reps_g = aggregate_replicates(X, y, groups=recipe_key)
        best_reproducible = float(yf.max())
        sigma2 = nr["noise_var"]
        # Only swap in the reproducible objective when there are enough distinct
        # recipes to fit a meaningful GP and a real, positive noise floor to feed;
        # otherwise fall through to the unchanged raw fit (still reported honestly).
        if nr["n_recipes"] >= 6 and np.isfinite(sigma2) and sigma2 > 0:
            replicate_aware = True
            per_point_var = float(sigma2) / np.maximum(n_reps_g, 1).astype(float)
            s = Surrogate().fit(Xf, yf, bounds=bounds, noise=per_point_var, cat_dims=cat_dims)
            incumbent = best_reproducible
            X_fit, y_fit = Xf, yf

    # proposed next batch, with predicted target + uncertainty + a why per row
    if not replicate_aware:
        s = Surrogate().fit(X, y, bounds=bounds, cat_dims=cat_dims)
    # In-flight recipes, encoded into this design so the acquisition treats them
    # as taken. Encoding can drop rows (a recipe naming an unknown categorical
    # level cannot be placed in the design at all), so the count actually used is
    # reported rather than assumed equal to what the caller passed.
    X_pending = _encode_pending(pending, cont_feats, kept_cats, code_maps, X.shape[1])
    n_pending = 0 if X_pending is None else int(X_pending.shape[0])
    batch = propose(
        s,
        bounds,
        q=5,
        cat_dims=cat_dims,
        cat_cardinalities=cat_cardinalities,
        pending=X_pending,
        # PART 1: only ever passed when GATE POLICY (above) cleared all three
        # conditions - `None` here is what keeps an ungated run's batch
        # byte-identical to the pre-gating acquisition.
        feasibility_classifier=feas_clf if gated else None,
        # PART 2: only ever passed when the constraint model actually fit.
        constraint_surrogate=constraint_surrogate if constraint_applied else None,
        constraint_floor=constraint_floor_val if constraint_applied else None,
    )
    show = [d["name"] for d in drv[:4]]
    show_idx = [d["_idx"] for d in drv[:4]]  # carry the feature index, not a name lookup
    for d in drv:
        d.pop("_idx", None)  # internal-only; not part of the returned API surface
    p_mean, p_std = s.posterior(batch)

    # PART 1 per-proposal P(feasible), read off the SAME classifier that gated
    # (or did not gate) the batch above - `feas_clf` was fit on the full sheet
    # regardless of `gated`, but its probability is only reported when the gate
    # was actually active (see `_annotate`'s docstring for why the ungated case
    # is `nan` here rather than a value nobody asked to trust).
    p_feasible_arr = (
        feas_clf.predict_proba(batch) if gated else np.full(len(batch), np.nan)
    )
    # PART 2 predicted constraint value +/- sd for each proposed recipe, read
    # off the constraint surrogate before it is released below - `None` (both
    # here and in `_annotate`) unless the constraint was actually applied.
    constraint_pred: tuple[str, np.ndarray, np.ndarray] | None = None
    if constraint_applied and constraint_surrogate is not None:
        c_mean, c_std = constraint_surrogate.posterior(batch)
        assert constraint_col is not None  # constraint_applied implies this was set
        constraint_pred = (constraint_col, c_mean, c_std)

    # Response SHAPES, read off this same fitted GP before it is released. This is
    # what closes the Spearman blind spot: a titer peaking at pH 7.0 gives a rank
    # correlation near zero, so the driver panel reports "no signal" for the most
    # important variable on the sheet, and the rho sign points the wrong way.
    #
    # Deliberately read from THIS surrogate rather than from a second model. The
    # shapes then describe the posterior that actually produced `proposals`, the
    # posterior's own standard deviation gates the interior-optimum claim, and ARD
    # lengthscales supply per-feature relevance already paid for during the fit.
    # Conditioned on `rho` (the out-of-fold Spearman the reliability verdict
    # already uses), so a model that cannot predict held-out runs reports no
    # shapes at all rather than describing the shape of its own overfitting.
    #
    # Continuous features only, matching the driver panel: a swept axis has to be
    # a measurement, not an integer category code.
    #
    # `X_fit`/`y_fit` are the rows this surrogate was actually fit on, which is
    # the contract `gp_shape_report` states and, on the replicate-aware path, is
    # NOT the raw sheet. It sweeps each feature through the incumbent - the row
    # with the best measured target - and on raw rows that is `best_single`, the
    # luckiest single well. Anchoring the shapes there re-introduces exactly the
    # noise spike the replicate-averaged fit exists to ignore, and it does it
    # inside a GP that never saw that row. The observed spread it compares peak
    # height against would be the raw spread too, assay noise included, which
    # makes a real bump read as flat.
    gp_shapes = gp_shape_report(
        s,
        X_fit[:, : len(cont_feats)] if cont_feats else np.empty((len(y_fit), 0)),
        y_fit,
        feature_names=[str(c) for c in cont_feats],
        cv_spearman=None if rho != rho else float(rho),
        rho_floor=RELIABILITY_SPEARMAN_FLOOR,
    ).to_dict()

    # Release the fitted GP(s) (hold torch/gpytorch tensors + parameter/prior
    # back-references that can form reference cycles refcounting alone won't
    # break) as soon as their last use is done, rather than waiting on
    # `_analyze` to return. Keeps a long-lived process (the portal, the
    # `--watch` poller) from accumulating fit memory across repeated analyses.
    del s
    if constraint_surrogate is not None:
        del constraint_surrogate
    gc.collect()

    # The explicit, LABELED alternative to modeling on the wrong scale: when
    # `het` (computed above, alongside `nr`) says a log transform would
    # materially help, actually run it - as a second, clearly-marked
    # evaluation pass, not a silent switch. Uses the SAME `groups` the raw
    # `rep`/`cv_spearman` above used (not `recipe_key`), so the two CV passes
    # are apples-to-apples and only differ by the target's scale. See
    # `_alternative_scale_block`'s docstring for the offset-semantics gotcha
    # and why this must not be `np.log1p(y + offset)`.
    #
    # DELIBERATELY LAST, after every other torch-consuming step (the
    # proposal's `Surrogate.fit`, `propose`, `gp_shape_report`) has already
    # run and its results are fixed: this block fits its OWN GP internally
    # (inside `grouped_cv_report`), which draws from the same seeded global
    # torch RNG everything else in this function shares. Running it any
    # earlier would advance that RNG state before the proposal fit and
    # `propose()` consume it, silently changing the proposed batch depending
    # on whether this diagnostic happened to run - i.e. this block would stop
    # being purely additive. Placed here, its extra draws can only affect
    # itself.
    alt_scale = _alternative_scale_block(
        het, X, y, groups=groups, bounds=bounds, cat_dims=cat_dims,
        grouped_cv_report=grouped_cv_report, interval_calibration=interval_calibration,
    )

    # per-column provenance: what was kept as a feature, used as the target, or
    # dropped (id / other output / constant / sparse), so the client is never left
    # guessing about a silently dropped column. Mirrors the selection logic above.
    kept_features = [str(c) for c in cont_feats] + [str(c) for c in kept_cats]
    provenance = provenance_dicts(
        column_provenance(
            df,
            target=str(target),
            features=kept_features,
            numeric_cols=[str(c) for c in num],
            id_hint=identifier_pattern(profile.id_hint.pattern),
            outcome_hint=profile.outcome_hint,
            constant_on_fitted_rows=[str(c) for c in constant_on_fitted]
            + [str(c) for c in constant_cats],
            declared={str(c) for c in declared},
            declared_ids={str(c) for c in roles.ids} if roles is not None else set(),
            declared_features={str(c) for c in roles.features} if roles is not None else set(),
        )
    )

    # Which acquisition optimizer actually served these proposals, so the audit
    # trail is explicit (mirrors seed / timestamp / engine_version): a large
    # categorical space silently switches from exact enumeration to the
    # alternating heuristic (kalos/core/optimize.py), and the client should know.
    if not cat_dims:
        proposal_optimizer = "continuous"
    else:
        n_combos = 1
        for k in cat_cardinalities or []:
            n_combos *= int(k)
        proposal_optimizer = "mixed_exact" if n_combos <= MAX_MIXED_COMBOS else "mixed_alternating"

    result = {
        "n": int(len(y)), "d": len(kept_features), "target": str(target), "group_col": gcol,
        "targets": [str(c) for c in candidate_targets], "features": kept_features,
        "categorical_features": [str(c) for c in kept_cats],
        "n_dropped_incomplete": n_dropped_incomplete,
        "proposal_optimizer": proposal_optimizer,
        # In-flight recipes the acquisition was told about. 0 means the batch was
        # computed as if nothing were running - true for a one-shot upload, and
        # the thing to check first if a round re-proposes work already started.
        "n_pending_considered": n_pending,
        "cv_spearman": None if rho != rho else round(rho, 3),
        "cv_ci95": ci95,
        "cv_n_groups": rep["n_groups"],
        # How many partitions actually went into `cv_ci95` above - the requested
        # ceiling (`CV_N_REPEATS`) collapses to 1 when the sheet has too few
        # groups for a second, genuinely different partition to exist (see
        # `grouped_cv_report`'s docstring), so this is the EFFECTIVE count, not
        # the requested one.
        "cv_n_repeats": rep["n_repeats"],
        # Per-repeat pooled Spearman, so the partition-to-partition spread that
        # widens `cv_ci95` is visible on its own rather than only as a width.
        # NaN -> None: this dict is serialized straight into an API response.
        "cv_spearman_per_repeat": [
            None if s != s else round(float(s), 3) for s in rep["spearman_per_repeat"]
        ],
        "conformal_q": conformal_q,
        # The band's ACTUAL coverage, so no consumer has to hardcode it.
        "conformal_coverage": round(1.0 - CONFORMAL_ALPHA, 4),
        # How the driver panel was selected and corrected, so the client can see
        # that many features were tested and only the strongest are shown.
        "driver_selection": {
            "n_tested": n_tested,
            "n_reported": len(drv),
            "top_k": DRIVER_TOP_K,
            "fdr_method": "benjamini_hochberg",
            "fdr_q": DRIVER_FDR_Q,
            "n_bootstrap": 200,
            "ranked_by": "abs_rho",
            # The unit of analysis, stated rather than assumed. Replicates of one
            # recipe are averaged before the panel runs, so `n_units` (not the row
            # count) is the independent evidence every p-value and CI is computed
            # against. On an unreplicated sheet the two are equal.
            "unit": "recipe",
            "n_units": n_driver_units,
            "n_rows": int(len(y)),
        },
        "reliability": reliability,
        # Fail-closed promotion verdict, REPORTED not enforced (see the block
        # above where `promotion` is assembled for why this must never change
        # what the API accepts).
        "promotion": promotion,
        # PART 1: whether/why the proposal acquisition was gated by P(feasible)
        # (see the GATE POLICY block above `promotion` is assembled near, for
        # the full three-condition reasoning). `threshold` is the label
        # threshold `feasible_labels` used to define "feasible" (strictly > 0
        # here, its own default), not the AUC/n_infeasible policy floors -
        # reported so a consumer never has to hardcode what "feasible" meant.
        "proposal_gating": {
            "gated": bool(gated),
            "reason": gate_reason,
            "n_infeasible": n_infeasible,
            "feasibility_auc": None if feas_auc != feas_auc else round(float(feas_auc), 3),
            "threshold": 0.0,
        },
        # PART 2: whether/why the requested constraint (if any) was applied to
        # the proposal below. Always present (even with no `constraint`
        # argument) so a consumer never has to special-case its absence.
        "constraint": constraint_report,
        # Leave-one-group-out, top-k overlap, and the group-mean baseline floor
        # - three evaluation-hygiene checks that ask harder or narrower
        # questions than the pooled grouped-CV Spearman above. Each is `None`
        # with a stated `reason` when it cannot be computed on this sheet,
        # rather than silently absent.
        "cv_logo": cv_logo,
        "cv_topk": cv_topk,
        "cv_group_mean_baseline": cv_group_mean_baseline,
        "best": round(best_single, 4),
        "noise": _noise_block(nr, best_single, best_reproducible, replicate_aware, het),
        # Explicit, labeled second evaluation pass on the log scale, run only
        # when `noise.scale.suggests_transform` is True (see
        # `_alternative_scale_block`'s docstring). Evaluation-only: proposals
        # and every other number in this response stay on the raw scale.
        "alternative_scale": alt_scale,
        "drivers": drv,
        # Response shape per feature, read off the same GP that proposed the
        # batch. Closes the interior-optimum blind spot in the rank drivers.
        "gp_shapes": gp_shapes,
        "proposal_features": show,
        "proposals": _annotate(
            batch, p_mean, p_std, incumbent, cols=show_idx, design=design,
            p_feasible=p_feasible_arr, constraint=constraint_pred,
        ),
        "oof": [[round(a, 4), round(p, 4)] for a, p in zip(oof_a, oof_p)],
        "provenance": provenance,
        # Orientation pre-pass fact from the upload path (kalos/portal/uploads.py):
        # present when the sheet was confidently detected as transposed and
        # normalized before analysis, so the client can see the frame they sent is
        # not byte-for-byte the frame that was modeled. None for a standard sheet.
        "orientation": df.attrs.get("kalos_orientation"),
        "validation": report_dict(validation),
        # Features whose search range was narrowed because some observed cells
        # were physically impossible. Reported, never silent: the client needs to
        # know the box they are being optimized over is not their full data range.
        "design_box_exclusions": box_exclusions,
        "seed": ANALYZE_SEED,
        "timestamp": int(time.time()),
        "engine_version": ENGINE_VERSION,
    }
    if anonymize:
        result = _anonymize_result(result)
    return result



def _encode_pending(
    pending: "pd.DataFrame | Sequence[Mapping[str, Any]] | None",
    cont_feats: list,
    kept_cats: list,
    code_maps: dict,
    width: int,
) -> np.ndarray | None:
    """Encode in-flight recipes into the fitted design's X layout, or None.

    Mirrors how the FITTED rows are built, so a pending recipe lands at the same
    coordinates its measured counterpart would: continuous components are coerced
    to numeric and zero-filled (the same `Xc_zf` convention), categoricals are
    mapped through the level codes derived from the fitted rows, and the columns
    are assembled continuous-first, categorical-last.

    A row is DROPPED when a kept categorical is missing, blank, or names a level
    that does not exist in this design: such a recipe has no coordinate here, and
    inventing one (say, code 0) would mark the wrong region of the space as taken.
    Returns None when nothing is left to report, which leaves the acquisition
    exactly as it is today. Never raises - a malformed in-flight record must
    degrade the proposal's information, not fail the analysis.
    """
    if pending is None:
        return None
    frame = pending if isinstance(pending, pd.DataFrame) else pd.DataFrame(list(pending))
    if frame.empty:
        return None
    cols: list[np.ndarray] = []
    for c in cont_feats:
        if c in frame.columns:
            cols.append(pd.to_numeric(frame[c], errors="coerce").fillna(0.0).to_numpy(float))
        else:
            cols.append(np.zeros(len(frame), dtype=float))
    usable = np.ones(len(frame), dtype=bool)
    for c in kept_cats:
        codes = np.full(len(frame), np.nan)
        if c in frame.columns:
            labels = frame[c].fillna("").astype(str).str.strip()
            codes = labels.map(code_maps[c]).to_numpy(dtype=float)
        usable &= np.isfinite(codes)
        cols.append(codes)
    if not cols:
        return None
    arr = np.column_stack(cols)
    usable &= np.isfinite(arr).all(axis=1)
    arr = arr[usable]
    if arr.shape[0] == 0 or arr.shape[1] != width:
        return None
    return arr


@functools.lru_cache(maxsize=8)
def identifier_pattern(id_hint_pattern: str) -> re.Pattern[str]:
    """A profile's id hint UNIONED with the compound-identifier pattern.

    A `DomainProfile.id_hint` is anchored to whole tokens
    (`^(id|name|run|batch|campaign|lot|...)$`), which matches a column called
    exactly `run` or `batch` but not `run_number`, `batch_id`, `campaign_id` or
    `lot_number` - and compound names are what real run sheets actually use.

    Two things went wrong because of that gap, and both are worse than a naming
    nit:

    1. A NUMERIC identifier column became a model FEATURE. Given `run_number`
       and `batch_id`, the engine fit on them and reported both as significant
       drivers at rho = 1.0, then proposed a recipe instructing the scientist to
       "set batch_id = 100.037". A run index rises monotonically with time, so it
       correlates with any drift or learning trend in the campaign and will
       almost always look like a top driver. That is a spurious correlation
       presented as a process insight, which is the failure mode this codebase
       works hardest everywhere else to prevent.
    2. `anonymize=True` published those same names verbatim while claiming to
       pseudonymize identifier columns. The codebase already contradicted itself
       here: the metadata scrubber's `HASH_EXACT` does list `campaign_id`, so one
       id was hashed as metadata and published as a column name.

    Both now resolve through this single pattern, so feature selection,
    provenance, and anonymization cannot disagree about what an identifier is.

    Union, never intersection - it can only ever classify MORE names as
    identifiers, never fewer. Precision still matters in the other direction,
    since dropping or aliasing a real process input would be its own defect:
    both halves reject `Methanol`, `pH`, `scale_L`, `lipase_titer`,
    `batch_titer` and `run_duration_days`.

    Cached because it is called per column per analysis; keyed on the pattern
    string so a different domain profile gets its own union.
    """
    return re.compile(
        f"(?:{id_hint_pattern})|(?:{_RUN_ID_RE.pattern})", re.IGNORECASE
    )


def _is_identifier_name(name: str) -> bool:
    """Whether a column NAME should be pseudonymized when `anonymize=True`."""
    return bool(identifier_pattern(_ID_HINT.pattern).match(str(name).strip()))


def _anonymize_result(result: dict) -> dict:
    """Replace identifier-type column names in the response with stable pseudonyms.

    Only identifier-like columns (matched by `_ID_HINT`) are renamed; process
    features and the target keep their real names because the authenticated owner
    UI legitimately shows them (a driver like "Methanol"). The mapping is stable
    (same name -> same pseudonym) via the anonymizer's irreversible hash, so an
    anonymized report is still internally consistent across fields.
    """
    def alias(name: str) -> str:
        return f"col_{_hash(name)[:8]}" if _is_identifier_name(name) else name

    out = dict(result)
    if out.get("group_col"):
        out["group_col"] = alias(out["group_col"])
    out["provenance"] = [
        {**row, "name": alias(row["name"])} for row in out.get("provenance", [])
    ]

    # The validation report names columns too, in two places: a `column` field and
    # the human-readable `message` built around it ("column 'Sample Name' carries
    # ..."). Aliasing only the field would leak the real name through the prose, so
    # both are rewritten together. Anonymization is worth nothing if it covers the
    # structured copy of a name and not the sentence next to it.
    validation = out.get("validation")
    if isinstance(validation, dict):
        out["validation"] = {
            **validation,
            "findings": [_alias_named_row(f, alias) for f in validation.get("findings", [])],
            "conversions": [
                _alias_named_row(c, alias) for c in validation.get("conversions", [])
            ],
        }
    out["design_box_exclusions"] = [
        _alias_named_row(e, alias) for e in out.get("design_box_exclusions", [])
    ]
    return out


def _alias_named_row(row: dict, alias: Callable[[str], str]) -> dict:
    """Alias `row["column"]` and scrub the real name out of `row["message"]`.

    Only rewrites the message when the alias actually differs, so a non-identifier
    column (a process feature like "Methanol", which the owner UI legitimately
    shows) is left completely untouched rather than being needlessly rewritten.
    """
    name = row.get("column")
    if not isinstance(name, str) or not name:
        return dict(row)
    aliased = alias(name)
    if aliased == name:
        return dict(row)
    out = {**row, "column": aliased}
    message = out.get("message")
    if isinstance(message, str):
        out["message"] = message.replace(name, aliased)
    return out
