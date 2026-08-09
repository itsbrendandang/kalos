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
from typing import Any, Callable

import numpy as np
import pandas as pd

from kalos.core.drivers import bootstrap_spearman, spearman_driver_matrix
from kalos import __version__ as ENGINE_VERSION
from kalos.core.conformal import q_from_residuals
from kalos.core.replicates import aggregate_replicates, noise_report
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


def _annotate(batch: np.ndarray, mean, std, best: float, cols=None, design: DesignSpace | None = None) -> list:
    """Attach predicted value, uncertainty, and an explore/exploit rationale to
    each proposed experiment. Explore = high model uncertainty (chosen to learn);
    exploit = high predicted value (chosen to win). Current human-in-the-loop BO
    research says a recommendation must carry exactly this.

    When a `design` is given, each row also carries `recipe`: the full proposed
    experiment decoded to `{feature: value}`, with categorical dimensions decoded
    back to their labels instead of raw integer codes."""
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


def _noise_block(
    nr: dict, best_single: float, best_reproducible: float | None, replicate_aware: bool
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
    def _num(x: float | None, nd: int = 4) -> float | None:
        if x is None or not np.isfinite(x):
            return None
        return round(float(x), nd)

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

    Deterministic: seeds torch + numpy up front so the same sheet gives the same
    proposals. Returns a per-column `provenance` report (what was kept/dropped and
    why) plus `seed`, `timestamp`, and `engine_version` for audit. When
    `anonymize` is True, identifier-type column names are replaced with stable
    pseudonyms in the response (the owner UI keeps real names when False)."""
    # Deferred: these transitively import torch/botorch/gpytorch (see the
    # module-level note above). This is the one place in this module that
    # actually needs them, so this is where the torch tax is paid.
    from kalos.core.evaluation import grouped_cv_report
    from kalos.core.optimize import MAX_MIXED_COMBOS, propose
    from kalos.core.surrogate import Surrogate

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

    if kept_cats:
        code_maps = {c: {lvl: i for i, lvl in enumerate(cat_dims_map[c])} for c in kept_cats}
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

    if gcol:
        groups = df.loc[keep_index, gcol].astype(str).tolist()
    else:
        # group on the RAW values (NaN preserved) so rows missing different
        # components are not merged into one replicate group by the zero-fill —
        # via the one leakage-checked, deterministic grouper. Categorical labels
        # join the grouping key so identical recipes stay one replicate group.
        gf = Xc_raw.copy()
        for c in kept_cats:
            gf[c] = df.loc[keep_index, c].fillna("").astype(str)
        groups = row_hash_groups(gf)

    # honest grouped cross-validation: pooled out-of-fold predictions + a
    # group-level bootstrap CI, all through the single leakage-checked splitter.
    rep = grouped_cv_report(X, y, groups=groups, n_splits=5, bounds=bounds, cat_dims=cat_dims)
    rho = rep["spearman"]
    oof_a, oof_p = rep["oof_actual"], rep["oof_pred"]

    # distribution-free +/- band from the pooled out-of-fold residuals (approximate
    # coverage under grouped CV). Honest alternative to the surrogate's own std,
    # which is often overconfident on small bioprocess datasets.
    resid = np.asarray(oof_a, float) - np.asarray(oof_p, float)
    conformal_q = round(q_from_residuals(resid, alpha=0.1), 4) if len(resid) else None

    # Honest reliability verdict: only what this path can actually assess. The
    # spearman floor mirrors GatesConfig.min_spearman (kalos/core/gates.py); we do
    # NOT assert feasibility or calibration gates, which are not measured here.
    ci95 = None if rho != rho else [round(rep["ci95"][0], 3), round(rep["ci95"][1], 3)]
    reliability = {
        "spearman": None if rho != rho else round(rho, 3),
        "ci95": ci95,
        "spearman_floor": 0.20,
        "clears_floor": bool(rho == rho and rho >= 0.20),
        "ci_excludes_zero": bool(ci95 is not None and ci95[0] > 0),
        "unmodeled": ["feasibility probability", "calibration (ECE)", "scale-up transfer"],
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
    drv: list[dict[str, Any]] = []
    if cont_feats:
        names = [str(c) for c in cont_feats]
        Xcont = X[:, : len(cont_feats)]
        point = np.asarray(spearman_driver_matrix(Xcont, y, feature_names=names)["rho"]).astype(float)
        boot = bootstrap_spearman(Xcont, y, feature_names=names)
        boot_lo = np.asarray(boot["lo"], dtype=float)
        boot_hi = np.asarray(boot["hi"], dtype=float)
        for j, c in enumerate(cont_feats):
            lo, hi = round(float(boot_lo[j]), 3), round(float(boot_hi[j]), 3)
            drv.append(
                {
                    "_idx": j,
                    "name": str(c),
                    "rho": round(float(point[j]), 3),
                    "ci95": [lo, hi],
                    # significance uses the SAME rounded bounds the client sees, so
                    # the flag never disagrees with the displayed CI. Two-sided: a
                    # strong negative driver is a finding too.
                    "significant": bool(lo > 0 or hi < 0),
                }
            )
    drv.sort(key=lambda d: -abs(float(d["rho"])))
    drv = drv[:8]

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
    nr = noise_report(X, y)
    best_single = float(y.max())
    best_reproducible: float | None = None
    replicate_aware = False
    incumbent = best_single
    if nr["n_replicated"] >= 1:
        Xf, yf, _yvar_g, n_reps_g = aggregate_replicates(X, y)
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

    # proposed next batch, with predicted target + uncertainty + a why per row
    if not replicate_aware:
        s = Surrogate().fit(X, y, bounds=bounds, cat_dims=cat_dims)
    batch = propose(s, bounds, q=5, cat_dims=cat_dims, cat_cardinalities=cat_cardinalities)
    show = [d["name"] for d in drv[:4]]
    show_idx = [d["_idx"] for d in drv[:4]]  # carry the feature index, not a name lookup
    for d in drv:
        d.pop("_idx", None)  # internal-only; not part of the returned API surface
    p_mean, p_std = s.posterior(batch)
    # Release the fitted GP (holds torch/gpytorch tensors + parameter/prior
    # back-references that can form reference cycles refcounting alone won't
    # break) as soon as its last use is done, rather than waiting on `_analyze`
    # to return. Keeps a long-lived process (the portal, the `--watch` poller)
    # from accumulating fit memory across repeated analyses.
    del s
    gc.collect()

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
        "cv_spearman": None if rho != rho else round(rho, 3),
        "cv_ci95": ci95,
        "cv_n_groups": rep["n_groups"],
        "conformal_q": conformal_q,
        "reliability": reliability,
        "best": round(best_single, 4),
        "noise": _noise_block(nr, best_single, best_reproducible, replicate_aware),
        "drivers": drv,
        "proposal_features": show,
        "proposals": _annotate(batch, p_mean, p_std, incumbent, cols=show_idx, design=design),
        "oof": [[round(a, 4), round(p, 4)] for a, p in zip(oof_a, oof_p)],
        "provenance": provenance,
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
