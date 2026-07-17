"""Kalos portal — the science: run the engine on an arbitrary uploaded run sheet.

Outcome-like columns are candidate TARGETS and are excluded from input features
so the model never "predicts" titer from another measured output (leakage).
"""
from __future__ import annotations

import gc
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kalos.core.drivers import bootstrap_spearman
from kalos import __version__ as ENGINE_VERSION
from kalos.core.conformal import q_from_residuals
from kalos.core.splits import row_hash_groups
from kalos.data.anonymizer import _hash
from kalos.portal.uploads import MAX_FIT_ROWS, UploadRejected, _ERR_TOO_MANY_FIT_ROWS
from kalos.portal.validate import column_provenance, provenance_dicts

# NOTE: `kalos.core.evaluation` (which imports `kalos.core.surrogate`),
# `kalos.core.optimize`, and `torch` itself are intentionally NOT imported at
# module level. This module is imported by `kalos.runner.singleton` (the
# `--watch` poller) and `kalos.portal.app` (the portal), so a top-level torch
# import here would tax the idle poller and portal boot with the whole
# torch/botorch/gpytorch stack (~220 MB) before any analysis ever runs. They
# are imported lazily inside `_analyze`/`_seed_everything`, the only places
# that actually need them.

_OUTCOME_HINT = re.compile(r"titer|titre|yield|conc|purity|lipase|biomass|od\d|product|response|output|score|kda|activity|titer", re.I)
_TARGET_PREF = re.compile(r"titer|titre|lipase|yield", re.I)
_ID_HINT = re.compile(r"^(id|name|sample.*|well|index|run|experiment|round|date|time|medium|strain|recipe|batch|campaign|group|lot|notes?)$", re.I)
_GROUP_HINT = re.compile(r"medium|strain|recipe|batch|campaign|group|lot", re.I)


def _annotate(batch: np.ndarray, mean, std, best: float, cols=None) -> list:
    """Attach predicted value, uncertainty, and an explore/exploit rationale to
    each proposed experiment. Explore = high model uncertainty (chosen to learn);
    exploit = high predicted value (chosen to win). Current human-in-the-loop BO
    research says a recommendation must carry exactly this."""
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


def _analyze(
    df: pd.DataFrame, target: str | None = None, *, anonymize: bool = False
) -> dict:
    """Run the engine on an arbitrary run sheet: pick the target (the value to
    maximize), use the process INPUTS as features (other measured outputs are
    excluded to avoid leakage), then honest grouped-CV, signed drivers, and a
    proposed next batch.

    Deterministic: seeds torch + numpy up front so the same sheet gives the same
    proposals. Returns a per-column `provenance` report (what was kept/dropped and
    why) plus `seed`, `timestamp`, and `engine_version` for audit. When
    `anonymize` is True, identifier-type column names are replaced with stable
    pseudonyms in the response (the owner UI keeps real names when False)."""
    # Deferred: these transitively import torch/botorch/gpytorch (see the
    # module-level note above). This is the one place in this module that
    # actually needs them, so this is where the torch tax is paid.
    from kalos.core.evaluation import grouped_cv_report
    from kalos.core.optimize import propose
    from kalos.core.surrogate import Surrogate

    _seed_everything()
    df = _dedupe_columns(df.dropna(axis=1, how="all"))
    num = _numeric_cols(df)
    if not num:
        raise ValueError("no numeric columns found")
    outcomes = [c for c in num if _OUTCOME_HINT.search(str(c))]
    if target is None or target not in df.columns:
        target = next((c for c in outcomes if _TARGET_PREF.search(str(c))), outcomes[0] if outcomes else num[-1])
    feats = []
    for c in num:
        if c == target or _ID_HINT.match(str(c).strip()) or _OUTCOME_HINT.search(str(c)):
            continue  # drop the target, ids, and OTHER measured outputs (anti-leakage)
        col = pd.to_numeric(df[c], errors="coerce")
        if col.std(skipna=True) and col.std() > 1e-9:
            feats.append(c)
    if len(feats) < 1:
        raise ValueError("no varying process-input columns found (only outputs/ids?)")
    candidate_targets = outcomes or [target]

    y_all = pd.to_numeric(df[target], errors="coerce")
    keep = y_all.notna()
    X_raw = df.loc[keep, feats].apply(pd.to_numeric, errors="coerce")   # NaN preserved (for grouping)
    X_zf = X_raw.fillna(0.0)                                             # zero-filled (for the GP + box)
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
    fitted_range = X_zf.max(axis=0) - X_zf.min(axis=0)
    constant_on_fitted = [c for c in feats if float(fitted_range[c]) <= 1e-9]
    if constant_on_fitted:
        feats = [c for c in feats if c not in set(constant_on_fitted)]
        if len(feats) < 1:
            raise ValueError("no varying process-input columns on the target-present rows")
        X_raw = X_raw.drop(columns=constant_on_fitted)
        X_zf = X_zf.drop(columns=constant_on_fitted)

    X = X_zf.to_numpy(float)
    bounds = np.vstack([X.min(0), X.max(0)])

    gcol = next((c for c in df.columns if _GROUP_HINT.search(str(c))), None)
    if gcol:
        groups = df.loc[keep, gcol].astype(str).tolist()
    else:
        # group on the RAW values (NaN preserved) so rows missing different
        # components are not merged into one replicate group by the zero-fill —
        # via the one leakage-checked, deterministic grouper.
        groups = row_hash_groups(X_raw)

    # honest grouped cross-validation: pooled out-of-fold predictions + a
    # group-level bootstrap CI, all through the single leakage-checked splitter.
    rep = grouped_cv_report(X, y, groups=groups, n_splits=5, bounds=bounds)
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
    drv = []
    if feats:
        Z = np.column_stack(
            [pd.to_numeric(df.loc[keep, c], errors="coerce").fillna(0.0).to_numpy() for c in feats]
        )
        boot = bootstrap_spearman(Z, y, feature_names=[str(c) for c in feats])
        for j, c in enumerate(feats):
            # NB: a distinct name (not `rho`) - the outer `rho` is the model's CV
            # Spearman that `cv_spearman`/`reliability` read below; shadowing it
            # here would clobber the headline reliability number.
            d_rho = float(boot["mean"][j])
            lo, hi = float(boot["lo"][j]), float(boot["hi"][j])
            if d_rho != d_rho:
                d_rho, lo, hi = 0.0, 0.0, 0.0
            drv.append(
                {
                    "name": str(c),
                    "rho": round(d_rho, 3),
                    "ci95": [round(lo, 3), round(hi, 3)],
                    "significant": bool(lo > 0 or hi < 0),  # 95% CI excludes zero
                }
            )
    drv.sort(key=lambda d: -abs(d["rho"]))
    drv = drv[:8]

    # proposed next batch, with predicted target + uncertainty + a why per row
    s = Surrogate().fit(X, y, bounds=bounds)
    batch = propose(s, bounds, q=5)
    show = [d["name"] for d in drv[:4]]
    show_idx = [feats.index(f) for f in show]
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
    provenance = provenance_dicts(
        column_provenance(
            df,
            target=str(target),
            features=[str(c) for c in feats],
            numeric_cols=[str(c) for c in num],
            id_hint=_ID_HINT,
            outcome_hint=_OUTCOME_HINT,
            constant_on_fitted_rows=[str(c) for c in constant_on_fitted],
        )
    )

    result = {
        "n": int(keep.sum()), "d": len(feats), "target": str(target), "group_col": gcol,
        "targets": [str(c) for c in candidate_targets], "features": [str(c) for c in feats],
        "cv_spearman": None if rho != rho else round(rho, 3),
        "cv_ci95": ci95,
        "cv_n_groups": rep["n_groups"],
        "conformal_q": conformal_q,
        "reliability": reliability,
        "best": round(float(y.max()), 4),
        "drivers": drv,
        "proposal_features": show,
        "proposals": _annotate(batch, p_mean, p_std, float(y.max()), cols=show_idx),
        "oof": [[round(a, 4), round(p, 4)] for a, p in zip(oof_a, oof_p)],
        "provenance": provenance,
        "seed": ANALYZE_SEED,
        "timestamp": int(time.time()),
        "engine_version": ENGINE_VERSION,
    }
    if anonymize:
        result = _anonymize_result(result)
    return result


def _anonymize_result(result: dict) -> dict:
    """Replace identifier-type column names in the response with stable pseudonyms.

    Only identifier-like columns (matched by `_ID_HINT`) are renamed; process
    features and the target keep their real names because the authenticated owner
    UI legitimately shows them (a driver like "Methanol"). The mapping is stable
    (same name -> same pseudonym) via the anonymizer's irreversible hash, so an
    anonymized report is still internally consistent across fields.
    """
    def alias(name: str) -> str:
        return f"col_{_hash(name)[:8]}" if _ID_HINT.match(str(name).strip()) else name

    out = dict(result)
    if out.get("group_col"):
        out["group_col"] = alias(out["group_col"])
    out["provenance"] = [
        {**row, "name": alias(row["name"])} for row in out.get("provenance", [])
    ]
    return out
