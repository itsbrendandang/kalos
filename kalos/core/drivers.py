"""Driver analysis: signed Spearman correlations with bootstrap CIs.

Ported from the lean engine. Gives each input a SIGNED, rank-based association
with the target plus an honest bootstrap confidence interval, so a "driver"
whose CI spans 0 is not oversold. This is the principled replacement for
RF-Gini importance (which is biased toward high-cardinality features and reports
neither sign nor confidence).
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import spearmanr


def _ensure_2d(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    return Z.reshape(-1, 1) if Z.ndim == 1 else Z


def spearman_driver_matrix(
    Z: np.ndarray, signal: ArrayLike, feature_names: Optional[Sequence[str]] = None
) -> Dict[str, object]:
    """Per-feature Spearman rho + p-value vs. a signal."""
    Z = _ensure_2d(Z)
    y = np.asarray(signal, dtype=float).reshape(-1)
    if Z.shape[0] != y.shape[0]:
        raise ValueError("Z rows must match signal length")
    d = Z.shape[1]
    rhos = np.zeros(d)
    pvals = np.ones(d)
    for j in range(d):
        rho, p = spearmanr(Z[:, j], y, nan_policy="omit")
        rhos[j] = 0.0 if np.isnan(rho) else float(rho)
        pvals[j] = 1.0 if np.isnan(p) else float(p)
    names = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(d)]
    return {"rho": rhos, "pvals": pvals, "feature_names": names}


def benjamini_hochberg(pvals: ArrayLike, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up: which of `pvals` survive at false-discovery rate `q`.

    Returns a boolean mask aligned to `pvals`.

    WHY THIS EXISTS. A driver panel tests every continuous feature on the sheet
    against the target and then reports the strongest. With 30 features and an
    uncorrected 95% per-feature threshold you expect about 1.5 features to look
    significant by chance alone in EVERY report, and because the panel then
    selects the largest |rho| it preferentially surfaces exactly those flukes.
    That is a mechanism for printing a false process insight, and a scientist
    acts on drivers and repeats them in meetings.

    BH controls the expected PROPORTION of false findings among those declared
    significant, which is the right error rate here. The alternative,
    Bonferroni, controls the probability of any false finding at all and at 30
    collinear media-DoE features would suppress nearly every real driver too -
    trading a false-positive problem for a false-negative one on data this
    small.

    Applied over ALL tested features, never over the surviving subset: running
    it after selecting the top 8 would correct for 8 tests when 30 were
    performed, which understates the multiplicity it exists to control.

    Assumption worth stating: BH assumes independent or positively-dependent
    tests. Media-DoE columns are often collinear by construction, which is
    positive dependence, so BH remains valid (Benjamini-Yekutieli would be the
    conservative choice under arbitrary dependence). This does NOT make a
    surviving driver causal - it remains an association, subject to confounding
    with any collinear column.
    """
    p = np.asarray(pvals, dtype=float).reshape(-1)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    # NaN cannot be ranked against real p-values; treat it as "no evidence".
    p = np.where(np.isnan(p), 1.0, p)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    thresholds = (np.arange(1, m + 1) / m) * q
    passing = ranked <= thresholds
    reject = np.zeros(m, dtype=bool)
    if passing.any():
        # Step-up: the largest rank that passes sets the cutoff, and every
        # smaller p-value is rejected with it - including any that individually
        # failed its own threshold. That is the step-up procedure, not a bug.
        k = int(np.nonzero(passing)[0].max())
        reject[order[: k + 1]] = True
    return reject


def bootstrap_spearman(
    Z: np.ndarray, signal: ArrayLike, B: int = 200, random_state: int = 42,
    ci: float = 0.95, feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Bootstrap Spearman per feature: mean, std, and percentile CI bounds.

    A resample draw can be degenerate for a feature (e.g. the bootstrap indices
    happen to make that column constant), which makes Spearman's rho undefined
    (`nan`). Those draws are recorded as `nan` rather than coerced to 0.0: 0.0 is
    a strong claim (this draw found no association), and folding "undefined"
    into "found no association" pulls the whole bootstrap distribution toward
    zero and understates the true CI width for exactly the noisiest features -
    substituting a fabricated data point, not a neutral default. `mean`/`std`
    and the `lo`/`hi` percentile CI are instead computed with nan-aware
    reductions (`np.nanmean`/`np.nanstd`/`np.nanquantile`) over only the valid
    draws per feature.

    If a feature has zero valid draws (every resample degenerate for that
    column - a fully constant or near-constant feature), the nan-reductions
    have nothing to reduce over; that case falls back to mean/std/lo/hi all
    0.0, reproducing today's behavior for that feature. This keeps every
    output finite by construction: `bootstrap_spearman`'s output feeds a
    strict-JSON API response (`kalos/portal/analysis.py`), and a `nan` in that
    payload would fail JSON encoding and 500 the request rather than degrade
    gracefully.
    """
    Z = _ensure_2d(Z)
    y = np.asarray(signal, dtype=float).reshape(-1)
    N, d = Z.shape
    if y.shape[0] != N:
        raise ValueError("Z rows must match signal length")
    rng = np.random.default_rng(random_state)
    R = np.full((B, d), np.nan)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        zb, yb = Z[idx], y[idx]
        for j in range(d):
            rho, _ = spearmanr(zb[:, j], yb, nan_policy="omit")
            R[b, j] = float(rho) if not np.isnan(rho) else np.nan
    alpha = (1.0 - ci) / 2.0
    names = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(d)]
    # A column with zero valid draws makes every nan-reduction operate on an
    # all-NaN slice, which numpy warns about (RuntimeWarning) even though the
    # all-NaN fallback below handles the result correctly - suppress it here
    # rather than let it leak into callers as spurious noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(R, axis=0)
        std = np.nanstd(R, axis=0, ddof=1) if B > 1 else np.zeros(d)
        lo = np.nanquantile(R, alpha, axis=0)
        hi = np.nanquantile(R, 1 - alpha, axis=0)
    # A column with zero valid draws leaves every reduction above as nan (an
    # all-NaN-slice input has no defined mean/std/quantile); a column with
    # exactly one valid draw leaves `std` as nan too (ddof=1 needs >= 2 points).
    # Both fall back to 0.0, reproducing the pre-fix value for a fully
    # degenerate feature and guaranteeing every output here is finite.
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=0.0)
    lo = np.nan_to_num(lo, nan=0.0)
    hi = np.nan_to_num(hi, nan=0.0)
    return {"mean": mean, "std": std, "lo": lo, "hi": hi, "feature_names": names}


def rank_drivers(summary: Dict[str, object], top_k: int = 5, direction: str = "abs") -> List[Tuple[str, float]]:
    """Rank features by bootstrap mean rho (or raw rho). direction: abs|pos|neg."""
    names = list(cast(Sequence[str], summary["feature_names"]))
    scores = np.asarray(summary.get("mean", summary.get("rho")), dtype=float)
    if direction == "abs":
        order = np.argsort(-np.abs(scores))
    elif direction == "pos":
        order = np.argsort(-scores)
    elif direction == "neg":
        order = np.argsort(scores)
    else:
        raise ValueError("direction must be abs|pos|neg")
    return [(names[i], float(scores[i])) for i in order[:top_k]]


__all__ = ["spearman_driver_matrix", "bootstrap_spearman", "rank_drivers", "benjamini_hochberg"]
