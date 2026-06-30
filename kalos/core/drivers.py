"""Driver analysis: signed Spearman correlations with bootstrap CIs.

Ported from the lean engine. Gives each input a SIGNED, rank-based association
with the target plus an honest bootstrap confidence interval, so a "driver"
whose CI spans 0 is not oversold. This is the principled replacement for
RF-Gini importance (which is biased toward high-cardinality features and reports
neither sign nor confidence).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr


def _ensure_2d(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    return Z.reshape(-1, 1) if Z.ndim == 1 else Z


def spearman_driver_matrix(
    Z: np.ndarray, signal: Sequence[float], feature_names: Optional[Sequence[str]] = None
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


def bootstrap_spearman(
    Z: np.ndarray, signal: Sequence[float], B: int = 200, random_state: int = 42,
    ci: float = 0.95, feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Bootstrap Spearman per feature: mean, std, and percentile CI bounds."""
    Z = _ensure_2d(Z)
    y = np.asarray(signal, dtype=float).reshape(-1)
    N, d = Z.shape
    if y.shape[0] != N:
        raise ValueError("Z rows must match signal length")
    rng = np.random.default_rng(random_state)
    R = np.zeros((B, d))
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        zb, yb = Z[idx], y[idx]
        for j in range(d):
            rho, _ = spearmanr(zb[:, j], yb, nan_policy="omit")
            R[b, j] = 0.0 if np.isnan(rho) else float(rho)
    alpha = (1.0 - ci) / 2.0
    names = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(d)]
    return {"mean": R.mean(axis=0), "std": R.std(axis=0, ddof=1) if B > 1 else np.zeros(d),
            "lo": np.quantile(R, alpha, axis=0), "hi": np.quantile(R, 1 - alpha, axis=0),
            "feature_names": names}


def rank_drivers(summary: Dict[str, object], top_k: int = 5, direction: str = "abs") -> List[Tuple[str, float]]:
    """Rank features by bootstrap mean rho (or raw rho). direction: abs|pos|neg."""
    names = list(summary["feature_names"])  # type: ignore[arg-type]
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


__all__ = ["spearman_driver_matrix", "bootstrap_spearman", "rank_drivers"]
