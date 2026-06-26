"""Honest evaluation: grouped cross-validation of the surrogate.

Replicates of one recipe must never straddle a fold, or the reported skill is
inflated. Group by an explicit key (recipe/strain id) when available, else by
identical (rounded) feature rows. Reports the cross-validated Spearman rank
correlation of the GP surrogate — the number that actually predicts ranking
quality on held-out recipes.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from .surrogate import Surrogate


def _row_groups(X: np.ndarray, decimals: int = 6) -> list:
    return [hash(tuple(np.round(r, decimals))) for r in X]


def grouped_folds(groups: Sequence, n_splits: int = 5) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    uniq = list(dict.fromkeys(groups))  # preserve first-seen order
    fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
    folds = np.array([fold_of[g] for g in groups])
    for f in range(min(n_splits, len(uniq))):
        te = folds == f
        tr = ~te
        if te.any() and tr.any():
            yield np.where(tr)[0], np.where(te)[0]


def grouped_cv_spearman(
    X, y, groups: Optional[Sequence] = None, n_splits: int = 5
) -> float:
    """Cross-validated Spearman of the surrogate, leakage-controlled by group."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if groups is None:
        groups = _row_groups(X)
    rhos: list[float] = []
    for tr, te in grouped_folds(groups, n_splits):
        if len(tr) < 4 or len(te) < 2:
            continue
        s = Surrogate().fit(X[tr], y[tr])
        mean, _ = s.posterior(X[te])
        if np.std(y[te]) > 0 and np.std(mean) > 0:
            rho = spearmanr(mean, y[te]).statistic
            if rho == rho:  # not NaN
                rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else float("nan")


__all__ = ["grouped_cv_spearman", "grouped_folds"]
