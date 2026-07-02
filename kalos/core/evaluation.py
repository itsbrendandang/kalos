"""Honest evaluation: grouped cross-validation of the surrogate.

All grouped CV routes through the single leakage-checked splitter in `splits.py`
(no second, weaker splitter lives here anymore). Each fold is fit under the SAME
input-normalization box the deployed model uses (pass `bounds`), out-of-fold
predictions are POOLED, and the rank correlation is reported with a group-level
bootstrap confidence band. At a few dozen rows a bare point estimate reads as far
more precise than it is, so `grouped_cv_report` is the number to quote.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from .splits import make_splits, row_hash_groups
from .surrogate import Surrogate


def grouped_folds(groups: Sequence, n_splits: int = 5) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Leakage-checked grouped folds. Kept for compatibility; delegates to the
    one splitter in `splits.make_splits`, so there is a single grouping code path.
    Yields nothing when there are too few groups for an honest split (the old
    round-robin version returned a train==validation dummy)."""
    groups = list(groups)
    n = len(groups)
    yield from make_splits(np.zeros((n, 1)), np.zeros(n), groups, n_splits=n_splits)


def _oof(X, y, groups, n_splits, bounds):
    """Pooled out-of-fold predictions under one fixed normalization box.
    Returns (pred, actual, group, n_folds)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    if groups is None:
        groups = row_hash_groups(X)
    groups = np.asarray(groups)
    # One normalization box for every fold AND for production. Fitting each fold
    # to its own training-data envelope (the old default) measures a different
    # input transform than the shipped model, so the CV number would not describe
    # what actually ships.
    if bounds is None and len(X):
        bounds = np.vstack([X.min(axis=0), X.max(axis=0)])
    pred, actual, grp, n_folds = [], [], [], 0
    for tr, te in make_splits(X, y, groups, n_splits=n_splits):
        if len(tr) < 4 or len(te) < 1:
            continue
        n_folds += 1
        s = Surrogate().fit(X[tr], y[tr], bounds=bounds)
        mean, _ = s.posterior(X[te])
        pred.extend(np.asarray(mean).ravel().tolist())
        actual.extend(y[te].tolist())
        grp.extend(groups[te].tolist())
    return np.asarray(pred), np.asarray(actual), np.asarray(grp, dtype=object), n_folds


def _spearman(pred, actual) -> float:
    if len(pred) < 3 or np.std(pred) == 0 or np.std(actual) == 0:
        return float("nan")
    r = spearmanr(pred, actual).statistic
    return float(r) if r == r else float("nan")


def grouped_cv_spearman(X, y, groups=None, n_splits: int = 5, bounds=None) -> float:
    """Pooled out-of-fold Spearman of the surrogate, leakage-controlled by group.
    Pools OOF predictions instead of averaging tiny per-fold rhos (which can only
    be +/-1 at this n), so the number reflects held-out ranking on real scale."""
    pred, actual, _, _ = _oof(X, y, groups, n_splits, bounds)
    return _spearman(pred, actual)


def grouped_cv_report(
    X, y, groups=None, n_splits: int = 5, bounds=None, n_boot: int = 1000, random_state: int = 0
) -> dict:
    """The honest CV number: pooled OOF Spearman plus a group-level bootstrap 95%
    CI (resample GROUPS, not rows), the count of held-out points and groups, and
    the fold count. The CI is wide on purpose at small n — relative ranking is
    more trustworthy than the absolute value."""
    pred, actual, grp, n_folds = _oof(X, y, groups, n_splits, bounds)
    point = _spearman(pred, actual)
    uniq = np.unique(grp)
    lo = hi = float("nan")
    if len(uniq) >= 3 and len(pred) >= 3:
        rng = np.random.default_rng(random_state)
        boot: list[float] = []
        for _ in range(int(n_boot)):
            take = rng.choice(uniq, size=len(uniq), replace=True)
            mask = np.concatenate([np.where(grp == g)[0] for g in take])
            r = _spearman(pred[mask], actual[mask])
            if r == r:
                boot.append(r)
        if boot:
            lo = float(np.percentile(boot, 2.5))
            hi = float(np.percentile(boot, 97.5))
    return {
        "spearman": point,
        "ci95": (lo, hi),
        "n_oof": int(len(pred)),
        "n_groups": int(len(uniq)),
        "n_folds": int(n_folds),
        "oof_actual": actual.tolist(),
        "oof_pred": pred.tolist(),
    }


__all__ = ["grouped_cv_spearman", "grouped_cv_report", "grouped_folds"]
