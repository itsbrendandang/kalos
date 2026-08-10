"""Honest evaluation: grouped cross-validation of the surrogate.

All grouped CV routes through the single leakage-checked splitter in `splits.py`
(no second, weaker splitter lives here anymore). Each fold is fit under the SAME
input-normalization box the deployed model uses (pass `bounds`), out-of-fold
predictions are POOLED, and the rank correlation is reported with a group-level
bootstrap confidence band. At a few dozen rows a bare point estimate reads as far
more precise than it is, so `grouped_cv_report` is the number to quote.
"""
from __future__ import annotations

from typing import Iterator, Sequence, Tuple

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


def _oof(X, y, groups, n_splits, bounds, cat_dims=None):
    """Pooled out-of-fold predictions under one fixed normalization box.
    Returns (pred, actual, group, n_folds).

    `cat_dims` (optional) marks categorical columns so each fold is fit with the
    same mixed GP the deployed model uses; omitted, every fold is the continuous
    SingleTaskGP (unchanged behavior)."""
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
        s = Surrogate().fit(X[tr], y[tr], bounds=bounds, cat_dims=cat_dims)
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
    X, y, groups=None, n_splits: int = 5, bounds=None, n_boot: int = 1000, random_state: int = 0,
    cat_dims=None,
) -> dict:
    """The honest CV number: pooled OOF Spearman plus a group-level bootstrap 95%
    CI (resample GROUPS, not rows), the count of held-out points and groups, and
    the fold count. The CI is wide on purpose at small n — relative ranking is
    more trustworthy than the absolute value.

    `cat_dims` (optional) marks categorical columns so the CV evaluates the same
    mixed GP the deployed model uses; omitted, behavior is unchanged."""
    pred, actual, grp, n_folds = _oof(X, y, groups, n_splits, bounds, cat_dims=cat_dims)
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


def producer_only_spearman(
    oof_actual, oof_pred, *, threshold: float = 0.0
) -> dict:
    """Rank agreement restricted to the rows that actually PRODUCED.

    WHY THE POOLED NUMBER IS NOT ENOUGH. `grouped_cv_report`'s Spearman is taken
    over every held-out row, producers and non-producers together. On a sheet with
    a substantial fraction of non-producers, a model can score well on that number
    by doing nothing more than separating zeros from non-zeros - which is a
    feasibility classifier, not a ranking of recipes. The client's decision is
    "which of my producing recipes is best", and that is a different question.

    `BENCHMARK.md` documents this concretely: on the real media DoE the pooled
    held-out Spearman was a plausible-looking 0.37-0.52 while feasibility was
    "mostly a stable property" and never the bottleneck, so most of that pooled
    agreement was the easy part of the problem.

    `threshold` is the value above which a measurement counts as production. The
    default 0.0 means "any non-zero measurement", which is a PROXY. The correct
    value is the assay's limit of detection: below LOD a reading is censored, not
    zero, and treating it as a true zero is a different modelling error. Pass the
    LOD when it is known and say so in the report.

    Returns `spearman` as `nan` when it cannot be computed - fewer than three
    producing rows, or no variation among them - rather than a number that looks
    like a measurement. `evaluable` says which case you are in.
    """
    actual = np.asarray(oof_actual, dtype=float).reshape(-1)
    pred = np.asarray(oof_pred, dtype=float).reshape(-1)
    if actual.shape != pred.shape:
        raise ValueError("oof_actual and oof_pred must be the same length")

    mask = np.isfinite(actual) & np.isfinite(pred) & (actual > threshold)
    n_prod = int(mask.sum())
    n_total = int(np.isfinite(actual).sum())
    rho = _spearman(pred[mask], actual[mask]) if n_prod >= 3 else float("nan")
    return {
        "spearman": rho,
        "n_producers": n_prod,
        "n_total": n_total,
        "threshold": float(threshold),
        # A pooled score can look fine while this is unmeasurable, so the caller
        # must be able to tell "no signal" from "not enough producers to ask".
        "evaluable": bool(n_prod >= 3 and rho == rho),
    }


__all__ = ["grouped_cv_spearman", "grouped_cv_report", "grouped_folds", "producer_only_spearman"]
