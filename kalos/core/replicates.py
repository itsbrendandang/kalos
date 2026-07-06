"""Replicate-aware aggregation and assay noise-floor estimation.

Media DoE run sheets are frequently heavily replicated: the same recipe
(feature row) is run several times because the assay itself is noisy, not
because the process is. Replicates are identified purely by feature identity —
two rows are the "same recipe" iff their feature vectors are equal after
rounding to `decimals` places. That rounding absorbs float round-trip noise
(e.g. a value written out and re-read from a spreadsheet) without merging
recipes that are genuinely different by design.

Aggregating replicates down to one row per recipe (mean titer) gives BO a
reproducible objective to optimize, and the within-replicate spread gives an
honest estimate of the assay noise floor (sigma^2) that a GP can be told about
explicitly instead of inferring from scratch on a handful of points.
"""
from __future__ import annotations

import numpy as np

__all__ = ["aggregate_replicates", "estimate_noise_floor", "noise_report"]


def _group_indices(X: np.ndarray, decimals: int) -> list[np.ndarray]:
    """Return index arrays grouping rows of `X` by equal rounded feature vectors.

    Groups are returned in first-occurrence order (the order each distinct
    rounded row first appears in `X`), so aggregation is deterministic.
    """
    Xr = np.round(np.asarray(X, dtype=float), decimals)
    n = Xr.shape[0]
    seen: dict[tuple[float, ...], list[int]] = {}
    order: list[tuple[float, ...]] = []
    for i in range(n):
        key = tuple(Xr[i].tolist())
        if key not in seen:
            seen[key] = []
            order.append(key)
        seen[key].append(i)
    return [np.asarray(seen[key], dtype=int) for key in order]


def aggregate_replicates(
    X: np.ndarray, y: np.ndarray, *, decimals: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse replicate rows (identical rounded feature vectors) to one row each.

    Replicates are identified as rows of `X` whose feature vectors are equal
    after `np.round(X, decimals)`. Returns `(X_unique, y_mean, y_var, n_reps)`
    where each output has one row per distinct recipe, in first-occurrence
    order (deterministic given the input order):
      - `X_unique`: the (unrounded) feature vector of the first occurrence of
        each recipe.
      - `y_mean`: the mean of `y` within each group.
      - `y_var`: the within-group sample variance (`ddof=1`); `0.0` for
        singleton groups (a variance needs at least 2 observations).
      - `n_reps`: the number of replicate rows folded into each group.
    """
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float).reshape(-1)
    if Xa.ndim != 2 or Xa.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D array (n x d)")
    if Xa.shape[0] != ya.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    groups = _group_indices(Xa, decimals)
    n_groups = len(groups)
    d = Xa.shape[1]
    X_unique = np.empty((n_groups, d), dtype=float)
    y_mean = np.empty(n_groups, dtype=float)
    y_var = np.empty(n_groups, dtype=float)
    n_reps = np.empty(n_groups, dtype=int)

    for g, idx in enumerate(groups):
        X_unique[g] = Xa[idx[0]]
        vals = ya[idx]
        y_mean[g] = vals.mean()
        y_var[g] = float(vals.var(ddof=1)) if vals.shape[0] >= 2 else 0.0
        n_reps[g] = vals.shape[0]

    return X_unique, y_mean, y_var, n_reps


def estimate_noise_floor(X: np.ndarray, y: np.ndarray, *, decimals: int = 6) -> float:
    """Estimate the pooled within-replicate (assay noise) variance, sigma^2.

    Assumes approximately homoscedastic assay noise across recipes: the
    estimate is the mean of the per-group sample variances, taken only over
    groups with at least 2 replicates (a singleton contributes no noise
    information). Returns `float("nan")` if no group has 2+ replicates.
    """
    _, _, y_var, n_reps = aggregate_replicates(X, y, decimals=decimals)
    replicated = n_reps >= 2
    if not np.any(replicated):
        return float("nan")
    return float(y_var[replicated].mean())


def noise_report(X: np.ndarray, y: np.ndarray, *, decimals: int = 6) -> dict:
    """Summarize replicate structure and signal-to-noise for a design matrix.

    Returns a dict with:
      - `n_rows`: total input rows.
      - `n_recipes`: distinct recipes (groups) found.
      - `n_replicated`: recipes with 2 or more replicate rows.
      - `noise_var`: pooled within-replicate variance (see
        `estimate_noise_floor`); `nan` if no recipe is replicated.
      - `signal_var`: sample variance (`ddof=1`) of the per-recipe means;
        `nan` if fewer than 2 recipes.
      - `icc`: intraclass correlation `signal_var / (signal_var + noise_var)`,
        the fraction of total variance attributable to real recipe-to-recipe
        differences rather than assay noise; `nan` if either input is `nan`
        or the denominator is 0.
    """
    X_unique, y_mean, y_var, n_reps = aggregate_replicates(X, y, decimals=decimals)
    n_recipes = X_unique.shape[0]
    replicated = n_reps >= 2
    n_replicated = int(replicated.sum())

    noise_var = float(y_var[replicated].mean()) if n_replicated > 0 else float("nan")
    signal_var = float(y_mean.var(ddof=1)) if n_recipes >= 2 else float("nan")

    if np.isnan(noise_var) or np.isnan(signal_var):
        icc = float("nan")
    else:
        denom = signal_var + noise_var
        icc = float(signal_var / denom) if denom > 0 else float("nan")

    return {
        "n_rows": int(np.asarray(X).shape[0]),
        "n_recipes": n_recipes,
        "n_replicated": n_replicated,
        "noise_var": noise_var,
        "signal_var": signal_var,
        "icc": icc,
    }
