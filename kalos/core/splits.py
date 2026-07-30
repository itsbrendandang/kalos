"""Group-aware cross-validation + a leakage tripwire.

Ported from the lean engine. Replicates of one recipe (or one strain/campaign)
must never land on both sides of a CV split, or the model is graded against a
near-identical twin it trained on and the reported skill is inflated.
`make_splits` enforces grouping; `assert_no_group_leakage` turns it into a
checked invariant. `row_hash_groups` is the safe fallback group key when no
explicit recipe/strain id is available (identical feature rows are replicates).
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def row_hash_groups(
    X: pd.DataFrame, feature_names: Optional[Sequence[str]] = None, decimals: int = 6
) -> np.ndarray:
    """Replicate-aware group ids: rows with identical (rounded) feature vectors
    share a group. Deterministic via pandas.factorize over a stringified row."""
    df = pd.DataFrame(X)
    cols = list(feature_names) if feature_names is not None else list(df.columns)
    df = df[cols].copy()
    num = df.columns[[pd.api.types.is_numeric_dtype(df[c]) for c in df.columns]]
    if len(num):
        # Round numeric values for replicate matching, but keep NaN DISTINCT from
        # a real 0.0. The old `nan_to_num(..., 0.0)` collapsed missingness to zero,
        # merging unrelated rows into one replicate group — which then leaked them
        # across every CV split. NaN gets a reserved "NA" token no real value hits.
        arr = np.round(df[num].to_numpy(float), decimals)
        df[num] = pd.DataFrame(
            np.where(np.isnan(arr), "NA", arr.astype("U32")),
            index=df.index,
            columns=list(num),
        )
    keys = np.array(["|".join(map(str, row)) for row in df.to_numpy(dtype=object)])
    return pd.factorize(keys)[0]


def _is_imbalanced(y: ArrayLike, threshold: float = 0.10) -> bool:
    counts = pd.Series(y).astype("category").value_counts(normalize=True)
    if len(counts) < 2:
        return False
    return bool(((counts - 1.0 / len(counts)).abs() > threshold).any())


def make_splits(
    X,
    y: ArrayLike,
    groups: ArrayLike,
    n_splits: int = 5,
    stratify: Optional[bool] = None,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Group-aware CV splits with small-data guards, verified leakage-free.

    `random_state` only bites on the stratified path, which needs a discrete `y`
    (`dtype.kind in "iubO"`). A continuous target always takes the unshuffled
    `GroupKFold` branch below, where the split is fully determined by the group
    labels: seeds 1 and 999 return identical folds. That is deterministic, not
    broken, but it means a bootstrap CI computed on these folds is conditional on
    one fixed partition and will understate how much the estimate moves with
    `n_splits` (on the real media DoE, 0.44 to 0.71 across n_splits 3 to 8).
    Quote the CI alongside a sensitivity sweep, not on its own.
    """
    y_arr = np.asarray(pd.Series(y).values)
    g_arr = np.asarray(pd.Series(groups).astype(str).values)
    n = len(g_arr)
    n_groups = len(pd.unique(g_arr))

    if stratify is None:
        stratify = _is_imbalanced(y_arr)
    if n_splits > n_groups:
        warnings.warn(f"n_splits={n_splits} > n_groups={n_groups}; reducing to {n_groups}.")
        n_splits = n_groups

    use_strat = bool(stratify) and y_arr.dtype.kind in "iubO" and len(pd.unique(y_arr)) >= 2
    if use_strat:
        n_splits = min(n_splits, int(pd.Series(y_arr).value_counts().min()))
    if n_splits < 2:
        warnings.warn(
            "n_splits < 2 after small-data guards: too few groups for a "
            "leakage-free split. Returning no splits (cross-validation "
            "unavailable) rather than a train==validation dummy."
        )
        return []

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    if use_strat:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for tr, va in cv.split(np.zeros(n), y_arr, groups=g_arr):
            splits.append((tr, va))
    else:
        cv = GroupKFold(n_splits=n_splits)
        for tr, va in cv.split(np.zeros(n), groups=g_arr):
            splits.append((tr, va))

    assert_no_group_leakage(splits, g_arr)
    return splits


def assert_no_group_leakage(splits: Sequence[Tuple[np.ndarray, np.ndarray]], groups: ArrayLike) -> None:
    """Raise AssertionError if any fold shares a group between train and val."""
    g = np.asarray(pd.Series(groups).astype(str).values)
    for i, (tr, va) in enumerate(splits):
        overlap = set(g[tr]) & set(g[va])
        if overlap:
            raise AssertionError(f"Group leakage in split {i}: {sorted(overlap)[:5]}")


__all__ = ["row_hash_groups", "make_splits", "assert_no_group_leakage"]
