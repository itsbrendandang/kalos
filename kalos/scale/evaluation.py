"""Honest cross-SCALE evaluation for the v0 transfer model.

The product claim ScaleBridge exists to make is "train on small-scale runs,
predict a large scale you have not run yet". A model can look excellent on an
ordinary shuffled or grouped k-fold CV while doing nothing of the sort - CV
that lets folds interpolate between scales it already has both smaller and
larger examples of is an easier question than the one that matters. This
module answers the harder one directly:

  - `leave_one_scale_out_report` holds out every value of `scale_L` in turn
    (never mixing a scale's own runs between train and test), reusing
    `kalos.core.evaluation.logo_report` for the headline pooled number (same
    leakage-checked splitter, same `Surrogate`) and then breaking that same
    per-fold evaluation down two ways the pooled number hides:

    1. PER HELD-OUT SCALE - Spearman and MAE at each of the (typically ~11)
       distinct scales, so one bad scale cannot hide inside an average.

    2. BY EXTRAPOLATION DIRECTION - every held-out scale is labeled
       `extrapolate_up` (every training scale is SMALLER - the product
       claim), `extrapolate_down` (every training scale is LARGER - the
       easier, opposite direction), or `interpolate` (training scales on
       both sides). Interpolating between two known scales is a materially
       easier problem than extrapolating past the largest scale ever run;
       reporting one pooled number across all three would let a model that
       only interpolates well read as if it had solved the harder problem.

  - Every report also carries two NAIVE baselines computed on the exact same
    per-scale train/test split, so the GP's lift is MEASURED, not asserted:

    - `naive_mean`: "carry the small-scale mean" - predict the held-out
      scale's target as the mean target at the SMALLEST scale present in the
      training fold. This is the null hypothesis that there is no scale
      effect at all.
    - `naive_nn`: "nearest matched recipe" - predict each held-out row as the
      target of its nearest training row in STANDARDIZED process-feature
      space (Euclidean, excluding the physics scale features), i.e. "find
      the most similar recipe run at another scale and assume it behaves the
      same".

HONESTY CLAUSE. This module does not assert that the physics-feature GP beats
either naive baseline. `beats_naive_mean_baseline` on the returned report
says whether it measurably did, on whatever data was passed in. Run against
the real synthetic dataset (see this package's tests for how), v0 did NOT
clearly beat the naive baselines on `extrapolate_up` - the one bucket that is
actually the product claim - even though it does beat them, sometimes
substantially, on `interpolate`. See this repo's final report for the
measured numbers; this is a finding about the evaluation harness and the
current feature set on a dataset with SIMULATED scale effects, not a claim
that scale-dependent physics is unlearnable from real data.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from kalos.core.evaluation import logo_report
from kalos.core.surrogate import Surrogate

from .transfer import DEFAULT_SCALE_FEATURE_CONFIG, ScaleFeatureConfig, build_feature_matrix

_MIN_SPEARMAN_N = 3
# Below three points a rank correlation is not a measurement (matches
# `kalos.core.evaluation._spearman`'s convention, restated here rather than
# importing a private helper).


def _spearman(pred: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    # `np.ptp` (max - min), not `np.std`, decides constancy: for an array of
    # bit-identical values `np.std` can come out a tiny nonzero float (e.g.
    # ~1e-15) from rounding in its own mean/variance computation, which would
    # let a genuinely constant array slip past an `== 0` check straight into
    # `spearmanr` - which then raises `ConstantInputWarning` and returns nan
    # anyway. `ptp` on identical values is exactly 0, no summation involved.
    if len(pred) < _MIN_SPEARMAN_N or np.ptp(pred) == 0 or np.ptp(actual) == 0:
        return float("nan")
    r = spearmanr(pred, actual).statistic
    return float(r) if r == r else float("nan")


def _direction(held_out_scale: float, train_scales: NDArray[np.float64]) -> str:
    """`extrapolate_up` if every training scale is smaller than the held-out
    one, `extrapolate_down` if every training scale is larger, else
    `interpolate`. See the module docstring."""
    smaller = bool(np.any(train_scales < held_out_scale))
    larger = bool(np.any(train_scales > held_out_scale))
    if smaller and not larger:
        return "extrapolate_up"
    if larger and not smaller:
        return "extrapolate_down"
    return "interpolate"


def _naive_carry_small_scale_mean(y_train: NDArray[np.float64], scale_train: NDArray[np.float64]) -> float:
    """Mean target among training rows AT THE SMALLEST TRAINING SCALE - the
    "no scale effect, whatever we saw small is what we'll get" null model."""
    smallest = scale_train.min()
    return float(np.mean(y_train[scale_train == smallest]))


def _naive_nearest_recipe(
    process_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    process_test: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Nearest-training-row prediction in standardized process-feature space
    (Euclidean distance; train-set mean/std, a zero-std column contributes
    0 rather than dividing by zero). A test row whose distance to EVERY
    training row is non-finite (e.g. a NaN process feature) predicts `nan`
    rather than an arbitrary match."""
    mu = process_train.mean(axis=0)
    sigma = process_train.std(axis=0)
    sigma_safe = np.where(sigma > 0, sigma, 1.0)
    train_z = (process_train - mu) / sigma_safe
    test_z = (process_test - mu) / sigma_safe

    preds = np.full(test_z.shape[0], np.nan)
    for i in range(test_z.shape[0]):
        dist = np.linalg.norm(train_z - test_z[i], axis=1)
        if np.isfinite(dist).any():
            preds[i] = y_train[int(np.nanargmin(dist))]
    return preds


def _bucket_metrics(
    pred: NDArray[np.float64],
    actual: NDArray[np.float64],
    naive_mean: NDArray[np.float64],
    naive_nn: NDArray[np.float64],
) -> dict:
    nn_finite = np.isfinite(naive_nn)
    out = {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, pred)) if len(actual) else float("nan"),
        "spearman": _spearman(pred, actual),
        "naive_mean_mae": float(mean_absolute_error(actual, naive_mean)) if len(actual) else float("nan"),
        "naive_mean_spearman": _spearman(naive_mean, actual),
        "naive_nn_mae": (
            float(mean_absolute_error(actual[nn_finite], naive_nn[nn_finite])) if nn_finite.any() else float("nan")
        ),
        "naive_nn_spearman": _spearman(naive_nn[nn_finite], actual[nn_finite]) if nn_finite.any() else float("nan"),
    }
    out["beats_naive_mean"] = bool(
        np.isfinite(out["mae"]) and np.isfinite(out["naive_mean_mae"]) and out["mae"] < out["naive_mean_mae"]
    )
    out["beats_naive_nn"] = bool(
        np.isfinite(out["mae"]) and np.isfinite(out["naive_nn_mae"]) and out["mae"] < out["naive_nn_mae"]
    )
    return out


def leave_one_scale_out_report(
    df: pd.DataFrame,
    target_column: str,
    process_columns: Sequence[str],
    *,
    config: ScaleFeatureConfig = DEFAULT_SCALE_FEATURE_CONFIG,
    bounds: NDArray[np.float64] | None = None,
) -> dict:
    """Leave-one-scale-out evaluation of the physics-feature GP against two
    naive baselines. See the module docstring for what each field means.

    Rows with a non-finite feature, target, or `scale_L` are dropped up
    front (`n_rows_dropped`), matching `ScaleUpTransferModel.fit`'s
    no-imputation contract.

    `bounds`: the SAME `(2, d)` box is used for every held-out scale's GP
    fit (default: the min/max of ALL surviving rows, train and held-out
    together) - mirroring `kalos.core.evaluation._oof`'s "one normalization
    box for every fold" contract, so the reported number describes one fixed
    model, not a different input transform per fold.

    Returns a dict:
      - `feature_names`, `n_rows_used`, `n_rows_dropped`, `n_scales`
      - `pooled_logo`: the headline pooled Spearman from
        `kalos.core.evaluation.logo_report` (cross-check against the
        from-scratch per-scale loop below - both consume the same X/y/groups
        and should agree on the pooled number).
      - `per_scale`: one dict per held-out scale L with `scale_L`, `n`,
        `direction`, `mae`, `spearman`, `naive_mean_mae`, `naive_nn_mae`.
      - `by_direction`: `{"extrapolate_up": {...} | None, "extrapolate_down":
        ..., "interpolate": ...}`, each bucket the same shape as `overall`
        below (`None` when no held-out scale fell in that bucket).
      - `overall`: pooled-across-all-scales GP vs. both naive baselines -
        `mae`, `spearman`, `naive_mean_mae`, `naive_mean_spearman`,
        `naive_nn_mae`, `naive_nn_spearman`, `beats_naive_mean`,
        `beats_naive_nn`.

    A held-out scale is skipped (mirrors `kalos.core.evaluation._oof`'s fold
    guard) if fewer than 4 rows remain to train on, or it has no rows itself
    - both meant for pathologically small inputs, not the 5-replicates-per-
    scale shape this module targets.
    """
    X, names = build_feature_matrix(df, process_columns, config)
    y = df[target_column].to_numpy(dtype=float)
    scale = df[config.scale_column].to_numpy(dtype=float)
    n_process = len(process_columns)

    finite = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(scale)
    n_dropped = int((~finite).sum())
    X, y, scale = X[finite], y[finite], scale[finite]
    process_only = X[:, :n_process]

    if bounds is None:
        box = np.vstack([X.min(axis=0), X.max(axis=0)]) if len(X) else None
    else:
        box = np.asarray(bounds, dtype=float)

    unique_scales = np.unique(scale)

    pooled_logo = logo_report(X, y, scale, bounds=box)

    per_scale: list[dict] = []
    all_pred: list[float] = []
    all_actual: list[float] = []
    all_naive_mean: list[float] = []
    all_naive_nn: list[float] = []
    all_direction: list[str] = []

    for s in unique_scales:
        test_mask = scale == s
        train_mask = ~test_mask
        n_test = int(test_mask.sum())
        n_train = int(train_mask.sum())
        if n_train < 4 or n_test < 1:
            continue

        direction = _direction(float(s), scale[train_mask])

        surrogate = Surrogate().fit(X[train_mask], y[train_mask], bounds=box)
        mean, _std = surrogate.posterior(X[test_mask], observation_noise=True)

        naive_mean_val = _naive_carry_small_scale_mean(y[train_mask], scale[train_mask])
        naive_mean_pred = np.full(n_test, naive_mean_val)
        naive_nn_pred = _naive_nearest_recipe(process_only[train_mask], y[train_mask], process_only[test_mask])

        actual = y[test_mask]
        scale_metrics = _bucket_metrics(mean, actual, naive_mean_pred, naive_nn_pred)
        per_scale.append({"scale_L": float(s), "direction": direction, **scale_metrics})

        all_pred.extend(np.asarray(mean).tolist())
        all_actual.extend(actual.tolist())
        all_naive_mean.extend(naive_mean_pred.tolist())
        all_naive_nn.extend(naive_nn_pred.tolist())
        all_direction.extend([direction] * n_test)

    pred_arr = np.asarray(all_pred)
    actual_arr = np.asarray(all_actual)
    naive_mean_arr = np.asarray(all_naive_mean)
    naive_nn_arr = np.asarray(all_naive_nn)
    direction_arr = np.asarray(all_direction)

    by_direction: dict[str, dict | None] = {}
    for label in ("extrapolate_up", "extrapolate_down", "interpolate"):
        mask = direction_arr == label
        by_direction[label] = (
            _bucket_metrics(pred_arr[mask], actual_arr[mask], naive_mean_arr[mask], naive_nn_arr[mask])
            if mask.any()
            else None
        )

    overall = _bucket_metrics(pred_arr, actual_arr, naive_mean_arr, naive_nn_arr)

    return {
        "feature_names": names,
        "n_rows_used": int(len(X)),
        "n_rows_dropped": n_dropped,
        "n_scales": int(len(unique_scales)),
        "pooled_logo": pooled_logo,
        "per_scale": per_scale,
        "by_direction": by_direction,
        "overall": overall,
    }


__all__ = ["leave_one_scale_out_report"]
