"""The v0 scale-up transfer model.

ARCHITECTURE, stated once because it is the whole design decision: this is
NOT a new model class. It is `kalos.core.surrogate.Surrogate` (the same GP
every other part of kalos fits) trained on
`[process features, physics scale features]`, where the physics features
(`kalos.scale.features`) are the layer that is supposed to carry the
scale-dependent signal. Scale-up prediction is a FEATURE-ENGINEERING problem
here, not a new-architecture problem - the prior company's mistake was a
hardcoded slope with no mechanism; this module's answer is to give the
proven GP a mechanistic feature instead of inventing a new model to replace
it.

`ScaleUpTransferModel.fit(df_small_scales)` / `.predict(df_new_scale)` is the
public contract. Internally it:
  1. builds `[process_columns..., physics feature columns...]` via
     `build_feature_matrix` (this module) + `compute_scale_features`
     (`kalos.scale.features`);
  2. drops rows with a non-finite feature or target (no imputation - see
     `fit`'s docstring for what is dropped and how the count is reported);
  3. fits `Surrogate` on what remains.

BOUNDS AND EXTRAPOLATION. `Surrogate.fit` requires an explicit `(2, d)` bounds
box and internally sanitizes it via `sanitize_bounds` (this module does not
reimplement that logic - see `kalos.core.surrogate.sanitize_bounds`'s
docstring for what "sanitize" means: finite, non-degenerate, lower <= upper).
When `fit(bounds=None)` (the default), the box is the TRAINING data's own
min/max envelope - the same fallback `kalos.core.evaluation._oof` uses. For a
model that is only ever asked to interpolate, that default is fine. For
genuine scale-UP prediction - the product claim - the box almost certainly
does NOT include the target scale, since by construction `df_small_scales`
does not contain it. BoTorch's `Normalize` input transform does not clamp
inputs to the box; features outside `[0, 1]` after normalization are simply
passed through, and how well the GP extrapolates there is an honest empirical
question, not a guarantee - which is exactly what
`kalos.scale.evaluation.leave_one_scale_out_report`'s "extrapolate_up" bucket
measures. A caller who already knows the target scale range should pass an
explicit `bounds` spanning both the training and target scales (matching how
`kalos.core.evaluation._oof` fixes ONE box across every CV fold, train and
held-out alike, so the reported CV number describes the same normalization
the deployed model would use).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from kalos.core.surrogate import Surrogate

from .features import (
    DEFAULT_GEOMETRY,
    DEFAULT_POWER_NUMBER,
    DEFAULT_VANT_RIET,
    PHYSICS_FEATURE_COLUMNS,
    GeometryAssumptions,
    PowerNumberAssumption,
    VantRietParams,
    compute_scale_features,
)


@dataclass(frozen=True)
class ScaleFeatureConfig:
    """Column names + physics-proxy parameters for building the feature
    matrix. Bundles `kalos.scale.features.compute_scale_features`'s
    arguments so a caller states them once and reuses the same config for
    `fit`, `predict`, and the evaluation harness.
    """

    scale_column: str = "scale_L"
    agitation_column: str = "agitation_rpm"
    airflow_column: str = "airflow_L_per_min"
    reference_scale_L: float = 1.0
    geometry: GeometryAssumptions = DEFAULT_GEOMETRY
    power: PowerNumberAssumption = DEFAULT_POWER_NUMBER
    kla_params: VantRietParams = DEFAULT_VANT_RIET


DEFAULT_SCALE_FEATURE_CONFIG = ScaleFeatureConfig()


def build_feature_matrix(
    df: pd.DataFrame,
    process_columns: Sequence[str],
    config: ScaleFeatureConfig = DEFAULT_SCALE_FEATURE_CONFIG,
) -> tuple[NDArray[np.float64], list[str]]:
    """Assemble `[process_columns..., physics feature columns...]` as a plain
    `(n, d)` float array, plus the matching column names in the same order.

    `process_columns` are taken from `df` AS-IS (cast to float; a non-numeric
    column raises `ValueError` from the cast itself rather than being
    silently coerced) - this function does not decide which raw columns
    belong in the model, the caller does. Raises `KeyError` up front, naming
    every missing column at once, if any `process_columns` entry is not in
    `df` (a column silently dropped from the matrix would change what the
    fitted model means without telling anyone).

    The physics columns are exactly `kalos.scale.features.PHYSICS_FEATURE_COLUMNS`,
    computed via `compute_scale_features` under `config`.
    """
    missing = [c for c in process_columns if c not in df.columns]
    if missing:
        raise KeyError(f"process_columns missing from df: {missing}")

    process_cols = list(process_columns)
    process_arr = df[process_cols].to_numpy(dtype=float) if process_cols else np.empty((len(df), 0))

    physics = compute_scale_features(
        df,
        scale_column=config.scale_column,
        agitation_column=config.agitation_column,
        airflow_column=config.airflow_column,
        reference_scale_L=config.reference_scale_L,
        geometry=config.geometry,
        power=config.power,
        kla_params=config.kla_params,
    )
    physics_arr = physics.to_numpy(dtype=float)

    X = np.hstack([process_arr, physics_arr])
    names = process_cols + list(PHYSICS_FEATURE_COLUMNS)
    return X, names


class ScaleUpTransferModel:
    """Predict a target KPI at a new scale from small-scale observations.

    See the module docstring for the architecture (a `Surrogate` GP over
    process + physics-scale features) and the bounds/extrapolation caveat.
    """

    def __init__(
        self,
        process_columns: Sequence[str],
        target_column: str,
        config: ScaleFeatureConfig = DEFAULT_SCALE_FEATURE_CONFIG,
    ) -> None:
        self.process_columns = list(process_columns)
        self.target_column = target_column
        self.config = config
        self.feature_names: list[str] | None = None
        self.bounds_: NDArray[np.float64] | None = None
        self.n_dropped_rows_: int = 0
        self._surrogate: Surrogate | None = None

    def fit(
        self,
        df_small_scales: pd.DataFrame,
        *,
        bounds: ArrayLike | None = None,
        noise: ArrayLike | float | None = None,
    ) -> "ScaleUpTransferModel":
        """Fit on `df_small_scales`. Rows with a non-finite process column,
        physics feature (see `kalos.scale.features` - a missing
        `agitation_rpm`/`airflow_L_per_min` column, or a NaN value in one
        present, propagates to NaN there), or target value are DROPPED, not
        imputed; the dropped count is recorded on `self.n_dropped_rows_`
        after the call, so a caller can tell a genuinely small dataset from
        one silently gutted by missing sensor columns.

        `bounds` (optional): see the module docstring's "BOUNDS AND
        EXTRAPOLATION" section. Passed straight through to `Surrogate.fit`,
        which sanitizes it (`sanitize_bounds`); this method does not
        reimplement that. Defaults to the (post-drop) training data's own
        min/max per feature when omitted.

        `noise` (optional): forwarded to `Surrogate.fit` unchanged - a known
        assay noise floor, if the caller has one; `None` infers noise as
        `Surrogate` always does.

        Raises `ValueError` if no rows survive the drop.
        """
        X, names = build_feature_matrix(df_small_scales, self.process_columns, self.config)
        y = df_small_scales[self.target_column].to_numpy(dtype=float)
        finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
        self.n_dropped_rows_ = int((~finite).sum())
        X_clean, y_clean = X[finite], y[finite]
        if X_clean.shape[0] == 0:
            raise ValueError(
                "no finite rows remain after dropping non-finite process/physics "
                "features or target values - nothing to fit"
            )

        if bounds is None:
            box = np.vstack([X_clean.min(axis=0), X_clean.max(axis=0)])
        else:
            box = np.asarray(bounds, dtype=float)

        self._surrogate = Surrogate().fit(X_clean, y_clean, bounds=box, noise=noise)
        self.feature_names = names
        self.bounds_ = box
        return self

    def predict(
        self, df_new_scale: pd.DataFrame, *, observation_noise: bool = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict the target at `df_new_scale`. Returns `(mean, std)` in the
        target's original units.

        `observation_noise=True` (default): the PREDICTIVE distribution for a
        new measurement (latent uncertainty + fitted assay noise) - the right
        default here because the use case is "what titer will this actual
        future batch read", a measurement, and scoring a measurement against
        the narrower latent band under-covers by construction (see
        `Surrogate.posterior`'s docstring, which this mirrors). Pass `False`
        for the latent process-surface uncertainty instead.

        Raises `RuntimeError` if called before `fit`. Raises whatever
        `build_feature_matrix` raises (`KeyError`) if `df_new_scale` is
        missing a required column. Rows with a non-finite feature are NOT
        dropped here (unlike `fit`) - `Surrogate.posterior` is a pure
        function of its input and will return `nan`/`inf`-tainted output for
        such a row rather than silently resizing the returned arrays out from
        under the caller's `df_new_scale` index.
        """
        if self._surrogate is None or self.feature_names is None:
            raise RuntimeError("call fit() before predict()")
        X, names = build_feature_matrix(df_new_scale, self.process_columns, self.config)
        assert names == self.feature_names, "feature layout changed between fit() and predict()"
        return self._surrogate.posterior(X, observation_noise=observation_noise)


__all__ = [
    "ScaleFeatureConfig",
    "DEFAULT_SCALE_FEATURE_CONFIG",
    "build_feature_matrix",
    "ScaleUpTransferModel",
]
