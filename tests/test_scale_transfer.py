"""Unit tests for `kalos.scale.transfer` - `ScaleUpTransferModel` and
`build_feature_matrix`. Runs entirely on tiny fabricated frames; no
dependency on the real dataset. Seeded for reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.scale.features import PHYSICS_FEATURE_COLUMNS
from kalos.scale.transfer import ScaleFeatureConfig, ScaleUpTransferModel, build_feature_matrix

_RNG = np.random.default_rng(20260825)


def _fabricate_batches(scales: list[float], n_per_scale: int = 4) -> pd.DataFrame:
    """A tiny synthetic batch sheet: several replicate rows per scale, with a
    target that has a genuine (fabricated, known) scale dependence so a fit
    has something real to find."""
    rows = []
    for s in scales:
        for _ in range(n_per_scale):
            ph = float(_RNG.uniform(6.9, 7.1))
            temp = float(_RNG.uniform(36.5, 37.0))
            rpm = float(_RNG.uniform(80, 250))
            airflow = float(max(s * 0.05, 0.05) * float(_RNG.uniform(0.8, 1.2)))
            # Fabricated titer: mildly decreasing with log-scale, small noise.
            titer = 5.0 - 0.2 * np.log10(s + 1e-6) + float(_RNG.normal(0, 0.05))
            rows.append(
                {
                    "scale_L": s,
                    "agitation_rpm": rpm,
                    "airflow_L_per_min": airflow,
                    "ph_setpoint": ph,
                    "temperature_C": temp,
                    "titer_g_per_L": titer,
                }
            )
    return pd.DataFrame(rows)


_PROCESS_COLUMNS = ["ph_setpoint", "temperature_C"]


def test_build_feature_matrix_shape_and_names():
    df = _fabricate_batches([1.0, 10.0])
    X, names = build_feature_matrix(df, _PROCESS_COLUMNS)
    assert X.shape == (len(df), len(_PROCESS_COLUMNS) + len(PHYSICS_FEATURE_COLUMNS))
    assert names == _PROCESS_COLUMNS + list(PHYSICS_FEATURE_COLUMNS)
    assert np.isfinite(X).all()


def test_build_feature_matrix_missing_process_column_raises_keyerror():
    df = _fabricate_batches([1.0])
    with pytest.raises(KeyError):
        build_feature_matrix(df, ["not_a_real_column"])


def test_build_feature_matrix_empty_process_columns_is_physics_only():
    df = _fabricate_batches([1.0])
    X, names = build_feature_matrix(df, [])
    assert names == list(PHYSICS_FEATURE_COLUMNS)
    assert X.shape == (len(df), len(PHYSICS_FEATURE_COLUMNS))


def test_fit_predict_roundtrip_shapes():
    train = _fabricate_batches([1.0, 10.0, 100.0])
    test = _fabricate_batches([50.0], n_per_scale=3)

    model = ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L")
    model.fit(train)
    mean, std = model.predict(test)

    assert mean.shape == (len(test),)
    assert std.shape == (len(test),)
    assert np.isfinite(mean).all()
    assert np.all(std > 0)
    assert model.n_dropped_rows_ == 0
    assert model.bounds_ is not None
    assert model.bounds_.shape == (2, len(_PROCESS_COLUMNS) + len(PHYSICS_FEATURE_COLUMNS))


def test_fit_default_bounds_are_the_training_envelope():
    train = _fabricate_batches([1.0, 10.0, 100.0])
    X, _ = build_feature_matrix(train, _PROCESS_COLUMNS)

    model = ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L")
    model.fit(train)

    np.testing.assert_allclose(model.bounds_[0], X.min(axis=0))
    np.testing.assert_allclose(model.bounds_[1], X.max(axis=0))


def test_fit_accepts_explicit_bounds_spanning_the_target_scale():
    train = _fabricate_batches([1.0, 10.0])
    test = _fabricate_batches([1000.0], n_per_scale=2)

    # Explicit bounds wide enough to cover the eventual large-scale target,
    # as the module docstring recommends for genuine extrapolation.
    X_train, names = build_feature_matrix(train, _PROCESS_COLUMNS)
    X_test, _ = build_feature_matrix(test, _PROCESS_COLUMNS)
    combined = np.vstack([X_train, X_test])
    bounds = np.vstack([combined.min(axis=0), combined.max(axis=0)])

    model = ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L")
    model.fit(train, bounds=bounds)
    np.testing.assert_allclose(model.bounds_, bounds)

    mean, std = model.predict(test)
    assert np.isfinite(mean).all()
    assert np.all(std > 0)


def test_fit_drops_nonfinite_rows_and_reports_count():
    train = _fabricate_batches([1.0, 10.0, 100.0])
    train.loc[0, "ph_setpoint"] = np.nan
    train.loc[1, "agitation_rpm"] = np.nan  # poisons specific_power/kLa for that row

    model = ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L")
    model.fit(train)

    assert model.n_dropped_rows_ == 2


def test_fit_raises_when_no_finite_rows_survive():
    train = _fabricate_batches([1.0])
    train["ph_setpoint"] = np.nan

    model = ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L")
    with pytest.raises(ValueError):
        model.fit(train)


def test_predict_before_fit_raises_runtime_error():
    model = ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L")
    with pytest.raises(RuntimeError):
        model.predict(_fabricate_batches([1.0]))


def test_predict_missing_column_raises_keyerror():
    train = _fabricate_batches([1.0, 10.0])
    model = ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L")
    model.fit(train)

    bad_test = _fabricate_batches([5.0]).drop(columns=["ph_setpoint"])
    with pytest.raises(KeyError):
        model.predict(bad_test)


def test_predict_observation_noise_widens_std():
    train = _fabricate_batches([1.0, 10.0, 100.0])
    test = _fabricate_batches([50.0], n_per_scale=2)

    model = ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L")
    model.fit(train)
    _mean_latent, std_latent = model.predict(test, observation_noise=False)
    _mean_pred, std_pred = model.predict(test, observation_noise=True)

    assert np.all(std_pred >= std_latent - 1e-9)


def test_custom_scale_feature_config_column_names():
    df = _fabricate_batches([1.0, 10.0]).rename(
        columns={"scale_L": "vessel_L", "agitation_rpm": "stir_rpm", "airflow_L_per_min": "gas_flow"}
    )
    config = ScaleFeatureConfig(scale_column="vessel_L", agitation_column="stir_rpm", airflow_column="gas_flow")
    X, names = build_feature_matrix(df, _PROCESS_COLUMNS, config)
    assert names == _PROCESS_COLUMNS + list(PHYSICS_FEATURE_COLUMNS)
    assert np.isfinite(X).all()
