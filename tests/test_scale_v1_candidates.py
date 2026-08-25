"""Unit tests for `kalos.scale.candidates` - `PhysicsMeanSurrogate` (candidate
A, physics-informed mean function) and `MultiFidelitySurrogate` (candidate B,
scale-as-fidelity). Runs entirely on tiny fabricated frames; no dependency on
the real dataset. Seeded for reproducibility.

See `kalos/scale/candidates.py`'s module docstring for the full rationale
and this repo's v1 report for the measured comparison against v0 - these
tests only cover fit/predict correctness and the harness contract, not the
promotion verdict (v0 stays default; see `test_scale_v1_evaluation.py` and
`kalos/scale/transfer.py`'s "V1 CANDIDATES" docstring section).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.scale.candidates import (
    DEFAULT_FIDELITY_FEATURE,
    DEFAULT_PHYSICS_MEAN_FEATURES,
    MultiFidelitySurrogate,
    PhysicsMeanSurrogate,
    ScaleCandidateModel,
)
from kalos.scale.transfer import build_feature_matrix

_RNG = np.random.default_rng(20260825)
_PROCESS_COLUMNS = ["ph_setpoint", "temperature_C"]


def _fabricate_batches(scales: list[float], n_per_scale: int = 5) -> pd.DataFrame:
    rows = []
    for s in scales:
        for _ in range(n_per_scale):
            ph = float(_RNG.uniform(6.9, 7.1))
            temp = float(_RNG.uniform(36.5, 37.0))
            rpm = float(_RNG.uniform(80, 250))
            airflow = float(max(s * 0.05, 0.05) * float(_RNG.uniform(0.8, 1.2)))
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


def _xy_bounds(df: pd.DataFrame):
    X, names = build_feature_matrix(df, _PROCESS_COLUMNS)
    y = df["titer_g_per_L"].to_numpy(dtype=float)
    bounds = np.vstack([X.min(axis=0), X.max(axis=0)])
    return X, y, names, bounds


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_both_candidates_satisfy_the_scale_candidate_model_protocol():
    df = _fabricate_batches([1.0, 10.0, 100.0])
    _X, _y, names, _bounds = _xy_bounds(df)
    assert isinstance(PhysicsMeanSurrogate(names), ScaleCandidateModel)
    assert isinstance(MultiFidelitySurrogate(names), ScaleCandidateModel)


# ---------------------------------------------------------------------------
# PhysicsMeanSurrogate (candidate A)
# ---------------------------------------------------------------------------


def test_physics_mean_default_mean_features_are_the_documented_three():
    assert DEFAULT_PHYSICS_MEAN_FEATURES == ("log_volume_ratio", "kla_proxy_per_s", "hydrostatic_pressure_mmHg")


def test_physics_mean_fit_predict_roundtrip_shapes():
    train = _fabricate_batches([1.0, 10.0, 100.0])
    test = _fabricate_batches([50.0], n_per_scale=3)
    X_train, y_train, names, bounds = _xy_bounds(train)
    X_test, _y_test, _n2, _b2 = _xy_bounds(test)

    model = PhysicsMeanSurrogate(names)
    model.fit(X_train, y_train, bounds=bounds)
    mean, std = model.posterior(X_test, observation_noise=True)

    assert mean.shape == (len(test),)
    assert std.shape == (len(test),)
    assert np.isfinite(mean).all()
    assert np.all(std > 0)


def test_physics_mean_rejects_mean_feature_not_in_feature_names():
    df = _fabricate_batches([1.0, 10.0])
    _X, _y, names, _bounds = _xy_bounds(df)
    with pytest.raises(KeyError):
        PhysicsMeanSurrogate(names, mean_feature_names=["not_a_real_feature"])


def test_physics_mean_rejects_x_with_wrong_column_count():
    df = _fabricate_batches([1.0, 10.0, 100.0])
    X, y, names, bounds = _xy_bounds(df)
    model = PhysicsMeanSurrogate(names)
    with pytest.raises(ValueError):
        model.fit(X[:, :-1], y, bounds=bounds[:, :-1])


def test_physics_mean_accepts_fixed_noise():
    train = _fabricate_batches([1.0, 10.0, 100.0])
    X, y, names, bounds = _xy_bounds(train)
    model = PhysicsMeanSurrogate(names)
    model.fit(X, y, bounds=bounds, noise=0.01)
    mean, std = model.posterior(X[:2], observation_noise=True)
    assert np.isfinite(mean).all()
    assert np.all(std > 0)


def test_physics_mean_posterior_before_fit_raises():
    df = _fabricate_batches([1.0])
    _X, _y, names, _bounds = _xy_bounds(df)
    model = PhysicsMeanSurrogate(names)
    with pytest.raises(AssertionError):
        model.posterior(np.zeros((1, len(names))))


# ---------------------------------------------------------------------------
# MultiFidelitySurrogate (candidate B)
# ---------------------------------------------------------------------------


def test_multi_fidelity_default_fidelity_feature_is_log_volume_ratio():
    assert DEFAULT_FIDELITY_FEATURE == "log_volume_ratio"


def test_multi_fidelity_fit_predict_roundtrip_shapes():
    train = _fabricate_batches([1.0, 10.0, 100.0])
    test = _fabricate_batches([50.0], n_per_scale=3)
    X_train, y_train, names, bounds = _xy_bounds(train)
    X_test, _y_test, _n2, _b2 = _xy_bounds(test)

    model = MultiFidelitySurrogate(names)
    model.fit(X_train, y_train, bounds=bounds)
    mean, std = model.posterior(X_test, observation_noise=True)

    assert mean.shape == (len(test),)
    assert std.shape == (len(test),)
    assert np.isfinite(mean).all()
    assert np.all(std > 0)


def test_multi_fidelity_rejects_unknown_fidelity_feature_name():
    df = _fabricate_batches([1.0, 10.0])
    _X, _y, names, _bounds = _xy_bounds(df)
    with pytest.raises(KeyError):
        MultiFidelitySurrogate(names, fidelity_feature_name="not_a_real_feature")


def test_multi_fidelity_accepts_fixed_noise():
    train = _fabricate_batches([1.0, 10.0, 100.0])
    X, y, names, bounds = _xy_bounds(train)
    model = MultiFidelitySurrogate(names)
    model.fit(X, y, bounds=bounds, noise=0.01)
    mean, std = model.posterior(X[:2], observation_noise=True)
    assert np.isfinite(mean).all()
    assert np.all(std > 0)


def test_multi_fidelity_extrapolation_beyond_training_box_is_finite():
    """The product claim's actual shape: fit on small scales, predict a
    scale the model has never seen. Not asserting a QUALITY bar here (that
    is the harness's job, see `test_scale_v1_evaluation.py` and this repo's
    v1 report) - just that a genuinely out-of-training-box query does not
    blow up to nan/inf."""
    train = _fabricate_batches([1.0, 10.0])
    test = _fabricate_batches([1000.0], n_per_scale=2)
    X_train, y_train, names, _b = _xy_bounds(train)
    X_test, _y_test, _n2, _b2 = _xy_bounds(test)
    combined = np.vstack([X_train, X_test])
    wide_bounds = np.vstack([combined.min(axis=0), combined.max(axis=0)])

    model = MultiFidelitySurrogate(names)
    model.fit(X_train, y_train, bounds=wide_bounds)
    mean, std = model.posterior(X_test, observation_noise=True)
    assert np.isfinite(mean).all()
    assert np.all(std > 0)
