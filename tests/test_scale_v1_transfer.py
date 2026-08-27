"""Tests for `ScaleUpTransferModel`'s new `candidate` constructor parameter
(default `"v0"`, unchanged behavior - see `kalos/scale/transfer.py`'s "V1
CANDIDATES" docstring section for why `"physics_mean"`/`"multi_fidelity"`
are opt-in rather than the default). Runs entirely on tiny fabricated
frames; no dependency on the real dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.scale.candidates import MultiFidelitySurrogate, PhysicsMeanSurrogate
from kalos.scale.transfer import ScaleUpTransferModel

_RNG = np.random.default_rng(20260825)
_PROCESS_COLUMNS = ["ph_setpoint", "temperature_C"]


def _fabricate_batches(scales: list[float], n_per_scale: int = 4) -> pd.DataFrame:
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


def test_default_candidate_is_v0():
    model = ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L")
    assert model.candidate == "v0"


def test_default_candidate_fits_a_plain_surrogate_not_a_v1_class():
    train = _fabricate_batches([1.0, 10.0, 100.0])
    model = ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L")
    model.fit(train)
    assert not isinstance(model._surrogate, (PhysicsMeanSurrogate, MultiFidelitySurrogate))


@pytest.mark.parametrize("candidate,cls", [("physics_mean", PhysicsMeanSurrogate), ("multi_fidelity", MultiFidelitySurrogate)])
def test_v1_candidate_fits_the_expected_class(candidate, cls):
    train = _fabricate_batches([1.0, 10.0, 100.0])
    model = ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L", candidate=candidate)
    model.fit(train)
    assert isinstance(model._surrogate, cls)


@pytest.mark.parametrize("candidate", ["v0", "physics_mean", "multi_fidelity"])
def test_every_candidate_fit_predict_roundtrip_shapes(candidate):
    train = _fabricate_batches([1.0, 10.0, 100.0])
    test = _fabricate_batches([50.0], n_per_scale=3)

    model = ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L", candidate=candidate)
    model.fit(train)
    mean, std = model.predict(test)

    assert mean.shape == (len(test),)
    assert std.shape == (len(test),)
    assert np.isfinite(mean).all()
    assert np.all(std > 0)


def test_unknown_candidate_raises_value_error_naming_the_valid_choices():
    with pytest.raises(ValueError, match="v0"):
        ScaleUpTransferModel(_PROCESS_COLUMNS, "titer_g_per_L", candidate="not_a_real_candidate")
