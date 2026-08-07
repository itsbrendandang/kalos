"""Mechanistic simulator sanity tests.

These pin the physics we rely on for the closed-loop benchmark: deterministic and
bounded trajectories, monotonically accumulating titer, plausible lactate range
(the diauxic re-uptake must keep it in a real regime), and genuine INTERIOR optima
in both feed rate and pH (so the optimizer has something real to find).
"""
from __future__ import annotations

import numpy as np

from pipeline.sim import (
    DESIGN_SPACE,
    ProcessParams,
    endpoint_titer,
    sample_doe,
    simulate,
    to_featurized_rows,
)


def test_trajectory_is_bounded_and_plausible():
    tr = simulate(ProcessParams(glc_feed_rate=1.0, glc_0=40, vcd_0=0.5, ph_setpoint=7.0))
    assert (tr[["VCD", "Glc", "Lac", "titer"]] >= 0).all().all()
    assert 10 < tr["VCD"].max() < 80           # a real fed-batch peak VCD band
    assert tr["Lac"].max() < 150               # diauxie keeps lactate realistic
    assert tr["titer"].is_monotonic_increasing  # production only accumulates
    assert tr.iloc[-1]["titer"] > 100


def test_endpoint_titer_is_deterministic():
    p = ProcessParams(1.0, 40, 0.5, 7.0)
    assert endpoint_titer(p) == endpoint_titer(p)


def test_feed_rate_has_an_interior_optimum():
    # starving (no feed) and over-feeding (lactate overflow) both lose to a mid feed.
    t_low = endpoint_titer(ProcessParams(0.0, 40, 0.5, 7.0))
    t_mid = endpoint_titer(ProcessParams(0.8, 40, 0.5, 7.0))
    t_high = endpoint_titer(ProcessParams(2.5, 40, 0.5, 7.0))
    assert t_mid > t_low and t_mid > t_high


def test_ph_has_an_interior_optimum():
    t_opt = endpoint_titer(ProcessParams(1.0, 40, 0.5, 7.0))
    assert t_opt > endpoint_titer(ProcessParams(1.0, 40, 0.5, 6.6))
    assert t_opt > endpoint_titer(ProcessParams(1.0, 40, 0.5, 7.4))


def test_noise_is_reproducible_and_perturbs():
    p = ProcessParams(1.0, 40, 0.5, 7.0)
    a = endpoint_titer(p, noise_cv=0.1, rng=np.random.default_rng(0))
    b = endpoint_titer(p, noise_cv=0.1, rng=np.random.default_rng(0))
    assert a == b                               # same seed -> same draw
    assert a != endpoint_titer(p)               # noise actually moves it


def test_doe_sample_respects_bounds():
    params = sample_doe(50, seed=3)
    assert len(params) == 50
    for p in params:
        for k, (lo, hi) in DESIGN_SPACE.items():
            assert lo <= getattr(p, k) <= hi


def test_featurized_rows_shape():
    params = sample_doe(5, seed=1)
    titers = [endpoint_titer(p) for p in params]
    df = to_featurized_rows(params, titers)
    assert len(df) == 5
    for col in ("well_id", "clone", "titer", *DESIGN_SPACE):
        assert col in df.columns
    assert df["titer"].gt(0).all()
