"""Unit tests for the `model_factory` extension to
`kalos.scale.evaluation.leave_one_scale_out_report` (see that function's
docstring). Covers: the default path is byte-for-byte unchanged (no
`model_factory` at all - the pre-existing `tests/test_scale_evaluation.py`
suite already locks this in end to end; this file only adds the swap-model
path), both v1 candidates run under the harness, `pooled_logo`'s
`independent_crosscheck` flag, and determinism. Fabricated frames only - no
dependency on the real dataset (see `test_scale_v1_real_data.py` for that).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from kalos.scale.candidates import MultiFidelitySurrogate, PhysicsMeanSurrogate
from kalos.scale.evaluation import leave_one_scale_out_report


def _same_value(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True  # NaN != NaN under `==`; per-scale naive_mean_spearman is legitimately NaN
    return a == b


def _same_report_rows(rows_a: list[dict], rows_b: list[dict]) -> bool:
    """Deep-compare two `per_scale`-shaped lists of dicts, treating NaN as
    equal to NaN (plain `==`/dict-equality does not - `nan != nan` by
    IEEE-754 - and `per_scale`'s `naive_mean_spearman` is legitimately NaN
    whenever a fold's naive-mean baseline is a constant, which is every
    fold: see `kalos.scale.evaluation._bucket_metrics`)."""
    if len(rows_a) != len(rows_b):
        return False
    for a, b in zip(rows_a, rows_b):
        if a.keys() != b.keys():
            return False
        if not all(_same_value(a[k], b[k]) for k in a):
            return False
    return True


_RNG = np.random.default_rng(20260825)
_SCALES = [1.0, 10.0, 100.0, 1000.0]
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


def test_model_factory_none_matches_default_v0_report_exactly():
    """`model_factory=None` (the default) must be pixel-identical to calling
    without the parameter at all - this is the "candidate C: v0 unchanged"
    contract, checked at the report level rather than just by inspection."""
    df = _fabricate_batches(_SCALES)
    explicit = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS, model_factory=None)
    implicit = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS)
    assert explicit["overall"] == implicit["overall"]
    assert explicit["pooled_logo"] == implicit["pooled_logo"]
    assert explicit["model"] == implicit["model"] == "v0_surrogate"


def test_model_factory_none_pooled_logo_is_independent_crosscheck():
    df = _fabricate_batches(_SCALES)
    report = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS)
    assert report["pooled_logo"]["independent_crosscheck"] is True


def test_physics_mean_candidate_runs_under_the_harness():
    df = _fabricate_batches(_SCALES)
    report = leave_one_scale_out_report(
        df,
        "titer_g_per_L",
        _PROCESS_COLUMNS,
        model_factory=lambda names: PhysicsMeanSurrogate(names),
        model_label="physics_mean",
    )
    assert report["model"] == "physics_mean"
    assert report["n_scales"] == len(_SCALES)
    assert len(report["per_scale"]) == len(_SCALES)
    assert report["by_direction"]["extrapolate_up"] is not None
    assert np.isfinite(report["overall"]["mae"])
    # Not an independent cross-check - `logo_report` hardcodes `Surrogate`.
    assert report["pooled_logo"]["independent_crosscheck"] is False
    assert report["pooled_logo"]["n_oof"] == report["overall"]["n"]


def test_multi_fidelity_candidate_runs_under_the_harness():
    df = _fabricate_batches(_SCALES)
    report = leave_one_scale_out_report(
        df,
        "titer_g_per_L",
        _PROCESS_COLUMNS,
        model_factory=lambda names: MultiFidelitySurrogate(names),
        model_label="multi_fidelity",
    )
    assert report["model"] == "multi_fidelity"
    assert report["n_scales"] == len(_SCALES)
    assert len(report["per_scale"]) == len(_SCALES)
    assert report["by_direction"]["extrapolate_up"] is not None
    assert np.isfinite(report["overall"]["mae"])
    assert report["pooled_logo"]["independent_crosscheck"] is False


def test_model_factory_receives_this_folds_feature_names():
    """The harness calls `model_factory(names)` - not a zero-arg factory -
    so a candidate can locate its physics/fidelity columns by name rather
    than a hardcoded position. Verified by a factory that asserts on what
    it receives."""
    df = _fabricate_batches(_SCALES)
    seen_names: list[list[str]] = []

    def factory(names):
        seen_names.append(list(names))
        return PhysicsMeanSurrogate(names)

    leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS, model_factory=factory)
    assert len(seen_names) == len(_SCALES)  # one fit call per held-out scale
    expected = _PROCESS_COLUMNS + [
        "log_volume_ratio",
        "specific_power_w_per_m3",
        "superficial_gas_velocity_m_per_s",
        "kla_proxy_per_s",
        "hydrostatic_pressure_mmHg",
    ]
    assert all(names == expected for names in seen_names)


# ---------------------------------------------------------------------------
# Determinism. `PhysicsMeanSurrogate` and `MultiFidelitySurrogate` are the
# only new sources of GP-parameter initialization in this repo's v1 change;
# both are documented (`candidates.py`) as fully deterministic (zero/fixed
# init, no `torch.randn` anywhere in either class). Locked in here by
# running the SAME full leave-one-scale-out report twice under different
# global `torch` RNG seeds and asserting byte-identical output - the
# honest "5 seeds" treatment this repo's v1 report describes: with no
# stochastic component to average over, running 5 seeds would just repeat
# one number five times, so this test proves that premise rather than
# assuming it.
# ---------------------------------------------------------------------------


def test_physics_mean_report_is_deterministic_across_torch_seeds():
    df = _fabricate_batches(_SCALES)
    torch.manual_seed(0)
    r1 = leave_one_scale_out_report(
        df, "titer_g_per_L", _PROCESS_COLUMNS, model_factory=lambda names: PhysicsMeanSurrogate(names)
    )
    torch.manual_seed(123456)
    r2 = leave_one_scale_out_report(
        df, "titer_g_per_L", _PROCESS_COLUMNS, model_factory=lambda names: PhysicsMeanSurrogate(names)
    )
    assert r1["overall"] == r2["overall"]
    assert _same_report_rows(r1["per_scale"], r2["per_scale"])


def test_multi_fidelity_report_is_deterministic_across_torch_seeds():
    df = _fabricate_batches(_SCALES)
    torch.manual_seed(0)
    r1 = leave_one_scale_out_report(
        df, "titer_g_per_L", _PROCESS_COLUMNS, model_factory=lambda names: MultiFidelitySurrogate(names)
    )
    torch.manual_seed(123456)
    r2 = leave_one_scale_out_report(
        df, "titer_g_per_L", _PROCESS_COLUMNS, model_factory=lambda names: MultiFidelitySurrogate(names)
    )
    assert r1["overall"] == r2["overall"]
    assert _same_report_rows(r1["per_scale"], r2["per_scale"])
