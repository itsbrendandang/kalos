"""Unit tests for `kalos.scale.evaluation.leave_one_scale_out_report`. Runs
on a tiny fabricated frame with a genuine, known scale dependence - no
dependency on the real dataset. Seeded for reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.scale.evaluation import leave_one_scale_out_report, _direction, _naive_carry_small_scale_mean

_RNG = np.random.default_rng(20260825)
_SCALES = [1.0, 10.0, 100.0, 1000.0]
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


def test_direction_labels():
    train_scales = np.array([1.0, 10.0])
    assert _direction(100.0, train_scales) == "extrapolate_up"
    assert _direction(0.1, train_scales) == "extrapolate_down"
    assert _direction(5.0, train_scales) == "interpolate"


def test_naive_carry_small_scale_mean():
    y = np.array([1.0, 2.0, 3.0, 30.0])
    scale = np.array([1.0, 1.0, 10.0, 10.0])
    assert _naive_carry_small_scale_mean(y, scale) == pytest.approx(1.5)


def test_report_structure_and_direction_buckets():
    df = _fabricate_batches(_SCALES, n_per_scale=4)
    report = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS)

    assert report["n_scales"] == len(_SCALES)
    assert report["n_rows_dropped"] == 0
    assert len(report["per_scale"]) == len(_SCALES)

    directions = {row["scale_L"]: row["direction"] for row in report["per_scale"]}
    assert directions[min(_SCALES)] == "extrapolate_down"
    assert directions[max(_SCALES)] == "extrapolate_up"
    for s in _SCALES[1:-1]:
        assert directions[s] == "interpolate"

    assert report["by_direction"]["extrapolate_up"] is not None
    assert report["by_direction"]["extrapolate_down"] is not None
    assert report["by_direction"]["interpolate"] is not None

    for row in report["per_scale"]:
        assert np.isfinite(row["mae"])
        assert np.isfinite(row["naive_mean_mae"])

    overall = report["overall"]
    assert np.isfinite(overall["mae"])
    assert np.isfinite(overall["naive_mean_mae"])
    assert isinstance(overall["beats_naive_mean"], bool)
    assert isinstance(overall["beats_naive_nn"], bool)


def test_report_pooled_logo_crosscheck_present():
    df = _fabricate_batches(_SCALES, n_per_scale=4)
    report = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS)
    assert "spearman" in report["pooled_logo"]
    assert report["pooled_logo"]["n_groups"] == len(_SCALES)


def test_report_drops_nonfinite_rows():
    df = _fabricate_batches(_SCALES, n_per_scale=4)
    df.loc[0, "ph_setpoint"] = np.nan
    df.loc[1, "titer_g_per_L"] = np.nan
    report = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS)
    assert report["n_rows_dropped"] == 2
    assert report["n_rows_used"] == len(df) - 2


def test_report_respects_explicit_bounds():
    df = _fabricate_batches(_SCALES, n_per_scale=4)
    from kalos.scale.transfer import build_feature_matrix

    X, _ = build_feature_matrix(df, _PROCESS_COLUMNS)
    wide_bounds = np.vstack([X.min(axis=0) - 1.0, X.max(axis=0) + 1.0])
    report = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS, bounds=wide_bounds)
    assert report["n_scales"] == len(_SCALES)
