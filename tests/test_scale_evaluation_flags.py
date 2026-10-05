"""Unit tests for the `include_oof` and `held_out_scales` extensions to
`kalos.scale.evaluation.leave_one_scale_out_report` (see that function's
docstring, and decision ledger entries D4/R1 and D11/R8 in the Scale-Up
Readout design doc). Fabricated frame only, no dependency on the real
dataset - matches the fixture style already used in
`tests/test_scale_evaluation.py` and `tests/test_scale_v1_evaluation.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.core.surrogate import Surrogate
from kalos.scale.evaluation import leave_one_scale_out_report

_RNG = np.random.default_rng(20260923)
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


def _same_value(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True  # NaN != NaN under `==`; per-scale naive_mean_spearman is legitimately NaN
    return a == b


def _deep_equal(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return _same_value(a, b)


class _CountingSurrogate:
    """Wraps `Surrogate`, counting `fit` calls, so a restricted
    `held_out_scales` call can be checked to fit exactly once per requested
    fold - the D11 perf motivation for this parameter."""

    fit_count = 0

    def __init__(self) -> None:
        self._inner = Surrogate()

    def fit(self, X, y, *, bounds, noise=None):
        type(self).fit_count += 1
        self._inner.fit(X, y, bounds=bounds, noise=noise)
        return self

    def posterior(self, X, *, observation_noise=False):
        return self._inner.posterior(X, observation_noise=observation_noise)


# ---------------------------------------------------------------------------
# CRITICAL regression: the default call path must stay byte-identical.
# ---------------------------------------------------------------------------


def test_default_call_unchanged_by_new_flags():
    df = _fabricate_batches(_SCALES)
    default = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS)
    explicit = leave_one_scale_out_report(
        df, "titer_g_per_L", _PROCESS_COLUMNS, include_oof=False, held_out_scales=None
    )
    assert _deep_equal(default, explicit)
    assert "oof" not in default
    assert "pooled_logo_skipped" not in default


# ---------------------------------------------------------------------------
# include_oof
# ---------------------------------------------------------------------------


def test_include_oof_arrays_have_equal_lengths_matching_per_scale_totals():
    df = _fabricate_batches(_SCALES)
    report = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS, include_oof=True)

    oof = report["oof"]
    keys = ("scale_L", "actual", "pred", "naive_mean_pred", "naive_nn_pred", "direction")
    assert set(oof.keys()) == set(keys)
    lengths = {len(oof[k]) for k in keys}
    assert len(lengths) == 1  # every array the same length

    expected_n = sum(row["n"] for row in report["per_scale"])
    assert lengths.pop() == expected_n

    for k in keys:
        assert isinstance(oof[k], list)  # plain, JSON-serializable lists


def test_include_oof_values_reproduce_per_scale_mae():
    df = _fabricate_batches(_SCALES)
    report = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS, include_oof=True)

    oof = report["oof"]
    scale_arr = np.asarray(oof["scale_L"])
    actual_arr = np.asarray(oof["actual"])
    pred_arr = np.asarray(oof["pred"])
    naive_mean_arr = np.asarray(oof["naive_mean_pred"])

    for row in report["per_scale"]:
        mask = scale_arr == row["scale_L"]
        assert int(mask.sum()) == row["n"]
        recomputed_mae = float(np.mean(np.abs(pred_arr[mask] - actual_arr[mask])))
        assert recomputed_mae == pytest.approx(row["mae"])
        recomputed_naive_mae = float(np.mean(np.abs(naive_mean_arr[mask] - actual_arr[mask])))
        assert recomputed_naive_mae == pytest.approx(row["naive_mean_mae"])


def test_include_oof_false_default_has_no_oof_key():
    df = _fabricate_batches(_SCALES)
    report = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS, include_oof=False)
    assert "oof" not in report


# ---------------------------------------------------------------------------
# held_out_scales
# ---------------------------------------------------------------------------


def test_held_out_scales_restricted_per_scale_matches_full_call():
    df = _fabricate_batches(_SCALES)
    full = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS)
    restricted = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS, held_out_scales=[1000.0])

    assert len(restricted["per_scale"]) == 1
    restricted_row = restricted["per_scale"][0]
    full_row = next(row for row in full["per_scale"] if row["scale_L"] == 1000.0)
    assert _deep_equal(restricted_row, full_row)


def test_held_out_scales_skips_pooled_logo_with_reason():
    df = _fabricate_batches(_SCALES)
    restricted = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS, held_out_scales=[1000.0])
    assert restricted["pooled_logo"] is None
    assert restricted["pooled_logo_skipped"] == (
        "held_out_scales restricts folds; pooled LOGO would require every fold"
    )


def test_held_out_scales_matches_rounding_convention():
    """A `held_out_scales` value carrying float noise below the 6-decimal
    grain still matches its scale, mirroring
    `test_float_noise_in_scale_does_not_split_a_scale_into_two_groups` in
    `tests/test_scale_evaluation.py`."""
    df = _fabricate_batches(_SCALES)
    restricted = leave_one_scale_out_report(
        df, "titer_g_per_L", _PROCESS_COLUMNS, held_out_scales=[1000.0 + 1e-9]
    )
    assert len(restricted["per_scale"]) == 1
    assert restricted["per_scale"][0]["scale_L"] == 1000.0


def test_held_out_scales_with_no_match_evaluates_no_folds():
    df = _fabricate_batches(_SCALES)
    restricted = leave_one_scale_out_report(df, "titer_g_per_L", _PROCESS_COLUMNS, held_out_scales=[99999.0])
    assert restricted["per_scale"] == []
    assert restricted["n_scales"] == 0
    assert restricted["pooled_logo"] is None


def test_held_out_scales_fit_count_with_counting_model_factory():
    df = _fabricate_batches(_SCALES)
    _CountingSurrogate.fit_count = 0
    leave_one_scale_out_report(
        df,
        "titer_g_per_L",
        _PROCESS_COLUMNS,
        model_factory=lambda names: _CountingSurrogate(),
        held_out_scales=[1000.0],
    )
    assert _CountingSurrogate.fit_count == 1
