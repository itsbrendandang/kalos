"""Tests for `kalos.scale.readout` (the Scale-Up Readout: gate -> ladder ->
decide -> build_readout). See that module's docstring, and the design doc's
decision ledger (R1-R8/D4-D11) this behavior implements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.scale import readout as ro
from kalos.scale.evaluation import leave_one_scale_out_report
from kalos.scale.transfer import ScaleUpTransferModel

PROCESS_COLUMNS = ["ph_setpoint", "temperature_C"]
TARGET_COLUMN = "titer_g_per_L"

_RNG_SEED = 20260923


def _make_sheet(
    scales: list[float],
    n_per_scale: int,
    *,
    seed: int = _RNG_SEED,
    ph_range: tuple[float, float] = (6.9, 7.1),
) -> pd.DataFrame:
    """A clean multi-scale sheet: a mild scale trend, no planted rank
    crossing (simple and fast, unlike the harder demo fixture) - good
    enough for the gate/ladder/decide plumbing this module tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in scales:
        for _ in range(n_per_scale):
            ph = float(rng.uniform(*ph_range))
            temp = float(rng.uniform(36.5, 37.0))
            rpm = float(rng.uniform(80, 250))
            airflow = float(max(s * 0.05, 0.05) * rng.uniform(0.8, 1.2))
            titer = 5.0 - 0.2 * np.log10(s + 1e-6) + float(rng.normal(0, 0.05))
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


def _target(scale_L: float = 500.0, **process) -> ro.TargetSpec:
    params = {"ph_setpoint": 7.0, "temperature_C": 36.7}
    params.update(process)
    return ro.TargetSpec(scale_L=scale_L, agitation_rpm=150.0, airflow_L_per_min=10.0, process_params=params)


def _demo_sheet() -> pd.DataFrame:
    return pd.read_csv("examples/synthetic_scaleup/synthetic_scaleup.csv")


# A sheet with exactly 3 scales, 10 rows each - meets every gate minimum
# (>=3 scales, >=3 runs/scale, >=10 rows at the third-smallest+ scale, since
# 100.0 is both the third-smallest AND the largest here).
_MINIMAL_SCALES = [1.0, 10.0, 100.0]
_MINIMAL_N = 10


# --------------------------------------------------------------------------- #
# gate: each of the six checks failing alone names that check
# --------------------------------------------------------------------------- #


def test_gate_passes_on_a_clean_minimal_sheet():
    df = _make_sheet(_MINIMAL_SCALES, _MINIMAL_N)
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=200.0))
    assert result.passed
    assert result.failed_check is None
    assert result.clean_df is not None


def test_gate_fails_missing_required_column():
    df = _make_sheet(_MINIMAL_SCALES, _MINIMAL_N).drop(columns=["agitation_rpm"])
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=200.0))
    assert not result.passed
    assert result.failed_check == "required_columns"


def test_gate_fails_too_few_distinct_scales():
    df = _make_sheet([1.0, 10.0], _MINIMAL_N)
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=200.0))
    assert not result.passed
    assert result.failed_check == "min_scales_and_runs"


def test_gate_fails_too_few_runs_at_one_scale():
    df = _make_sheet(_MINIMAL_SCALES, _MINIMAL_N)
    # thin one scale down below MIN_RUNS_PER_SCALE
    keep = df[df["scale_L"] != 10.0].index.tolist() + df[df["scale_L"] == 10.0].index[:2].tolist()
    df = df.loc[keep]
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=200.0))
    assert not result.passed
    assert result.failed_check == "min_scales_and_runs"


def test_gate_fails_too_few_rung_target_rows():
    # 3 scales but only MIN_RUNS_PER_SCALE (3) rows each: passes the basic
    # scale/run minimum but the third-smallest-and-up row count (3) is
    # below MIN_RUNG_TARGET_ROWS (10).
    df = _make_sheet(_MINIMAL_SCALES, ro.MIN_RUNS_PER_SCALE)
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=200.0))
    assert not result.passed
    assert result.failed_check == "min_rung_target_rows"


def test_gate_fails_non_finite_dropped_fraction():
    df = _demo_sheet()  # 9 scales x 10 rows = 90 rows
    # NaN out ~22% of rows, spread so every scale still keeps enough rows to
    # clear the earlier count checks.
    rng = np.random.default_rng(0)
    idx = rng.choice(df.index, size=20, replace=False)
    df.loc[idx, "ph_setpoint"] = np.nan
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=6000.0))
    assert not result.passed
    assert result.failed_check == "non_finite_dropped_fraction"
    assert result.n_rows_dropped == 20
    assert result.dropped_fraction == pytest.approx(20 / 90)


def test_gate_fails_target_not_interpolation():
    df = _make_sheet(_MINIMAL_SCALES, _MINIMAL_N)
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=50.0))  # <= max trained (100)
    assert not result.passed
    assert result.failed_check == "target_not_interpolation"
    assert result.detail == "interpolation is out of scope for this readout"


def test_gate_fails_target_ratio_hard_cap():
    df = _make_sheet(_MINIMAL_SCALES, _MINIMAL_N)
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=100.0 * 25))  # 25x > 20x cap
    assert not result.passed
    assert result.failed_check == "target_ratio_hard_cap"


def test_gate_rejects_mismatched_process_params():
    df = _make_sheet(_MINIMAL_SCALES, _MINIMAL_N)
    bad_target = ro.TargetSpec(
        scale_L=200.0, agitation_rpm=150.0, airflow_L_per_min=10.0, process_params={"ph_setpoint": 7.0}
    )
    with pytest.raises(ValueError):
        ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, bad_target)


# --------------------------------------------------------------------------- #
# ladder: property-style (several passing sheets), skipped first rung, and
# same bounds as the final fit
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "scales,n_per_scale",
    [
        ([1.0, 10.0, 100.0], 10),
        ([1.0, 3.0, 10.0, 30.0], 6),
        ([1.0, 5.0, 25.0, 125.0, 625.0], 4),
    ],
)
def test_gate_passing_sheets_yield_at_least_min_residuals(scales, n_per_scale):
    """Property: any sheet that clears the gate lets the ladder reach at
    least MIN_RESIDUALS pooled residuals (R4/D7's whole point)."""
    df = _make_sheet(scales, n_per_scale)
    target = _target(scale_L=scales[-1] * 1.5)
    gate_result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    assert gate_result.passed, gate_result.detail

    X, _ = ro.build_feature_matrix(gate_result.clean_df, PROCESS_COLUMNS, ro.DEFAULT_SCALE_FEATURE_CONFIG)
    Xt, _ = ro.build_feature_matrix(target.to_frame(ro.DEFAULT_SCALE_FEATURE_CONFIG), PROCESS_COLUMNS, ro.DEFAULT_SCALE_FEATURE_CONFIG)
    bounds = np.vstack([np.minimum(X.min(axis=0), Xt.min(axis=0)), np.maximum(X.max(axis=0), Xt.max(axis=0))])

    ladder_result = ro.ladder(gate_result.clean_df, TARGET_COLUMN, PROCESS_COLUMNS, bounds)
    assert len(ladder_result.residuals) >= ro.MIN_RESIDUALS


def test_one_scale_first_rung_is_skipped_and_reported():
    df = _make_sheet(_MINIMAL_SCALES, _MINIMAL_N)
    X, _ = ro.build_feature_matrix(df, PROCESS_COLUMNS, ro.DEFAULT_SCALE_FEATURE_CONFIG)
    bounds = np.vstack([X.min(axis=0), X.max(axis=0)])
    ladder_result = ro.ladder(df, TARGET_COLUMN, PROCESS_COLUMNS, bounds)

    # scale_L=10.0 (second-smallest) trains on exactly one smaller scale
    # (1.0) - skipped for too few distinct training scales, not rows.
    assert any(s.scale_L == 10.0 for s in ladder_result.skipped)
    skipped = next(s for s in ladder_result.skipped if s.scale_L == 10.0)
    assert "1 distinct scale" in skipped.reason
    # the eligible rung (100.0, trained on {1.0, 10.0}) still ran.
    assert any(r.scale_L == 100.0 for r in ladder_result.rungs)


def test_ladder_and_final_fit_use_the_same_bounds(monkeypatch):
    df = _demo_sheet()
    target = _target(scale_L=7500.0)

    captured_eval_bounds: list[np.ndarray] = []
    captured_fit_bounds: list[np.ndarray] = []

    original_report = leave_one_scale_out_report

    def _spy_report(*args, **kwargs):
        captured_eval_bounds.append(np.asarray(kwargs["bounds"]))
        return original_report(*args, **kwargs)

    original_fit = ScaleUpTransferModel.fit

    def _spy_fit(self, df_arg, *, bounds=None, noise=None):
        captured_fit_bounds.append(np.asarray(bounds))
        return original_fit(self, df_arg, bounds=bounds, noise=noise)

    monkeypatch.setattr(ro, "leave_one_scale_out_report", _spy_report)
    monkeypatch.setattr(ScaleUpTransferModel, "fit", _spy_fit)

    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    assert out["decision"] in (ro.NUMBER, ro.NUMBER_WITH_WARNING)
    assert captured_eval_bounds, "ladder should have called the evaluation harness"
    assert captured_fit_bounds, "the final model should have been fit"
    for b in captured_eval_bounds:
        assert np.allclose(b, captured_fit_bounds[0])
    assert np.allclose(captured_fit_bounds[0], captured_eval_bounds[0])


# --------------------------------------------------------------------------- #
# decide: NUMBER / NUMBER_WITH_WARNING / REFUSAL branches
# --------------------------------------------------------------------------- #


def _rung(scale_L: float, step_ratio: float, beats_both: bool, mae: float = 0.1) -> ro.RungResult:
    return ro.RungResult(
        scale_L=scale_L,
        step_ratio=step_ratio,
        n=10,
        mae=mae,
        naive_mean_mae=mae + 1.0,
        naive_nn_mae=mae + 1.0,
        beats_both=beats_both,
    )


def _ladder_result(rungs: list[ro.RungResult], *, pooled_beats_both: bool = True, n_residuals: int = 20) -> ro.LadderResult:
    residuals = np.full(n_residuals, 0.1)
    return ro.LadderResult(
        rungs=rungs,
        skipped=[],
        residuals=residuals,
        pooled_mae=0.1,
        pooled_naive_mean_mae=1.0 if pooled_beats_both else 0.05,
        pooled_naive_nn_mae=1.0 if pooled_beats_both else 0.05,
        pooled_beats_both=pooled_beats_both,
    )


def test_decide_number_when_everything_is_in_range():
    lr = _ladder_result([_rung(10.0, 3.0, True), _rung(100.0, 3.0, True)])
    result = ro.decide(lr, requested_ratio=4.0, target_process_params={"ph_setpoint": 7.0}, trained_param_ranges={"ph_setpoint": (6.5, 7.5)})
    assert result.decision == ro.NUMBER
    assert result.reasons == []
    assert result.reference_ratio == 3.0


def test_decide_warns_on_ratio_between_2x_and_5x_reference():
    lr = _ladder_result([_rung(10.0, 2.0, True)])
    result = ro.decide(lr, requested_ratio=5.0, target_process_params={"ph_setpoint": 7.0}, trained_param_ranges={"ph_setpoint": (6.5, 7.5)})
    # 5.0 is between 2x (4.0) and 5x (10.0) of the reference ratio 2.0
    assert result.decision == ro.NUMBER_WITH_WARNING
    assert any("ratio" in r for r in result.reasons)


def test_decide_warns_on_out_of_range_process_param():
    lr = _ladder_result([_rung(10.0, 2.0, True)])
    result = ro.decide(
        lr, requested_ratio=3.0, target_process_params={"ph_setpoint": 9.0}, trained_param_ranges={"ph_setpoint": (6.5, 7.5)}
    )
    assert result.decision == ro.NUMBER_WITH_WARNING
    assert result.out_of_range_params == ["ph_setpoint"]
    assert any("ph_setpoint" in r for r in result.reasons)


def test_decide_refuses_when_no_rung_beats_both_baselines():
    lr = _ladder_result([_rung(10.0, 3.0, False), _rung(100.0, 3.0, False)])
    result = ro.decide(lr, requested_ratio=4.0, target_process_params={"ph_setpoint": 7.0}, trained_param_ranges={"ph_setpoint": (6.5, 7.5)})
    assert result.decision == ro.REFUSAL
    assert result.reference_ratio is None


def test_decide_refuses_when_pooled_mae_loses():
    lr = _ladder_result([_rung(10.0, 3.0, True)], pooled_beats_both=False)
    result = ro.decide(lr, requested_ratio=2.0, target_process_params={"ph_setpoint": 7.0}, trained_param_ranges={"ph_setpoint": (6.5, 7.5)})
    assert result.decision == ro.REFUSAL


def test_decide_refuses_below_min_residuals():
    lr = _ladder_result([_rung(10.0, 3.0, True)], n_residuals=ro.MIN_RESIDUALS - 1)
    result = ro.decide(lr, requested_ratio=2.0, target_process_params={"ph_setpoint": 7.0}, trained_param_ranges={"ph_setpoint": (6.5, 7.5)})
    assert result.decision == ro.REFUSAL


def test_decide_refuses_beyond_5x_reference():
    lr = _ladder_result([_rung(10.0, 3.0, True)])
    result = ro.decide(lr, requested_ratio=16.0, target_process_params={"ph_setpoint": 7.0}, trained_param_ranges={"ph_setpoint": (6.5, 7.5)})
    assert result.decision == ro.REFUSAL


def test_losing_longest_rung_falls_back_to_next_winning_rung():
    """The longest step ratio rung LOSES; the reference must come from the
    next rung that actually won (R5/D8)."""
    losing_long = _rung(1000.0, step_ratio=10.0, beats_both=False)
    winning_short = _rung(100.0, step_ratio=3.0, beats_both=True)
    lr = _ladder_result([winning_short, losing_long])
    result = ro.decide(lr, requested_ratio=4.0, target_process_params={"ph_setpoint": 7.0}, trained_param_ranges={"ph_setpoint": (6.5, 7.5)})
    assert result.reference_ratio == 3.0
    assert result.decision != ro.REFUSAL


def test_all_rungs_losing_refuses():
    lr = _ladder_result([_rung(10.0, 3.0, False), _rung(100.0, 10.0, False)])
    result = ro.decide(lr, requested_ratio=2.0, target_process_params={"ph_setpoint": 7.0}, trained_param_ranges={"ph_setpoint": (6.5, 7.5)})
    assert result.decision == ro.REFUSAL


# --------------------------------------------------------------------------- #
# build_readout: end to end, never a number without an interval, provenance
# hashes stable across runs
# --------------------------------------------------------------------------- #


def test_demo_sheet_1p5x_target_yields_a_number_with_interval():
    df = _demo_sheet()
    target = _target(scale_L=7500.0)  # 1.5x the largest trained scale (5000)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    assert out["decision"] == ro.NUMBER
    assert out["prediction"] is not None
    assert out["interval"] is not None
    assert out["interval"][0] < out["prediction"] < out["interval"][1]


@pytest.mark.parametrize(
    "scale_L",
    [200.0, 5000.0 * 1.5, 5000.0 * 16.7],
)
def test_never_a_number_without_an_interval(scale_L):
    df = _demo_sheet()
    target = _target(scale_L=scale_L)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    if out["decision"] in (ro.NUMBER, ro.NUMBER_WITH_WARNING):
        assert out["prediction"] is not None
        assert out["interval"] is not None
    else:
        assert out["decision"] == ro.REFUSAL
        assert out["prediction"] is None
        assert out["interval"] is None


def test_provenance_hashes_are_stable_across_two_runs():
    df = _demo_sheet()
    target = _target(scale_L=7500.0)
    raw = df.to_csv(index=False).encode("utf-8")

    out1 = ro.build_readout(df.copy(), TARGET_COLUMN, PROCESS_COLUMNS, target, raw_bytes=raw)
    out2 = ro.build_readout(df.copy(), TARGET_COLUMN, PROCESS_COLUMNS, target, raw_bytes=raw)

    assert out1["provenance"]["raw_upload_sha256"] == out2["provenance"]["raw_upload_sha256"]
    assert out1["provenance"]["normalized_frame_sha256"] == out2["provenance"]["normalized_frame_sha256"]
    assert out1["provenance"]["normalize_plan_json"] == out2["provenance"]["normalize_plan_json"]
    assert out1["prediction"] == out2["prediction"]


def test_physics_assumptions_tagged_default_and_user_supplied():
    df = _demo_sheet()
    target = _target(scale_L=7500.0)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target, physics_overrides={"power_number": 6.0})
    assumptions = out["physics_assumptions"]
    assert assumptions["power_number"]["source"] == "user-supplied"
    assert assumptions["power_number"]["value"] == 6.0
    assert assumptions["a"]["source"] == "default"  # a van't Riet param, untouched


def test_gate_failure_short_circuits_before_any_fit():
    df = _make_sheet(_MINIMAL_SCALES, _MINIMAL_N)
    target = _target(scale_L=50.0)  # interpolation -> gate failure
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    assert out["decision"] == ro.REFUSAL
    assert out["gate"]["failed_check"] == "target_not_interpolation"
    assert out["prediction"] is None
    assert out["interval"] is None
