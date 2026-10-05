"""Tests for `kalos.scale.readout` (the Scale-Up Readout: gate -> ladder ->
decide -> build_readout). See that module's docstring, and the design doc's
decision ledger (R1-R8/D4-D11) this behavior implements.
"""
from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from kalos.scale import readout as ro
from kalos.scale.evaluation import leave_one_scale_out_report
from kalos.scale.transfer import ScaleFeatureConfig, ScaleUpTransferModel

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


# A sheet with exactly 3 scales, 10 rows each - meets the gate's only
# structural minimum (>=3 distinct scales; there is no per-scale run-count
# or rung-row-count requirement any more, see kalos/scale/readout.py).
_MINIMAL_SCALES = [1.0, 10.0, 100.0]
_MINIMAL_N = 10


def _sparse_sheet() -> pd.DataFrame:
    """Real tech-transfer sheets look like this: many bench runs, then one
    or two runs at each large scale. 96 runs at the smallest scale, 2 at
    the next, then a single run each at three larger scales - the exact
    shape the removed `min_scales_and_runs`/`min_rung_target_rows` gate
    checks used to reject with a bare 422 before any evidence was
    computed."""
    scales = [0.003, 3.0, 15.0, 3000.0, 35000.0]
    counts = [96, 2, 1, 1, 1]
    rng = np.random.default_rng(_RNG_SEED)
    rows = []
    for s, n in zip(scales, counts):
        for _ in range(n):
            ph = float(rng.uniform(6.9, 7.1))
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


# --------------------------------------------------------------------------- #
# gate: each of the five checks failing alone names that check
# --------------------------------------------------------------------------- #


def test_gate_passes_on_a_clean_minimal_sheet():
    df = _make_sheet(_MINIMAL_SCALES, _MINIMAL_N)
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=200.0))
    assert result.passed
    assert result.failed_check is None
    assert result.clean_df is not None


def test_gate_fails_missing_required_column():
    # `agitation_rpm`/`airflow_L_per_min` are deliberately NOT in this list
    # any more (see `_required_columns` - the scale-only fallback this
    # module now supports means a sheet missing either is not a gate
    # failure), so this exercises a column that is still genuinely
    # required: a process column.
    df = _make_sheet(_MINIMAL_SCALES, _MINIMAL_N).drop(columns=["ph_setpoint"])
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=200.0))
    assert not result.passed
    assert result.failed_check == "required_columns"


def test_gate_passes_when_agitation_and_airflow_columns_are_absent():
    """The behavior this module exists to fix (owner decision 2026-09-24):
    a sheet missing `agitation_rpm`/`airflow_L_per_min` no longer 422s at
    the gate - `build_readout` falls back to the scale_only feature set
    instead (see `test_scale_readout.py`'s scale-only fallback section).
    `config.feature_set="scale_only"` here mirrors what `build_readout`
    itself would have selected for this sheet before calling `gate`."""
    df = _make_sheet(_MINIMAL_SCALES, _MINIMAL_N).drop(columns=["agitation_rpm", "airflow_L_per_min"])
    target = ro.TargetSpec(scale_L=200.0, process_params={"ph_setpoint": 7.0, "temperature_C": 36.7})
    config = ScaleFeatureConfig(feature_set="scale_only")
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, target, config=config)
    assert result.passed, result.detail


def test_gate_fails_too_few_distinct_scales():
    df = _make_sheet([1.0, 10.0], _MINIMAL_N)
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=200.0))
    assert not result.passed
    assert result.failed_check == "min_distinct_scales"
    # the detail says exactly how many MORE distinct scales are needed.
    assert "1 more" in result.detail


def test_gate_passes_a_sheet_with_only_one_run_at_each_large_scale():
    """The removed per-scale run-count check used to reject this sheet
    outright; it is now the gate's whole reason for existing in this
    shape - real tech-transfer data (see `_sparse_sheet`)."""
    df = _sparse_sheet()
    result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=35000.0 * 1.2))
    assert result.passed, result.detail


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


def test_gate_passing_does_not_guarantee_min_residuals():
    """Unlike before, the gate no longer enforces a minimum row count at
    the rung-eligible scales, so passing the gate no longer guarantees the
    ladder reaches MIN_RESIDUALS pooled residuals - that is now `decide`'s
    job, and too few residuals is a REFUSAL, not a 422 (see
    `_sparse_sheet` and the `build_readout` tests below)."""
    df = _sparse_sheet()
    target = _target(scale_L=35000.0 * 1.2)
    gate_result = ro.gate(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    assert gate_result.passed, gate_result.detail

    X, _ = ro.build_feature_matrix(gate_result.clean_df, PROCESS_COLUMNS, ro.DEFAULT_SCALE_FEATURE_CONFIG)
    Xt, _ = ro.build_feature_matrix(target.to_frame(ro.DEFAULT_SCALE_FEATURE_CONFIG), PROCESS_COLUMNS, ro.DEFAULT_SCALE_FEATURE_CONFIG)
    bounds = np.vstack([np.minimum(X.min(axis=0), Xt.min(axis=0)), np.maximum(X.max(axis=0), Xt.max(axis=0))])

    ladder_result = ro.ladder(gate_result.clean_df, TARGET_COLUMN, PROCESS_COLUMNS, bounds)
    assert len(ladder_result.residuals) < ro.MIN_RESIDUALS


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


def _rung(scale_L: float, step_ratio: float, beats_both: bool, mae: float = 0.1, n: int = 10) -> ro.RungResult:
    return ro.RungResult(
        scale_L=scale_L,
        step_ratio=step_ratio,
        n=n,
        mae=mae,
        naive_mean_mae=mae + 1.0,
        naive_nn_mae=mae + 1.0,
        beats_both=beats_both,
        too_few_to_judge=n < ro.MIN_RUNG_N_FOR_LICENSE,
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


def test_a_winning_rung_with_too_few_runs_cannot_license_a_reference():
    """A rung with n=2 that beats both baselines still cannot set the
    reference step ratio (MIN_RUNG_N_FOR_LICENSE=3) - the only rung here
    wins but is too thin, so there is no license at all."""
    lr = _ladder_result([_rung(10.0, 3.0, True, n=2)])
    result = ro.decide(lr, requested_ratio=2.0, target_process_params={"ph_setpoint": 7.0}, trained_param_ranges={"ph_setpoint": (6.5, 7.5)})
    assert result.decision == ro.REFUSAL
    assert result.reference_ratio is None


def test_a_winning_rung_with_too_few_runs_is_skipped_in_favor_of_a_licensing_one():
    """Even when a larger, thin (n=2) rung wins, the reference must come
    from the smaller rung that actually has enough runs to license it."""
    thin_but_winning = _rung(1000.0, step_ratio=10.0, beats_both=True, n=2)
    licensing = _rung(100.0, step_ratio=3.0, beats_both=True, n=3)
    lr = _ladder_result([licensing, thin_but_winning])
    result = ro.decide(lr, requested_ratio=4.0, target_process_params={"ph_setpoint": 7.0}, trained_param_ranges={"ph_setpoint": (6.5, 7.5)})
    assert result.reference_ratio == 3.0
    assert result.decision != ro.REFUSAL


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
    # the GP fit itself is not bit-for-bit deterministic (BoTorch's L-BFGS
    # restarts touch global RNG/BLAS state) - the provenance HASHES above are
    # the reproducibility contract this test is really pinning.
    assert out1["prediction"] == pytest.approx(out2["prediction"])


def test_generated_at_utc_is_present_and_iso_with_no_fractional_seconds():
    df = _demo_sheet()
    target = _target(scale_L=7500.0)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    stamp = out["generated_at_utc"]
    assert isinstance(stamp, str)
    assert datetime.fromisoformat(stamp).tzinfo is not None
    assert "." not in stamp  # seconds precision only


def test_git_sha_uses_kalos_git_sha_env_override(monkeypatch):
    monkeypatch.setenv("KALOS_GIT_SHA", "deadbeef")
    df = _demo_sheet()
    target = _target(scale_L=7500.0)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    assert out["provenance"]["kalos_git_sha"] == "deadbeef"


def test_normalize_plan_appendix_provenance_fields():
    df = _demo_sheet()
    target = _target(scale_L=7500.0)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    p = out["provenance"]
    assert p["normalize_plan_n_columns"] == len(df.columns)
    assert p["normalize_plan_sha256"] == hashlib.sha256(p["normalize_plan_json"].encode("utf-8")).hexdigest()


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


def test_provenance_carries_engine_version_like_api_run():
    """Parity with `/api/run`'s audit fields: the readout names the kalos
    package version that produced it."""
    from kalos import __version__

    out = ro.build_readout(_demo_sheet(), TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=7500.0))
    assert out["provenance"]["engine_version"] == __version__


# --------------------------------------------------------------------------- #
# sparse tech-transfer sheet: 200 REFUSAL, not a 422, with a data plan
# --------------------------------------------------------------------------- #


def test_sparse_sheet_is_a_refusal_not_a_gate_failure():
    """The gate passes (>=3 distinct scales is the only structural
    minimum); the ladder's own evidence is what's too thin, so this is a
    REFUSAL from `decide`, never a 422 from `gate`."""
    df = _sparse_sheet()
    target = _target(scale_L=35000.0 * 1.2)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    assert out["gate"]["passed"]
    assert out["decision"] == ro.REFUSAL
    assert out["prediction"] is None
    assert out["interval"] is None


def test_sparse_sheet_rungs_with_n_1_are_too_few_to_judge():
    df = _sparse_sheet()
    target = _target(scale_L=35000.0 * 1.2)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    rungs = out["rungs"]
    assert rungs, "expected at least one evaluated rung"
    assert all(r["n"] == 1 for r in rungs)
    assert all(r["too_few_to_judge"] for r in rungs)


def test_sparse_sheet_pooled_residuals_equal_the_rung_rows():
    df = _sparse_sheet()
    target = _target(scale_L=35000.0 * 1.2)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    assert out["baseline_comparison"]["n_residuals"] == sum(r["n"] for r in out["rungs"])


def test_sparse_sheet_data_plan_states_the_residual_shortfall():
    df = _sparse_sheet()
    target = _target(scale_L=35000.0 * 1.2)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)

    n_residuals = out["baseline_comparison"]["n_residuals"]
    expected_n = ro.MIN_RESIDUALS - n_residuals
    # X is the third-smallest distinct scale (15.0 L): the first two scales
    # (0.003 L x96, 3.0 L x2) already clear the rung training minimums.
    expected_x = ro.format_liters_plain(15.0)

    assert expected_n > 0
    plan = out["data_plan"]
    assert any(f"{expected_n} more run" in line and f"{expected_x} L or larger" in line for line in plan)
    assert any("evidence that licenses larger steps" in line for line in plan)


def test_sparse_sheet_data_plan_states_the_no_licensing_rung_requirement():
    df = _sparse_sheet()
    target = _target(scale_L=35000.0 * 1.2)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    plan = out["data_plan"]
    assert any(
        f"at least {ro.MIN_RUNG_N_FOR_LICENSE} runs" in line and "beats both baselines" in line for line in plan
    )


def test_ratio_warning_data_plan_gives_only_the_clean_target():
    """A target ratio between 2x and 5x the reference step ratio is a
    warning, not a refusal: the plan names T/(2r) for a clean prediction
    and must NOT tell the reader how to "avoid refusal" - it already did."""
    df = _demo_sheet()
    target_scale = 5000.0 * 10.0  # 10x the largest trained scale (5000 L)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=target_scale))
    assert out["decision"] == ro.NUMBER_WITH_WARNING, out["reasons"]
    reference = out["reference_ratio"]
    assert reference is not None
    assert out["requested_ratio"] > 2.0 * reference  # exercises the ratio line

    clean = ro.format_liters_plain(target_scale / (2.0 * reference))
    ratio_lines = [line for line in out["data_plan"] if "for a clean prediction" in line]
    assert len(ratio_lines) == 1, out["data_plan"]
    assert clean in ratio_lines[0]
    assert "avoid refusal" not in ratio_lines[0]


def test_ratio_refusal_data_plan_gives_clean_and_refusal_targets():
    """A target ratio above 5x the reference is a refusal: the plan gives
    both T/(2r) (clean) and T/(5r) (avoid refusal)."""
    df = _demo_sheet()
    target_scale = 5000.0 * 17.0  # 17x the largest trained scale
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=target_scale))
    assert out["decision"] == ro.REFUSAL, out["reasons"]
    reference = out["reference_ratio"]
    assert reference is not None
    assert out["requested_ratio"] > 5.0 * reference  # exercises the refusal clause

    clean = ro.format_liters_plain(target_scale / (2.0 * reference))
    refusal = ro.format_liters_plain(target_scale / (5.0 * reference))
    assert any(
        clean in line and f"avoid refusal, {refusal} L" in line for line in out["data_plan"]
    ), out["data_plan"]


# --------------------------------------------------------------------------- #
# scale-only fallback: agitation_rpm/airflow_L_per_min optional when the
# sheet does not record them (owner decision 2026-09-24)
# --------------------------------------------------------------------------- #


def test_dropped_columns_demo_runs_end_to_end_as_scale_only():
    df = _demo_sheet().drop(columns=["agitation_rpm", "airflow_L_per_min"])
    target = ro.TargetSpec(scale_L=7500.0, process_params={"ph_setpoint": 7.2, "temperature_C": 37.0})
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)

    assert out["gate"]["passed"], out["gate"]["detail"]
    assert out["feature_set"] == "scale_only"
    assert set(out["missing_physics_inputs"]) == {"agitation_rpm", "airflow_L_per_min"}
    assert out["provenance"]["feature_set"] == "scale_only"
    assert set(out["provenance"]["missing_physics_inputs"]) == {"agitation_rpm", "airflow_L_per_min"}
    # never a crash, whatever the evidence-driven decision turns out to be.
    assert out["decision"] in (ro.NUMBER, ro.NUMBER_WITH_WARNING, ro.REFUSAL)


def test_dropped_columns_demo_ladder_uses_the_scale_only_feature_set(monkeypatch):
    """The ladder backtest must evaluate the SAME feature set that produces
    the final number - checked here via the feature names
    `leave_one_scale_out_report` actually reported, not just the config
    passed in."""
    from kalos.scale.features import SCALE_ONLY_FEATURE_COLUMNS

    df = _demo_sheet().drop(columns=["agitation_rpm", "airflow_L_per_min"])
    target = ro.TargetSpec(scale_L=7500.0, process_params={"ph_setpoint": 7.2, "temperature_C": 37.0})

    captured_feature_names: list[list[str]] = []
    original_report = leave_one_scale_out_report

    def _spy_report(*args, **kwargs):
        report = original_report(*args, **kwargs)
        captured_feature_names.append(report["feature_names"])
        return report

    monkeypatch.setattr(ro, "leave_one_scale_out_report", _spy_report)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)

    assert out["feature_set"] == "scale_only"
    assert captured_feature_names, "the ladder should have called the evaluation harness at least once"
    expected_names = PROCESS_COLUMNS + list(SCALE_ONLY_FEATURE_COLUMNS)
    for names in captured_feature_names:
        assert names == expected_names


def test_demo_sheet_with_both_columns_is_still_physics_mode():
    """Physics-path parity: with both columns present, `feature_set` stays
    `"physics"` and the decision/prediction are unchanged from before the
    scale-only fallback existed (see `test_demo_sheet_1p5x_target_yields_a_number_with_interval`)."""
    df = _demo_sheet()
    target = _target(scale_L=7500.0)
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)

    assert out["feature_set"] == "physics"
    assert out["missing_physics_inputs"] == []
    assert out["decision"] == ro.NUMBER
    assert out["prediction"] is not None
    assert out["interval"] is not None


def test_sheet_missing_only_airflow_is_scale_only_and_lists_only_airflow():
    df = _demo_sheet().drop(columns=["airflow_L_per_min"])
    target = ro.TargetSpec(scale_L=7500.0, process_params={"ph_setpoint": 7.2, "temperature_C": 37.0})
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)

    assert out["feature_set"] == "scale_only"
    assert out["missing_physics_inputs"] == ["airflow_L_per_min"]


def test_physics_mode_target_missing_agitation_raises_value_error_naming_it():
    """The sheet records both columns (physics mode); the target must then
    supply both values, or `gate` raises `ValueError` (the portal route
    maps this to a 422 `invalid_target`) naming the missing one."""
    df = _demo_sheet()
    target = ro.TargetSpec(
        scale_L=7500.0,
        process_params={"ph_setpoint": 7.2, "temperature_C": 37.0},
        airflow_L_per_min=300.0,
    )
    with pytest.raises(ValueError, match="agitation_rpm"):
        ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)


def test_scale_only_ignores_a_caller_supplied_agitation_value():
    """In scale_only mode a caller-supplied value is accepted (no error)
    but never fed to the physics features - it is recorded in
    `target_inputs` as-is for the page to mark "ignored"."""
    df = _demo_sheet().drop(columns=["agitation_rpm", "airflow_L_per_min"])
    target = ro.TargetSpec(
        scale_L=7500.0,
        process_params={"ph_setpoint": 7.2, "temperature_C": 37.0},
        agitation_rpm=999.0,
    )
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    assert out["feature_set"] == "scale_only"
    assert out["target_inputs"]["agitation_rpm"] == 999.0


def test_scale_only_physics_assumptions_mark_unused_constants():
    df = _demo_sheet().drop(columns=["agitation_rpm", "airflow_L_per_min"])
    target = ro.TargetSpec(scale_L=7500.0, process_params={"ph_setpoint": 7.2, "temperature_C": 37.0})
    out = ro.build_readout(df, TARGET_COLUMN, PROCESS_COLUMNS, target)
    assumptions = out["physics_assumptions"]

    assert assumptions["power_number"]["source"] == "not used (scale-only)"
    assert assumptions["a"]["source"] == "not used (scale-only)"
    assert assumptions["alpha"]["source"] == "not used (scale-only)"
    assert assumptions["beta"]["source"] == "not used (scale-only)"
    assert assumptions["impeller_to_tank_diameter_ratio"]["source"] == "not used (scale-only)"
    # still reported as actually used, since log_volume_ratio and
    # hydrostatic_pressure_mmHg both need them.
    assert assumptions["aspect_ratio_h_over_t"]["source"] == "default"
    assert assumptions["liquid_density_kg_per_m3"]["source"] == "default"
    assert assumptions["reference_scale_L"]["source"] == "default"


def test_production_phase_headers_do_not_block_the_readout():
    """Regression from the first real-data run (lipase sheet): process
    columns named after the production phase were guessed as extra targets
    and the whole readout failed. The caller's named target is authoritative."""
    df = _demo_sheet().rename(columns={"ph_setpoint": "pH_production", "temperature_C": "temp_production_C"})
    target = ro.TargetSpec(
        scale_L=7500.0,
        agitation_rpm=150.0,
        airflow_L_per_min=300.0,
        process_params={"pH_production": 7.2, "temp_production_C": 37.0},
    )
    out = ro.build_readout(df, TARGET_COLUMN, ["pH_production", "temp_production_C"], target)
    assert out["decision"] == ro.NUMBER, out["reasons"]


# --------------------------------------------------------------------------- #
# license window: bench-scale wins cannot license plant-scale steps
# --------------------------------------------------------------------------- #


def test_bench_scale_rung_below_the_license_window_cannot_set_the_reference():
    """A 5x step won at 0.25 L must not license a 5x step at plant scale:
    only rungs within LICENSE_WINDOW_DECADES of the largest trained scale
    may set the reference ratio."""
    bench = ro.RungResult(
        scale_L=0.25, step_ratio=5.0, n=10, mae=0.1, naive_mean_mae=1.1, naive_nn_mae=1.1,
        beats_both=True, too_few_to_judge=False, in_license_window=False,
    )
    plant = _rung(2000.0, 2.0, beats_both=True)
    result = ro.decide(
        _ladder_result([bench, plant]),
        requested_ratio=4.5,
        target_process_params={"ph_setpoint": 7.0},
        trained_param_ranges={"ph_setpoint": (6.5, 7.5)},
    )
    assert result.reference_ratio == pytest.approx(2.0)
    # 4.5x > 2 x 2.0 warns; with the bench rung licensing (5.0) it would be clean
    assert result.decision == ro.NUMBER_WITH_WARNING


def test_only_bench_scale_wins_is_a_refusal_naming_the_window():
    bench = ro.RungResult(
        scale_L=0.25, step_ratio=5.0, n=10, mae=0.1, naive_mean_mae=1.1, naive_nn_mae=1.1,
        beats_both=True, too_few_to_judge=False, in_license_window=False,
    )
    result = ro.decide(
        _ladder_result([bench, _rung(2000.0, 2.0, beats_both=False)]),
        requested_ratio=2.0,
        target_process_params={"ph_setpoint": 7.0},
        trained_param_ranges={"ph_setpoint": (6.5, 7.5)},
    )
    assert result.decision == ro.REFUSAL
    assert any("decade(s) of the largest trained scale" in r for r in result.reasons)


def test_ladder_marks_rungs_outside_the_license_window():
    """Demo sheet: largest trained scale 5000 L, window 1 decade -> rungs at
    500 L and above are inside, the rest outside."""
    out = ro.build_readout(_demo_sheet(), TARGET_COLUMN, PROCESS_COLUMNS, _target(scale_L=7500.0))
    for r in out["rungs"]:
        assert r["in_license_window"] == (r["scale_L"] >= 500.0), r
