"""Tests for the bioprocess data-validation gate (`kalos.validation`).

Each check is exercised on a small, hand-built frame where the "right
answer" is known in advance, plus a handful of end-to-end `validate_frame`
tests covering the engine's own hard floor (6 rows), a fully clean frame,
and the runner's crash-safety contract.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kalos.domains import BIOPROCESS_PROFILE
from kalos.validation import runner as runner_module
from kalos.validation.checks import (
    check_constant_columns,
    check_controls_present,
    check_duplicate_rows,
    check_missingness,
    check_outliers,
    check_physical_bounds,
    check_provenance_metadata,
    check_replicate_adequacy,
    check_units_consistency,
)
from kalos.validation.report import UnitConversion, report_dict
from kalos.validation.runner import apply_unit_conversions, validate_frame


def _severities(findings, check=None):
    return [f.severity for f in findings if check is None or f.check == check]


# --- 1. unit consistency ----------------------------------------------------- #


def test_units_consistency_flags_genuinely_inconsistent_mixed_concentration():
    # 1 mg/mL == 1 g/L numerically in units.py's own conversion table, so a
    # column mixing "1.2 g/L" with "1.2 mg/mL" would (mis-leadingly) look
    # numerically consistent after conversion. Pick values whose base-unit
    # numbers genuinely disagree (5.0 mg/mL converts to 5.0 g/L, not 1.2)
    # so the test demonstrates a real, dangerous mixed-unit column.
    df = pd.DataFrame({"titer": ["1.2 g/L", "5.0 mg/mL", "1.3 g/L"]})
    findings, conversions = check_units_consistency(df)
    errors = [f for f in findings if f.severity == "error"]
    assert len(errors) == 1
    assert errors[0].column == "titer"
    assert conversions == []  # ambiguous column: no auto-conversion record


def test_units_consistency_flags_mixed_temperature_and_converts_to_celsius():
    df = pd.DataFrame({"temp": ["34.6 C", "94.3 F", "35.0 C"]})
    findings, conversions = check_units_consistency(df)
    errors = [f for f in findings if f.severity == "error"]
    assert len(errors) == 1
    assert errors[0].column == "temp"

    # apply_unit_conversions converts each cell by its OWN parsed token
    # (not the aggregate record), so it correctly normalizes a mixed-unit
    # column to base Celsius even though check_units_consistency itself
    # declined to emit an automatic UnitConversion for an ambiguous column.
    manual_conversion = [
        UnitConversion(column="temp", from_units=("C", "F"), to_unit="_c", cells_converted=3)
    ]
    converted = apply_unit_conversions(df, manual_conversion)
    values = converted["temp"].tolist()
    assert values[0] == 34.6
    assert values[2] == 35.0
    assert abs(values[1] - 34.611) < 0.01  # (94.3-32)*5/9


def test_units_consistency_single_unit_is_info_and_produces_conversion():
    df = pd.DataFrame({"flow": ["10 mL/h", "20 mL/h", "30 mL/h"]})
    findings, conversions = check_units_consistency(df)
    assert [f.severity for f in findings] == ["info"]
    assert len(conversions) == 1
    assert conversions[0].column == "flow"
    assert conversions[0].cells_converted == 3


def test_units_consistency_bare_numbers_produce_no_findings():
    df = pd.DataFrame({"temp": [20.0, 21.5, 19.0]})
    findings, conversions = check_units_consistency(df)
    assert findings == []
    assert conversions == []


# --- 2. physical bounds ------------------------------------------------------- #


def test_physical_bounds_ph():
    df = pd.DataFrame({"ph": [40.0, 2.5, 7.1]})
    findings = check_physical_bounds(df)
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]
    assert len(errors) == 1 and 0 in errors[0].rows
    assert len(warnings) == 1 and 1 in warnings[0].rows
    for f in findings:
        assert 2 not in f.rows  # 7.1 is clean


def test_physical_bounds_negative_titer_is_error():
    df = pd.DataFrame({"titer_g_l": [-1.0, 5.0, 10.0]})
    findings = check_physical_bounds(df)
    errors = [f for f in findings if f.severity == "error"]
    assert len(errors) == 1
    assert 0 in errors[0].rows


def test_physical_bounds_skips_columns_with_no_inferable_dimension():
    df = pd.DataFrame({"batch": ["B1", "B2", "B3"]})
    assert check_physical_bounds(df) == []


# --- 3. duplicate rows --------------------------------------------------------- #


def test_duplicate_rows_exact_and_replicate():
    df = pd.DataFrame(
        {
            "run_id": [1, 1, 2, 2],
            "medium": ["A", "A", "B", "B"],
            "titer": [5.0, 5.0, 6.0, 7.0],  # rows 0/1 exact dup; rows 2/3 replicate
        }
    )
    findings = check_duplicate_rows(df, outcome_hint=BIOPROCESS_PROFILE.outcome_hint)
    exact = [f for f in findings if f.severity == "warning"]
    replicate = [f for f in findings if f.severity == "info"]
    assert len(exact) == 1
    assert set(exact[0].rows) == {0, 1}
    assert len(replicate) == 1
    assert set(replicate[0].rows) == {2, 3}


# --- 4. missingness ------------------------------------------------------------- #


def test_missingness_flags_95pct_blank_column():
    n = 20
    df = pd.DataFrame(
        {
            "full": list(range(n)),
            "sparse": [None] * (n - 1) + [1.0],  # 95% blank
        }
    )
    findings = check_missingness(df)
    col_findings = [f for f in findings if f.column == "sparse"]
    assert len(col_findings) == 1
    assert col_findings[0].severity == "warning"
    assert col_findings[0].detail["frac_blank"] == 0.95


def test_missingness_flags_50_90pct_as_sparse_not_unusable():
    n = 10
    df = pd.DataFrame(
        {
            "full": list(range(n)),
            "sparse": [None] * 6 + [1.0, 2.0, 3.0, 4.0],  # 60% blank
        }
    )
    findings = check_missingness(df)
    col_findings = [f for f in findings if f.column == "sparse"]
    assert len(col_findings) == 1
    assert "sparse" in col_findings[0].message


def test_missingness_flags_rows_more_than_half_blank():
    df = pd.DataFrame(
        {
            "a": [1.0, None, 3.0],
            "b": [1.0, None, 3.0],
            "c": [1.0, 2.0, 3.0],
        }
    )
    findings = check_missingness(df)
    row_findings = [f for f in findings if f.column is None]
    assert len(row_findings) == 1
    assert set(row_findings[0].rows) == {1}


# --- 5. outliers ------------------------------------------------------------------ #


def test_outliers_mad_catches_extreme_value():
    # A single wild outlier (1000) against a modestly-spread background; MAD
    # stays small (median deviation barely moves) so the outlier's z-score is
    # enormous. A mean/std based z-score would have its denominator inflated
    # by the very outlier it is trying to detect, which is exactly the
    # failure mode this check is designed to avoid.
    values = [9, 10, 10, 11, 12, 13, 14, 1000]
    df = pd.DataFrame({"x": values})
    findings = check_outliers(df)
    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert 7 in findings[0].rows  # index of 1000


def test_outliers_skips_zero_mad_column():
    df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0, 500.0]})
    # median=5, MAD=0 (more than half the values equal the median) -> guarded
    assert check_outliers(df) == []


# --- 6. provenance metadata -------------------------------------------------------- #


def test_provenance_metadata_warns_when_no_run_id_column():
    df = pd.DataFrame({"temp": [20.0, 21.0], "titer": [5.0, 6.0]})
    findings = check_provenance_metadata(df, BIOPROCESS_PROFILE.id_hint)
    missing = {f.detail["missing"] for f in findings}
    assert "run_or_batch_id" in missing
    assert "date" in missing
    assert "operator_or_instrument" in missing
    assert all(f.severity == "warning" for f in findings)


def test_provenance_metadata_clean_when_all_present():
    df = pd.DataFrame(
        {
            "batch": ["B1", "B2"],
            "date": ["2026-01-01", "2026-01-02"],
            "operator": ["opA", "opA"],
        }
    )
    findings = check_provenance_metadata(df, BIOPROCESS_PROFILE.id_hint)
    assert findings == []


# --- 7. replicate adequacy ---------------------------------------------------------- #


def test_replicate_adequacy_below_hard_floor_is_error():
    df = pd.DataFrame({"temp": [20.0, 21.0, 22.0, 23.0, 24.0], "titer": [1, 2, 3, 4, 5]})
    findings = check_replicate_adequacy(df, target="titer", features=["temp"])
    errors = [f for f in findings if f.severity == "error"]
    assert len(errors) == 1
    assert errors[0].detail["n_rows"] == 5


def test_replicate_adequacy_no_replication_warns():
    df = pd.DataFrame({"temp": [float(i) for i in range(10)], "titer": [float(i) for i in range(10)]})
    findings = check_replicate_adequacy(df, target="titer", features=["temp"])
    messages = [f.message for f in findings if f.severity == "warning"]
    assert any("no condition is replicated" in m for m in messages)


# --- 8. controls present --------------------------------------------------------------- #


def test_controls_present_warns_when_absent():
    df = pd.DataFrame({"condition": ["test1", "test2"]})
    findings = check_controls_present(df)
    assert len(findings) == 1
    assert findings[0].severity == "warning"


def test_controls_present_clean_when_value_present():
    df = pd.DataFrame({"condition": ["control", "test1", "test2"]})
    assert check_controls_present(df) == []


# --- 9. constant columns ------------------------------------------------------------------ #


def test_constant_columns_flags_zero_variance():
    df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0], "y": [1.0, 2.0, 3.0, 4.0]})
    findings = check_constant_columns(df)
    assert len(findings) == 1
    assert findings[0].column == "x"
    assert findings[0].severity == "info"


# --- end-to-end runner --------------------------------------------------------------------- #


def _clean_frame() -> pd.DataFrame:
    """A frame built to pass every one of the eleven checks clean (INFO allowed)."""
    temps = [20.0, 20.0, 25.0, 25.0, 30.0, 30.0, 35.0, 35.0, 40.0, 40.0]
    titers = [5.0, 5.2, 6.0, 6.1, 7.0, 7.3, 8.0, 8.2, 9.0, 9.4]
    n = len(temps)
    return pd.DataFrame(
        {
            "batch": [f"B{i}" for i in range(n)],
            "date": [f"2026-01-{i + 1:02d}" for i in range(n)],
            "operator": ["opA"] * n,
            "condition": ["control"] + ["test"] * (n - 1),
            "temp": temps,
            "titer": titers,
        }
    )


def test_validate_frame_clean_frame_passes():
    df = _clean_frame()
    report = validate_frame(df, target="titer", features=["temp"], profile=BIOPROCESS_PROFILE)
    assert report.status == "pass"
    assert report.counts["error"] == 0
    assert report.counts["warning"] == 0


def test_validate_frame_five_rows_is_fail():
    df = _clean_frame().iloc[:5].reset_index(drop=True)
    report = validate_frame(df, target="titer", features=["temp"])
    assert report.status == "fail"
    assert report.counts["error"] >= 1
    assert any(f.check == "replicate_adequacy" and f.severity == "error" for f in report.findings)


def test_validate_frame_never_raises_when_a_check_explodes(monkeypatch):
    def _boom(df):
        raise TypeError("simulated failure: numeric op on all-string column")

    monkeypatch.setattr(runner_module, "check_outliers", _boom)
    df = _clean_frame()
    report = validate_frame(df, target="titer", features=["temp"])
    assert report.status in {"pass", "pass_with_warnings", "fail"}
    failed = [f for f in report.findings if f.check == "outliers" and f.severity == "warning"]
    assert len(failed) == 1
    assert "check failed to run: TypeError" in failed[0].message


def test_validate_frame_survives_weird_object_column():
    df = _clean_frame()
    df["weird"] = [{"a": 1}, [1, 2], "x", None, object(), 3.14, "%%", "", "n/a", (1, 2)]
    # Should not raise regardless of what the extra column does to any
    # individual check; the runner's own try/except is the safety net.
    report = validate_frame(df, target="titer", features=["temp"])
    assert report.n_rows == len(df)


def test_report_row_indices_are_plain_ints_and_json_safe():
    df = pd.DataFrame({"ph": [40.0, 2.5, 7.1, np.float64(50.0)]})
    report = validate_frame(df)
    for finding in report.findings:
        for row in finding.rows:
            assert type(row) is int
    payload = report_dict(report)
    dumped = json.dumps(payload)
    assert isinstance(dumped, str)
