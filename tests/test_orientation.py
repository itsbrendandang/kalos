"""Tests for `kalos.normalize.orientation`: the tier-1 CSV/xlsx orientation
pre-pass (standard vs. transposed vs. ambiguous detection, and normalizing a
confidently-transposed sheet back to standard orientation), plus its wiring
into `kalos.portal.uploads._parse_upload` via `_apply_orientation_prepass`.

No real experimental data is used anywhere in this file - every frame is
hand-built/fabricated, mirroring the style of the ported adapter's own tests
(`ports/csv-orientation/test_csv_adapter.py` in the kalos-transition repo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.normalize.orientation import OrientationReport, detect_orientation
from kalos.portal.uploads import _apply_orientation_prepass, _parse_upload

# --- fixtures ----------------------------------------------------------------- #


def _standard_frame() -> pd.DataFrame:
    """Rows = runs, columns = parameters + objective."""
    return pd.DataFrame(
        {
            "temp": [30.0, 32.0, 35.0, 37.0, 28.0, 31.0],
            "ph": [6.5, 7.0, 7.2, 6.8, 6.6, 6.9],
            "agitation_rpm": [150, 180, 200, 175, 160, 190],
            "titer": [8.5, 9.2, 10.1, 9.8, 8.1, 9.5],
        }
    )


def _transposed_frame() -> pd.DataFrame:
    """The same shape of data, transposed: rows = parameters (first column
    holds their names), columns = runs."""
    return pd.DataFrame(
        {
            "parameter": ["temperature", "ph_setpoint", "induction_duration", "titer_output"],
            "exp1": [30.0, 6.5, 4.0, 8.5],
            "exp2": [32.0, 7.0, 5.0, 9.2],
            "exp3": [35.0, 7.2, 4.5, 10.1],
            "exp4": [37.0, 6.8, 6.0, 9.8],
            "exp5": [28.0, 6.6, 3.5, 8.1],
            "exp6": [31.0, 6.9, 5.5, 9.5],
        }
    )


def _narrow_tall_standard_frame() -> pd.DataFrame:
    """A common real bioprocess shape: few numeric parameter columns, many
    run rows. This is the shape that regressed kalos's own upload hardening
    tests (test_hardening.py, test_production_hardening.py) the first time
    the ported heuristic ran unmodified - see orientation.py's
    `_FIRST_COLUMN_NUMERIC_STANDARD_RATE` guard, which fixes it."""
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame(
        {
            "methanol": rng.uniform(0, 4, n).round(3),
            "ph": rng.uniform(5, 7, n).round(2),
            "lipase_titer": rng.uniform(1, 6, n).round(3),
        }
    )


# --- detect_orientation: standard --------------------------------------------- #


def test_standard_frame_detected_as_standard():
    report = detect_orientation(_standard_frame())
    assert report.orientation == "standard"
    assert report.normalized_frame is None


def test_narrow_tall_numeric_frame_never_misdetected_as_transposed():
    # The regression case: a first column that IS numeric run data (not
    # parameter labels) must never trip the transposed heuristics, no matter
    # how tall/narrow the sheet is.
    report = detect_orientation(_narrow_tall_standard_frame())
    assert report.orientation == "standard"
    assert report.normalized_frame is None


def test_type_header_column_forces_standard():
    df = pd.DataFrame(
        {
            "type": ["feature", "feature", "target"],
            "temp": [30.0, 32.0, 35.0],
            "titer": [8.5, 9.2, 10.1],
        }
    )
    report = detect_orientation(df)
    assert report.orientation == "standard"


def test_tiny_frame_reported_standard_not_crashed():
    report = detect_orientation(pd.DataFrame({"a": [1]}))
    assert report.orientation == "standard"
    assert report.normalized_frame is None


# --- detect_orientation: transposed -------------------------------------------- #


def test_transposed_frame_detected_with_positive_confidence_and_signals():
    report = detect_orientation(_transposed_frame())
    assert report.orientation == "transposed"
    assert report.confidence > 0
    assert report.signals.get("first_column_keywords", 0) > 0
    assert report.normalized_frame is not None


def test_transposed_normalizes_to_standard_shape():
    report = detect_orientation(_transposed_frame())
    normalized = report.normalized_frame
    assert normalized is not None
    # 6 runs (exp1..exp6) become rows; 4 parameters become columns.
    assert len(normalized) == 6
    assert set(normalized.columns) == {
        "temperature",
        "ph_setpoint",
        "induction_duration",
        "titer_output",
    }


def test_transposed_values_survive_round_trip():
    report = detect_orientation(_transposed_frame())
    normalized = report.normalized_frame
    assert normalized is not None
    # exp1's titer_output was 8.5 in the raw (transposed) frame.
    assert normalized.iloc[0]["titer_output"] == pytest.approx(8.5)
    # exp3's temperature was 35.0.
    assert normalized.iloc[2]["temperature"] == pytest.approx(35.0)


def test_transposed_disambiguates_duplicate_parameter_names():
    df = pd.DataFrame(
        {
            "parameter": ["temperature setpoint", "temperature setpoint", "titer_output"],
            "exp1": [30.0, 31.0, 8.5],
            "exp2": [32.0, 33.0, 9.2],
            "exp3": [35.0, 36.0, 10.1],
            "exp4": [37.0, 38.0, 9.8],
        }
    )
    report = detect_orientation(df)
    assert report.orientation == "transposed"
    normalized = report.normalized_frame
    assert normalized is not None
    assert "temperature setpoint" in normalized.columns
    assert "temperature setpoint__1" in normalized.columns


def test_transposed_filters_non_numeric_label_rows():
    df = _transposed_frame()
    # Insert a non-numeric annotation row (e.g. a units/notes row) - it must
    # be dropped, not coerced to NaN and kept.
    df.loc[len(df)] = ["notes", "ok", "ok", "flagged", "ok", "ok", "ok"]
    report = detect_orientation(df)
    assert report.orientation == "transposed"
    normalized = report.normalized_frame
    assert normalized is not None
    assert "notes" not in normalized.columns


# --- detect_orientation: ambiguous --------------------------------------------- #


def test_ambiguous_never_produces_a_normalized_frame():
    # Short non-numeric text codes with no keyword match, roughly square
    # shape - not enough signal either way.
    df = pd.DataFrame({"code": ["a1", "a2", "a3"], "x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    report = detect_orientation(df)
    assert report.orientation in ("standard", "ambiguous")
    if report.orientation == "ambiguous":
        assert report.normalized_frame is None


# --- never raises --------------------------------------------------------------- #


def test_never_raises_on_empty_frame():
    report = detect_orientation(pd.DataFrame())
    assert isinstance(report, OrientationReport)
    assert report.normalized_frame is None


def test_never_raises_on_all_nan_frame():
    df = pd.DataFrame({"a": [None, None, None], "b": [None, None, None]})
    report = detect_orientation(df)
    assert isinstance(report, OrientationReport)


def test_never_raises_on_mixed_object_dtypes():
    df = pd.DataFrame(
        {
            "weird": [{"a": 1}, [1, 2], object(), "text", 5.0],
            "b": [1, 2, 3, 4, 5],
        }
    )
    report = detect_orientation(df)
    assert isinstance(report, OrientationReport)


# --- uploads.py wiring ----------------------------------------------------------- #


def test_apply_orientation_prepass_records_signal_on_standard_frame():
    df = _standard_frame()
    out = _apply_orientation_prepass(df)
    assert out is df  # untouched, same object - byte-identical for standard sheets
    assert out.attrs["kalos_orientation"]["orientation"] == "standard"


def test_apply_orientation_prepass_normalizes_transposed_frame():
    df = _transposed_frame()
    out = _apply_orientation_prepass(df)
    assert out is not df
    assert out.attrs["kalos_orientation"]["orientation"] == "transposed"
    assert len(out) == 6
    assert len(out.columns) == 4


def test_apply_orientation_prepass_leaves_ambiguous_frame_untouched():
    df = pd.DataFrame({"code": ["a1", "a2", "a3"], "x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    out = _apply_orientation_prepass(df)
    assert out is df
    assert out.attrs["kalos_orientation"]["orientation"] in ("standard", "ambiguous")


def test_parse_upload_transposed_csv_is_normalized_before_return():
    text = _transposed_frame().to_csv(index=False)
    df = _parse_upload(text.encode("utf-8"))
    assert df.attrs["kalos_orientation"]["orientation"] == "transposed"
    assert len(df) == 6
    assert set(df.columns) == {"temperature", "ph_setpoint", "induction_duration", "titer_output"}


def test_parse_upload_standard_csv_is_byte_identical_in_values():
    raw_df = _standard_frame()
    text = raw_df.to_csv(index=False)
    parsed = _parse_upload(text.encode("utf-8"))
    assert parsed.attrs["kalos_orientation"]["orientation"] == "standard"
    pd.testing.assert_frame_equal(
        parsed.reset_index(drop=True), raw_df.reset_index(drop=True), check_dtype=False
    )
