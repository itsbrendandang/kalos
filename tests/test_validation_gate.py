"""The validation gate wired into the live `_analyze` path.

`tests/test_validation.py` covers the checks in isolation. This file covers the
integration: that the gate actually runs on the sheet a client uploads, that its
report reaches the response, that strict mode can refuse an upload, and above
all that the recommendation engine can no longer propose a physically impossible
recipe.

The centerpiece is `test_sensor_sentinel_cannot_produce_impossible_recipe`. It is
a regression test for an observed defect, not a hypothetical: a 12-row sheet with
one `-999` "sensor offline" sentinel in its temperature column produced proposals
at -422 C, -350 C and -858 C - below absolute zero - each with a confidence
interval, because the design box was the raw observed [min, max] of every
feature. Nothing in the response flagged it, and the per-column provenance
report actively said the column was clean (`coerced_cells=0`). Any change that
lets that recur must fail here.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from kalos.portal.analysis import _analyze, validation_mode  # noqa: E402
from kalos.portal.uploads import UploadRejected  # noqa: E402

# Absolute zero in Celsius. Nothing below this is a temperature.
_ABSOLUTE_ZERO_C = -273.15


def _corrupt_sheet() -> pd.DataFrame:
    """A 12-row sheet carrying the defects a real client export actually ships.

    Planted, one per class:
      feed_mL_h  row 8  - negative flow, impossible for a pump
      temp_C     row 5  - -999, a "sensor offline" sentinel, not a reading
      pH         row 3  - 40.0, outside the 0-14 pH scale
      titer_g_L  rows 6+ - a 1000x scale break (g/L values then mg/mL values)
      (no run id / date / operator column anywhere)
    """
    return pd.DataFrame(
        {
            "feed_mL_h": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, -0.60, 0.65, 0.70, 0.75],
            "temp_C": [37.0, 37.2, 36.8, 37.1, 36.9, -999.0, 37.0, 37.3, 36.7, 37.2, 36.9, 37.1],
            "pH": [7.0, 7.1, 6.9, 40.0, 7.05, 7.0, 6.95, 7.1, 7.0, 7.15, 6.9, 7.05],
            "titer_g_L": [1.2, 1.4, 1.3, 1.5, 1.6, 1.55, 1600.0, 1700.0, 1650.0, 1800.0, 1750.0, 1900.0],
        }
    )


def _clean_sheet(n: int = 24) -> pd.DataFrame:
    """A physically plausible sheet with a batch id, so the gate has nothing to
    report as an error. Deterministic, no RNG."""
    return pd.DataFrame(
        {
            "batch_id": [f"BQ-{i:04d}" for i in range(n)],
            "feed_mL_h": [0.20 + 0.02 * i for i in range(n)],
            "temp_C": [36.5 + 0.05 * (i % 8) for i in range(n)],
            "pH": [6.85 + 0.02 * (i % 10) for i in range(n)],
            "titer_g_L": [1.0 + 0.15 * i - 0.004 * i * i for i in range(n)],
        }
    )


# --- the regression -------------------------------------------------------- #


def test_sensor_sentinel_cannot_produce_impossible_recipe():
    """One -999 sentinel must not widen the search space to impossible recipes."""
    out = _analyze(_corrupt_sheet(), target="titer_g_L")

    recipes = [p["recipe"] for p in out["proposals"]]
    assert recipes, "expected at least one proposal to inspect"
    for recipe in recipes:
        # The original defect: temperatures hundreds of degrees below zero.
        assert recipe["temp_C"] > _ABSOLUTE_ZERO_C, f"proposed below absolute zero: {recipe}"
        # Tighter, and the property that actually matters: every proposal sits
        # inside the range of PHYSICALLY VALID observations, not the raw span.
        assert 36.7 <= recipe["temp_C"] <= 37.3, f"temp outside valid observed range: {recipe}"
        assert 0.0 <= recipe["pH"] <= 14.0, f"impossible pH proposed: {recipe}"
        assert recipe["feed_mL_h"] >= 0.0, f"negative feed rate proposed: {recipe}"


def test_impossible_values_are_reported_as_errors_not_swallowed():
    """Fixing the proposals is not enough - the client must be told why."""
    out = _analyze(_corrupt_sheet(), target="titer_g_L")
    v = out["validation"]

    assert v["status"] == "fail"
    assert v["counts"]["error"] >= 3

    bounds_errors = [
        f for f in v["findings"] if f["check"] == "physical_bounds" and f["severity"] == "error"
    ]
    flagged = {f["column"] for f in bounds_errors}
    assert {"temp_C", "pH", "feed_mL_h"} <= flagged, f"missed a column: {flagged}"

    # Each error names the offending row, so the client can go fix that cell.
    by_col = {f["column"]: f for f in bounds_errors}
    assert 5 in by_col["temp_C"]["rows"]
    assert 3 in by_col["pH"]["rows"]
    assert 8 in by_col["feed_mL_h"]["rows"]


def test_design_box_narrowing_is_reported_never_silent():
    """Narrowing the search space is a real change to what was optimized, so it
    has to appear in the response rather than happening behind the client."""
    out = _analyze(_corrupt_sheet(), target="titer_g_L")
    exclusions = {e["column"]: e for e in out["design_box_exclusions"]}

    assert {"temp_C", "pH", "feed_mL_h"} <= set(exclusions)
    temp = exclusions["temp_C"]
    assert temp["n_excluded"] == 1
    assert temp["lower"] == pytest.approx(36.7)
    assert temp["upper"] == pytest.approx(37.3)


def test_missing_provenance_is_flagged():
    """A sheet with no run id / date / operator cannot be traced to a physical
    run, which is a traceability failure worth naming even when the numbers are
    all plausible."""
    v = _analyze(_corrupt_sheet(), target="titer_g_L")["validation"]
    missing = {
        f["detail"].get("missing")
        for f in v["findings"]
        if f["check"] == "provenance_metadata"
    }
    assert {"run_or_batch_id", "date", "operator_or_instrument"} <= missing


# --- mode policy ----------------------------------------------------------- #


def test_default_mode_is_warn_and_does_not_reject():
    """Adding the gate must not change what the API accepts by default."""
    assert validation_mode() == "warn"
    out = _analyze(_corrupt_sheet(), target="titer_g_L")  # must not raise
    assert out["validation"]["mode"] == "warn"
    assert out["validation"]["status"] == "fail"


def test_strict_mode_rejects_a_failing_sheet(monkeypatch):
    monkeypatch.setenv("KALOS_VALIDATION_MODE", "strict")
    assert validation_mode() == "strict"
    with pytest.raises(UploadRejected):
        _analyze(_corrupt_sheet(), target="titer_g_L")


def test_strict_mode_accepts_a_clean_sheet(monkeypatch):
    """Strict must reject only on error-severity findings; warnings pass."""
    monkeypatch.setenv("KALOS_VALIDATION_MODE", "strict")
    out = _analyze(_clean_sheet(), target="titer_g_L")
    assert out["validation"]["counts"]["error"] == 0
    assert out["validation"]["status"] in {"pass", "pass_with_warnings"}


def test_unrecognized_mode_falls_back_to_warn(monkeypatch):
    """A typo in deployment config must not silently enable rejection."""
    monkeypatch.setenv("KALOS_VALIDATION_MODE", "STRICTT")
    assert validation_mode() == "warn"


# --- unit conversion in the live path -------------------------------------- #


def test_unit_tagged_column_is_converted_instead_of_dropped():
    """A temperature column written "36.8 C" fails the numeric-parse test and was
    being dropped as non-numeric, throwing away a real process input. Converting
    before feature selection recovers it, in Celsius."""
    df = _clean_sheet()
    df["temp_C"] = [f"{v:.1f} C" for v in df["temp_C"]]
    out = _analyze(df, target="titer_g_L")

    assert "temp_C" in out["features"], "unit-tagged column was dropped, not converted"
    conversions = {c["column"]: c for c in out["validation"]["conversions"]}
    assert conversions["temp_C"]["to_unit_label"] == "Celsius"
    for recipe in (p["recipe"] for p in out["proposals"]):
        assert 36.0 <= recipe["temp_C"] <= 38.0


def test_fahrenheit_and_celsius_in_one_column_is_an_error():
    """The classic silent-corruption case: one column, two units."""
    df = _clean_sheet()
    df["temp_C"] = [
        f"{v:.1f} C" if i % 2 else f"{v * 9 / 5 + 32:.1f} F"
        for i, v in enumerate(df["temp_C"])
    ]
    v = _analyze(df, target="titer_g_L")["validation"]
    unit_errors = [
        f for f in v["findings"] if f["check"] == "units_consistency" and f["severity"] == "error"
    ]
    assert unit_errors, "mixed C/F in one column was not reported as an error"
    assert unit_errors[0]["column"] == "temp_C"


def test_unrecognized_unit_is_not_silently_converted():
    """A vessel label like "5L" uses a token the registry cannot convert.
    Rewriting it to a bare 5 would turn an identifier into a measurement, so the
    column must be left alone and the client told."""
    df = _clean_sheet()
    df["vessel"] = ["5L" if i % 2 else "500L" for i in range(len(df))]
    v = _analyze(df, target="titer_g_L")["validation"]

    assert "vessel" not in {c["column"] for c in v["conversions"]}
    warned = [
        f
        for f in v["findings"]
        if f["check"] == "units_consistency" and f["column"] == "vessel"
    ]
    assert warned and warned[0]["severity"] == "warning"


# --- response shape -------------------------------------------------------- #


def test_clean_sheet_reports_no_errors_and_no_narrowing():
    out = _analyze(_clean_sheet(), target="titer_g_L")
    assert out["validation"]["counts"]["error"] == 0
    assert out["design_box_exclusions"] == []


def test_validation_block_is_json_serializable():
    """The report is serialized into an HTTP response, so no numpy scalars."""
    out = _analyze(_corrupt_sheet(), target="titer_g_L")
    encoded = json.dumps(out["validation"])
    assert '"schema_version": 1' in encoded
    for finding in out["validation"]["findings"]:
        for row in finding["rows"]:
            assert type(row) is int


def test_row_lists_are_capped_but_the_true_total_is_kept():
    """A finding must never imply its row list is complete when it is not."""
    n = 60
    df = pd.DataFrame(
        {
            "batch_id": [f"BQ-{i:04d}" for i in range(n)],
            "feed_mL_h": [0.2 + 0.01 * i for i in range(n)],
            # every pH impossible: 60 offending rows, far past the 20-row cap
            "pH": [40.0 + 0.1 * i for i in range(n)],
            "titer_g_L": [1.0 + 0.1 * i for i in range(n)],
        }
    )
    v = _analyze(df, target="titer_g_L")["validation"]
    ph_error = next(
        f
        for f in v["findings"]
        if f["check"] == "physical_bounds" and f["column"] == "pH"
    )
    assert len(ph_error["rows"]) == 20
    assert ph_error["detail"]["n_rows_affected"] == n
