"""Tests for `kalos.normalize`: unit parsing/conversion, canonical naming,
offline role guessing, plan validation/round-trip, and end-to-end `apply_plan`
determinism on a hand-built messy run sheet.
"""
from __future__ import annotations

import pandas as pd
import pytest

from kalos.data.anonymizer import Anonymizer
from kalos.normalize import (
    ColumnPlan,
    NormalizationPlan,
    apply_plan,
    canonical_suffix,
    convert,
    guess_role,
    parse_value,
    snake_canonical,
)

# --- units ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "raw, expected_value, expected_unit",
    [
        ("34.6 C", 34.6, "C"),
        ("88%", 88.0, "%"),
        ("0.45 mL/h", 0.45, "mL/h"),
        ("37 C", 37.0, "C"),
        ("72", 72.0, None),
        (72, 72.0, None),
        ("n/a", None, None),
        ("", None, None),
    ],
)
def test_parse_value(raw, expected_value, expected_unit):
    value, unit = parse_value(raw)
    assert value == expected_value
    assert unit == expected_unit


def test_convert_temperature():
    assert convert(34.6, "C") == (34.6, "_c")
    assert convert(98.6, "F")[0] == pytest.approx(37.0, abs=1e-6)
    assert convert(310.15, "K")[0] == pytest.approx(37.0, abs=1e-6)


def test_convert_percent():
    assert convert(88.0, "%") == (88.0, "_pct")
    assert convert(72.0, "pct") == (72.0, "_pct")


def test_convert_concentration():
    assert convert(1.5, "g/L") == (1.5, "_g_l")
    assert convert(1.5, "mg/mL") == (1.5, "_g_l")  # 1 mg/mL == 1 g/L


def test_convert_flow_rate():
    assert convert(0.45, "mL/h") == (0.45, "_ml_h")
    assert convert(450.0, "uL/h") == (0.45, "_ml_h")
    assert convert(450.0, "µL/h") == (0.45, "_ml_h")  # micro sign variant
    assert convert(0.001, "L/h") == (1.0, "_ml_h")


def test_convert_time():
    assert convert(2.0, "h") == (2.0, "_h")
    assert convert(120.0, "min") == (2.0, "_h")


def test_convert_dimensionless_and_unknown_passthrough():
    assert convert(7.2, "pH") == (7.2, "")
    assert convert(0.8, None) == (0.8, "")
    assert convert(5.0, "furlongs") == (5.0, "")  # unknown unit -> passthrough


def test_canonical_suffix():
    assert canonical_suffix("C") == "_c"
    assert canonical_suffix("mg/mL") == "_g_l"
    assert canonical_suffix(None) == ""
    assert canonical_suffix("bogus") == ""


# --- synonyms: snake_canonical ---------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Titer (g/L)", "titer"),
        ("pH_final", "ph_final"),
        ("DO %", "do"),
        ("Client Sample Name", "client_sample_name"),
    ],
)
def test_snake_canonical(raw, expected):
    assert snake_canonical(raw) == expected


# --- synonyms: guess_role ---------------------------------------------------- #


def test_guess_role_identity_column():
    anon = Anonymizer()
    clean = anon.anonymize_meta({"Client Sample Name": "acme-01"})
    assert "Client Sample Name" not in clean  # confirms this header IS an identity col
    assert guess_role("Client Sample Name", is_numeric=False, parse_rate=0.0) == "identity"


def test_guess_role_titer_target():
    assert guess_role("Titer (g/L)", is_numeric=True, parse_rate=1.0) == "target"


def test_guess_role_group_column():
    assert guess_role("Campaign", is_numeric=False, parse_rate=0.0) == "group"


def test_guess_role_feature_and_freetext_and_metadata():
    assert guess_role("Temperature", is_numeric=True, parse_rate=1.0) == "feature"
    assert guess_role("Notes", is_numeric=False, parse_rate=0.1) == "freetext"
    assert guess_role("Run Date", is_numeric=False, parse_rate=0.9) == "metadata"


# --- plan: validate() --------------------------------------------------------- #


def test_validate_rejects_two_targets():
    plan = NormalizationPlan(
        columns=[
            ColumnPlan(raw_name="Titer (g/L)", canonical_name="titer", role="target"),
            ColumnPlan(raw_name="Yield (%)", canonical_name="yield_pct", role="target"),
        ],
        created_by="offline",
    )
    with pytest.raises(ValueError, match="target"):
        plan.validate()


def test_validate_rejects_duplicate_canonical_names():
    plan = NormalizationPlan(
        columns=[
            ColumnPlan(raw_name="Temp", canonical_name="temperature", role="feature"),
            ColumnPlan(raw_name="Temperature", canonical_name="temperature", role="feature"),
        ],
        created_by="offline",
    )
    with pytest.raises(ValueError, match="duplicate"):
        plan.validate()


def test_validate_rejects_identity_with_canonical_name():
    plan = NormalizationPlan(
        columns=[
            ColumnPlan(
                raw_name="Client Sample Name",
                canonical_name="sample_name",
                role="identity",
            ),
        ],
        created_by="offline",
    )
    with pytest.raises(ValueError, match="identity"):
        plan.validate()


def test_validate_passes_on_a_clean_plan():
    plan = NormalizationPlan(
        columns=[
            ColumnPlan(raw_name="Titer (g/L)", canonical_name="titer", role="target"),
            ColumnPlan(raw_name="Temp", canonical_name="temperature", role="feature"),
            ColumnPlan(raw_name="Client Sample Name", canonical_name=None, role="identity"),
        ],
        created_by="offline",
    )
    plan.validate()  # no raise


# --- plan: to_json/from_json round-trip -------------------------------------- #


def test_plan_json_round_trip():
    plan = NormalizationPlan(
        columns=[
            ColumnPlan(
                raw_name="Temp",
                canonical_name="temperature_c",
                role="feature",
                unit_token="C",
                to_base=True,
                note="parsed from cell text",
            ),
            ColumnPlan(raw_name="Client Sample Name", canonical_name=None, role="identity", is_identity=True),
        ],
        created_by="llm",
        model="claude-sonnet-5",
    )
    restored = NormalizationPlan.from_json(plan.to_json())
    assert restored == plan
    assert restored.to_json() == plan.to_json()


def test_plan_summary_counts():
    plan = NormalizationPlan(
        columns=[
            ColumnPlan(raw_name="Titer (g/L)", canonical_name="titer", role="target"),
            ColumnPlan(raw_name="Temp", canonical_name="temperature_c", role="feature", to_base=True),
            ColumnPlan(raw_name="Campaign", canonical_name="campaign", role="group"),
            ColumnPlan(raw_name="Client Sample Name", canonical_name=None, role="identity"),
            ColumnPlan(raw_name="Notes", canonical_name=None, role="freetext"),
        ],
        created_by="offline",
    )
    summary = plan.summary()
    assert summary["n_in"] == 5
    assert summary["n_canonical"] == 3
    assert summary["n_target"] == 1
    assert summary["n_feature"] == 1
    assert summary["n_group"] == 1
    assert summary["n_dropped"] == 2
    assert summary["n_unit_conversions"] == 1


# --- end-to-end apply_plan ---------------------------------------------------- #


def _messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Client Sample Name": ["acme-01", "acme-02", "acme-03"],
            "Campaign": ["camp-A", "camp-A", "camp-B"],
            "Titer (g/L)": [4.2, 5.1, 3.9],
            "Temp": ["34.6 C", "37 C", "36.2 C"],
            "DO %": ["88%", "72 %", "90%"],
            "Operator": ["Jane", "Jane", "Sam"],
            "Notes": ["ran fine", "slight foam", "n/a, check probe"],
        }
    )


def _messy_plan() -> NormalizationPlan:
    return NormalizationPlan(
        columns=[
            ColumnPlan(
                raw_name="Client Sample Name", canonical_name=None, role="identity"
            ),
            ColumnPlan(raw_name="Campaign", canonical_name="campaign", role="group"),
            ColumnPlan(raw_name="Titer (g/L)", canonical_name="titer", role="target"),
            ColumnPlan(
                raw_name="Temp",
                canonical_name="temperature_c",
                role="feature",
                unit_token="C",
                to_base=True,
            ),
            ColumnPlan(
                raw_name="DO %",
                canonical_name="dissolved_oxygen_pct",
                role="feature",
                unit_token="%",
                to_base=True,
            ),
            ColumnPlan(raw_name="Operator", canonical_name=None, role="identity"),
            ColumnPlan(raw_name="Notes", canonical_name=None, role="freetext"),
        ],
        created_by="offline",
    )


def test_apply_plan_end_to_end():
    df = _messy_df()
    plan = _messy_plan()
    result = apply_plan(df, plan, anonymizer=Anonymizer(salt="test-salt"))

    # identity + freetext dropped
    assert set(result.dropped) == {"Client Sample Name", "Operator", "Notes"}
    assert "client_sample_name" not in result.frame.columns
    assert "operator" not in result.frame.columns
    assert "notes" not in result.frame.columns

    # campaign hashed: value differs from raw, same raw value -> same hash
    assert "campaign" in result.frame.columns
    hashed = result.frame["campaign"].tolist()
    assert hashed[0] != "camp-A"
    assert hashed[0] == hashed[1]  # both rows were "camp-A"
    assert hashed[0] != hashed[2]  # "camp-B" hashes differently
    assert all(v.startswith("cmp_") for v in hashed)

    # temp -> temperature_c numeric 34.6 (already Celsius, base = passthrough)
    assert "temperature_c" in result.frame.columns
    assert result.frame["temperature_c"].tolist() == pytest.approx([34.6, 37.0, 36.2])

    # DO % -> dissolved_oxygen_pct == 88 (percent base is the bare number)
    assert "dissolved_oxygen_pct" in result.frame.columns
    assert result.frame["dissolved_oxygen_pct"].tolist() == pytest.approx([88.0, 72.0, 90.0])

    # titer kept as numeric target
    assert result.frame["titer"].tolist() == pytest.approx([4.2, 5.1, 3.9])

    # provenance actions
    actions = {p.raw_name: p.action for p in result.provenance}
    assert actions["Client Sample Name"] == "dropped_identity"
    assert actions["Operator"] == "dropped_identity"
    assert actions["Notes"] == "dropped_freetext"
    assert actions["Campaign"] == "hashed"
    assert actions["Temp"] == "converted"
    assert actions["DO %"] == "converted"
    assert actions["Titer (g/L)"] == "coerced"

    # row order and index preserved
    assert list(result.frame.index) == list(df.index)

    # determinism: same df + same plan -> identical frame on a second call
    result2 = apply_plan(df, plan, anonymizer=Anonymizer(salt="test-salt"))
    pd.testing.assert_frame_equal(result.frame, result2.frame)


def test_apply_plan_rejects_invalid_plan():
    df = _messy_df()
    bad_plan = NormalizationPlan(
        columns=[
            ColumnPlan(raw_name="Titer (g/L)", canonical_name="titer", role="target"),
            ColumnPlan(raw_name="Temp", canonical_name="titer", role="target"),
        ],
        created_by="offline",
    )
    with pytest.raises(ValueError):
        apply_plan(df, bad_plan)
