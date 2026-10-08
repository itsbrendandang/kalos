"""Tests for the LLM tier of `kalos.normalize`: the identity-stripping
payload boundary, the offline deterministic fallback, and the mocked live
LLM path. Fully hermetic - no network call is ever made, and no real
Anthropic API key is used anywhere in this file.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from kalos.normalize import apply_plan, build_payload, load_config
from kalos.normalize.config import credentials_available
from kalos.normalize.llm import offline_plan, propose_plan

# --- shared fixture ---------------------------------------------------------- #


def _messy_df() -> pd.DataFrame:
    """A hand-built messy run sheet: two identity columns, one free-text
    column, one grouping column, one numeric target, and one numeric+unit
    feature - the same shape a real client run sheet takes."""
    return pd.DataFrame(
        {
            "Client Sample Name": [
                "Acme-BF-018",
                "Acme-BF-019",
                "Acme-BF-020",
                "Acme-BF-021",
                "Acme-BF-022",
                "Acme-BF-023",
            ],
            "Operator": ["J. Rivera", "J. Rivera", "M. Chen", "M. Chen", "J. Rivera", "M. Chen"],
            "Notes": [
                "Foam observed at hour 12, adjusted antifoam accordingly during the run.",
                "Nothing unusual, run proceeded per protocol without any deviation at all.",
                "Slight pH drift corrected manually at the 24h timepoint by the operator.",
                "Client requested an early harvest due to a scheduling conflict this week.",
                "DO probe recalibrated mid run after a suspicious reading was noticed today.",
                "Feed pump replaced after an unexpected stall during the overnight shift.",
            ],
            "Campaign": ["C-100", "C-100", "C-100", "C-101", "C-101", "C-101"],
            "Titer (g/L)": [3.2, 3.6, 4.1, 3.9, 4.4, 4.0],
            "Temp": ["34.6 C", "34.8 C", "35.0 C", "34.9 C", "35.1 C", "34.7 C"],
        }
    )


# --- 1. privacy: build_payload is the identity-stripping boundary ---------- #


def test_build_payload_privacy():
    df = _messy_df()
    payload, dropped_identity = build_payload(df)

    headers = {c["header"] for c in payload["columns"]}
    assert "Client Sample Name" not in headers
    assert "Operator" not in headers
    assert set(dropped_identity) == {"Client Sample Name", "Operator"}

    by_header = {c["header"]: c for c in payload["columns"]}
    assert by_header["Notes"]["sample"] == "<redacted>"

    titer_sample = by_header["Titer (g/L)"]["sample"]
    assert titer_sample, "expected non-empty sample for a numeric column"
    assert all(isinstance(v, (int, float)) for v in titer_sample)

    dumped = json.dumps(payload)
    assert dumped  # round-trips without raising

    for raw_value in ("Acme-BF-018", "Acme-BF-019", "J. Rivera", "M. Chen"):
        assert raw_value not in dumped


def test_build_payload_is_deterministic_and_guards_identity_headers():
    df = _messy_df()
    payload_a, dropped_a = build_payload(df)
    payload_b, dropped_b = build_payload(df)
    assert payload_a == payload_b
    assert dropped_a == dropped_b

    # Defense-in-depth guard: build_payload asserts no header in the result
    # matches the anonymizer's DROP rules - exercise it directly to ensure
    # the check runs (it should never fire in normal operation).
    from kalos.normalize import payload as payload_module

    assert not payload_module._is_identity_header("Titer (g/L)")
    assert payload_module._is_identity_header("Operator")


# --- 2. offline fallback ----------------------------------------------------- #


def test_offline_plan_created_by_and_validates():
    df = _messy_df()
    plan = offline_plan(df)
    assert plan.created_by == "offline"
    assert plan.model is None
    plan.validate()  # must not raise

    by_raw = {c.raw_name: c for c in plan.columns}
    assert by_raw["Client Sample Name"].role == "identity"
    assert by_raw["Client Sample Name"].canonical_name is None
    assert by_raw["Operator"].role == "identity"
    assert by_raw["Notes"].role == "freetext"
    assert by_raw["Notes"].canonical_name is None
    assert by_raw["Campaign"].role == "group"
    assert by_raw["Titer (g/L)"].role == "target"
    assert by_raw["Temp"].role == "feature"
    assert by_raw["Temp"].to_base is True
    assert by_raw["Temp"].unit_token == "C"


def test_propose_plan_offline_when_no_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    df = _messy_df()

    plan = propose_plan(df)
    assert plan.created_by == "offline"
    plan.validate()

    result = apply_plan(df, plan)
    assert "Client Sample Name" in result.dropped
    assert "Operator" in result.dropped
    assert "Notes" in result.dropped

    # temp -> temperature_c, numeric, unchanged value (already Celsius)
    assert "temperature_c" in result.frame.columns
    assert result.frame["temperature_c"].tolist() == pytest.approx([34.6, 34.8, 35.0, 34.9, 35.1, 34.7])

    # group column hashed to a stable pseudonym, not the raw campaign label
    group_col = [c for c in plan.columns if c.raw_name == "Campaign"][0]
    hashed_name = group_col.canonical_name
    assert hashed_name in result.frame.columns
    assert not result.frame[hashed_name].isin(["C-100", "C-101"]).any()


# --- 3. mocked live path ------------------------------------------------------ #


class _FakeMessages:
    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class _FakeClient:
    def __init__(self, messages: _FakeMessages):
        self.messages = messages


def _canned_llm_columns(df: pd.DataFrame) -> list[SimpleNamespace]:
    """A canned `parsed_output.columns` list mirroring what the model would
    return for the surviving (non-identity) columns of `_messy_df()`."""
    return [
        SimpleNamespace(
            raw_name="Notes",
            canonical_name=None,
            role="freetext",
            unit_token=None,
            is_identity=False,
            note="free text notes",
        ),
        SimpleNamespace(
            raw_name="Campaign",
            canonical_name="campaign",
            role="group",
            unit_token=None,
            is_identity=False,
            note="campaign grouping id",
        ),
        SimpleNamespace(
            raw_name="Titer (g/L)",
            canonical_name="titer",
            role="target",
            unit_token="g/L",
            is_identity=False,
            note="product titer",
        ),
        SimpleNamespace(
            raw_name="Temp",
            canonical_name="temperature_c",
            role="feature",
            unit_token="C",
            is_identity=False,
            note="culture temperature",
        ),
    ]


def test_propose_plan_live_mocked(monkeypatch):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-not-real")
    df = _messy_df()

    canned = SimpleNamespace(columns=_canned_llm_columns(df))
    fake_messages = _FakeMessages(response=SimpleNamespace(parsed_output=canned))
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: _FakeClient(fake_messages))

    config = load_config()
    plan = propose_plan(df, config=config)

    assert plan.created_by == "llm"
    assert plan.model == config.model
    assert config.model == "claude-sonnet-5"
    plan.validate()

    # Identity columns merged back in from the deterministic pre-screen,
    # never sent to (or returned by) the model.
    by_raw = {c.raw_name: c for c in plan.columns}
    assert by_raw["Client Sample Name"].role == "identity"
    assert by_raw["Operator"].role == "identity"

    # Re-assert the privacy guarantee on the live path: capture exactly what
    # was sent to the (fake) model and confirm no identity header and no raw
    # free-text value made it into the request.
    assert len(fake_messages.calls) == 1
    sent_content = fake_messages.calls[0]["messages"][0]["content"]
    assert "Client Sample Name" not in sent_content
    assert "Operator" not in sent_content
    for raw_value in ("Acme-BF-018", "J. Rivera", "M. Chen"):
        assert raw_value not in sent_content
    sent_payload = json.loads(sent_content)
    by_header = {c["header"]: c for c in sent_payload["columns"]}
    assert by_header["Notes"]["sample"] == "<redacted>"


def test_propose_plan_falls_back_on_api_error(monkeypatch):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-not-real")
    df = _messy_df()

    error = anthropic.APIConnectionError(
        message="simulated connection failure",
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    fake_messages = _FakeMessages(error=error)
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: _FakeClient(fake_messages))

    plan = propose_plan(df)  # must not raise
    assert plan.created_by == "offline"
    plan.validate()


# --- 4. config ---------------------------------------------------------------- #


def test_load_config_defaults(monkeypatch):
    monkeypatch.delenv("KALOS_NORMALIZE_MODEL", raising=False)
    monkeypatch.delenv("KALOS_NORMALIZE_MAX_SAMPLE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config = load_config()
    assert config.model == "claude-sonnet-5"
    assert config.max_sample == 5
    assert config.enabled_live is False
    assert credentials_available() is False


def test_load_config_model_override(monkeypatch):
    monkeypatch.setenv("KALOS_NORMALIZE_MODEL", "claude-opus-4-8")
    config = load_config()
    assert config.model == "claude-opus-4-8"


def test_credentials_available_reflects_env_var_only(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert credentials_available() is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-not-real")
    assert credentials_available() is True

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert credentials_available() is False


def test_max_sample_clamped(monkeypatch):
    monkeypatch.setenv("KALOS_NORMALIZE_MAX_SAMPLE", "999")
    assert load_config().max_sample == 20

    monkeypatch.setenv("KALOS_NORMALIZE_MAX_SAMPLE", "0")
    assert load_config().max_sample == 1

    monkeypatch.setenv("KALOS_NORMALIZE_MAX_SAMPLE", "not-a-number")
    assert load_config().max_sample == 5


def test_offline_plan_production_phase_headers_do_not_break_validation():
    """Bioprocess sheets name process conditions after the production phase
    ("temp_production_C", "pH_production"). The outcome heuristic matches
    "product" inside those headers; without a named target the plan must
    still validate (no target, every candidate kept as a feature) instead
    of raising "more than one target column"."""
    df = pd.DataFrame(
        {
            "temp_production_C": [25.0, 25.0, 30.0],
            "pH_production": [6.0, 7.0, 7.0],
            "lipase_g.L": [0.6, 1.59, 1.31],
        }
    )
    plan = offline_plan(df)
    assert [c.raw_name for c in plan.columns if c.role == "target"] == []
    notes = {c.raw_name: c.note for c in plan.columns}
    assert "caller must name the target" in notes["lipase_g.L"]


def test_offline_plan_explicit_target_wins_over_the_header_heuristic():
    df = pd.DataFrame(
        {
            "temp_production_C": [25.0, 25.0, 30.0],
            "pH_production": [6.0, 7.0, 7.0],
            "lipase_g.L": [0.6, 1.59, 1.31],
        }
    )
    plan = offline_plan(df, target_column="lipase_g.L")
    roles = {c.raw_name: c.role for c in plan.columns}
    assert roles == {"temp_production_C": "feature", "pH_production": "feature", "lipase_g.L": "target"}
