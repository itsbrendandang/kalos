"""Tests for `kalos.ingest.llm.fill_unresolved_fields`: the schema-constrained
Ollama structured-output tier, and its fallback discipline on every failure
mode. Fully hermetic - `urllib.request.urlopen` is monkeypatched throughout
(mirroring `tests/test_llm_providers.py`'s own fake-urlopen pattern for the
`kalos.normalize` Ollama provider); no live Ollama server or network call is
ever made.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

from kalos.ingest.llm import _build_schema, fill_unresolved_fields
from kalos.ingest.schema import Measurement
from kalos.normalize.config import NormalizeConfig


def _config(**overrides: Any) -> NormalizeConfig:
    base = dict(
        model="unused",
        max_sample=5,
        enabled_live=True,
        provider="ollama",
        ollama_url="http://localhost:11434",
        ollama_model="llama3.1",
    )
    base.update(overrides)
    return NormalizeConfig(**base)


class _FakeHTTPResponse:
    """A context-manager stand-in for `http.client.HTTPResponse`, matching
    `tests/test_llm_providers.py::_FakeHTTPResponse` exactly."""

    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _fake_urlopen(
    *,
    health_status: int = 200,
    generate_status: int = 200,
    generate_response_text: str | None = None,
    raise_on: str | None = None,
    capture: dict[str, Any] | None = None,
):
    """Build a fake `urllib.request.urlopen`, distinguishing the health check
    (bare URL, GET) from the generate call (a `urllib.request.Request`, POST)
    by argument type - exactly how `_check_ollama_server`/`_generate_json`
    call it. `capture`, if given, records the decoded JSON body of the
    generate request under `capture["body"]` for the caller to inspect.
    """

    def _urlopen(url_or_request: Any, timeout: float | None = None) -> _FakeHTTPResponse:
        if isinstance(url_or_request, urllib.request.Request):
            if capture is not None:
                capture["body"] = json.loads(url_or_request.data.decode("utf-8"))
            if raise_on == "generate":
                raise TimeoutError("simulated generate timeout")
            body = json.dumps({"response": generate_response_text or "{}"}).encode("utf-8")
            return _FakeHTTPResponse(generate_status, body)
        if raise_on == "health":
            raise TimeoutError("simulated health-check timeout")
        return _FakeHTTPResponse(health_status, b"{}")

    return _urlopen


# --- provider gating: never touch the network unless provider == "ollama" ------ #


def test_no_fields_requested_returns_empty_without_any_call(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("must never call urlopen with no fields requested")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert fill_unresolved_fields("some text", [], config=_config()) == {}


def test_provider_anthropic_never_touches_the_network(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("must never call urlopen when provider='anthropic'")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    result = fill_unresolved_fields("text", ["final_titer"], config=_config(provider="anthropic"))
    assert result == {}


def test_provider_none_never_touches_the_network(monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError("must never call urlopen when provider='none'")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    result = fill_unresolved_fields("text", ["final_titer"], config=_config(provider="none"))
    assert result == {}


# --- happy path + the structured-output upgrade --------------------------------- #


def test_happy_path_returns_validated_measurements(monkeypatch):
    canned = {"final_titer": {"value": 6.1, "unit": "g/L"}, "duration": {"value": 96, "unit": "hours"}}
    fake = _fake_urlopen(generate_response_text=json.dumps(canned))
    monkeypatch.setattr(urllib.request, "urlopen", fake)

    result = fill_unresolved_fields("raw pdf text", ["final_titer", "duration"], config=_config())

    assert result == {
        "final_titer": Measurement(value=6.1, unit="g/L", method="llm_ollama"),
        "duration": Measurement(value=96.0, unit="hours", method="llm_ollama"),
    }


def test_format_field_carries_the_json_schema_not_the_string_json(monkeypatch):
    # The upgrade this module implements over kalos.normalize.llm's Ollama
    # provider: `format` must be the actual target JSON Schema object (so
    # Ollama constrains generation to it), never the bare string "json".
    capture: dict[str, Any] = {}
    fake = _fake_urlopen(
        generate_response_text=json.dumps({"final_titer": {"value": 6.1, "unit": "g/L"}}),
        capture=capture,
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake)

    fill_unresolved_fields("raw pdf text", ["final_titer"], config=_config())

    sent_format = capture["body"]["format"]
    assert sent_format == _build_schema(["final_titer"])
    assert sent_format != "json"
    assert sent_format["type"] == "object"
    assert set(sent_format["properties"]) == {"final_titer"}
    assert sent_format["properties"]["final_titer"]["required"] == ["value", "unit"]
    # Matches kalos.normalize.llm's own low-temperature / non-streaming shape.
    assert capture["body"]["stream"] is False
    assert capture["body"]["options"]["temperature"] == 0.0
    assert capture["body"]["model"] == "llama3.1"


def test_prompt_names_exactly_the_requested_fields(monkeypatch):
    capture: dict[str, Any] = {}
    fake = _fake_urlopen(generate_response_text="{}", capture=capture)
    monkeypatch.setattr(urllib.request, "urlopen", fake)

    fill_unresolved_fields("raw pdf text with a titer of 6.1 g/L", ["final_titer", "final_od"], config=_config())

    prompt = capture["body"]["prompt"]
    assert "final_titer" in prompt
    assert "final_od" in prompt
    assert "raw pdf text with a titer of 6.1 g/L" in prompt


# --- validation: schema-shaped is not the same as accepted ---------------------- #


def test_partial_response_keeps_only_well_formed_fields(monkeypatch):
    canned = {
        "final_titer": {"value": 6.1, "unit": "g/L"},
        "final_od": {"value": "not-a-number", "unit": "OD"},  # wrong type -> dropped
        "duration": {"value": 96.0},  # missing unit -> dropped
    }
    fake = _fake_urlopen(generate_response_text=json.dumps(canned))
    monkeypatch.setattr(urllib.request, "urlopen", fake)

    result = fill_unresolved_fields(
        "text", ["final_titer", "final_od", "duration"], config=_config()
    )
    assert list(result) == ["final_titer"]


def test_field_not_requested_is_ignored_even_if_present(monkeypatch):
    canned = {"final_titer": {"value": 6.1, "unit": "g/L"}, "yield_value": {"value": 99.0, "unit": "g/L"}}
    fake = _fake_urlopen(generate_response_text=json.dumps(canned))
    monkeypatch.setattr(urllib.request, "urlopen", fake)

    result = fill_unresolved_fields("text", ["final_titer"], config=_config())
    assert list(result) == ["final_titer"]


# --- fallback discipline: any failure -> {} , deterministic tier's result stands  #


def test_health_check_failure_falls_back_to_empty_without_posting(monkeypatch):
    fake = _fake_urlopen(raise_on="health")
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    result = fill_unresolved_fields("text", ["final_titer"], config=_config())
    assert result == {}


def test_generate_timeout_falls_back_to_empty(monkeypatch):
    fake = _fake_urlopen(raise_on="generate")
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    result = fill_unresolved_fields("text", ["final_titer"], config=_config())
    assert result == {}


def test_non_200_generate_status_falls_back_to_empty(monkeypatch):
    fake = _fake_urlopen(generate_status=500, generate_response_text="{}")
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    result = fill_unresolved_fields("text", ["final_titer"], config=_config())
    assert result == {}


def test_unparsable_response_text_falls_back_to_empty(monkeypatch):
    fake = _fake_urlopen(generate_response_text="not json at all {{{")
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    result = fill_unresolved_fields("text", ["final_titer"], config=_config())
    assert result == {}


def test_non_dict_response_json_falls_back_to_empty(monkeypatch):
    fake = _fake_urlopen(generate_response_text=json.dumps([1, 2, 3]))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    result = fill_unresolved_fields("text", ["final_titer"], config=_config())
    assert result == {}
