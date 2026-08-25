"""Tests for the Ollama provider seam in `kalos.normalize.llm`/`config`:
`KALOS_LLM_PROVIDER` selection, the happy-path Ollama column-plan mapping,
and its fallback behavior on malformed/schema-invalid JSON and on a
timeout/unreachable server. Fully hermetic - `urllib.request.urlopen` is
monkeypatched throughout; no live Ollama server or network call is ever
made. The existing Anthropic-path tests in `test_normalize_llm.py` are left
untouched and must keep passing unmodified (verified by the shared test
run, not duplicated here) - this file only re-confirms that the default,
no-`KALOS_LLM_PROVIDER` path still resolves to the offline plan exactly as
before this provider seam existed.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from kalos.normalize.config import load_config
from kalos.normalize.llm import propose_plan


def _messy_df() -> pd.DataFrame:
    """A hand-built messy run sheet: two identity columns, one grouping
    column, one numeric target, and one numeric+unit feature - the same
    shape of fixture `test_normalize_llm.py` uses for the Anthropic path."""
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
            "Campaign": ["C-100", "C-100", "C-100", "C-101", "C-101", "C-101"],
            "Titer (g/L)": [3.2, 3.6, 4.1, 3.9, 4.4, 4.0],
            "Temp": ["34.6 C", "34.8 C", "35.0 C", "34.9 C", "35.1 C", "34.7 C"],
        }
    )


def _canned_ollama_columns() -> list[dict[str, Any]]:
    """A canned column-plan response mirroring what a local model would
    return for the surviving (non-identity) columns of `_messy_df()`."""
    return [
        {
            "raw_name": "Campaign",
            "canonical_name": "campaign",
            "role": "group",
            "unit_token": None,
            "is_identity": False,
            "note": "campaign grouping id",
        },
        {
            "raw_name": "Titer (g/L)",
            "canonical_name": "titer",
            "role": "target",
            "unit_token": "g/L",
            "is_identity": False,
            "note": "product titer",
        },
        {
            "raw_name": "Temp",
            "canonical_name": "temperature_c",
            "role": "feature",
            "unit_token": "C",
            "is_identity": False,
            "note": "culture temperature",
        },
    ]


class _FakeHTTPResponse:
    """A context-manager stand-in for `http.client.HTTPResponse`, as
    returned by `urllib.request.urlopen`."""

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
    generate_body: dict[str, Any] | None = None,
    raise_on: str | None = None,
):
    """Build a fake `urllib.request.urlopen` that distinguishes the health
    check (a bare string URL -> GET `/api/tags`) from the generate call (a
    `urllib.request.Request` -> POST `/api/generate`) by argument type,
    exactly the way `_check_ollama_server`/`_ollama_generate` call it.

    `raise_on` is `"health"` or `"generate"` to simulate a timeout/connection
    failure at that specific call. `calls` (an attribute on the returned
    function) records which endpoint was hit, in order, for assertions.
    """
    calls: list[str] = []

    def _urlopen(url_or_request: Any, timeout: float | None = None) -> _FakeHTTPResponse:
        import urllib.request

        if isinstance(url_or_request, urllib.request.Request):
            calls.append("generate")
            if raise_on == "generate":
                raise TimeoutError("simulated generate timeout")
            body = json.dumps({"response": json.dumps(generate_body or {})}).encode("utf-8")
            return _FakeHTTPResponse(generate_status, body)
        calls.append("health")
        if raise_on == "health":
            raise TimeoutError("simulated health-check timeout")
        return _FakeHTTPResponse(health_status, b"{}")

    _urlopen.calls = calls  # type: ignore[attr-defined]
    return _urlopen


# --- provider selection (config.py) -------------------------------------------- #


def test_default_provider_is_anthropic(monkeypatch):
    monkeypatch.delenv("KALOS_LLM_PROVIDER", raising=False)
    config = load_config()
    assert config.provider == "anthropic"


def test_provider_none_forces_enabled_live_false_even_with_api_key(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "none")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-not-real")
    config = load_config()
    assert config.provider == "none"
    assert config.enabled_live is False


def test_provider_ollama_enabled_live_true_without_any_credential(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = load_config()
    assert config.provider == "ollama"
    assert config.enabled_live is True


def test_unrecognized_provider_falls_back_to_anthropic_default(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "not-a-real-provider")
    config = load_config()
    assert config.provider == "anthropic"


def test_ollama_url_and_model_env_overrides(monkeypatch):
    monkeypatch.setenv("KALOS_OLLAMA_URL", "http://example-host:9999")
    monkeypatch.setenv("KALOS_OLLAMA_MODEL", "mistral")
    config = load_config()
    assert config.ollama_url == "http://example-host:9999"
    assert config.ollama_model == "mistral"


def test_ollama_url_and_model_have_sensible_defaults(monkeypatch):
    monkeypatch.delenv("KALOS_OLLAMA_URL", raising=False)
    monkeypatch.delenv("KALOS_OLLAMA_MODEL", raising=False)
    config = load_config()
    assert config.ollama_url == "http://localhost:11434"
    assert config.ollama_model  # non-empty default


def test_provider_none_never_touches_the_network(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "none")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-not-real")

    import urllib.request

    def _boom(*args, **kwargs):
        raise AssertionError("must never call urlopen when provider='none'")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)

    plan = propose_plan(_messy_df())
    assert plan.created_by == "offline"


# --- happy path ------------------------------------------------------------------ #


def test_ollama_happy_path_mapping(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("KALOS_OLLAMA_MODEL", "llama3.1")

    import urllib.request

    fake = _fake_urlopen(generate_body={"columns": _canned_ollama_columns()})
    monkeypatch.setattr(urllib.request, "urlopen", fake)

    config = load_config()
    plan = propose_plan(_messy_df(), config=config)

    assert plan.created_by == "llm"
    assert plan.model == "llama3.1"
    plan.validate()

    by_raw = {c.raw_name: c for c in plan.columns}
    # Identity columns merged back in from the deterministic pre-screen,
    # never sent to (or returned by) the model - same guarantee the
    # Anthropic path gives.
    assert by_raw["Client Sample Name"].role == "identity"
    assert by_raw["Client Sample Name"].canonical_name is None
    assert by_raw["Operator"].role == "identity"
    assert by_raw["Titer (g/L)"].canonical_name == "titer"
    assert by_raw["Titer (g/L)"].role == "target"
    assert by_raw["Temp"].to_base is True
    assert by_raw["Temp"].unit_token == "C"
    assert fake.calls == ["health", "generate"]  # health-check before any payload


def test_ollama_prompt_never_contains_raw_identity_values(monkeypatch):
    # Same privacy guarantee test_normalize_llm.py asserts for the Anthropic
    # path: the payload sent to ANY live provider must already be screened.
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")

    import urllib.request

    sent_prompts: list[str] = []

    def _urlopen(url_or_request, timeout=None):
        if isinstance(url_or_request, urllib.request.Request):
            sent_prompts.append(url_or_request.data.decode("utf-8"))
            body = json.dumps(
                {"response": json.dumps({"columns": _canned_ollama_columns()})}
            ).encode("utf-8")
            return _FakeHTTPResponse(200, body)
        return _FakeHTTPResponse(200, b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    propose_plan(_messy_df(), config=load_config())
    assert len(sent_prompts) == 1
    for raw_value in ("Acme-BF-018", "J. Rivera", "M. Chen"):
        assert raw_value not in sent_prompts[0]
    assert "Client Sample Name" not in sent_prompts[0]
    assert "Operator" not in sent_prompts[0]


# --- fallback: malformed / schema-invalid JSON ------------------------------------ #


def test_ollama_unparsable_response_falls_back_to_offline(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")

    import urllib.request

    def _urlopen(url_or_request, timeout=None):
        if isinstance(url_or_request, urllib.request.Request):
            # The model wrapped its answer in prose instead of pure JSON,
            # and there is no JSON object anywhere in the response.
            body = json.dumps({"response": "not json at all, just prose"}).encode("utf-8")
            return _FakeHTTPResponse(200, body)
        return _FakeHTTPResponse(200, b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    plan = propose_plan(_messy_df(), config=load_config())  # must not raise
    assert plan.created_by == "offline"
    plan.validate()


def test_ollama_schema_invalid_json_falls_back_to_offline(monkeypatch):
    # Valid JSON, but missing the required "columns" key entirely - must
    # fail the SAME pydantic validation the Anthropic path applies, not be
    # accepted as-is.
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")

    import urllib.request

    def _urlopen(url_or_request, timeout=None):
        if isinstance(url_or_request, urllib.request.Request):
            body = json.dumps({"response": json.dumps({"not_columns": []})}).encode("utf-8")
            return _FakeHTTPResponse(200, body)
        return _FakeHTTPResponse(200, b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)

    plan = propose_plan(_messy_df(), config=load_config())
    assert plan.created_by == "offline"


def test_ollama_non_200_generate_status_falls_back_to_offline(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")

    import urllib.request

    fake = _fake_urlopen(generate_status=500, generate_body={"columns": []})
    monkeypatch.setattr(urllib.request, "urlopen", fake)

    plan = propose_plan(_messy_df(), config=load_config())
    assert plan.created_by == "offline"


# --- fallback: timeout / unreachable server --------------------------------------- #


def test_ollama_health_check_timeout_falls_back_to_offline_without_posting(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")

    import urllib.request

    fake = _fake_urlopen(raise_on="health")
    monkeypatch.setattr(urllib.request, "urlopen", fake)

    plan = propose_plan(_messy_df(), config=load_config())
    assert plan.created_by == "offline"
    # An unreachable server must never be POSTed to - the health-check gate
    # is the whole point of the ported pattern.
    assert fake.calls == ["health"]


def test_ollama_generate_timeout_falls_back_to_offline(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")

    import urllib.request

    fake = _fake_urlopen(raise_on="generate")
    monkeypatch.setattr(urllib.request, "urlopen", fake)

    plan = propose_plan(_messy_df(), config=load_config())
    assert plan.created_by == "offline"
    assert fake.calls == ["health", "generate"]


# --- anthropic-path default is unaffected ----------------------------------------- #


def test_default_no_provider_env_set_still_falls_back_offline_without_key(monkeypatch):
    # Sanity check that the default (no KALOS_LLM_PROVIDER set) path's
    # offline behavior - fully covered by tests/test_normalize_llm.py, left
    # untouched by this change - still holds after this module's refactor
    # into shared provider helpers.
    monkeypatch.delenv("KALOS_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    plan = propose_plan(_messy_df())
    assert plan.created_by == "offline"
    plan.validate()
