"""External-provider / API-key seam (kalos/providers/): every provider is a
SLOT - a name, a lazy credential-presence check, and an honest status. Covers:
clean-env unavailability, availability once required vars are set, exact
missing_env lists, that a key VALUE never appears in status() output, that
provider_status() is JSON-safe, the GET /api/providers route, and that a key
set AFTER import is still picked up (no import-time caching).
"""
from __future__ import annotations

import json

import pytest

from kalos.providers import AnthropicProvider, BenchlingProvider, BioNemoProvider
from kalos.providers.registry import all_providers, get, provider_status

_ALL_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "NVIDIA_API_KEY",
    "HF_TOKEN",
    "BENCHLING_API_KEY",
    "BENCHLING_TENANT",
)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    """Every test starts with none of the provider env vars set, regardless
    of what the ambient shell/CI environment carries."""
    for var in _ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# --- clean env: nothing available ------------------------------------------- #

def test_all_providers_unavailable_with_clean_env():
    for provider in all_providers():
        assert provider.available() is False
        assert provider.status().available is False


# --- anthropic ---------------------------------------------------------------- #

def test_anthropic_available_once_key_set(monkeypatch):
    provider = AnthropicProvider()
    assert provider.available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key")
    assert provider.available() is True
    status = provider.status()
    assert status.available is True
    assert status.missing_env == ()
    assert status.required_env == ("ANTHROPIC_API_KEY",)


def test_anthropic_missing_env_lists_exactly_the_absent_name():
    status = AnthropicProvider().status()
    assert status.missing_env == ("ANTHROPIC_API_KEY",)


# --- bionemo ------------------------------------------------------------------- #

def test_bionemo_requires_only_nvidia_api_key(monkeypatch):
    provider = BioNemoProvider()
    assert provider.available() is False
    # HF_TOKEN alone must not make it available - NVIDIA_API_KEY is required.
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token")
    assert provider.available() is False
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-fake-key")
    assert provider.available() is True


def test_bionemo_status_is_honest_about_being_a_slot():
    status = BioNemoProvider().status()
    assert status.missing_env == ("NVIDIA_API_KEY",)
    # honest framing: the status text must say this is a slot, not imply a
    # working BioNeMo client exists.
    assert "slot" in status.capability.lower()
    assert status.fallback  # a concrete keyless fallback is always stated


# --- benchling ------------------------------------------------------------------ #

def test_benchling_requires_both_key_and_tenant(monkeypatch):
    provider = BenchlingProvider()
    assert provider.available() is False
    monkeypatch.setenv("BENCHLING_API_KEY", "fake-key")
    assert provider.available() is False  # tenant still missing
    monkeypatch.setenv("BENCHLING_TENANT", "acme")
    assert provider.available() is True


def test_benchling_missing_env_lists_exactly_the_absent_names(monkeypatch):
    status = BenchlingProvider().status()
    assert status.missing_env == ("BENCHLING_API_KEY", "BENCHLING_TENANT")

    monkeypatch.setenv("BENCHLING_API_KEY", "fake-key")
    status = BenchlingProvider().status()
    assert status.missing_env == ("BENCHLING_TENANT",)


# --- no key VALUE ever appears in status() output ----------------------------- #

def test_status_never_leaks_a_credential_value(monkeypatch):
    secret = "sk-ant-super-secret-do-not-leak-12345"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-another-secret-98765")
    monkeypatch.setenv("BENCHLING_API_KEY", "benchling-secret-abcde")
    monkeypatch.setenv("BENCHLING_TENANT", "acme")

    serialized = json.dumps(provider_status())
    assert secret not in serialized
    assert "nvapi-another-secret-98765" not in serialized
    assert "benchling-secret-abcde" not in serialized


# --- registry ------------------------------------------------------------------- #

def test_all_providers_returns_the_three_known_providers():
    names = {p.name for p in all_providers()}
    assert names == {"anthropic", "bionemo", "benchling"}


def test_get_returns_provider_by_name_or_none():
    assert get("anthropic") is not None
    assert get("bionemo") is not None
    assert get("benchling") is not None
    assert get("does-not-exist") is None


def test_provider_status_is_json_dumps_able():
    payload = provider_status()
    assert isinstance(payload, list)
    assert len(payload) == 3
    # must not raise
    text = json.dumps(payload)
    data = json.loads(text)
    assert {entry["name"] for entry in data} == {"anthropic", "bionemo", "benchling"}
    for entry in data:
        assert set(entry) == {
            "name", "available", "required_env", "missing_env", "capability", "fallback",
        }


# --- no import-time caching: a key set AFTER import is picked up ------------- #

def test_key_set_after_import_is_picked_up(monkeypatch):
    # kalos.providers is already imported at module scope above; setting the
    # env var now (well after import) must still flip availability - proves
    # available() reads os.environ lazily rather than caching at import time.
    provider = AnthropicProvider()
    assert provider.available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-set-after-import")
    assert provider.available() is True
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert provider.available() is False


# --- GET /api/providers -------------------------------------------------------- #

def test_providers_route_returns_all_three(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalos.portal.app import app

    secret = "sk-ant-route-should-never-leak-this-99999"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    client = TestClient(app)
    resp = client.get("/api/providers")
    assert resp.status_code == 200
    body = resp.json()
    assert "providers" in body
    names = {entry["name"] for entry in body["providers"]}
    assert names == {"anthropic", "bionemo", "benchling"}
    anthropic_entry = next(e for e in body["providers"] if e["name"] == "anthropic")
    assert anthropic_entry["available"] is True
    # a live key must never leak into the response, even though it's set
    assert secret not in resp.text
