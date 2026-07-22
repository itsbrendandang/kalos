"""In-house auth layer (kalos/portal/auth.py, docs/HARDENING.md Phase 1):
bearer tokens -> Principal (subject, tenant, scopes); open mode when unconfigured;
constant-time token compare; scope gating on the mutating portal endpoints.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from kalos.portal.auth import (
    ADMIN,
    READ,
    WRITE,
    AuthConfigError,
    Authenticator,
    Principal,
    _bearer,
)


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _configure(monkeypatch, records: list[dict]) -> None:
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(records))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)


# --- open mode (no tokens configured) --------------------------------------- #

def test_open_mode_returns_anonymous_readwrite_principal(monkeypatch):
    monkeypatch.delenv("KALOS_AUTH_TOKENS", raising=False)
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    auth = Authenticator()
    assert auth.is_configured() is False
    p = auth.principal_for(None)  # no header needed in open mode
    assert p.anonymous is True
    assert p.tenant == "default"
    assert p.has(READ) and p.has(WRITE)
    assert not p.has(ADMIN)  # admin is never granted without a real token


# --- configured mode: valid / invalid / missing ----------------------------- #

def test_valid_token_resolves_to_its_principal(monkeypatch):
    _configure(monkeypatch, [
        {"token_sha256": _sha("s3cret-A"), "subject": "svc-ingest", "tenant": "acme",
         "scopes": [READ, WRITE]},
    ])
    auth = Authenticator()
    assert auth.is_configured() is True
    p = auth.principal_for("Bearer s3cret-A")
    assert p.anonymous is False
    assert p.subject == "svc-ingest"
    assert p.tenant == "acme"
    assert p.has(WRITE) and not p.has(ADMIN)


def test_two_tenants_are_isolated_by_token(monkeypatch):
    _configure(monkeypatch, [
        {"token_sha256": _sha("tok-acme"), "subject": "a", "tenant": "acme", "scopes": [WRITE]},
        {"token_sha256": _sha("tok-globex"), "subject": "b", "tenant": "globex", "scopes": [READ]},
    ])
    auth = Authenticator()
    assert auth.principal_for("Bearer tok-acme").tenant == "acme"
    assert auth.principal_for("Bearer tok-globex").tenant == "globex"


@pytest.mark.parametrize("header", [None, "", "token abc", "Bearer ", "Basic xyz"])
def test_configured_mode_rejects_missing_or_malformed_bearer_with_401(monkeypatch, header):
    _configure(monkeypatch, [
        {"token_sha256": _sha("good"), "subject": "s", "tenant": "t", "scopes": [WRITE]},
    ])
    auth = Authenticator()
    with pytest.raises(Exception) as exc:
        auth.principal_for(header)
    assert getattr(exc.value, "status_code", None) == 401


def test_wrong_token_is_401(monkeypatch):
    _configure(monkeypatch, [
        {"token_sha256": _sha("right"), "subject": "s", "tenant": "t", "scopes": [WRITE]},
    ])
    auth = Authenticator()
    with pytest.raises(Exception) as exc:
        auth.principal_for("Bearer wrong")
    assert getattr(exc.value, "status_code", None) == 401


# --- file config takes precedence over inline env --------------------------- #

def test_token_file_takes_precedence(monkeypatch, tmp_path):
    f = tmp_path / "tokens.json"
    f.write_text(json.dumps([
        {"token_sha256": _sha("file-tok"), "subject": "from-file", "tenant": "ft", "scopes": [READ]},
    ]))
    monkeypatch.setenv("KALOS_AUTH_TOKENS_FILE", str(f))
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps([
        {"token_sha256": _sha("env-tok"), "subject": "from-env", "tenant": "et", "scopes": [READ]},
    ]))
    auth = Authenticator()
    assert auth.principal_for("Bearer file-tok").subject == "from-file"
    with pytest.raises(Exception):  # the env token is ignored when the file is set
        auth.principal_for("Bearer env-tok")


# --- malformed config is a clear error, never a silent open --------------- #

@pytest.mark.parametrize("bad", [
    "{not json",
    json.dumps({"not": "a list"}),
    json.dumps([{"subject": "s", "tenant": "t"}]),           # missing token_sha256
    json.dumps([{"token_sha256": "abc", "tenant": "t"}]),    # missing subject
    json.dumps([{"token_sha256": "abc", "subject": "s"}]),   # missing tenant
    json.dumps([{"token_sha256": "abc", "subject": "s", "tenant": "t", "scopes": ["superuser"]}]),
])
def test_malformed_config_raises_authconfigerror(monkeypatch, bad):
    monkeypatch.setenv("KALOS_AUTH_TOKENS", bad)
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    with pytest.raises(AuthConfigError):
        Authenticator().principal_for("Bearer whatever")


# --- helpers ---------------------------------------------------------------- #

@pytest.mark.parametrize("header,expected", [
    ("Bearer abc", "abc"),
    ("Bearer   abc  ", "abc"),
    ("bearer abc", None),   # case-sensitive scheme
    ("abc", None),
    (None, None),
    ("Bearer ", None),
])
def test_bearer_extraction(header, expected):
    assert _bearer(header) == expected


def test_principal_scope_membership():
    p = Principal(subject="s", tenant="t", scopes=frozenset({READ, WRITE}))
    assert p.has(READ) and p.has(WRITE) and not p.has(ADMIN)


# --- HTTP: the write gate on a real portal endpoint ------------------------- #

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402


@pytest.fixture
def client(tmp_path):
    # empty campaign store so start() 400s ("no campaign") once auth is passed -
    # that 400 (not 401) is exactly how we detect the request cleared the gate.
    app.dependency_overrides[get_campaign_store] = lambda: CampaignStore(tmp_path)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


_BODY = {"recipes": [{"recipe": {"pH": 6.5}, "pred": 1.0, "std": 0.1, "mode": "explore", "reason": "x"}]}


def test_open_mode_endpoint_is_reachable_without_a_token(monkeypatch, client):
    monkeypatch.delenv("KALOS_AUTH_TOKENS", raising=False)
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    r = client.post("/api/campaign/start", json=_BODY)
    assert r.status_code == 400  # past auth, then "no active campaign"
    assert r.status_code != 401


def test_configured_mode_rejects_unauthenticated_write(monkeypatch, client):
    _configure(monkeypatch, [
        {"token_sha256": _sha("wtok"), "subject": "s", "tenant": "t", "scopes": [WRITE]},
    ])
    r = client.post("/api/campaign/start", json=_BODY)
    assert r.status_code == 401


def test_configured_mode_accepts_valid_write_token(monkeypatch, client):
    _configure(monkeypatch, [
        {"token_sha256": _sha("wtok"), "subject": "s", "tenant": "t", "scopes": [WRITE]},
    ])
    r = client.post("/api/campaign/start", json=_BODY, headers={"Authorization": "Bearer wtok"})
    assert r.status_code == 400  # cleared auth, then "no active campaign"


def test_read_only_token_is_forbidden_from_writing(monkeypatch, client):
    _configure(monkeypatch, [
        {"token_sha256": _sha("rtok"), "subject": "s", "tenant": "t", "scopes": [READ]},
    ])
    r = client.post("/api/campaign/start", json=_BODY, headers={"Authorization": "Bearer rtok"})
    assert r.status_code == 403
