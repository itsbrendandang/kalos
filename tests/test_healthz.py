"""GET /healthz and GET /readyz (deploy/Dockerfile.engine's HEALTHCHECK target).

Before this, the only unauthenticated route the deploy healthcheck could hit
was `/`, which renders the full portal page - heavier than a healthcheck
needs, and coupled to the UI. `/healthz` is a liveness probe (process is up,
never touches the database); `/readyz` is a readiness probe (the campaign
store is actually reachable).

Every test here uses a `tmp_path` `CampaignStore` via `app.dependency_overrides`,
per tests/test_tenant_isolation.py's convention - the real `~/.kalos/portal.db`
is never touched.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal.app import app, get_campaign_store  # noqa: E402
from kalos.portal.campaign import CampaignStore  # noqa: E402


@pytest.fixture
def client(tmp_path):
    store = CampaignStore(tmp_path)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch):
    """Every test starts from genuinely open, matching
    tests/test_open_access_guard.py's fixture - no leftover auth config from
    another test's monkeypatch."""
    for var in ("KALOS_AUTH_TOKENS", "KALOS_AUTH_TOKENS_FILE", "KALOS_ALLOW_OPEN_ACCESS"):
        monkeypatch.delenv(var, raising=False)


# --- /healthz ----------------------------------------------------------- #


def test_healthz_returns_exactly_the_documented_body(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_healthz_needs_no_auth_token_even_when_auth_is_enforced(client, monkeypatch):
    """Unauthenticated BY DESIGN: provisioning real tokens (which makes every
    scoped route below start requiring one) must not affect /healthz."""
    monkeypatch.setenv(
        "KALOS_AUTH_TOKENS",
        '[{"subject": "acme", "tenant": "acme", "scopes": ["read"], '
        '"token_sha256": "' + "a" * 64 + '"}]',
    )
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_auth_required_routes_still_401_without_a_token_once_enforced(client, monkeypatch):
    """The control case for the test above: /healthz being open must not mean
    auth stopped being enforced everywhere else."""
    monkeypatch.setenv(
        "KALOS_AUTH_TOKENS",
        '[{"subject": "acme", "tenant": "acme", "scopes": ["read"], '
        '"token_sha256": "' + "a" * 64 + '"}]',
    )
    resp = client.get("/api/latest")
    assert resp.status_code == 401


def test_healthz_does_not_touch_the_database(client, monkeypatch):
    """Liveness must survive a store that would raise on any real query -
    proves /healthz never calls into the store at all."""

    def _boom(*args, **kwargs):
        raise AssertionError("healthz must never touch the store")

    monkeypatch.setattr(CampaignStore, "get", _boom)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- /readyz -------------------------------------------------------------- #


def test_readyz_is_ok_on_a_working_store(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_degrades_to_503_with_a_reason_when_the_store_is_broken(client, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("database disk image is malformed")

    monkeypatch.setattr(CampaignStore, "get", _boom)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "unavailable"
    assert "malformed" in body["reason"]


def test_readyz_performs_a_read_not_a_write(client):
    """Readiness must not mutate the store it is checking - no campaign
    should appear as a side effect of polling /readyz."""
    store = app.dependency_overrides[get_campaign_store]()
    assert store.get() is None  # nothing seeded yet
    client.get("/readyz")
    assert store.get() is None  # /readyz did not seed one
