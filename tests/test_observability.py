"""Portal observability (kalos/portal/observability.py, docs/HARDENING.md Phase 2):
liveness/readiness endpoints, request-id + structured access logging, and a
text /metrics feed."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402
from kalos.portal.experiments import get_store  # noqa: E402
from kalos.store import SqliteStore  # noqa: E402


@pytest.fixture
def client(tmp_path):
    # Point both stores at tmp DBs so readiness never touches the real ~/.kalos.
    app.dependency_overrides[get_campaign_store] = lambda: CampaignStore(tmp_path)
    app.dependency_overrides[get_store] = lambda: SqliteStore(tmp_path / "exp.db")
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        app.dependency_overrides.pop(get_store, None)


# --- liveness / readiness --------------------------------------------------- #

def test_healthz_is_ok_and_needs_no_auth(client, monkeypatch):
    monkeypatch.setenv("KALOS_AUTH_TOKENS", '[]')  # even "auth configured", health is open
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readyz_ok_when_stores_reachable(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["checks"] == {"portal_db": True, "experiments_db": True}


def test_readyz_503_when_a_store_is_down(tmp_path):
    class _BrokenStore:
        def ping(self) -> None:
            raise RuntimeError("db unreachable")

    app.dependency_overrides[get_campaign_store] = lambda: CampaignStore(tmp_path)
    app.dependency_overrides[get_store] = lambda: _BrokenStore()
    try:
        r = TestClient(app).get("/readyz")
        assert r.status_code == 503
        body = r.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["experiments_db"] is False
        assert body["checks"]["portal_db"] is True
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        app.dependency_overrides.pop(get_store, None)


# --- request id + access log ------------------------------------------------ #

def test_every_response_carries_a_request_id(client):
    r = client.get("/healthz")
    assert r.headers.get("x-request-id")


def test_inbound_request_id_is_propagated(client):
    r = client.get("/healthz", headers={"X-Request-ID": "trace-abc"})
    assert r.headers.get("x-request-id") == "trace-abc"


def test_access_log_line_carries_no_secrets(client, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="kalos.portal.access"):
        client.get("/api/latest", headers={"Authorization": "Bearer super-secret-token"})
    joined = " ".join(r.message for r in caplog.records)
    assert "/api/latest" in joined  # the path IS logged
    assert "super-secret-token" not in joined  # the token is NEVER logged
    assert "Authorization" not in joined


# --- metrics ---------------------------------------------------------------- #

def test_metrics_is_prometheus_text_and_counts_requests(client):
    client.get("/healthz")
    client.get("/api/latest")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    text = r.text
    assert "kalos_uptime_seconds" in text
    assert "kalos_requests_total" in text
    assert 'kalos_requests_by_class{class="2xx"}' in text
