"""Per-tenant persistence (docs/HARDENING.md, Phase 1b): two tenants can never
see or overwrite each other's campaign or latest analysis - at the store level
and end-to-end through the authenticated portal."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from kalos.portal.campaign import CampaignStore, get_campaign_store


def _df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, 8)
    ph = rng.uniform(5, 7, 8)
    return pd.DataFrame({"Methanol": m.round(3), "pH": ph.round(2),
                         "lipase_titer": (1.5 * m + rng.normal(0, 0.1, 8)).round(3)})


# --- store level ------------------------------------------------------------ #

def test_campaigns_are_isolated_per_tenant(tmp_path):
    store = CampaignStore(tmp_path)
    store.seed(_df(1), "lipase_titer", ["Methanol", "pH"], tenant="acme")
    # globex has no campaign even though acme does
    assert store.summary(tenant="acme")["has_campaign"] is True
    assert store.summary(tenant="globex")["has_campaign"] is False

    store.seed(_df(2), "lipase_titer", ["Methanol", "pH"], tenant="globex")
    # a write to one tenant never leaks into the other
    store.start([{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 1.0, "std": 0.1,
                  "mode": "explore", "reason": "x"}], tenant="acme")
    assert store.summary(tenant="acme")["n_awaiting"] == 1
    assert store.summary(tenant="globex")["n_awaiting"] == 0


def test_reseeding_one_tenant_does_not_touch_another(tmp_path):
    store = CampaignStore(tmp_path)
    store.seed(_df(1), "lipase_titer", ["Methanol", "pH"], tenant="acme")
    started = store.start([{"recipe": {"Methanol": 1.0, "pH": 5.5}, "pred": 1.0, "std": 0.1,
                            "mode": "explore", "reason": "x"}], tenant="acme")
    store.set_result(started[0]["id"], 3.3, tenant="acme")

    store.seed(_df(9), "lipase_titer", ["Methanol", "pH"], tenant="globex")  # fresh globex

    acme = store.get(tenant="acme")
    assert acme is not None and acme["pending"][0]["result"] == 3.3  # acme's measured run intact
    assert store.get(tenant="globex")["pending"] == []


# --- /api/latest per tenant ------------------------------------------------- #

def test_latest_is_isolated_per_tenant(tmp_path, monkeypatch):
    from kalos.portal import app as portal

    monkeypatch.setattr(portal, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(portal, "_LATEST", {})

    portal._save_latest({"target": "a"}, "acme.csv", tenant="acme")
    portal._save_latest({"target": "b"}, "globex.csv", tenant="globex")

    assert portal._load_latest("acme")["dataset"] == "acme.csv"
    assert portal._load_latest("globex")["dataset"] == "globex.csv"
    assert portal._load_latest("nobody") is None
    # a fresh process (empty in-memory cache) still reloads each tenant's own file
    monkeypatch.setattr(portal, "_LATEST", {})
    assert portal._load_latest("acme")["dataset"] == "acme.csv"


# --- HTTP end-to-end: auth tenant scopes every read/write ------------------- #

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal.app import app  # noqa: E402


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps([
        {"token_sha256": _sha("tok-acme"), "subject": "a", "tenant": "acme", "scopes": ["read", "write"]},
        {"token_sha256": _sha("tok-globex"), "subject": "g", "tenant": "globex", "scopes": ["read", "write"]},
    ]))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def test_one_tenants_campaign_is_invisible_to_another_over_http(client):
    tc, store = client
    store.seed(_df(1), "lipase_titer", ["Methanol", "pH"], tenant="acme")  # only acme has a campaign

    acme = tc.get("/api/campaign", headers={"Authorization": "Bearer tok-acme"}).json()
    globex = tc.get("/api/campaign", headers={"Authorization": "Bearer tok-globex"}).json()
    assert acme["has_campaign"] is True
    assert globex["has_campaign"] is False

    # globex writing to /start hits ITS OWN (empty) campaign -> 400, never acme's
    body = {"recipes": [{"recipe": {"pH": 6.5}, "pred": 1.0, "std": 0.1, "mode": "explore", "reason": "x"}]}
    r = tc.post("/api/campaign/start", json=body, headers={"Authorization": "Bearer tok-globex"})
    assert r.status_code == 400
    # acme's campaign is untouched by globex's attempt
    assert store.summary(tenant="acme")["has_campaign"] is True
    assert store.summary(tenant="acme")["n_awaiting"] == 0
