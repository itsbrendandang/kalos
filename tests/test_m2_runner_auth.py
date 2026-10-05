"""A remote Singleton runner (`HttpBackendAdapter`) against a portal that
enforces auth (docs/HARDENING.md, Phase 1 + P1).

Since P1 the `/api/experiments*` routes require `read`/`write` scope, so once
`KALOS_AUTH_TOKENS` is provisioned - any real deployment - a runner that sends
no credential is 401'd on its very first poll. And even with a plain
read+write token the client allowlist on `PATCH` 409'd the runner's own
`READY -> PROCESSING` claim, and `/result` only ever looked on the `default`
tenant. The runner now authenticates as a provisioned principal with the
`runner` scope (`KALOS_RUNNER_API_TOKEN`), which unlocks exactly the runner
transitions and the result push, on that principal's tenant.

Every test overrides `get_store`/`get_lock_path` to `tmp_path`, and drives the
real FastAPI app through `TestClient` - no network I/O, no `~/.kalos` writes.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal.app import app, get_lock_path, get_store  # noqa: E402
from kalos.runner.adapter import HttpBackendAdapter  # noqa: E402
from kalos.runner.singleton import run_ready  # noqa: E402
from kalos.store import SqliteStore, Status  # noqa: E402

_USER = "tok-acme-user"            # a client on acme: read+write, no runner scope
_RUNNER = "tok-acme-runner"        # the runner on acme, least privilege: read+runner, no write
_GLOBEX_RUNNER = "tok-globex-runner"
_READER = "tok-acme-reader"        # read only
_LEGACY = "legacy-runner-token"    # the pre-P1 KALOS_RUNNER_TOKEN machine channel


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps([
        {"token_sha256": _sha(_USER), "subject": "alice", "tenant": "acme",
         "scopes": ["read", "write"]},
        {"token_sha256": _sha(_RUNNER), "subject": "runner", "tenant": "acme",
         "scopes": ["read", "runner"]},
        {"token_sha256": _sha(_GLOBEX_RUNNER), "subject": "runner", "tenant": "globex",
         "scopes": ["read", "runner"]},
        {"token_sha256": _sha(_READER), "subject": "auditor", "tenant": "acme",
         "scopes": ["read"]},
    ]))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.delenv("KALOS_RUNNER_TOKEN", raising=False)
    store = SqliteStore(tmp_path / "experiments.db")
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_lock_path] = lambda: tmp_path / "portal-runner.lock"
    try:
        yield store
    finally:
        app.dependency_overrides.pop(get_store, None)
        app.dependency_overrides.pop(get_lock_path, None)


@pytest.fixture
def client(store):
    return TestClient(app)


class _PortalError(Exception):
    """A non-2xx portal response, raised the way urllib's `HTTPError` would be."""

    def __init__(self, status: int, text: str) -> None:
        super().__init__(f"HTTP {status}: {text}")
        self.status = status


class _RecordingTransport:
    """`HttpBackendAdapter` transport over `TestClient`: the real FastAPI
    routes, auth and all, with every request's method + headers recorded."""

    def __init__(self, client: TestClient) -> None:
        self._client = client
        self.sent: list[tuple[str, dict[str, str] | None]] = []

    def __call__(self, method: str, url: str, json_body: Any, headers: dict[str, str] | None = None) -> Any:
        self.sent.append((method, headers))
        resp = self._client.request(method, url, json=json_body, headers=headers)
        if resp.status_code >= 400:
            raise _PortalError(resp.status_code, resp.text)
        return resp.json() if resp.content else None


def _good_sheet(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Mirrors tests/test_m2_portal.py::_good_sheet - a small, honest, varying
    run sheet that `_analyze` can actually fit."""
    rng = np.random.default_rng(seed)
    methanol = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    titer = 1.5 * methanol - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "medium": rng.choice(["A", "B", "C"], n),
        "Methanol": methanol.round(3),
        "pH": ph.round(2),
        "lipase_titer": titer.round(3),
    })


def _create_ready_as_user(client: TestClient) -> str:
    """The front end's half: alice uploads a run sheet and flips it READY."""
    resp = client.post(
        "/api/experiments",
        headers=_bearer(_USER),
        files={"file": ("runs.csv", _good_sheet().to_csv(index=False).encode(), "text/csv")},
        data={"name": "acme run", "target": "lipase_titer"},
    )
    assert resp.status_code == 201, resp.text
    exp_id: str = resp.json()["id"]
    patch = client.patch(f"/api/experiments/{exp_id}", headers=_bearer(_USER), json={"status": "READY"})
    assert patch.status_code == 200, patch.text
    return exp_id


_PAYLOAD = {"columns": ["Methanol", "lipase_titer"], "rows": [{"Methanol": 1.0, "lipase_titer": 0.5}]}


def _seed(store: SqliteStore, status: Status, *, tenant: str = "acme") -> str:
    exp = store.create("seeded", _PAYLOAD, {"target": "lipase_titer"}, tenant=tenant)
    for step in (Status.READY, Status.PROCESSING):
        if exp.status == status:
            break
        exp = store.set_status(exp.id, step, tenant=tenant)
    assert exp.status == status
    return exp.id


# --- the round trip ---------------------------------------------------------- #

def test_runner_round_trip_over_http_with_auth_enforced(client, store, tmp_path):
    """The whole remote loop on a non-default tenant: list READY -> fetch ->
    claim PROCESSING -> analyze locally -> push the result -> DONE, every
    request carrying the runner's API token."""
    exp_id = _create_ready_as_user(client)
    transport = _RecordingTransport(client)
    adapter = HttpBackendAdapter("http://testserver", api_token=_RUNNER, transport=transport)

    results = run_ready(adapter, lock_path=tmp_path / "runner.lock")

    assert [(r.id, r.status, r.error) for r in results] == [(exp_id, "DONE", None)]
    done = client.get(f"/api/experiments/{exp_id}", headers=_bearer(_USER)).json()
    assert done["status"] == "DONE"
    assert done["result"] is not None
    assert done["provenance"]["processed_at"]
    assert [method for method, _ in transport.sent] == ["GET", "GET", "PATCH", "POST"]
    assert all(headers == _bearer(_RUNNER) for _, headers in transport.sent)


def test_runner_without_api_token_is_rejected_on_its_first_poll(client, store, tmp_path):
    """The original bug, pinned: no credential -> 401 on `list_ready`, before
    the runner touches any experiment."""
    exp_id = _create_ready_as_user(client)
    adapter = HttpBackendAdapter(
        "http://testserver", token=_LEGACY, transport=_RecordingTransport(client)
    )

    with pytest.raises(_PortalError) as err:
        run_ready(adapter, lock_path=tmp_path / "runner.lock")

    assert err.value.status == 401
    assert store.get(exp_id, tenant="acme").status == Status.READY


def test_a_client_token_cannot_act_as_the_runner(client, store, tmp_path):
    """A read+write client token can list and fetch, but the claim is a runner
    transition: 409, and the experiment is left READY for a real runner."""
    exp_id = _create_ready_as_user(client)
    adapter = HttpBackendAdapter("http://testserver", api_token=_USER, transport=_RecordingTransport(client))

    results = run_ready(adapter, lock_path=tmp_path / "runner.lock")

    assert [(r.id, r.status) for r in results] == [(exp_id, "FAILED")]
    assert "HTTP 409" in (results[0].error or "")
    assert store.get(exp_id, tenant="acme").status == Status.READY


# --- PATCH: the runner transitions ------------------------------------------- #

def test_runner_principal_may_claim_and_fail_an_experiment(client, store):
    exp_id = _seed(store, Status.READY)

    claim = client.patch(f"/api/experiments/{exp_id}", headers=_bearer(_RUNNER), json={"status": "PROCESSING"})
    assert claim.status_code == 200, claim.text
    fail = client.patch(
        f"/api/experiments/{exp_id}", headers=_bearer(_RUNNER), json={"status": "FAILED", "error": "boom"}
    )
    assert fail.status_code == 200, fail.text

    failed = store.get(exp_id, tenant="acme")
    assert failed.status == Status.FAILED and failed.error == "boom"


def test_runner_principal_still_cannot_patch_to_done(client, store):
    """DONE is only ever reached through `/result`, which carries the result -
    the `runner` scope does not reopen the empty-result hole."""
    exp_id = _seed(store, Status.PROCESSING)

    resp = client.patch(f"/api/experiments/{exp_id}", headers=_bearer(_RUNNER), json={"status": "DONE"})

    assert resp.status_code == 409
    assert store.get(exp_id, tenant="acme").status == Status.PROCESSING


def test_runner_transitions_still_obey_the_lifecycle(client, store):
    """The scope widens the allowlist, not `legal_transition`: DRAFT cannot
    jump straight to PROCESSING, and READY cannot be FAILED."""
    draft = _seed(store, Status.DRAFT)
    ready = _seed(store, Status.READY)

    assert client.patch(
        f"/api/experiments/{draft}", headers=_bearer(_RUNNER), json={"status": "PROCESSING"}
    ).status_code == 409
    assert client.patch(
        f"/api/experiments/{ready}", headers=_bearer(_RUNNER), json={"status": "FAILED"}
    ).status_code == 409
    assert store.get(draft, tenant="acme").status == Status.DRAFT
    assert store.get(ready, tenant="acme").status == Status.READY


def test_client_principal_cannot_claim_or_fail_with_auth_enforced(client, store):
    ready = _seed(store, Status.READY)
    processing = _seed(store, Status.PROCESSING)

    claim = client.patch(f"/api/experiments/{ready}", headers=_bearer(_USER), json={"status": "PROCESSING"})
    fail = client.patch(f"/api/experiments/{processing}", headers=_bearer(_USER), json={"status": "FAILED"})

    assert claim.status_code == 409 and "not directly settable" in claim.json()["detail"]
    assert fail.status_code == 409 and "not directly settable" in fail.json()["detail"]


def test_runner_principal_holds_no_client_powers(client, store):
    """Least privilege: `read` + `runner` is everything a polling runner needs,
    and nothing a client has - no upload, no READY flip."""
    draft = _seed(store, Status.DRAFT)

    flip = client.patch(f"/api/experiments/{draft}", headers=_bearer(_RUNNER), json={"status": "READY"})
    upload = client.post(
        "/api/experiments",
        headers=_bearer(_RUNNER),
        files={"file": ("runs.csv", _good_sheet().to_csv(index=False).encode(), "text/csv")},
        data={"name": "nope"},
    )

    assert flip.status_code == 403
    assert upload.status_code == 403
    assert store.get(draft, tenant="acme").status == Status.DRAFT


def test_read_only_principal_cannot_patch_anything(client, store):
    ready = _seed(store, Status.READY)
    for target in ("READY", "PROCESSING", "FAILED", "DONE"):
        resp = client.patch(f"/api/experiments/{ready}", headers=_bearer(_READER), json={"status": target})
        assert resp.status_code == 403, target
    assert store.get(ready, tenant="acme").status == Status.READY


def test_runner_on_another_tenant_cannot_claim(client, store):
    exp_id = _seed(store, Status.READY)

    resp = client.patch(
        f"/api/experiments/{exp_id}", headers=_bearer(_GLOBEX_RUNNER), json={"status": "PROCESSING"}
    )

    assert resp.status_code == 404  # a cross-tenant id reads as not-found
    assert store.get(exp_id, tenant="acme").status == Status.READY


# --- /result: tenant-aware push ---------------------------------------------- #

def _push(client: TestClient, exp_id: str, headers: dict[str, str] | None = None):
    return client.post(
        f"/api/experiments/{exp_id}/result",
        headers=headers,
        json={"result": {"n": 1}, "provenance": {"seed": 1}},
    )


def test_runner_principal_pushes_a_result_on_its_own_tenant(client, store):
    exp_id = _seed(store, Status.PROCESSING)

    resp = _push(client, exp_id, _bearer(_RUNNER))

    assert resp.status_code == 200, resp.text
    done = store.get(exp_id, tenant="acme")
    assert done.status == Status.DONE and done.result == {"n": 1}


def test_result_push_needs_the_runner_scope(client, store):
    """A plain client token must never be able to fabricate a result."""
    exp_id = _seed(store, Status.PROCESSING)

    resp = _push(client, exp_id, _bearer(_USER))

    assert resp.status_code == 403
    assert store.get(exp_id, tenant="acme").result is None


def test_result_push_without_a_bearer_is_401_when_auth_enforces(client, store):
    exp_id = _seed(store, Status.PROCESSING)

    assert _push(client, exp_id).status_code == 401
    assert _push(client, exp_id, _bearer("not-a-real-token")).status_code == 401
    assert store.get(exp_id, tenant="acme").result is None


def test_result_push_from_another_tenants_runner_reads_as_not_found(client, store):
    exp_id = _seed(store, Status.PROCESSING)

    resp = _push(client, exp_id, _bearer(_GLOBEX_RUNNER))

    assert resp.status_code == 404
    assert store.get(exp_id, tenant="acme").result is None


def test_legacy_runner_token_still_reaches_only_the_default_tenant(client, store, monkeypatch):
    """`KALOS_RUNNER_TOKEN` keeps working alongside provisioned tokens, but it
    carries no identity, so it is confined to the `default` tenant."""
    monkeypatch.setenv("KALOS_RUNNER_TOKEN", _LEGACY)
    default_exp = _seed(store, Status.PROCESSING, tenant="default")
    acme_exp = _seed(store, Status.PROCESSING)

    assert _push(client, default_exp, _bearer(_LEGACY)).status_code == 200
    assert _push(client, acme_exp, _bearer(_LEGACY)).status_code == 404

    assert store.get(default_exp, tenant="default").status == Status.DONE
    assert store.get(acme_exp, tenant="acme").result is None


def test_a_non_ascii_bearer_is_a_401_not_a_500(client, store, monkeypatch):
    """`hmac.compare_digest` raises TypeError on non-ASCII `str` input; the
    legacy check compares bytes so a crafted header is just a wrong token."""
    monkeypatch.setenv("KALOS_RUNNER_TOKEN", _LEGACY)
    exp_id = _seed(store, Status.PROCESSING, tenant="default")

    resp = client.post(
        f"/api/experiments/{exp_id}/result",
        headers={"Authorization": "Bearer été".encode("latin-1")},
        json={"result": {"n": 1}, "provenance": {"seed": 1}},
    )

    assert resp.status_code == 401
    assert store.get(exp_id, tenant="default").result is None
