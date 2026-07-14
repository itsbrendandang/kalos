"""Portal /api/experiments endpoints (M2.2, docs/M2_INTEGRATION.md "Portal API
additions"): thin wrappers over the M2.1 store + Singleton runner.

Every test overrides `get_store`/`get_lock_path` to a `tmp_path` store and
lock, so the real `~/.kalos/experiments.db` and `~/.kalos/runner.lock` are
never touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal.app import app, get_lock_path, get_store  # noqa: E402
from kalos.store import SqliteStore, Status  # noqa: E402


def _good_sheet(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Mirrors tests/test_hardening.py::_good_sheet - a small, honest, varying
    run sheet that `_analyze` can actually fit and produce proposals for."""
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


def _zero_variance_sheet(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "medium": rng.choice(["A", "B"], n),
        "Methanol": rng.uniform(0, 4, n).round(3),
        "lipase_titer": np.full(n, 5.0),  # zero variance
    })


def _missing_cell_sheet(n: int = 40, seed: int = 1) -> pd.DataFrame:
    """A valid, fittable sheet with genuine missing cells (NaN) in a feature
    column - reproduces the create-response NaN serialization bug."""
    df = _good_sheet(n, seed)
    df.loc[df.index[:5], "pH"] = np.nan
    return df


def test_missing_cells_survive_the_full_loop(client):
    """Regression + round-trip: a sheet with genuine missing cells (float NaN)
    must (a) create at 201 with NaN coerced to JSON null - Starlette's
    JSONResponse encodes with allow_nan=False, so an un-sanitized NaN 500s the
    create response - and (b) run to DONE, since None round-trips back to NaN
    when the Singleton rebuilds the DataFrame."""
    r = _create(client, _missing_cell_sheet())
    assert r.status_code == 201, r.text
    exp = r.json()
    rows = exp["payload"]["rows"]
    assert rows[0]["pH"] is None                       # NaN -> JSON null
    assert any(row["pH"] is not None for row in rows)  # real values preserved
    eid = exp["id"]
    assert client.patch(f"/api/experiments/{eid}", json={"status": "READY"}).status_code == 200
    run = client.post(f"/api/experiments/{eid}/run")
    assert run.status_code == 200, run.text
    assert run.json()["status"] == "DONE"


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@pytest.fixture
def client(tmp_path):
    store = SqliteStore(tmp_path / "experiments.db")
    lock_path = tmp_path / "runner.lock"
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_lock_path] = lambda: lock_path
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_store, None)
        app.dependency_overrides.pop(get_lock_path, None)


def _create(client, df, name="run 1", target="lipase_titer", **extra_form):
    data = {"name": name, "target": target, **extra_form}
    return client.post(
        "/api/experiments",
        files={"file": ("runs.csv", _csv_bytes(df), "text/csv")},
        data=data,
    )


# --- full round trip: create -> READY -> run -> DONE with result ----------- #

def test_full_round_trip_create_ready_run_done(client):
    resp = _create(client, _good_sheet())
    assert resp.status_code == 201
    exp = resp.json()
    assert exp["status"] == "DRAFT"
    assert exp["config"]["target"] == "lipase_titer"
    exp_id = exp["id"]

    patch = client.patch(f"/api/experiments/{exp_id}", json={"status": "READY"})
    assert patch.status_code == 200
    assert patch.json()["status"] == "READY"

    run = client.post(f"/api/experiments/{exp_id}/run")
    assert run.status_code == 200
    done = run.json()
    assert done["status"] == "DONE"
    assert done["result"] is not None
    assert done["result"]["target"] == "lipase_titer"
    assert done["provenance"] is not None
    assert done["provenance"]["engine_version"]
    assert "processed_at" in done["provenance"]
    assert done["error"] is None

    get_resp = client.get(f"/api/experiments/{exp_id}")
    assert get_resp.status_code == 200
    got = get_resp.json()
    assert got["status"] == "DONE"
    assert got["result"] == done["result"]

    listing = client.get("/api/experiments")
    assert listing.status_code == 200
    rows = listing.json()
    row = next(r for r in rows if r["id"] == exp_id)
    assert set(row) == {"id", "name", "status", "updated_at"}
    assert row["status"] == "DONE"
    assert row["name"] == "run 1"


def test_create_with_outcomes_and_anonymize_config(client):
    resp = _create(
        client, _good_sheet(), target="lipase_titer",
        outcomes="lipase_titer, other_output", anonymize="true",
    )
    assert resp.status_code == 201
    config = resp.json()["config"]
    assert config["outcomes"] == ["lipase_titer", "other_output"]
    assert config["anonymize"] is True


# --- illegal PATCH transition -> 409 ---------------------------------------- #

def test_illegal_patch_transition_returns_409(client):
    resp = _create(client, _good_sheet())
    exp_id = resp.json()["id"]

    patch = client.patch(f"/api/experiments/{exp_id}", json={"status": "DONE"})
    assert patch.status_code == 409
    # untouched - still DRAFT
    assert client.get(f"/api/experiments/{exp_id}").json()["status"] == "DRAFT"


# --- FIX 1 (BLOCKER): PATCH client allowlist - only READY is client-settable #
# Before the fix, PATCH only checked `legal_transition`, so a client could walk
# DRAFT -> READY -> PROCESSING -> DONE via PATCH alone, reaching a DONE
# experiment with `result: null` and never actually running the analysis.

def test_patch_to_processing_is_rejected_409_not_run_by_client(client):
    resp = _create(client, _good_sheet())
    exp_id = resp.json()["id"]
    client.patch(f"/api/experiments/{exp_id}", json={"status": "READY"})

    patch = client.patch(f"/api/experiments/{exp_id}", json={"status": "PROCESSING"})
    assert patch.status_code == 409
    assert "not directly settable" in patch.json()["detail"]
    # untouched - still READY, never silently advanced to PROCESSING
    assert client.get(f"/api/experiments/{exp_id}").json()["status"] == "READY"


def test_client_cannot_patch_all_the_way_to_done_with_empty_result(client):
    """The full attack this fix closes: DRAFT -> READY -> PROCESSING -> DONE,
    entirely via PATCH, with no analysis ever run - `result` would stay null."""
    resp = _create(client, _good_sheet())
    exp_id = resp.json()["id"]

    assert client.patch(f"/api/experiments/{exp_id}", json={"status": "READY"}).status_code == 200
    assert client.patch(f"/api/experiments/{exp_id}", json={"status": "PROCESSING"}).status_code == 409
    assert client.patch(f"/api/experiments/{exp_id}", json={"status": "DONE"}).status_code == 409

    got = client.get(f"/api/experiments/{exp_id}").json()
    assert got["status"] == "READY"  # never advanced past the one legal client flip
    assert got["result"] is None


def test_patch_to_failed_is_rejected_409(client):
    resp = _create(client, _good_sheet())
    exp_id = resp.json()["id"]
    patch = client.patch(f"/api/experiments/{exp_id}", json={"status": "FAILED"})
    assert patch.status_code == 409
    assert "not directly settable" in patch.json()["detail"]


def test_patch_draft_to_ready_still_succeeds(client):
    resp = _create(client, _good_sheet())
    exp_id = resp.json()["id"]
    patch = client.patch(f"/api/experiments/{exp_id}", json={"status": "READY"})
    assert patch.status_code == 200
    assert patch.json()["status"] == "READY"


def test_patch_done_to_ready_requires_force(client):
    resp = _create(client, _good_sheet())
    exp_id = resp.json()["id"]
    client.patch(f"/api/experiments/{exp_id}", json={"status": "READY"})
    run = client.post(f"/api/experiments/{exp_id}/run")
    assert run.json()["status"] == "DONE"

    without_force = client.patch(f"/api/experiments/{exp_id}", json={"status": "READY"})
    assert without_force.status_code == 409
    assert client.get(f"/api/experiments/{exp_id}").json()["status"] == "DONE"

    with_force = client.patch(f"/api/experiments/{exp_id}", json={"status": "READY", "force": True})
    assert with_force.status_code == 200
    assert with_force.json()["status"] == "READY"
    assert with_force.json()["result"] is None  # discarded by the forced re-queue


def test_patch_to_ready_while_processing_is_rejected_even_though_store_allows_reclaim(client):
    """`PROCESSING -> READY` is legal at the store layer ONLY for the
    Singleton's own orphan recovery (FIX 4, `reclaim_stale`) - the PATCH
    endpoint must still reject a client trying to reach it directly, even
    with `force=true`."""
    resp = _create(client, _good_sheet())
    exp_id = resp.json()["id"]
    store = app.dependency_overrides[get_store]()
    store.set_status(exp_id, Status.READY)
    store.set_status(exp_id, Status.PROCESSING)

    patch = client.patch(f"/api/experiments/{exp_id}", json={"status": "READY", "force": True})
    assert patch.status_code == 409
    assert client.get(f"/api/experiments/{exp_id}").json()["status"] == "PROCESSING"


def test_patch_unknown_status_value_returns_400(client):
    resp = _create(client, _good_sheet())
    exp_id = resp.json()["id"]
    patch = client.patch(f"/api/experiments/{exp_id}", json={"status": "NOT_A_STATUS"})
    assert patch.status_code == 400


def test_patch_missing_experiment_returns_404(client):
    resp = client.patch("/api/experiments/exp_does_not_exist", json={"status": "READY"})
    assert resp.status_code == 404


def test_get_missing_experiment_returns_404(client):
    resp = client.get("/api/experiments/exp_does_not_exist")
    assert resp.status_code == 404


def test_run_missing_experiment_returns_404(client):
    resp = client.post("/api/experiments/exp_does_not_exist/run")
    assert resp.status_code == 404


# --- zero-variance target -> FAILED at HTTP 200, not a 500 ------------------ #

def test_zero_variance_target_run_fails_at_http_200(client):
    resp = _create(client, _zero_variance_sheet())
    exp_id = resp.json()["id"]
    client.patch(f"/api/experiments/{exp_id}", json={"status": "READY"})

    run = client.post(f"/api/experiments/{exp_id}/run")
    assert run.status_code == 200
    body = run.json()
    assert body["status"] == "FAILED"
    assert body["result"] is None
    assert body["error"] == (
        "The target column 'lipase_titer' has no variance (all values are "
        "identical); it cannot be modeled. Check you selected the "
        "measured-output column."
    )


# --- run-ready processes every READY experiment ----------------------------- #

def test_run_ready_processes_all_ready_experiments(client):
    exp1 = _create(client, _good_sheet(seed=1), name="run 1").json()
    exp2 = _create(client, _good_sheet(seed=2), name="run 2").json()
    draft = _create(client, _good_sheet(seed=3), name="run 3 (draft)").json()
    client.patch(f"/api/experiments/{exp1['id']}", json={"status": "READY"})
    client.patch(f"/api/experiments/{exp2['id']}", json={"status": "READY"})

    resp = client.post("/api/experiments/run-ready")
    assert resp.status_code == 200
    results = resp.json()
    by_id = {r["id"]: r["status"] for r in results}
    assert by_id == {exp1["id"]: "DONE", exp2["id"]: "DONE"}

    # the DRAFT experiment was never picked up
    assert client.get(f"/api/experiments/{draft['id']}").json()["status"] == "DRAFT"
    assert client.get(f"/api/experiments/{exp1['id']}").json()["status"] == "DONE"
    assert client.get(f"/api/experiments/{exp2['id']}").json()["status"] == "DONE"


# --- existing endpoints stay byte-for-byte unchanged ------------------------ #

def test_existing_run_and_latest_endpoints_still_work(client, tmp_path, monkeypatch):
    from kalos.portal import app as portal_module

    # isolate the pre-existing /api/latest cache so this smoke test never
    # touches the real ~/.kalos/latest_analysis.json either.
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_PATH", tmp_path / "latest.json")
    monkeypatch.setattr(portal_module, "_LATEST", None)

    assert client.get("/api/latest").json() == {"has_data": False}

    resp = client.post(
        "/api/run",
        files={"file": ("runs.csv", _csv_bytes(_good_sheet()), "text/csv")},
        data={},
    )
    assert resp.status_code == 200
    assert resp.json()["target"] == "lipase_titer"

    latest = client.get("/api/latest")
    assert latest.status_code == 200
    assert latest.json()["has_data"] is True


# --- FIX 5 (HIGH): GET /api/experiments honors ?status=, and the new
# POST /api/experiments/{id}/result push endpoint really exists -------------- #

def test_list_experiments_status_filter(client):
    ready = _create(client, _good_sheet(seed=1), name="ready one").json()
    draft = _create(client, _good_sheet(seed=2), name="draft one").json()
    client.patch(f"/api/experiments/{ready['id']}", json={"status": "READY"})

    all_rows = client.get("/api/experiments").json()
    assert {r["id"] for r in all_rows} == {ready["id"], draft["id"]}  # unchanged default

    ready_rows = client.get("/api/experiments", params={"status": "READY"}).json()
    assert [r["id"] for r in ready_rows] == [ready["id"]]

    draft_rows = client.get("/api/experiments", params={"status": "DRAFT"}).json()
    assert [r["id"] for r in draft_rows] == [draft["id"]]


def test_list_experiments_unknown_status_filter_returns_400(client):
    resp = client.get("/api/experiments", params={"status": "NOT_A_STATUS"})
    assert resp.status_code == 400


def test_push_result_endpoint_marks_done(client):
    resp = _create(client, _good_sheet())
    exp_id = resp.json()["id"]
    store = app.dependency_overrides[get_store]()
    store.set_status(exp_id, Status.READY)
    store.set_status(exp_id, Status.PROCESSING)

    push = client.post(
        f"/api/experiments/{exp_id}/result",
        json={"result": {"n": 1}, "provenance": {"seed": 1}},
    )
    assert push.status_code == 200, push.text
    body = push.json()
    assert body["status"] == "DONE"
    assert body["result"] == {"n": 1}
    assert body["provenance"] == {"seed": 1}


def test_push_result_endpoint_illegal_from_draft_returns_409(client):
    resp = _create(client, _good_sheet())
    exp_id = resp.json()["id"]
    push = client.post(
        f"/api/experiments/{exp_id}/result",
        json={"result": {"n": 1}, "provenance": {"seed": 1}},
    )
    assert push.status_code == 409


def test_push_result_endpoint_missing_experiment_returns_404(client):
    push = client.post(
        "/api/experiments/exp_does_not_exist/result",
        json={"result": {"n": 1}, "provenance": {"seed": 1}},
    )
    assert push.status_code == 404


# --- FIX 5 contract test: HttpBackendAdapter against the REAL FastAPI app --- #
# Not the mock transport (tests/test_m2_adapter.py) - this drives the actual
# portal endpoints through TestClient, proving list_ready's status filter and
# push_result's endpoint are real, not just documented.

def test_http_adapter_against_real_portal_app(client):
    from kalos.runner.adapter import HttpBackendAdapter

    store = app.dependency_overrides[get_store]()

    def _transport(method, url, json_body):
        resp = client.request(method, url, json=json_body)
        assert resp.status_code < 400, (method, url, resp.status_code, resp.text)
        return resp.json() if resp.content else None

    adapter = HttpBackendAdapter("http://testserver", transport=_transport)

    ready = _create(client, _good_sheet(seed=1), name="ready one").json()
    draft = _create(client, _good_sheet(seed=2), name="draft one").json()
    client.patch(f"/api/experiments/{ready['id']}", json={"status": "READY"})

    ids = adapter.list_ready()
    assert ids == [ready["id"]]  # the real /api/experiments?status=READY filter
    assert draft["id"] not in ids

    store.set_status(ready["id"], Status.PROCESSING)
    adapter.push_result(ready["id"], {"n": 1}, {"seed": 1})  # the real /result endpoint

    done = store.get(ready["id"])
    assert done.status == Status.DONE
    assert done.result == {"n": 1}
