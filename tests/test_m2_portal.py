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
from kalos.store import SqliteStore  # noqa: E402


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
