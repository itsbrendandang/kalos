"""Campaign loop tests (docs/CAMPAIGN_LOOP.md): the closed optimization loop
behind `/api/campaign*` — propose -> run -> log outcome -> re-propose.

Every test overrides `get_campaign_store` to a fresh `CampaignStore(tmp_path)`
(same pattern as `tests/test_m2_portal.py`'s `get_store` override) and sets
`KALOS_STATE_DIR` to a `tmp_path` too, so the real `~/.kalos/campaign.json`
is never touched either way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402


def _tiny_df(n: int = 8, seed: int = 0) -> pd.DataFrame:
    """A small, honest, varying run sheet - 2 numeric features + a numeric
    target - just big enough for `_analyze` to fit (needs >= 6 rows) while
    staying fast."""
    rng = np.random.default_rng(seed)
    methanol = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    titer = 1.5 * methanol - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, n)
    return pd.DataFrame({
        "Methanol": methanol.round(3),
        "pH": ph.round(2),
        "lipase_titer": titer.round(3),
    })


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    return CampaignStore(tmp_path)


@pytest.fixture
def client(store, tmp_path, monkeypatch):
    # `_STATE_DIR`/`_LATEST_PATH` in kalos.portal.app are fixed at import
    # time, so setting KALOS_STATE_DIR alone would not stop `reanalyze`'s
    # `_save_latest` from writing to the real ~/.kalos/latest_analysis.json.
    # Patch the module's globals directly instead, mirroring
    # tests/test_m2_portal.py::test_existing_run_and_latest_endpoints_still_work.
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(portal_module, "_LATEST", {})
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


# --- GET /api/campaign: has_campaign false/true ----------------------------- #

def test_get_campaign_before_any_seed_has_campaign_false(client):
    resp = client.get("/api/campaign")
    assert resp.status_code == 200
    assert resp.json() == {"has_campaign": False}


def test_seed_then_get_campaign_has_campaign_true(store, client):
    df = _tiny_df()
    store.seed(df, "lipase_titer", ["Methanol", "pH"])

    resp = client.get("/api/campaign")
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_campaign"] is True
    assert body["target"] == "lipase_titer"
    assert body["features"] == ["Methanol", "pH"]
    assert body["n_base"] == len(df)
    assert body["best"] == pytest.approx(df["lipase_titer"].max())
    assert body["round"] == 0
    assert body["pending"] == []
    assert body["n_awaiting"] == 0
    assert body["n_measured"] == 0


def test_seed_overwrites_any_existing_campaign(store, client):
    store.seed(_tiny_df(seed=1), "lipase_titer", ["Methanol", "pH"])
    client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 1.0, "std": 0.1,
         "mode": "explore", "reason": "test"},
    ]})

    # A fresh upload seeds a fresh campaign: new base, empty pending, round 0.
    fresh = _tiny_df(seed=2)
    store.seed(fresh, "lipase_titer", ["Methanol", "pH"])

    body = client.get("/api/campaign").json()
    assert body["n_base"] == len(fresh)
    assert body["round"] == 0
    assert body["pending"] == []


# --- POST /api/campaign/start: appends pending runs with ids --------------- #

def test_start_appends_pending_runs_with_ids(store, client):
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    recipes = [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "diversifies the batch"},
        {"recipe": {"Methanol": 3.0, "pH": 6.5}, "pred": 6.2, "std": 0.2,
         "mode": "exploit", "reason": "predicted high"},
    ]
    resp = client.post("/api/campaign/start", json={"recipes": recipes})
    assert resp.status_code == 200
    started = resp.json()["started"]
    assert len(started) == 2
    ids = {run["id"] for run in started}
    assert len(ids) == 2  # unique ids assigned
    for run, recipe in zip(started, recipes):
        assert run["recipe"] == recipe["recipe"]
        assert run["pred"] == recipe["pred"]
        assert run["result"] is None
        assert run["measured_at"] is None

    body = client.get("/api/campaign").json()
    assert body["n_awaiting"] == 2
    assert body["n_measured"] == 0
    assert {p["id"] for p in body["pending"]} == ids
    assert all(p["awaiting"] for p in body["pending"])


def test_start_without_a_campaign_returns_400(client):
    resp = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0}, "pred": 1.0, "std": 0.1, "mode": "explore", "reason": "x"},
    ]})
    assert resp.status_code == 400
    assert "error" in resp.json()


# --- POST /api/campaign/result: sets value, rejects bad input -------------- #

def test_result_sets_value_and_measured_at(store, client):
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    started = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "test"},
    ]}).json()["started"]
    run_id = started[0]["id"]

    resp = client.post("/api/campaign/result", json={"id": run_id, "value": 7.3})
    assert resp.status_code == 200
    run = resp.json()
    assert run["id"] == run_id
    assert run["result"] == 7.3
    assert run["measured_at"] is not None

    body = client.get("/api/campaign").json()
    pending = next(p for p in body["pending"] if p["id"] == run_id)
    assert pending["result"] == 7.3
    assert pending["awaiting"] is False
    assert body["n_measured"] == 1
    assert body["n_awaiting"] == 0


def test_result_unknown_id_returns_400(store, client):
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    resp = client.post("/api/campaign/result", json={"id": "not-a-real-id", "value": 1.0})
    assert resp.status_code == 400
    assert "error" in resp.json()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_result_non_finite_value_returns_400(store, client, bad_value):
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    started = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "test"},
    ]}).json()["started"]
    run_id = started[0]["id"]

    # NaN/Infinity are not valid JSON literals; pydantic (via ujson/json)
    # still parses them the way Python's json module does when sent as bare
    # tokens, so build the request body ourselves rather than relying on
    # `json=` (which would raise on a non-finite float before the request
    # ever reaches the server).
    import json as _json
    payload = _json.dumps({"id": run_id, "value": bad_value})
    resp = client.post(
        "/api/campaign/result", content=payload, headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


# --- POST /api/campaign/reanalyze: the loop closing ------------------------- #

def test_reanalyze_folds_measured_runs_and_keeps_awaiting(store, client):
    df = _tiny_df()
    store.seed(df, "lipase_titer", ["Methanol", "pH"])
    started = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "test A"},
        {"recipe": {"Methanol": 3.0, "pH": 6.5}, "pred": 6.2, "std": 0.2,
         "mode": "exploit", "reason": "test B"},
        {"recipe": {"Methanol": 1.0, "pH": 5.5}, "pred": 3.0, "std": 0.5,
         "mode": "explore", "reason": "test C - stays awaiting"},
    ]}).json()["started"]

    # measure two of the three; the third stays awaiting
    client.post("/api/campaign/result", json={"id": started[0]["id"], "value": 6.0})
    client.post("/api/campaign/result", json={"id": started[1]["id"], "value": 7.5})
    awaiting_id = started[2]["id"]

    resp = client.post("/api/campaign/reanalyze")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    analysis = body["analysis"]
    assert analysis["has_data"] is True
    assert analysis["target"] == "lipase_titer"

    campaign = body["campaign"]
    assert campaign["has_campaign"] is True
    assert campaign["n_base"] == len(df) + 2  # the two measured runs folded in
    assert campaign["round"] == 1
    assert campaign["n_awaiting"] == 1
    assert campaign["n_measured"] == 0
    assert [p["id"] for p in campaign["pending"]] == [awaiting_id]
    assert campaign["pending"][0]["awaiting"] is True

    # progress trajectory: a point at round 0 (seed) and round 1 (this fold),
    # best-so-far non-decreasing because base_rows only grow
    history = campaign["history"]
    assert [h["round"] for h in history] == [0, 1]
    assert history[1]["n_base"] == len(df) + 2
    assert history[0]["best"] is not None and history[1]["best"] is not None
    assert history[1]["best"] >= history[0]["best"]

    # GET /api/campaign reflects the same folded-in state
    follow_up = client.get("/api/campaign").json()
    assert follow_up["n_base"] == len(df) + 2
    assert follow_up["round"] == 1

    # /api/latest reflects the fresh analysis too (docs/CAMPAIGN_LOOP.md,
    # "Re-analyze = the loop closing")
    latest = client.get("/api/latest").json()
    assert latest["has_data"] is True
    assert latest["dataset"] == "campaign round 1"


def test_reanalyze_without_a_campaign_returns_400(client):
    resp = client.post("/api/campaign/reanalyze")
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_reanalyze_with_no_measured_runs_returns_400_and_does_not_advance_round(store, client):
    """Re-analyzing with nothing newly measured is rejected up front — it would
    only inflate `round` and the progress trajectory over an unchanged base."""
    df = _tiny_df()
    store.seed(df, "lipase_titer", ["Methanol", "pH"])
    # one awaiting run, never measured
    client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "awaiting"},
    ]})

    resp = client.post("/api/campaign/reanalyze")
    assert resp.status_code == 400
    assert "error" in resp.json()

    body = client.get("/api/campaign").json()
    assert body["round"] == 0
    assert body["n_base"] == len(df)
    assert [h["round"] for h in body["history"]] == [0]  # no phantom round appended


# --- malformed recipe rejected at the point of input (not a later 500) ------ #

@pytest.mark.parametrize("bad_recipe", [
    {"pred": 1.0},                 # no "recipe" key at all
    {"recipe": None},              # explicit null
    {"recipe": "not-a-mapping"},   # wrong type
    {"recipe": {}},                # empty mapping
])
def test_start_rejects_malformed_recipe_with_400(store, client, bad_recipe):
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    resp = client.post("/api/campaign/start", json={"recipes": [bad_recipe]})
    assert resp.status_code == 400
    assert "error" in resp.json()
    # nothing was appended
    assert client.get("/api/campaign").json()["pending"] == []


# --- transactional reanalyze: failure and reseed-race leave state intact ---- #

def test_reanalyze_analyze_failure_does_not_advance_round_or_fold(store, client, monkeypatch):
    """If `_analyze` raises mid-reanalyze, nothing is committed: the measured
    run stays pending, `round`/`base_rows` are untouched, and a retry (once
    `_analyze` works) folds normally. Regression for the persist-then-validate
    ordering bug."""
    df = _tiny_df()
    store.seed(df, "lipase_titer", ["Methanol", "pH"])
    started = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "measured"},
    ]}).json()["started"]
    client.post("/api/campaign/result", json={"id": started[0]["id"], "value": 6.0})

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated GP fit failure")

    monkeypatch.setattr(portal_module, "_analyze", _boom)
    resp = client.post("/api/campaign/reanalyze")
    assert resp.status_code == 400
    assert "error" in resp.json()

    # nothing folded, nothing advanced — the measured run is still pending
    body = client.get("/api/campaign").json()
    assert body["round"] == 0
    assert body["n_base"] == len(df)
    assert body["n_measured"] == 1
    assert [h["round"] for h in body["history"]] == [0]

    # retry with a working _analyze now folds the run in
    monkeypatch.undo()
    resp = client.post("/api/campaign/reanalyze")
    assert resp.status_code == 200, resp.text
    assert resp.json()["campaign"]["round"] == 1
    assert resp.json()["campaign"]["n_base"] == len(df) + 1


def test_reanalyze_on_legacy_campaign_without_generation_key(store, client):
    """A campaign row written before the `generation` token existed has no such
    key. The first reanalyze after upgrade must not crash (KeyError -> 500) —
    plan_fold reads it with .get(), so it folds normally. Regression for the
    migration crash."""
    import json as _json

    df = _tiny_df()
    store.seed(df, "lipase_titer", ["Methanol", "pH"])
    started = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "measured"},
    ]}).json()["started"]
    client.post("/api/campaign/result", json={"id": started[0]["id"], "value": 6.0})

    # simulate a pre-generation row: strip the key as the last write before reanalyze
    state = store.get()
    assert state is not None
    state.pop("generation", None)
    with store._conn:
        store._conn.execute(
            "UPDATE campaigns SET state = ? WHERE tenant = ?", (_json.dumps(state), "default")
        )

    resp = client.post("/api/campaign/reanalyze")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["campaign"]["round"] == 1
    assert body["campaign"]["n_base"] == len(df) + 1
    # the migration is self-healing: the commit re-stamped a real generation
    final = store.get()
    assert final is not None and "generation" in final


def test_commit_fold_aborts_when_campaign_reseeded_underneath(store):
    """A `seed()` (fresh `/api/run` upload) landing between `plan_fold` and
    `commit_fold` bumps the generation, so `commit_fold` refuses to write —
    the caller then skips `_save_latest`, so `/api/latest` is never clobbered
    with the stale analysis. Regression for the concurrent-upload data-loss
    BLOCKER."""
    from kalos.portal.campaign import CampaignError

    store.seed(_tiny_df(seed=1), "lipase_titer", ["Methanol", "pH"])
    started = store.start([
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "measured"},
    ])
    store.set_result(started[0]["id"], 6.0)

    # plan the fold (captures the current generation), then a fresh upload
    # reseeds the whole campaign underneath the in-flight re-analysis
    _df, _target, generation = store.plan_fold()
    fresh = _tiny_df(n=10, seed=2)
    store.seed(fresh, "lipase_titer", ["Methanol", "pH"])

    with pytest.raises(CampaignError):
        store.commit_fold(generation)

    # the fresh upload survives intact: still round 0, base == the new upload,
    # no round-1 fold written over it
    state = store.get()
    assert state is not None
    assert state["round"] == 0
    assert len(state["base_rows"]) == len(fresh)
    assert state["pending"] == []
