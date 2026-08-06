"""Campaign loop tests (docs/CAMPAIGN_LOOP.md): the closed optimization loop
behind `/api/campaign*` — propose -> run -> log outcome -> re-propose — plus
campaign identity (many named campaigns per tenant), lineage, and the
activity series.

Every test overrides `get_campaign_store` to a fresh `CampaignStore(tmp_path)`
(same pattern as `tests/test_m2_portal.py`'s `get_store` override) and sets
`KALOS_STATE_DIR` to a `tmp_path` too, so the real `~/.kalos/portal.db` is
never touched either way.
"""
from __future__ import annotations

import json as _json
import logging
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignError, CampaignStore, get_campaign_store  # noqa: E402


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


def _upload(client, df: pd.DataFrame, *, filename: str = "runs.csv", campaign_id: str | None = None):
    """POST a run sheet to `/api/run` the way the browser does - a multipart
    form body, with `campaign_id` (when given) as a QUERY parameter. That
    parameter is the only thing that switches an upload from "seed a new
    campaign" to "replace this one" (docs/CAMPAIGN_LOOP.md, "Replacing a
    campaign's data")."""
    params = {} if campaign_id is None else {"campaign_id": campaign_id}
    return client.post(
        "/api/run",
        files={"file": (filename, df.to_csv(index=False).encode("utf-8"), "text/csv")},
        data={},
        params=params,
    )


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
    # `POST /api/run` reaches the campaign store through the module-level
    # `get_campaign_store()` singleton, NOT through FastAPI's dependency
    # injection (it runs deep inside `_run_uploaded_sync`, off the request
    # scope), so the override above does not cover it. Patch the module
    # attribute too, or an upload test would seed/replace inside the real
    # ~/.kalos/portal.db.
    monkeypatch.setattr(campaign_module, "get_campaign_store", lambda: store)
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def _set_state(store: CampaignStore, state: dict) -> None:
    """Write `state` back verbatim via raw SQL - for tests that need to stamp
    a deterministic timestamp (e.g. a UTC day boundary) that `time.time()`
    cannot reliably produce."""
    with store._conn:
        store._conn.execute(
            "UPDATE campaigns SET state = ? WHERE tenant = ? AND campaign_id = ?",
            (_json.dumps(state), "default", state["id"]),
        )


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
    # campaign identity: every campaign carries a stable id and a name
    assert isinstance(body["id"], str) and body["id"]
    assert body["name"]
    # lineage is present from the start, empty until something is folded
    assert body["lineage"] == {"folded_rows": []}


def test_seed_creates_a_new_campaign_that_becomes_the_default(store, client):
    """A fresh upload does not mutate the previous campaign in place — it
    creates its own new campaign (docs/CAMPAIGN_LOOP.md, "Campaign
    identity"). The default (no campaign_id) view still shows exactly what
    the pre-identity single-campaign behavior showed: the freshest upload,
    with an empty pending list and round 0 — because "default" always
    resolves to the tenant's most-recently-updated campaign, which a fresh
    seed always is."""
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


# --- Campaign identity: many campaigns per tenant, addressable by id -------- #

def test_list_campaigns_returns_all_newest_first_with_briefs(store, client):
    id_a = store.seed(_tiny_df(seed=1), "lipase_titer", ["Methanol", "pH"], name="Batch A")
    id_b = store.seed(_tiny_df(seed=2), "lipase_titer", ["Methanol", "pH"], name="Batch B")

    resp = client.get("/api/campaigns")
    assert resp.status_code == 200
    campaigns = resp.json()
    assert [c["id"] for c in campaigns] == [id_b, id_a]
    assert [c["name"] for c in campaigns] == ["Batch B", "Batch A"]
    assert campaigns[0]["n_base"] == len(_tiny_df(seed=2))
    assert campaigns[0]["round"] == 0
    assert campaigns[0]["n_awaiting"] == 0


def test_list_campaigns_empty_for_a_tenant_with_no_uploads(client):
    assert client.get("/api/campaigns").json() == []


def test_seed_without_a_name_falls_back_to_a_generic_default(store, client):
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    body = client.get("/api/campaign").json()
    assert body["name"] == "Untitled campaign"


def test_campaign_id_query_param_addresses_a_specific_non_default_campaign(store, client):
    id_a = store.seed(_tiny_df(seed=1), "lipase_titer", ["Methanol", "pH"], name="Older")
    id_b = store.seed(_tiny_df(seed=2), "lipase_titer", ["Methanol", "pH"], name="Newer")

    # default (no campaign_id) resolves to the most recently updated
    default_body = client.get("/api/campaign").json()
    assert default_body["id"] == id_b

    # an explicit campaign_id reaches the older one instead
    older_body = client.get("/api/campaign", params={"campaign_id": id_a}).json()
    assert older_body["id"] == id_a
    assert older_body["name"] == "Older"

    # mutating with an explicit campaign_id touches only that campaign
    resp = client.post(
        "/api/campaign/start",
        params={"campaign_id": id_a},
        json={"recipes": [{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 1.0, "std": 0.1,
                            "mode": "explore", "reason": "x"}]},
    )
    assert resp.status_code == 200
    assert client.get("/api/campaign", params={"campaign_id": id_a}).json()["n_awaiting"] == 1
    assert client.get("/api/campaign", params={"campaign_id": id_b}).json()["n_awaiting"] == 0


def test_unknown_campaign_id_on_get_returns_has_campaign_false(store, client):
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    resp = client.get("/api/campaign", params={"campaign_id": "not-a-real-id"})
    assert resp.status_code == 200
    assert resp.json() == {"has_campaign": False}


def test_unknown_campaign_id_on_start_returns_400(store, client):
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    resp = client.post(
        "/api/campaign/start",
        params={"campaign_id": "not-a-real-id"},
        json={"recipes": [{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 1.0, "std": 0.1,
                            "mode": "explore", "reason": "x"}]},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_campaign_id_stable_across_a_store_reload(tmp_path, monkeypatch):
    """A campaign's id must survive a process restart — a fresh
    `CampaignStore` pointed at the same db file must see the SAME id, not
    mint a new one."""
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    store1 = CampaignStore(tmp_path)
    cid = store1.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"], name="Run A")

    store2 = CampaignStore(tmp_path)  # simulates a process restart against the same db file
    state = store2.get(campaign_id=cid)
    assert state is not None
    assert state["id"] == cid
    assert state["name"] == "Run A"
    # and it is still the default (most-recent) campaign after reopening
    assert store2.summary()["id"] == cid


def test_legacy_pre_identity_schema_migrates_forward_and_id_stays_stable(tmp_path):
    """Backward compatibility: a `portal.db` written before campaign identity
    existed has `campaigns(tenant PRIMARY KEY, state, updated_at)` — one
    unnamed, unidentified campaign per tenant, no `campaign_id` column at
    all. Opening a `CampaignStore` against it must migrate that row forward
    (mint an id + a generic name) rather than fail to load, and the minted id
    must be STABLE across a second open — the migration runs exactly once."""
    db_path = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE campaigns (tenant TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at REAL NOT NULL)"
    )
    legacy_state = {
        "target": "lipase_titer",
        "features": ["Methanol", "pH"],
        "base_rows": [{"Methanol": 1.0, "pH": 6.0, "lipase_titer": 4.0}],
        "pending": [],
        "round": 0,
        "history": [{"round": 0, "best": 4.0, "n_base": 1}],
        "updated_at": 1_700_000_000.0,
        "generation": "legacy-token",
    }
    conn.execute(
        "INSERT INTO campaigns (tenant, state, updated_at) VALUES (?, ?, ?)",
        ("default", _json.dumps(legacy_state), 1_700_000_000.0),
    )
    conn.commit()
    conn.close()

    store1 = CampaignStore(tmp_path)
    summary = store1.summary(tenant="default")
    assert summary["has_campaign"] is True
    assert isinstance(summary["id"], str) and summary["id"]
    assert summary["name"]  # a generic default, never blank
    assert summary["n_base"] == 1
    assert summary["target"] == "lipase_titer"
    minted_id = summary["id"]

    # the migrated campaign is also listable and still supports the loop -
    # a reanalyze needs no measured runs to be REJECTED cleanly, not crash
    campaigns = store1.list_campaigns(tenant="default")
    assert [c["id"] for c in campaigns] == [minted_id]

    # stable across a second open of the same file - the migration must not
    # re-run and mint a DIFFERENT id on restart
    store2 = CampaignStore(tmp_path)
    summary2 = store2.summary(tenant="default")
    assert summary2["id"] == minted_id


# --- Migration atomicity: BLOCKER regression (crash mid-migration must not - #
# --- permanently orphan legacy data), plus the other migration edge cases -- #

def _write_legacy_db(db_path, rows: dict) -> None:
    """Write a raw pre-identity-schema `portal.db`:
    `campaigns(tenant PRIMARY KEY, state, updated_at)` - one row per
    `(tenant, raw_state_json)` in `rows`. `raw_state_json` is written
    VERBATIM, not re-encoded, so a caller can pass invalid JSON to exercise
    the corrupt-row path."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE campaigns (tenant TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at REAL NOT NULL)"
    )
    for tenant, raw_state in rows.items():
        conn.execute(
            "INSERT INTO campaigns (tenant, state, updated_at) VALUES (?, ?, ?)",
            (tenant, raw_state, 1_700_000_000.0),
        )
    conn.commit()
    conn.close()


def _legacy_state_json(base_value: float = 4.0, n_rows: int = 1) -> str:
    """A realistic legacy (pre-identity) campaign state, JSON-encoded - real
    base rows, a target, a generation token, everything a genuine tenant's
    only campaign would have had before campaign identity existed."""
    state = {
        "target": "lipase_titer",
        "features": ["Methanol", "pH"],
        "base_rows": [
            {"Methanol": 1.0 + i, "pH": 6.0, "lipase_titer": base_value + i} for i in range(n_rows)
        ],
        "pending": [],
        "round": 0,
        "history": [{"round": 0, "best": base_value + n_rows - 1, "n_base": n_rows}],
        "updated_at": 1_700_000_000.0,
        "generation": "legacy-token",
    }
    return _json.dumps(state)


def test_migration_crash_partway_through_leaves_all_tenants_data_recoverable(tmp_path, monkeypatch):
    """BLOCKER regression: `_migrate_legacy_schema` must be one genuinely
    atomic unit (explicit `BEGIN IMMEDIATE` / `COMMIT` / `ROLLBACK`), not a
    `with self._conn:` block - which does NOT cover the DDL (`ALTER TABLE`
    RENAME, `CREATE TABLE`), since Python's sqlite3 module runs DDL in
    autocommit and only opens an implicit transaction before DML. Before the
    fix, a crash between the CREATE and the DROP left an empty new
    `campaigns` table plus the real data orphaned, unreachable, in
    `campaigns_legacy` forever - the migration guard sees the fresh
    `campaign_id` column and treats it as already migrated on every later
    restart, silently.

    Reproduces the review's exact recipe: two tenants' real campaign data in
    a legacy db, `uuid.uuid4` monkeypatched to raise on the SECOND call
    (i.e. after tenant1's row is migrated in-loop, before tenant2's) -
    simulating a crash/OOM/kill -9 partway through the migration loop.
    """
    db_path = tmp_path / "portal.db"
    _write_legacy_db(db_path, {
        "tenant1": _legacy_state_json(base_value=4.0, n_rows=1),
        "tenant2": _legacy_state_json(base_value=9.0, n_rows=2),
    })

    import kalos.portal.campaign as campaign_module

    real_uuid4 = campaign_module.uuid.uuid4
    calls = {"n": 0}

    def _uuid4_crash_on_second_call():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash mid-migration")
        return real_uuid4()

    monkeypatch.setattr(campaign_module.uuid, "uuid4", _uuid4_crash_on_second_call)

    with pytest.raises(RuntimeError, match="simulated crash mid-migration"):
        CampaignStore(tmp_path)

    # The crash must NOT have left the db half-migrated: still the legacy
    # shape, no leftover "campaigns_legacy" table, no partially-populated
    # new table sitting around fooling the next attempt's "already
    # migrated" guard - this is the actual atomicity assertion, not just
    # "it eventually recovers."
    raw = sqlite3.connect(str(db_path))
    tables = {row[0] for row in raw.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert tables == {"campaigns"}
    cols = {row[1] for row in raw.execute("PRAGMA table_info(campaigns)").fetchall()}
    assert "campaign_id" not in cols  # rolled all the way back to the legacy shape
    remaining = raw.execute("SELECT tenant FROM campaigns ORDER BY tenant").fetchall()
    assert [r[0] for r in remaining] == ["tenant1", "tenant2"]
    raw.close()

    monkeypatch.undo()  # the crash was only meant to simulate ONE interruption

    # The next construction retries and fully recovers BOTH tenants' data.
    store = CampaignStore(tmp_path)
    campaigns1 = store.list_campaigns(tenant="tenant1")
    campaigns2 = store.list_campaigns(tenant="tenant2")
    assert len(campaigns1) == 1
    assert campaigns1[0]["n_base"] == 1
    assert campaigns1[0]["best"] == pytest.approx(4.0)
    assert len(campaigns2) == 1
    assert campaigns2[0]["n_base"] == 2
    assert campaigns2[0]["best"] == pytest.approx(10.0)

    # ids are real, non-empty, distinct, and stable across yet another reopen
    id1, id2 = campaigns1[0]["id"], campaigns2[0]["id"]
    assert id1 and id2 and id1 != id2
    store2 = CampaignStore(tmp_path)
    assert store2.list_campaigns(tenant="tenant1")[0]["id"] == id1
    assert store2.list_campaigns(tenant="tenant2")[0]["id"] == id2


def test_migration_is_a_noop_on_a_fresh_db_with_no_existing_table(tmp_path):
    """No `campaigns` table exists yet (a brand new state dir): the
    migration guard's `if not cols` branch returns immediately - nothing to
    rename, nothing to roll back."""
    store = CampaignStore(tmp_path)
    assert store.summary(tenant="default") == {"has_campaign": False}
    conn = sqlite3.connect(str(tmp_path / "portal.db"))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(campaigns)").fetchall()}
    assert "campaign_id" in cols  # the CURRENT schema, created fresh, not migrated
    conn.close()


def test_migration_is_a_noop_on_an_already_migrated_db(tmp_path, caplog):
    """A second `CampaignStore` against a db that has ALREADY been migrated
    (the `campaign_id` column is already present) must not re-run the
    migration - the `cols` guard short-circuits before the explicit `BEGIN
    IMMEDIATE`, so a second open mints no new id and logs no migration."""
    db_path = tmp_path / "portal.db"
    _write_legacy_db(db_path, {"default": _legacy_state_json()})
    store1 = CampaignStore(tmp_path)
    minted_id = store1.summary(tenant="default")["id"]

    with caplog.at_level(logging.INFO, logger="kalos.portal"):
        caplog.clear()
        store2 = CampaignStore(tmp_path)
    assert "migrat" not in caplog.text.lower()
    assert store2.summary(tenant="default")["id"] == minted_id

    conn = sqlite3.connect(str(db_path))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert tables == {"campaigns"}  # no campaigns_legacy table left lying around
    conn.close()


def test_migration_migrates_every_tenants_legacy_row_in_one_pass(tmp_path):
    """The legacy schema was `tenant PRIMARY KEY` - multiple tenants share
    the same legacy table. All of them must survive migration, not just the
    first one encountered, and stay isolated from each other afterward."""
    db_path = tmp_path / "portal.db"
    _write_legacy_db(db_path, {
        "alpha": _legacy_state_json(base_value=1.0, n_rows=3),
        "beta": _legacy_state_json(base_value=2.0, n_rows=1),
        "gamma": _legacy_state_json(base_value=3.0, n_rows=5),
    })

    store = CampaignStore(tmp_path)
    for tenant, expected_n in [("alpha", 3), ("beta", 1), ("gamma", 5)]:
        campaigns = store.list_campaigns(tenant=tenant)
        assert len(campaigns) == 1
        assert campaigns[0]["n_base"] == expected_n

    alpha_id = store.list_campaigns(tenant="alpha")[0]["id"]
    assert store.get(tenant="beta", campaign_id=alpha_id) is None  # tenants stay isolated


def test_migration_logs_a_warning_for_each_corrupt_legacy_row(tmp_path, caplog):
    """A legacy row that fails to parse must not silently vanish. The
    tenant still ends up with no campaigns (the honest outcome for
    genuinely unreadable data - there is nothing recoverable to show), but
    an operator must be able to find out FROM THE LOGS that this happened,
    rather than it being indistinguishable from a tenant that never
    uploaded anything."""
    db_path = tmp_path / "portal.db"
    _write_legacy_db(db_path, {
        "good_tenant": _legacy_state_json(base_value=5.0),
        "corrupt_tenant": "not valid json{{{",
        "wrong_type_tenant": _json.dumps([1, 2, 3]),  # valid JSON, but not an object
    })

    with caplog.at_level(logging.WARNING, logger="kalos.portal"):
        store = CampaignStore(tmp_path)

    assert store.list_campaigns(tenant="good_tenant") != []
    assert store.list_campaigns(tenant="corrupt_tenant") == []
    assert store.list_campaigns(tenant="wrong_type_tenant") == []

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    warning_text = " ".join(r.getMessage() for r in warnings)
    assert "corrupt_tenant" in warning_text
    assert "wrong_type_tenant" in warning_text


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
    # "Re-analyze = the loop closing"), and now also carries the campaign id
    # it came from - two campaigns' analyses must be distinguishable to a
    # client reading /api/latest alone.
    latest = client.get("/api/latest").json()
    assert latest["has_data"] is True
    assert latest["dataset"] == "campaign round 1"
    assert latest["campaign_id"] == campaign["id"]


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

    real_analyze = portal_module._analyze

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

    # Retry with a working _analyze now folds the run in. Restore ONLY
    # `_analyze` - a blanket `monkeypatch.undo()` would also revert the
    # `client` fixture's patches (pytest hands both the fixture and the test
    # the same function-scoped monkeypatch instance), pointing `_save_latest`
    # at the real ~/.kalos for the rest of the test.
    monkeypatch.setattr(portal_module, "_analyze", real_analyze)
    resp = client.post("/api/campaign/reanalyze")
    assert resp.status_code == 200, resp.text
    assert resp.json()["campaign"]["round"] == 1
    assert resp.json()["campaign"]["n_base"] == len(df) + 1


def test_reanalyze_on_legacy_campaign_without_generation_key(store, client):
    """A campaign row written before the `generation` token existed has no such
    key. The first reanalyze after upgrade must not crash (KeyError -> 500) —
    plan_fold reads it with .get(), so it folds normally. Regression for the
    migration crash."""
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
    _set_state(store, state)

    resp = client.post("/api/campaign/reanalyze")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["campaign"]["round"] == 1
    assert body["campaign"]["n_base"] == len(df) + 1
    # the migration is self-healing: the commit re-stamped a real generation
    final = store.get()
    assert final is not None and "generation" in final


def test_commit_fold_without_an_explicit_id_fails_safe_if_a_fresh_upload_lands(store):
    """`plan_fold()`/`commit_fold()` called with no `campaign_id` both default
    to the tenant's most-recently-updated campaign. If a fresh `seed()`
    creates a NEW, unrelated campaign between the two calls, that new
    campaign — not the one `plan_fold` actually planned against — becomes
    "most recent", so a `commit_fold()` that (incorrectly) re-resolves the
    default lands on the WRONG campaign, whose generation naturally does not
    match. It fails safe (`CampaignError`, nothing corrupted in either
    campaign) rather than silently committing into the wrong one. This is
    exactly why `campaign_routes.reanalyze_campaign` threads the id
    `plan_fold` returns through to `commit_fold` explicitly instead of
    relying on this fallback — see
    `test_reanalyze_of_one_campaign_survives_a_fresh_upload_creating_another`
    for the correctly-threaded, non-error path."""
    campaign_a = store.seed(_tiny_df(seed=1), "lipase_titer", ["Methanol", "pH"])
    started = store.start([
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "measured"},
    ], campaign_id=campaign_a)
    store.set_result(started[0]["id"], 6.0, campaign_id=campaign_a)

    # plan the fold against campaign_a specifically (captures ITS generation)...
    _df, _target, generation, cid = store.plan_fold(campaign_id=campaign_a)
    assert cid == campaign_a
    # ...then a fresh upload creates an UNRELATED campaign B, which becomes
    # the tenant's new "most recent"
    fresh = _tiny_df(n=10, seed=2)
    campaign_b = store.seed(fresh, "lipase_titer", ["Methanol", "pH"])

    with pytest.raises(CampaignError):
        store.commit_fold(generation)  # no campaign_id -> mis-resolves to B, not A

    # campaign A survives completely untouched — the actual isolation guarantee
    state_a = store.get(campaign_id=campaign_a)
    assert state_a is not None
    assert state_a["round"] == 0
    assert len(state_a["pending"]) == 1  # still measured, not yet folded

    # campaign B (what commit_fold safely refused to corrupt) is untouched too
    state_b = store.get(campaign_id=campaign_b)
    assert state_b is not None
    assert state_b["round"] == 0
    assert len(state_b["base_rows"]) == len(fresh)
    assert state_b["pending"] == []


def test_same_campaign_concurrent_write_during_reanalyze_aborts_commit(store):
    """The realistic concurrency hazard now that a fresh upload no longer
    shares a row with any existing campaign: a `start()`/`set_result()`
    write landing on the SAME campaign between `plan_fold` and `commit_fold`
    bumps ITS generation, so `commit_fold` still refuses to write — the core
    guarantee `docs/CAMPAIGN_LOOP.md` documents is unchanged by campaign
    identity."""
    df = _tiny_df()
    campaign_id = store.seed(df, "lipase_titer", ["Methanol", "pH"])
    started = store.start([
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "measured"},
    ], campaign_id=campaign_id)
    store.set_result(started[0]["id"], 6.0, campaign_id=campaign_id)

    _df, _target, generation, cid = store.plan_fold(campaign_id=campaign_id)

    # a concurrent write to the SAME campaign lands here, mid-analysis
    store.start([
        {"recipe": {"Methanol": 1.0, "pH": 5.5}, "pred": 3.0, "std": 0.5,
         "mode": "explore", "reason": "concurrent"},
    ], campaign_id=campaign_id)

    with pytest.raises(CampaignError):
        store.commit_fold(generation, campaign_id=cid)

    # nothing folded: still round 0, base_rows untouched, both pending runs intact
    state = store.get(campaign_id=campaign_id)
    assert state is not None
    assert state["round"] == 0
    assert len(state["base_rows"]) == len(df)
    assert len(state["pending"]) == 2


def test_reanalyze_of_one_campaign_survives_a_fresh_upload_creating_another(store, client, monkeypatch):
    """The key isolation improvement from campaign identity: an /api/run-style
    upload elsewhere no longer shares a row with an in-flight campaign's
    reanalyze, so it can no longer force a spurious 409 on it — as long as
    the resolved campaign_id is threaded through end to end, which
    `reanalyze_campaign` does."""
    campaign_a = store.seed(_tiny_df(seed=1), "lipase_titer", ["Methanol", "pH"], name="A")
    started = client.post(
        "/api/campaign/start", params={"campaign_id": campaign_a},
        json={"recipes": [{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
                            "mode": "explore", "reason": "x"}]},
    ).json()["started"]
    client.post(
        "/api/campaign/result", params={"campaign_id": campaign_a},
        json={"id": started[0]["id"], "value": 6.0},
    )

    real_analyze = portal_module._analyze

    def _analyze_and_race(*args, **kwargs):
        # a concurrent /api/run upload lands while THIS reanalyze's _analyze
        # is in flight, seeding a brand new, unrelated campaign B
        store.seed(_tiny_df(seed=2), "lipase_titer", ["Methanol", "pH"], name="B")
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr(portal_module, "_analyze", _analyze_and_race)

    resp = client.post("/api/campaign/reanalyze", params={"campaign_id": campaign_a})
    assert resp.status_code == 200, resp.text
    campaign = resp.json()["campaign"]
    assert campaign["id"] == campaign_a
    assert campaign["round"] == 1

    campaigns = client.get("/api/campaigns").json()
    assert len(campaigns) == 2
    names_by_id = {c["id"]: c["name"] for c in campaigns}
    assert names_by_id[campaign_a] == "A"
    assert "B" in names_by_id.values()


# --- Lineage: folded rows carry run_id/timestamps; proposals carry parents - #

def test_commit_fold_carries_run_id_and_timestamps_into_lineage(store, client):
    df = _tiny_df()
    store.seed(df, "lipase_titer", ["Methanol", "pH"])
    started = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "test"},
    ]}).json()["started"]
    run_id = started[0]["id"]
    client.post("/api/campaign/result", json={"id": run_id, "value": 6.0})

    resp = client.post("/api/campaign/reanalyze")
    assert resp.status_code == 200, resp.text
    campaign = resp.json()["campaign"]

    folded = campaign["lineage"]["folded_rows"]
    assert len(folded) == 1
    assert folded[0]["run_id"] == run_id
    assert folded[0]["created_at"] == started[0]["created_at"]
    assert folded[0]["measured_at"] is not None


def test_pending_run_parent_ids_trace_prior_folded_runs_honestly(store, client):
    """Round 0 proposals have no traceable parent ids — the original upload
    rows carry no run id — but DO report the true row count they were
    conditioned on, rather than fabricating ids for rows that have none
    (docs/CAMPAIGN_LOOP.md, "Lineage"). Round 1+ proposals, started after a
    fold, trace back to the run(s) folded in before them."""
    df = _tiny_df()
    store.seed(df, "lipase_titer", ["Methanol", "pH"])

    first_batch = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "round 0"},
    ]}).json()["started"]
    assert first_batch[0]["parent_ids"] == []
    assert first_batch[0]["parent_row_count"] == len(df)

    client.post("/api/campaign/result", json={"id": first_batch[0]["id"], "value": 6.0})
    reanalyze_resp = client.post("/api/campaign/reanalyze")
    assert reanalyze_resp.status_code == 200, reanalyze_resp.text

    second_batch = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 3.0, "pH": 6.5}, "pred": 6.0, "std": 0.3,
         "mode": "exploit", "reason": "round 1"},
    ]}).json()["started"]
    assert second_batch[0]["parent_ids"] == [first_batch[0]["id"]]
    assert second_batch[0]["parent_row_count"] == len(df) + 1


# --- Activity series: runs measured per UTC day + derived stats ------------ #

def test_activity_no_campaign_returns_has_campaign_false(client):
    resp = client.get("/api/campaign/activity")
    assert resp.status_code == 200
    assert resp.json() == {"has_campaign": False}


def test_activity_empty_campaign_has_zero_stats(store, client):
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    resp = client.get("/api/campaign/activity")
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_campaign"] is True
    assert body["days"] == []
    assert body["stats"] == {"days_running": 0, "days_with_runs": 0, "longest_pause_days": 0}
    assert "POST /api/campaign/start" in body["note"]


def test_activity_excludes_awaiting_never_measured_runs(store, client):
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "never measured"},
    ]})
    body = client.get("/api/campaign/activity").json()
    assert body["days"] == []
    assert body["stats"]["days_with_runs"] == 0


def test_activity_buckets_by_utc_calendar_day_at_the_boundary(store, client):
    """A run measured one second before UTC midnight and another one second
    after must land in two different day buckets — the exact boundary a
    naive or local-timezone bucketing would get wrong (docs/CAMPAIGN_LOOP.md,
    "Activity series")."""
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    started = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4, "mode": "explore", "reason": "a"},
        {"recipe": {"Methanol": 3.0, "pH": 6.5}, "pred": 6.0, "std": 0.3, "mode": "exploit", "reason": "b"},
    ]}).json()["started"]

    before_midnight = datetime(2026, 1, 1, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    after_midnight = datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc).timestamp()

    state = store.get()
    for run in state["pending"]:
        if run["id"] == started[0]["id"]:
            run["result"], run["measured_at"] = 6.0, before_midnight
        elif run["id"] == started[1]["id"]:
            run["result"], run["measured_at"] = 7.0, after_midnight
    _set_state(store, state)

    body = client.get("/api/campaign/activity").json()
    assert body["days"] == [
        {"date": "2026-01-01", "runs": 1},
        {"date": "2026-01-02", "runs": 1},
    ]
    assert body["stats"] == {"days_running": 2, "days_with_runs": 2, "longest_pause_days": 0}


def test_activity_counts_folded_runs_and_computes_longest_pause(store, client):
    """After a reanalyze folds a measured run into base_rows, its
    measured_at still counts toward the series (via `base_row_meta`, not
    just `pending`). A multi-day gap between active days is reported as the
    longest pause."""
    df = _tiny_df()
    store.seed(df, "lipase_titer", ["Methanol", "pH"])
    first = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4, "mode": "explore", "reason": "a"},
    ]}).json()["started"]
    client.post("/api/campaign/result", json={"id": first[0]["id"], "value": 6.0})

    day1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    state = store.get()
    state["pending"][0]["measured_at"] = day1
    _set_state(store, state)

    reanalyze_resp = client.post("/api/campaign/reanalyze")
    assert reanalyze_resp.status_code == 200, reanalyze_resp.text  # folds it into base_row_meta

    second = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 1.5, "pH": 5.8}, "pred": 4.0, "std": 0.4, "mode": "explore", "reason": "b"},
    ]}).json()["started"]
    client.post("/api/campaign/result", json={"id": second[0]["id"], "value": 5.0})
    day6 = datetime(2026, 1, 6, 9, 0, 0, tzinfo=timezone.utc).timestamp()
    state = store.get()
    for run in state["pending"]:
        if run["id"] == second[0]["id"]:
            run["measured_at"] = day6
    _set_state(store, state)

    body = client.get("/api/campaign/activity").json()
    assert body["days"] == [
        {"date": "2026-01-01", "runs": 1},
        {"date": "2026-01-06", "runs": 1},
    ]
    assert body["stats"] == {"days_running": 6, "days_with_runs": 2, "longest_pause_days": 4}


# --- Replace: POST /api/run?campaign_id=X corrects one campaign in place ---- #

def test_upload_without_a_campaign_id_still_creates_a_new_campaign(store, client):
    """The default is unchanged: an upload with no `campaign_id` seeds its own
    new campaign, so two uploads leave two campaigns. Replacing is strictly
    opt-in (docs/CAMPAIGN_LOOP.md, "Replacing a campaign's data")."""
    assert _upload(client, _tiny_df(seed=1), filename="first.csv").status_code == 200
    assert _upload(client, _tiny_df(seed=2), filename="second.csv").status_code == 200

    campaigns = client.get("/api/campaigns").json()
    assert [c["name"] for c in campaigns] == ["second.csv", "first.csv"]


def test_upload_with_a_campaign_id_replaces_that_campaign_in_place(store, client):
    """The corrected-CSV path: re-uploading with `?campaign_id=` fixes the
    campaign that already exists instead of creating a second one. Its `id`
    survives, its base data is the corrected file's, and the correction is
    recorded in `revisions`."""
    original = _tiny_df(n=8, seed=1)
    cid = store.seed(original, "lipase_titer", ["Methanol", "pH"], name="typo.csv")

    corrected = _tiny_df(n=11, seed=2)
    resp = _upload(client, corrected, filename="corrected.csv", campaign_id=cid)
    assert resp.status_code == 200, resp.text

    # exactly ONE campaign, still the same one
    campaigns = client.get("/api/campaigns").json()
    assert [c["id"] for c in campaigns] == [cid]

    body = client.get("/api/campaign", params={"campaign_id": cid}).json()
    assert body["id"] == cid
    assert body["name"] == "corrected.csv"
    assert body["n_base"] == len(corrected)
    assert body["best"] == pytest.approx(corrected["lipase_titer"].max())
    assert body["round"] == 0
    # history is reset to a single round-0 point over the NEW base - it must not
    # keep describing a dataset that no longer exists
    assert [h["round"] for h in body["history"]] == [0]
    assert body["history"][0]["n_base"] == len(corrected)
    # lineage is reset too: every row is a fresh upload row, none traceable
    assert body["lineage"] == {"folded_rows": []}

    # the correction itself is on the record, even though the superseded rows
    # are not - nothing is silently rewritten
    assert len(body["revisions"]) == 1
    assert body["revisions"][0]["name"] == "typo.csv"
    assert body["revisions"][0]["n_base"] == len(original)

    # /api/latest points at the SAME campaign, not a newly minted one
    assert client.get("/api/latest").json()["campaign_id"] == cid


def test_replace_preserves_awaiting_pending_runs(store, client):
    """An awaiting run is already physically in the lab, so a correction to
    the sheet it was proposed from must not erase the record that it was
    started (docs/CAMPAIGN_LOOP.md, "Replacing a campaign's data")."""
    cid = store.seed(_tiny_df(seed=1), "lipase_titer", ["Methanol", "pH"], name="typo.csv")
    started = client.post(
        "/api/campaign/start", params={"campaign_id": cid},
        json={"recipes": [{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
                           "mode": "explore", "reason": "already in the lab"}]},
    ).json()["started"]

    resp = _upload(client, _tiny_df(n=11, seed=2), filename="corrected.csv", campaign_id=cid)
    assert resp.status_code == 200, resp.text

    body = client.get("/api/campaign", params={"campaign_id": cid}).json()
    assert [p["id"] for p in body["pending"]] == [started[0]["id"]]
    assert body["n_awaiting"] == 1
    assert body["pending"][0]["awaiting"] is True
    # its recorded parentage is history, not a live pointer - it still reports
    # the base it was genuinely proposed from
    assert body["pending"][0]["parent_row_count"] == len(_tiny_df(seed=1))


def test_replace_refuses_to_discard_a_measured_result(store, client):
    """THE data-loss path: a measured result is an outcome a scientist logged
    from the lab, and a corrected run sheet has no business deleting it. The
    replace is refused with an explicit reason and nothing at all is written -
    not the campaign, not /api/latest."""
    df = _tiny_df(seed=1)
    cid = store.seed(df, "lipase_titer", ["Methanol", "pH"], name="typo.csv")
    started = client.post(
        "/api/campaign/start", params={"campaign_id": cid},
        json={"recipes": [{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
                           "mode": "explore", "reason": "measured"}]},
    ).json()["started"]
    client.post(
        "/api/campaign/result", params={"campaign_id": cid},
        json={"id": started[0]["id"], "value": 6.0},
    )

    resp = _upload(client, _tiny_df(n=11, seed=2), filename="corrected.csv", campaign_id=cid)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert "measured" in error
    assert "archive" in error  # the refusal names the way forward, not just "no"

    # the campaign is untouched: original base, the measured run still pending
    body = client.get("/api/campaign", params={"campaign_id": cid}).json()
    assert body["n_base"] == len(df)
    assert body["name"] == "typo.csv"
    assert body["n_measured"] == 1
    assert body["revisions"] == []
    # and no second campaign was created as a consolation prize
    assert [c["id"] for c in client.get("/api/campaigns").json()] == [cid]
    # /api/latest was never written either - the refusal happens before it
    assert client.get("/api/latest").json() == {"has_data": False}


def test_replace_refuses_after_a_reanalyze_folded_a_round(store, client):
    """Same refusal once the measured runs have been FOLDED: the folded rows
    descend from a proposal the GP conditioned on the old base, so replacing
    that base underneath them would leave the campaign describing a dataset
    that never existed."""
    df = _tiny_df(seed=1)
    cid = store.seed(df, "lipase_titer", ["Methanol", "pH"], name="typo.csv")
    started = client.post(
        "/api/campaign/start", params={"campaign_id": cid},
        json={"recipes": [{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
                           "mode": "explore", "reason": "measured"}]},
    ).json()["started"]
    client.post(
        "/api/campaign/result", params={"campaign_id": cid},
        json={"id": started[0]["id"], "value": 6.0},
    )
    assert client.post("/api/campaign/reanalyze", params={"campaign_id": cid}).status_code == 200

    resp = _upload(client, _tiny_df(n=11, seed=2), filename="corrected.csv", campaign_id=cid)
    assert resp.status_code == 400
    assert "measured" in resp.json()["error"]

    body = client.get("/api/campaign", params={"campaign_id": cid}).json()
    assert body["round"] == 1
    assert body["n_base"] == len(df) + 1
    assert len(body["lineage"]["folded_rows"]) == 1


def test_replace_refuses_a_different_target(store):
    """A campaign IS one optimization target plus a growing dataset, so a file
    that analyzes a DIFFERENT target is a different experiment, not a
    correction - and accepting it would leave any preserved pending run due to
    be folded in under an objective it was never proposed for."""
    cid = store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"], name="original.csv")
    with pytest.raises(CampaignError, match="different target"):
        store.replace(cid, _tiny_df(seed=2), "purity_pct", ["Methanol", "pH"], name="other.csv")

    state = store.get(campaign_id=cid)
    assert state is not None
    assert state["target"] == "lipase_titer"
    assert state["name"] == "original.csv"
    assert state.get("revisions") is None


def test_replace_of_an_unknown_campaign_id_returns_400_and_creates_nothing(store, client):
    """A `campaign_id` that does not exist for this tenant is a refusal, NOT a
    silent fall back to seeding a new campaign - the caller asked to correct
    one specific campaign and has to be told that it was not found."""
    cid = store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"], name="real.csv")

    resp = _upload(client, _tiny_df(seed=2), filename="corrected.csv", campaign_id="not-a-real-id")
    assert resp.status_code == 400
    assert "not-a-real-id" in resp.json()["error"]

    # no new campaign, and the real one is untouched
    assert [c["id"] for c in client.get("/api/campaigns").json()] == [cid]
    assert client.get("/api/campaign", params={"campaign_id": cid}).json()["name"] == "real.csv"
    assert client.get("/api/latest").json() == {"has_data": False}


def test_replace_refuses_on_an_archived_campaign(store, client):
    """An archived campaign is a closed record: unarchive it before rewriting
    its data."""
    cid = store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"], name="original.csv")
    assert client.post("/api/campaign/archive", params={"campaign_id": cid}).status_code == 200

    resp = _upload(client, _tiny_df(seed=2), filename="corrected.csv", campaign_id=cid)
    assert resp.status_code == 400
    assert "archived" in resp.json()["error"]
    assert client.get("/api/campaign", params={"campaign_id": cid}).json()["name"] == "original.csv"


# --- Archiving: hide from the rail, keep every row queryable ---------------- #

def test_archive_hides_a_campaign_from_the_list_but_include_archived_shows_it(store, client):
    id_a = store.seed(_tiny_df(seed=1), "lipase_titer", ["Methanol", "pH"], name="Keep")
    id_b = store.seed(_tiny_df(seed=2), "lipase_titer", ["Methanol", "pH"], name="Retire")

    resp = client.post("/api/campaign/archive", params={"campaign_id": id_b})
    assert resp.status_code == 200
    assert resp.json()["archived"] is True

    active = client.get("/api/campaigns").json()
    assert [c["id"] for c in active] == [id_a]
    assert active[0]["archived"] is False

    everything = client.get("/api/campaigns", params={"include_archived": "true"}).json()
    assert {c["id"] for c in everything} == {id_a, id_b}
    assert {c["id"]: c["archived"] for c in everything} == {id_a: False, id_b: True}


def test_archived_campaign_is_still_fully_queryable_by_id(store, client):
    """Archiving hides a campaign; it never makes its data unreadable. This is
    the pharma-traceability requirement: the record that a run happened has to
    survive (docs/CAMPAIGN_LOOP.md, "Archiving")."""
    df = _tiny_df()
    cid = store.seed(df, "lipase_titer", ["Methanol", "pH"], name="Retired")
    started = client.post(
        "/api/campaign/start", params={"campaign_id": cid},
        json={"recipes": [{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
                           "mode": "explore", "reason": "ran in the lab"}]},
    ).json()["started"]
    client.post(
        "/api/campaign/result", params={"campaign_id": cid},
        json={"id": started[0]["id"], "value": 6.0},
    )
    client.post("/api/campaign/archive", params={"campaign_id": cid})

    body = client.get("/api/campaign", params={"campaign_id": cid}).json()
    assert body["has_campaign"] is True
    assert body["archived"] is True
    assert body["n_base"] == len(df)
    assert body["n_measured"] == 1
    assert [p["id"] for p in body["pending"]] == [started[0]["id"]]

    activity = client.get("/api/campaign/activity", params={"campaign_id": cid}).json()
    assert activity["has_campaign"] is True
    assert activity["stats"]["days_with_runs"] == 1


def test_archived_campaign_never_becomes_the_default(store, client):
    """Every id-less resolution path has to skip archived campaigns, not just
    the listing - otherwise a `start`/`result`/`reanalyze` that omitted the id
    could still land on the campaign the scientist just put away."""
    id_a = store.seed(_tiny_df(seed=1), "lipase_titer", ["Methanol", "pH"], name="Older")
    id_b = store.seed(_tiny_df(seed=2), "lipase_titer", ["Methanol", "pH"], name="Newer")
    assert client.get("/api/campaign").json()["id"] == id_b  # B is the default while active

    client.post("/api/campaign/archive", params={"campaign_id": id_b})

    # reads fall through to the older ACTIVE campaign, not the archived newer one
    assert client.get("/api/campaign").json()["id"] == id_a
    assert client.get("/api/campaign/activity").json()["id"] == id_a

    # ...and so do writes: an id-less start lands on A, never on archived B
    resp = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 1.0, "std": 0.1,
         "mode": "explore", "reason": "id-less"},
    ]})
    assert resp.status_code == 200
    assert client.get("/api/campaign", params={"campaign_id": id_a}).json()["n_awaiting"] == 1
    assert client.get("/api/campaign", params={"campaign_id": id_b}).json()["n_awaiting"] == 0

    # with EVERY campaign archived, an id-less caller gets "nothing", not the
    # most recently archived one
    client.post("/api/campaign/archive", params={"campaign_id": id_a})
    assert client.get("/api/campaign").json() == {"has_campaign": False}
    assert client.get("/api/campaigns").json() == []
    assert client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 1.0, "std": 0.1,
         "mode": "explore", "reason": "id-less"},
    ]}).status_code == 400


def test_unarchive_restores_a_campaign_to_the_rail_and_the_default(store, client):
    id_a = store.seed(_tiny_df(seed=1), "lipase_titer", ["Methanol", "pH"], name="Older")
    id_b = store.seed(_tiny_df(seed=2), "lipase_titer", ["Methanol", "pH"], name="Newer")
    client.post("/api/campaign/archive", params={"campaign_id": id_a})
    assert [c["id"] for c in client.get("/api/campaigns").json()] == [id_b]

    resp = client.post("/api/campaign/unarchive", params={"campaign_id": id_a})
    assert resp.status_code == 200
    assert resp.json()["archived"] is False

    # back in the rail, and - because unarchiving is a mutation like any other
    # - it is now the most-recently-updated campaign, so it is the default again
    assert {c["id"] for c in client.get("/api/campaigns").json()} == {id_a, id_b}
    assert client.get("/api/campaign").json()["id"] == id_a
    # and it accepts writes again
    assert client.post("/api/campaign/start", params={"campaign_id": id_a}, json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 1.0, "std": 0.1,
         "mode": "explore", "reason": "resumed"},
    ]}).status_code == 200


@pytest.mark.parametrize("path", ["/api/campaign/archive", "/api/campaign/unarchive"])
def test_archive_and_unarchive_reject_an_unknown_campaign_id(store, client, path):
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    resp = client.post(path, params={"campaign_id": "not-a-real-id"})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_archive_requires_an_explicit_campaign_id(store, client):
    """Unlike every other route on this router, archive does NOT fall back to
    "the current campaign" - retiring the wrong one because a caller forgot a
    parameter is exactly the mistake worth designing out. FastAPI rejects the
    missing query parameter with a 422."""
    store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    assert client.post("/api/campaign/archive").status_code == 422
    assert client.post("/api/campaign/unarchive").status_code == 422


def test_writes_against_an_archived_campaign_are_refused(store, client):
    """An archived campaign is a closed record: it accepts no new pending run,
    no result, and no re-analyze until it is unarchived. Refusing here is what
    stops an archived campaign from silently advancing rounds while invisible
    in the rail."""
    cid = store.seed(_tiny_df(), "lipase_titer", ["Methanol", "pH"])
    started = client.post(
        "/api/campaign/start", params={"campaign_id": cid},
        json={"recipes": [{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
                           "mode": "explore", "reason": "before archiving"}]},
    ).json()["started"]
    client.post("/api/campaign/archive", params={"campaign_id": cid})

    start_resp = client.post(
        "/api/campaign/start", params={"campaign_id": cid},
        json={"recipes": [{"recipe": {"Methanol": 3.0, "pH": 6.5}, "pred": 6.0, "std": 0.3,
                           "mode": "exploit", "reason": "after archiving"}]},
    )
    assert start_resp.status_code == 400
    assert "archived" in start_resp.json()["error"]

    result_resp = client.post(
        "/api/campaign/result", params={"campaign_id": cid},
        json={"id": started[0]["id"], "value": 6.0},
    )
    assert result_resp.status_code == 400
    assert "archived" in result_resp.json()["error"]

    reanalyze_resp = client.post("/api/campaign/reanalyze", params={"campaign_id": cid})
    assert reanalyze_resp.status_code == 400
    assert "archived" in reanalyze_resp.json()["error"]

    # nothing landed: still exactly the one run started before archiving
    body = client.get("/api/campaign", params={"campaign_id": cid}).json()
    assert [p["id"] for p in body["pending"]] == [started[0]["id"]]
    assert body["n_measured"] == 0
    assert body["round"] == 0


# --- Archive / replace racing an in-flight reanalyze ------------------------ #

def test_archive_landing_mid_reanalyze_aborts_the_commit(store):
    """An archive is a write like any other: it mints a fresh `generation`, so
    a `commit_fold` planned before it refuses to land. The message is the
    ARCHIVED one rather than the generic "please re-analyze again", because
    the retry that wording invites would keep failing until unarchived."""
    df = _tiny_df()
    cid = store.seed(df, "lipase_titer", ["Methanol", "pH"])
    started = store.start([
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "measured"},
    ], campaign_id=cid)
    store.set_result(started[0]["id"], 6.0, campaign_id=cid)

    _df, _target, generation, resolved = store.plan_fold(campaign_id=cid)

    # the archive lands here, while _analyze is still running
    store.archive(cid)

    with pytest.raises(CampaignError, match="archived"):
        store.commit_fold(generation, campaign_id=resolved)

    # nothing folded, nothing lost - the measured run is still exactly where it was
    state = store.get(campaign_id=cid)
    assert state is not None
    assert state["round"] == 0
    assert len(state["base_rows"]) == len(df)
    assert len(state["pending"]) == 1
    assert state["pending"][0]["result"] == 6.0

    # and once unarchived the same fold completes normally
    store.unarchive(cid)
    _df2, _t2, generation2, resolved2 = store.plan_fold(campaign_id=cid)
    committed = store.commit_fold(generation2, campaign_id=resolved2)
    assert committed["round"] == 1
    assert len(committed["base_rows"]) == len(df) + 1


def test_reanalyze_racing_an_archive_returns_409_and_leaves_latest_untouched(store, client, monkeypatch):
    """The same race through the real route: an archive lands while `_analyze`
    is in flight. The commit is refused, the campaign is untouched, and
    `/api/latest` is never written with an analysis of a campaign that was put
    away mid-flight."""
    df = _tiny_df()
    cid = store.seed(df, "lipase_titer", ["Methanol", "pH"])
    started = client.post(
        "/api/campaign/start", params={"campaign_id": cid},
        json={"recipes": [{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
                           "mode": "explore", "reason": "measured"}]},
    ).json()["started"]
    client.post(
        "/api/campaign/result", params={"campaign_id": cid},
        json={"id": started[0]["id"], "value": 6.0},
    )

    real_analyze = portal_module._analyze

    def _analyze_and_archive(*args, **kwargs):
        store.archive(cid)
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr(portal_module, "_analyze", _analyze_and_archive)

    resp = client.post("/api/campaign/reanalyze", params={"campaign_id": cid})
    assert resp.status_code == 409
    assert "archived" in resp.json()["error"]
    # restore ONLY `_analyze` - see the note in
    # `test_reanalyze_analyze_failure_does_not_advance_round_or_fold` on why a
    # blanket `monkeypatch.undo()` here would un-isolate the whole test
    monkeypatch.setattr(portal_module, "_analyze", real_analyze)

    body = client.get("/api/campaign", params={"campaign_id": cid}).json()
    assert body["round"] == 0
    assert body["n_base"] == len(df)
    assert body["n_measured"] == 1
    assert client.get("/api/latest").json() == {"has_data": False}

    # unarchiving makes the retry meaningful, exactly as the error says
    client.post("/api/campaign/unarchive", params={"campaign_id": cid})
    retry = client.post("/api/campaign/reanalyze", params={"campaign_id": cid})
    assert retry.status_code == 200, retry.text
    assert retry.json()["campaign"]["round"] == 1


def test_replace_and_reanalyze_are_mutually_exclusive_by_construction(store, client, monkeypatch):
    """A replace can never corrupt an in-flight reanalyze, and not by timing:
    `plan_fold` only succeeds when at least one run is MEASURED, and a
    measured run is exactly what makes `replace` refuse. So the replace
    attempted mid-analysis is rejected, and the reanalyze it raced commits
    untouched."""
    df = _tiny_df()
    cid = store.seed(df, "lipase_titer", ["Methanol", "pH"], name="original.csv")
    started = client.post(
        "/api/campaign/start", params={"campaign_id": cid},
        json={"recipes": [{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
                           "mode": "explore", "reason": "measured"}]},
    ).json()["started"]
    client.post(
        "/api/campaign/result", params={"campaign_id": cid},
        json={"id": started[0]["id"], "value": 6.0},
    )

    real_analyze = portal_module._analyze
    attempted: list[str] = []

    def _analyze_and_replace(*args, **kwargs):
        # a corrected-CSV replace lands while THIS reanalyze's _analyze runs
        try:
            store.replace(cid, _tiny_df(n=11, seed=9), "lipase_titer", ["Methanol", "pH"],
                          name="corrected.csv")
        except CampaignError as exc:
            attempted.append(str(exc))
        return real_analyze(*args, **kwargs)

    monkeypatch.setattr(portal_module, "_analyze", _analyze_and_replace)

    resp = client.post("/api/campaign/reanalyze", params={"campaign_id": cid})
    assert resp.status_code == 200, resp.text
    assert len(attempted) == 1 and "measured" in attempted[0]

    campaign = resp.json()["campaign"]
    assert campaign["round"] == 1
    assert campaign["n_base"] == len(df) + 1
    assert campaign["name"] == "original.csv"  # the refused replace changed nothing
    assert campaign["revisions"] == []
