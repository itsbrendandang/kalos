"""The campaign store repairs an incompatible `campaigns` table instead of
failing every write.

`CREATE TABLE IF NOT EXISTS` is not a migration: it matches on table NAME and
ignores shape. When the shape is wrong, `_write_locked`'s
`ON CONFLICT(tenant) DO UPDATE` raises `sqlite3.OperationalError`, `/api/run`
catches and logs it, and the portal goes on looking healthy while persisting no
campaign at all - forever, because nothing repairs the table.

This is a regression test for an observed state, not a hypothetical one. A
developer database carried `PRIMARY KEY (tenant, campaign_id)` from an unmerged
multi-campaign branch, so every campaign write on `main` silently failed.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from kalos.portal.campaign import CampaignStore, _campaigns_table_is_compatible

# The exact shape found in the wild: the multi-campaign schema from an unmerged
# branch, whose composite key cannot satisfy ON CONFLICT(tenant).
_LEGACY_MULTI_CAMPAIGN = """
CREATE TABLE campaigns (
    tenant TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    state TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (tenant, campaign_id)
)
"""


def _sheet(n: int = 12) -> pd.DataFrame:
    """A minimal sheet the store can seed a campaign from."""
    return pd.DataFrame(
        {
            "feed_mL_h": [0.2 + 0.05 * i for i in range(n)],
            "titer_g_L": [1.0 + 0.2 * i for i in range(n)],
        }
    )


def _tables(path) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {
            str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()


# --- compatibility detection ----------------------------------------------- #


def test_composite_primary_key_is_detected_as_incompatible(tmp_path):
    """The specific trap: `tenant` is IN the key but is not the sole key, so
    ON CONFLICT(tenant) cannot bind to it."""
    conn = sqlite3.connect(str(tmp_path / "probe.db"))
    conn.execute(_LEGACY_MULTI_CAMPAIGN)
    assert _campaigns_table_is_compatible(conn) is False
    conn.close()


def test_absent_table_is_incompatible(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "probe.db"))
    assert _campaigns_table_is_compatible(conn) is False
    conn.close()


def test_correct_table_is_compatible(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "probe.db"))
    conn.execute(
        "CREATE TABLE campaigns (tenant TEXT PRIMARY KEY, state TEXT NOT NULL, "
        "updated_at REAL NOT NULL)"
    )
    assert _campaigns_table_is_compatible(conn) is True
    conn.close()


# --- migration ------------------------------------------------------------- #


def test_legacy_table_is_migrated_and_writes_succeed(tmp_path, caplog):
    """The end-to-end repair: a legacy DB must not merely be detected, it must
    end up writable."""
    db = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_LEGACY_MULTI_CAMPAIGN)
    conn.execute(
        "INSERT INTO campaigns (tenant, campaign_id, state, updated_at) VALUES (?,?,?,?)",
        ("default", "c-old", '{"round": 1, "marker": "older"}', 100.0),
    )
    conn.execute(
        "INSERT INTO campaigns (tenant, campaign_id, state, updated_at) VALUES (?,?,?,?)",
        ("default", "c-new", '{"round": 7, "marker": "newest"}', 900.0),
    )
    conn.commit()
    conn.close()

    with caplog.at_level("WARNING"):
        store = CampaignStore(state_dir=tmp_path)

    # The operator is told, because a table was renamed under them.
    assert any("campaigns table was incompatible" in r.getMessage() for r in caplog.records)

    # The write that used to raise OperationalError now works.
    store.seed(_sheet(), target="titer_g_L", features=["feed_mL_h"])
    state = store.get()
    assert state is not None


def test_migration_keeps_the_newest_campaign_per_tenant(tmp_path):
    """Collapsing many campaigns per tenant into one must keep the campaign the
    single-campaign API would have been serving: the most recent."""
    db = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_LEGACY_MULTI_CAMPAIGN)
    rows = [
        ("default", "c-old", '{"marker": "older"}', 100.0),
        ("default", "c-new", '{"marker": "newest"}', 900.0),
        ("acme", "c-a", '{"marker": "acme-newest"}', 500.0),
        ("acme", "c-b", '{"marker": "acme-older"}', 400.0),
    ]
    conn.executemany(
        "INSERT INTO campaigns (tenant, campaign_id, state, updated_at) VALUES (?,?,?,?)", rows
    )
    conn.commit()
    conn.close()

    CampaignStore(state_dir=tmp_path)

    conn = sqlite3.connect(str(db))
    migrated = dict(conn.execute("SELECT tenant, state FROM campaigns").fetchall())
    conn.close()
    assert "newest" in migrated["default"]
    assert "acme-newest" in migrated["acme"]
    # one row per tenant now, keyed by tenant alone
    assert len(migrated) == 2


def test_migration_is_non_destructive(tmp_path):
    """Nothing is dropped: the original rows stay readable in a backup table."""
    db = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_LEGACY_MULTI_CAMPAIGN)
    conn.execute(
        "INSERT INTO campaigns (tenant, campaign_id, state, updated_at) VALUES (?,?,?,?)",
        ("default", "c-1", '{"marker": "keep-me"}', 100.0),
    )
    conn.commit()
    conn.close()

    CampaignStore(state_dir=tmp_path)

    assert "campaigns_backup_1" in _tables(db)
    conn = sqlite3.connect(str(db))
    preserved = conn.execute(
        "SELECT campaign_id, state FROM campaigns_backup_1"
    ).fetchall()
    conn.close()
    assert preserved == [("c-1", '{"marker": "keep-me"}')]


def test_repeated_migration_does_not_clobber_an_earlier_backup(tmp_path):
    """A second incompatible table must land beside the first backup, not on it."""
    db = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_LEGACY_MULTI_CAMPAIGN)
    conn.commit()
    conn.close()

    CampaignStore(state_dir=tmp_path)  # -> campaigns_backup_1

    # Put an incompatible table back, as a downgrade/upgrade ping-pong would.
    conn = sqlite3.connect(str(db))
    conn.execute("DROP TABLE campaigns")
    conn.execute(_LEGACY_MULTI_CAMPAIGN)
    conn.commit()
    conn.close()

    CampaignStore(state_dir=tmp_path)  # -> campaigns_backup_2

    tables = _tables(db)
    assert {"campaigns", "campaigns_backup_1", "campaigns_backup_2"} <= tables


def test_unsalvageable_table_still_yields_a_working_store(tmp_path, caplog):
    """A table sharing only the name, with none of the needed columns, must not
    block startup - back it up, start clean, and say so."""
    db = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE campaigns (something_else TEXT)")
    conn.execute("INSERT INTO campaigns (something_else) VALUES ('junk')")
    conn.commit()
    conn.close()

    with caplog.at_level("WARNING"):
        store = CampaignStore(state_dir=tmp_path)
    assert any("no state carried over" in r.getMessage() for r in caplog.records)

    store.seed(_sheet(), target="titer_g_L", features=["feed_mL_h"])
    assert store.get() is not None
    assert "campaigns_backup_1" in _tables(db)


# --- the happy paths must not regress ------------------------------------- #


def test_fresh_database_needs_no_migration(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        store = CampaignStore(state_dir=tmp_path)
    assert not any("incompatible" in r.getMessage() for r in caplog.records)
    store.seed(_sheet(), target="titer_g_L", features=["feed_mL_h"])
    assert store.get() is not None
    assert "campaigns_backup_1" not in _tables(tmp_path / "portal.db")


def test_reopening_a_correct_database_is_a_no_op(tmp_path, caplog):
    """Opening the store twice must not migrate anything the second time, or a
    long-lived process would churn backups on every restart."""
    first = CampaignStore(state_dir=tmp_path)
    first.seed(_sheet(), target="titer_g_L", features=["feed_mL_h"])

    with caplog.at_level("WARNING"):
        second = CampaignStore(state_dir=tmp_path)
    assert not any("incompatible" in r.getMessage() for r in caplog.records)
    assert second.get() is not None
    assert "campaigns_backup_1" not in _tables(tmp_path / "portal.db")


@pytest.mark.parametrize("tenant", ["default", "acme"])
def test_write_read_roundtrip_after_migration(tmp_path, tenant):
    db = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute(_LEGACY_MULTI_CAMPAIGN)
    conn.commit()
    conn.close()

    store = CampaignStore(state_dir=tmp_path)
    store.seed(_sheet(), target="titer_g_L", features=["feed_mL_h"], tenant=tenant)
    assert store.get(tenant=tenant) is not None
