"""KALOS_STATE_DIR must be honored by every module that persists state, not
just kalos.portal.campaign.CampaignStore and kalos.portal.app's `_LATEST`
cache. Before this fix, kalos/store/sqlite_store.py's SqliteStore and
kalos/runner/singleton.py's SingletonLock hardcoded `Path.home() / ".kalos"`
and ignored the env var entirely - a split-brain where half the state
followed KALOS_STATE_DIR and half did not.

The real `~/.kalos` is never touched here (matching tests/test_m2_store.py
and tests/test_m2_singleton.py's convention): the "env unset" case is
checked at the path-resolution-function level only, never by actually
constructing a store/lock with no path and no env var set.
"""
from __future__ import annotations

import kalos.runner.singleton as singleton_module
import kalos.store.sqlite_store as sqlite_store_module
from kalos.runner.singleton import DEFAULT_LOCK_PATH, SingletonLock
from kalos.store.sqlite_store import DEFAULT_DB_PATH, SqliteStore


# --- env unset: identical to today, unchanged ------------------------------- #


def test_store_default_path_unset_env_matches_today(monkeypatch):
    monkeypatch.delenv("KALOS_STATE_DIR", raising=False)
    assert sqlite_store_module._default_db_path() == DEFAULT_DB_PATH == (
        singleton_module.Path.home() / ".kalos" / "experiments.db"
    )


def test_lock_default_path_unset_env_matches_today(monkeypatch):
    monkeypatch.delenv("KALOS_STATE_DIR", raising=False)
    assert singleton_module._default_lock_path() == DEFAULT_LOCK_PATH == (
        singleton_module.Path.home() / ".kalos" / "runner.lock"
    )


# --- env set: both the store and the lock land under it --------------------- #


def test_store_and_lock_both_land_under_kalos_state_dir_when_set(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))

    store = SqliteStore()
    lock = SingletonLock()

    assert store.path == tmp_path / "experiments.db"
    assert lock.path == tmp_path / "runner.lock"
    # Not just path arithmetic - the store actually created its db file there.
    assert store.path.exists()


def test_store_and_lock_functionally_work_under_the_env_dir(tmp_path, monkeypatch):
    """Not just that the path is computed correctly - the lock can actually be
    acquired and released, and the store can actually be written to, entirely
    under the env-var-provided directory."""
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))

    lock = SingletonLock()
    assert lock.acquire() is True
    assert lock.path.exists()
    lock.release()
    assert not lock.path.exists()

    store = SqliteStore()
    exp = store.create("demo", payload={"rows": []}, config={})
    assert store.get(exp.id).id == exp.id


# --- call-time, not import-time: a monkeypatched env needs no restart ------- #


def test_env_read_is_call_time_not_import_time(tmp_path, monkeypatch):
    """Matches kalos.portal.campaign.CampaignStore's convention: the env var is
    read fresh inside __init__ / the path-resolution helper, not baked into a
    module-level constant at import. Two constructions in the SAME process,
    with the env var changed in between via monkeypatch (no reimport, no
    process restart), must resolve to two different directories."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    monkeypatch.setenv("KALOS_STATE_DIR", str(first_dir))
    store_a = SqliteStore()
    lock_a = SingletonLock()
    assert store_a.path == first_dir / "experiments.db"
    assert lock_a.path == first_dir / "runner.lock"

    monkeypatch.setenv("KALOS_STATE_DIR", str(second_dir))
    store_b = SqliteStore()
    lock_b = SingletonLock()
    assert store_b.path == second_dir / "experiments.db"
    assert lock_b.path == second_dir / "runner.lock"


def test_explicit_path_still_overrides_the_env_var(tmp_path, monkeypatch):
    """An explicit constructor argument must still win over KALOS_STATE_DIR -
    unchanged contract, used throughout the test suite to point at a
    `tmp_path` regardless of the environment."""
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path / "ignored"))
    explicit_db = tmp_path / "explicit" / "experiments.db"
    explicit_lock = tmp_path / "explicit" / "runner.lock"

    store = SqliteStore(explicit_db)
    lock = SingletonLock(explicit_lock)

    assert store.path == explicit_db
    assert lock.path == explicit_lock
