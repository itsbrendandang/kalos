"""The kalos home sandbox itself (tests/conftest.py): the suite must never be
able to read or write the developer's real `~/.kalos`.

A guard that has never been shown to fire is not a guard, so these tests
deliberately aim writes at the real home and assert they are blocked and
recorded. Each such test drains the recorded violations, otherwise the autouse
teardown check would (correctly) fail it.
"""
from __future__ import annotations

import pathlib
import sqlite3
import threading
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # imported for typing only: pytest owns the single conftest module
    from conftest import HomeGuard


def test_home_resolves_into_the_sandbox(home_guard: "HomeGuard") -> None:
    """Every kalos entry point funnels through `Path.home()`; it must not be
    the real home for the duration of the session."""
    assert pathlib.Path.home() == home_guard.sandbox
    assert pathlib.Path.home() != home_guard.real_home
    assert home_guard.sandbox.is_dir()


def test_default_state_paths_are_inside_the_sandbox(home_guard: "HomeGuard") -> None:
    """The four import-time defaults - portal `_STATE_DIR`/`_LATEST_DIR`, the
    experiments db, the runner lock - and the runtime `CampaignStore` default
    must all have bound to the sandbox, not to the real home."""
    pytest.importorskip("fastapi")
    from kalos.portal import app as portal_module
    from kalos.runner.singleton import DEFAULT_LOCK_PATH
    from kalos.store.sqlite_store import DEFAULT_DB_PATH

    for path in (
        portal_module._STATE_DIR,
        portal_module._LATEST_DIR,
        DEFAULT_DB_PATH,
        DEFAULT_LOCK_PATH,
    ):
        assert home_guard.sandbox in pathlib.Path(path).parents or path == home_guard.sandbox, path
        assert home_guard.real_kalos not in pathlib.Path(path).parents


def test_write_to_real_home_is_blocked(home_guard: "HomeGuard") -> None:
    """A plain file write aimed at the real `~/.kalos` raises and is recorded."""
    target = home_guard.real_kalos / "guard-probe.json"
    with pytest.raises(home_guard.violation):
        target.write_text("this must never reach disk")
    assert not target.exists()
    assert home_guard.pop_violations(), "the violation was not recorded"


def test_sqlite_connect_to_real_home_is_blocked(home_guard: "HomeGuard") -> None:
    """sqlite opens its file in C, below every Python-level file wrapper, so
    `sqlite3.connect` is guarded on its own - this is the exact call that wrote
    campaigns into the real `~/.kalos/portal.db`."""
    with pytest.raises(home_guard.violation):
        sqlite3.connect(str(home_guard.real_kalos / "portal.db"))
    assert home_guard.pop_violations()


def test_mkdir_of_real_home_is_blocked(home_guard: "HomeGuard") -> None:
    """`CampaignStore.__init__` mkdir's its state dir before opening the db."""
    with pytest.raises(home_guard.violation):
        (home_guard.real_kalos / "nested").mkdir(parents=True, exist_ok=True)
    assert home_guard.pop_violations()


def test_guard_survives_a_leaked_monkeypatch_undo(
    home_guard: "HomeGuard", monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression this whole sandbox exists for.

    `monkeypatch.undo()` reverts a test's own patches - including, when the
    same function-scoped instance is shared with a fixture, the fixture's
    patches. The sandbox is installed as session state at conftest import, so
    an undo can neither restore the real home nor remove the guard.
    """
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: home_guard.real_home))
    assert pathlib.Path.home() == home_guard.real_home
    monkeypatch.undo()

    assert pathlib.Path.home() == home_guard.sandbox
    with pytest.raises(home_guard.violation):
        sqlite3.connect(str(home_guard.real_kalos / "portal.db"))
    assert home_guard.pop_violations()


def test_violation_on_a_worker_thread_is_still_recorded(home_guard: "HomeGuard") -> None:
    """`_analyze` runs in a worker thread, where a raise cannot fail the test.
    The recorded violation is what makes the autouse teardown fail it."""

    def _write() -> None:
        try:
            (home_guard.real_kalos / "latest_analysis.json").write_text("{}")
        except BaseException:  # noqa: BLE001 - a thread swallowing it is the point
            pass

    worker = threading.Thread(target=_write)
    worker.start()
    worker.join()

    assert home_guard.pop_violations(), "a thread's violation must still be recorded"
