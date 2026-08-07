"""Portal boundary guards.

1. `kalos.portal` must never import `kalos.normalize` - normalize's payload
   pipeline (`kalos/normalize/payload.py`) deliberately strips identifying
   metadata for an LLM-assisted normalization step and is not (yet) wired into
   any live portal route. A portal module reaching for it would be a silent
   step toward shipping that path to real clients before it is ready.
2. `_LATEST_LOCK` (`kalos.portal.app`) must actually serialize concurrent
   `/api/run` uploads - `_load_latest`/`_save_latest` read-modify-write the
   `_LATEST` global and the on-disk cache, so two overlapping uploads without
   the lock could interleave a half-written state.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest

pytest.importorskip("fastapi")


def test_portal_never_imports_normalize():
    import kalos.portal as portal_pkg

    portal_dir = Path(portal_pkg.__file__).parent
    import_re = re.compile(r"^\s*(?:from|import)\s+(?:\.*\s*)?kalos\.normalize\b|^\s*from\s+\.+\s*normalize\b", re.M)
    offenders = []
    for py_file in portal_dir.rglob("*.py"):
        text = py_file.read_text()
        if import_re.search(text) or re.search(r"\bimport\s+.*normalize\b", text):
            offenders.append(py_file.relative_to(portal_dir))
    assert not offenders, (
        f"kalos.portal must never import kalos.normalize (found in: {offenders}); "
        "the identity-stripping payload pipeline must not be wired into a live portal route"
    )


def test_latest_lock_serializes_concurrent_saves(tmp_path, monkeypatch):
    from kalos.portal import app as portal

    monkeypatch.setattr(portal, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal, "_LATEST_PATH", tmp_path / "latest.json")
    monkeypatch.setattr(portal, "_LATEST", None)

    entered = threading.Event()
    release = threading.Event()

    def holder() -> None:
        # Simulate one upload mid-`_save_latest`/`_load_latest`, holding the
        # exact lock those functions take, for as long as `release` is unset.
        with portal._LATEST_LOCK:
            entered.set()
            release.wait(timeout=2)

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert entered.wait(timeout=2), "holder thread never acquired _LATEST_LOCK"

    saved = threading.Event()

    def saver() -> None:
        portal._save_latest({"target": "lipase_titer"}, "concurrent-runs.csv")
        saved.set()

    saver_thread = threading.Thread(target=saver)
    saver_thread.start()
    try:
        # A second, concurrent save must block behind the held lock rather
        # than racing it - give it ample time to (wrongly) finish if the lock
        # were not actually serializing the two calls.
        saver_thread.join(timeout=0.3)
        assert not saved.is_set(), (
            "_save_latest completed while _LATEST_LOCK was held elsewhere - "
            "concurrent uploads are not actually serialized"
        )
    finally:
        release.set()

    holder_thread.join(timeout=2)
    saver_thread.join(timeout=2)
    assert saved.is_set()
    assert portal._LATEST["dataset"] == "concurrent-runs.csv"
