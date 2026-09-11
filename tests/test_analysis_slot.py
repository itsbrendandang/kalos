"""The single analysis slot: one heavy fit at a time, honest 503 for the rest.

WHY. Found on the first real containerized run (2026-09-10): `_analyze` is
CPU-bound for tens of seconds, `run_in_threadpool` work cannot be cancelled,
and a client that gives up (a proxy timeout) orphans a thread that keeps
computing. With no admission control every retry contended with the ghosts of
its predecessors - measured as 598% engine CPU with zero connected clients.
The slot makes the failure mode impossible: a second analysis gets HTTP 503
with Retry-After, and the slot is released by the WORKER THREAD when the fit
actually finishes, so a disconnected client's orphan keeps holding it and the
503 stays truthful.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import busy  # noqa: E402
from kalos.portal.busy import AnalysisBusy, run_exclusively  # noqa: E402


def test_slot_serializes_and_releases_in_the_worker():
    """The unit contract: second acquire while held raises; release happens
    when the wrapped callable finishes, not before."""
    started = threading.Event()
    release = threading.Event()

    def slow() -> str:
        started.set()
        release.wait(timeout=5)
        return "done"

    locked = run_exclusively(slow)
    t = threading.Thread(target=locked)
    t.start()
    assert started.wait(timeout=5)
    with pytest.raises(AnalysisBusy):
        run_exclusively(lambda: "never")
    release.set()
    t.join(timeout=5)
    # slot free again once the worker actually finished
    run_exclusively(lambda: None)()


def test_slot_survives_worker_exception():
    """A failing analysis must free the slot - a crash that leaks the slot
    would turn every later request into a permanent 503."""
    def boom() -> None:
        raise RuntimeError("fit exploded")

    with pytest.raises(RuntimeError):
        run_exclusively(boom)()
    run_exclusively(lambda: None)()  # acquires cleanly


def _tiny_sheet() -> bytes:
    rng = np.random.default_rng(0)
    n = 12
    df = pd.DataFrame(
        {
            "Methanol": rng.uniform(0, 4, n).round(3),
            "pH": rng.uniform(5, 7, n).round(2),
            "lipase_titer": rng.uniform(1, 5, n).round(3),
        }
    )
    return df.to_csv(index=False).encode()


def test_api_run_returns_503_with_retry_after_while_slot_held(tmp_path, monkeypatch):
    """The endpoint contract, through the real route: while an analysis holds
    the slot, POST /api/run answers 503 + Retry-After + the JSON error
    envelope the frontend parses - it does not queue, block, or 500."""
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    from kalos.portal.app import app

    client = TestClient(app)
    started = threading.Event()
    release = threading.Event()

    def hold() -> None:
        started.set()
        release.wait(timeout=10)

    locked = run_exclusively(hold)
    t = threading.Thread(target=locked)
    t.start()
    try:
        assert started.wait(timeout=5)
        resp = client.post(
            "/api/run", files={"file": ("t.csv", _tiny_sheet(), "text/csv")}
        )
        assert resp.status_code == 503
        assert "Retry-After" in resp.headers
        assert int(resp.headers["Retry-After"]) == busy.RETRY_AFTER_SECONDS
        assert "already running" in resp.json()["error"]
    finally:
        release.set()
        t.join(timeout=10)

    # and once free, the same upload analyzes normally end to end
    resp = client.post("/api/run", files={"file": ("t.csv", _tiny_sheet(), "text/csv")})
    assert resp.status_code == 200
    assert "proposals" in resp.json()
