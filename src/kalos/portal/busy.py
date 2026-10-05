"""One analysis at a time: the portal's heavy-work admission gate.

WHY THIS EXISTS, from the first real containerized run (2026-09-10): a full
`_analyze` is CPU-bound for tens of seconds, `run_in_threadpool` cannot be
cancelled, and a client that gives up (a proxy timeout, a closed tab) leaves
an ORPHAN thread computing to nowhere. With no admission control, each retry
then contended with the ghosts of its predecessors - measured as 598% engine
CPU with zero connected clients, and every request slower than the proxy
timeout that spawned the previous orphan. A one-user thundering herd.

The gate is a single slot, matching the engine's own architecture (one
single-writer store, one Singleton runner, one uvicorn worker): a second
analysis arriving while one runs gets an honest HTTP 503 with Retry-After,
never a silent queue. The slot is released BY THE WORKER THREAD when the
analysis actually finishes - not when the request's await unwinds - so a
disconnected client's orphan keeps holding the slot and later requests are
told "busy" instead of being invited to pile on. That makes the 503 the
truthful description of the machine's state, which is the whole point.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

_ERR_BUSY = (
    "An analysis is already running. This engine runs one analysis at a time; "
    "retry after the current one finishes."
)
# Advisory client hint, roughly one containerized cold analyze (measured 47s
# on the 160-run fixture on the reference deployment).
RETRY_AFTER_SECONDS = 60

_slot = threading.BoundedSemaphore(1)


class AnalysisBusy(Exception):
    """The single analysis slot is held (possibly by an orphaned run)."""

    def __init__(self) -> None:
        super().__init__(_ERR_BUSY)


def run_exclusively(fn: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap `fn` for threadpool dispatch under the single analysis slot.

    Acquires non-blocking HERE (in the event loop, before any thread is
    spawned) so a busy engine answers instantly; raises `AnalysisBusy` when
    the slot is held. The returned callable releases the slot in ITS OWN
    finally - inside the worker thread - so cancellation of the awaiting
    request cannot free the slot while the computation is still running.
    """
    if not _slot.acquire(blocking=False):
        raise AnalysisBusy()

    def _locked() -> Any:
        try:
            return fn()
        finally:
            _slot.release()

    return _locked
