"""The Experiment record and its status lifecycle.

`Experiment` is the stable JSON contract that crosses the `BackendAdapter` seam
(see `docs/M2_INTEGRATION.md`). Both the local SQLite store and any future HTTP
backend serialize to exactly this shape via `to_dict()` / `from_dict()`.

The status lifecycle is encoded once, as a pure predicate (`legal_transition`),
so the store, the adapters, and the Singleton runner all enforce the same rule
instead of re-deriving it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    """The Experiment status flag. `READY` is the "ready to process" flag from
    the Voyager ML plan; the rest of the lifecycle wraps the Singleton run."""

    DRAFT = "DRAFT"
    READY = "READY"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


# Legal forward transitions, keyed by the CURRENT status. `DONE -> READY` and
# `PROCESSING -> READY` are deliberately absent here: both are only legal with
# `force=True`, so `legal_transition` special-cases them below rather than
# listing them as unconditional edges.
_LEGAL_TRANSITIONS: dict[Status, frozenset[Status]] = {
    Status.DRAFT: frozenset({Status.READY}),
    Status.READY: frozenset({Status.PROCESSING}),
    Status.PROCESSING: frozenset({Status.DONE, Status.FAILED}),
    Status.FAILED: frozenset({Status.READY}),
    Status.DONE: frozenset(),
}


def legal_transition(old: Status, new: Status, *, force: bool = False) -> bool:
    """Is `old -> new` a legal status transition?

    Encodes exactly the lifecycle in `docs/M2_INTEGRATION.md`:
    `DRAFT -> READY -> PROCESSING -> (DONE | FAILED)`; `FAILED -> READY`
    (retry); `DONE -> READY` only with `force` (re-run, discards prior
    result). A same-status "transition" is never legal (including under
    `force`) - callers that want idempotent no-ops must check that themselves.

    `PROCESSING -> READY` is ALSO only legal with `force` - it exists purely
    for orphan recovery (`kalos.runner.singleton.reclaim_stale`): the
    Singleton lock guarantees single-instance execution, so a runner that
    just acquired the lock and finds a `PROCESSING` row knows it is an orphan
    from a run that crashed mid-analysis, never a live one. This is a
    store-level legality only - the client-facing `PATCH
    /api/experiments/{id}` endpoint (`kalos/portal/app.py`) rejects ANY
    request whose current status is `PROCESSING` regardless of `force`, so a
    client can never reach this edge through the API.
    """
    if old == new:
        return False
    if force and new == Status.READY and old in (Status.DONE, Status.PROCESSING):
        return True
    return new in _LEGAL_TRANSITIONS.get(old, frozenset())


@dataclass
class Experiment:
    """Matches the Experiment JSON contract in `docs/M2_INTEGRATION.md` field
    for field. `config` and `payload` are free-form JSON-serializable dicts
    (the config knobs and the run-sheet payload respectively); `result` and
    `provenance` are populated on `DONE`; `error` is populated on `FAILED`."""

    id: str
    name: str
    status: Status
    created_at: str
    updated_at: str
    config: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to exactly the documented JSON shape."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": self.config,
            "payload": self.payload,
            "result": self.result,
            "provenance": self.provenance,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experiment":
        """Round-trip the documented JSON shape back into an `Experiment`."""
        return cls(
            id=data["id"],
            name=data["name"],
            status=Status(data["status"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            config=data.get("config") or {},
            payload=data.get("payload") or {},
            result=data.get("result"),
            provenance=data.get("provenance"),
            error=data.get("error"),
        )
