"""SQLite-backed experiment store.

Default location: `~/.kalos/experiments.db`. The path is a constructor
argument specifically so tests can point at a `tmp_path` and never touch the
real user database. `payload` / `config` / `result` / `provenance` are stored
as JSON text columns; every mutation runs inside its own single-statement
transaction (one connection per call, opened and closed within the method),
which is transaction-safe enough for the lock'd, single-instance Singleton
this store backs - there is never more than one writer at a time by design.
"""
from __future__ import annotations

import contextlib
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Experiment, Status, legal_transition

DEFAULT_DB_PATH = Path.home() / ".kalos" / "experiments.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    config TEXT NOT NULL,
    payload TEXT NOT NULL,
    result TEXT,
    provenance TEXT,
    error TEXT
)
"""
_TENANT_INDEX = "CREATE INDEX IF NOT EXISTS idx_experiments_tenant ON experiments(tenant)"


class ExperimentNotFound(KeyError):
    """No experiment with the given id exists in the store."""


class IllegalTransition(ValueError):
    """A status transition was requested that `legal_transition` rejects."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_sanitize(value: Any) -> Any:
    """Recursively replace non-finite floats (NaN/+inf/-inf) with `None`.

    `json.dumps` defaults to `allow_nan=True`, which happily writes the
    literal tokens `NaN`/`Infinity`/`-Infinity` into a TEXT column - those are
    NOT valid JSON (no `json.loads` implementation, strict or otherwise,
    accepts them per the spec), so a downstream reader doing plain
    `json.loads` on the stored text breaks. Applied recursively over
    dict/list/tuple so a NaN buried anywhere in `payload`/`config`/`result`/
    `provenance` - e.g. `result["reliability"]["ci95"] == [nan, nan]` - is
    caught, not just a top-level float.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    return value


def _dumps(value: Any) -> str:
    """`json.dumps` over a `_json_sanitize`d copy - the ONE place every JSON
    column write goes through, so `experiments.db` always holds strict,
    valid JSON independent of whatever the API layer already sanitized."""
    return json.dumps(_json_sanitize(value))


def _row_to_experiment(row: sqlite3.Row) -> Experiment:
    return Experiment(
        id=row["id"],
        name=row["name"],
        status=Status(row["status"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        config=json.loads(row["config"]),
        payload=json.loads(row["payload"]),
        result=json.loads(row["result"]) if row["result"] is not None else None,
        provenance=json.loads(row["provenance"]) if row["provenance"] is not None else None,
        error=row["error"],
    )


class SqliteStore:
    """CRUD + lifecycle mutations over the `experiments` table."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(self._connect()) as conn:
            with conn:
                conn.execute(_SCHEMA)
                self._migrate_tenant(conn)
                conn.execute(_TENANT_INDEX)

    @staticmethod
    def _migrate_tenant(conn: sqlite3.Connection) -> None:
        """Backfill the `tenant` column on a pre-multi-tenant database.

        A `experiments.db` created before Phase 1b has no `tenant` column;
        `ADD COLUMN ... DEFAULT 'default'` adds it and backfills every existing
        row to the `default` tenant (docs/HARDENING.md, Phase 1b). No-op once the
        column exists, so it is safe to run on every startup."""
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(experiments)")}
        if "tenant" not in cols:
            conn.execute("ALTER TABLE experiments ADD COLUMN tenant TEXT NOT NULL DEFAULT 'default'")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.row_factory = sqlite3.Row
        return conn

    def ping(self) -> None:
        """Readiness probe: a trivial query proving the DB is reachable
        (docs/HARDENING.md, Phase 2). Raises on failure."""
        with contextlib.closing(self._connect()) as conn:
            conn.execute("SELECT 1").fetchone()

    # --- CRUD ------------------------------------------------------------ #

    def create(
        self, name: str, payload: dict[str, Any], config: dict[str, Any], *, tenant: str = "default"
    ) -> Experiment:
        """Create a new experiment in `DRAFT`, owned by `tenant`."""
        now = _now_iso()
        exp = Experiment(
            id=f"exp_{uuid.uuid4()}",
            name=name,
            status=Status.DRAFT,
            created_at=now,
            updated_at=now,
            config=dict(config),
            payload=dict(payload),
        )
        with contextlib.closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    "INSERT INTO experiments "
                    "(id, tenant, name, status, created_at, updated_at, config, payload, result, provenance, error) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        exp.id, tenant, exp.name, exp.status.value, exp.created_at, exp.updated_at,
                        _dumps(exp.config), _dumps(exp.payload), None, None, None,
                    ),
                )
        return exp

    def get(self, exp_id: str, *, tenant: str = "default") -> Experiment:
        """Fetch one experiment owned by `tenant`. Raises `ExperimentNotFound` if
        no such experiment exists FOR THIS TENANT - a cross-tenant id is
        indistinguishable from a missing one, so tenants can never probe each
        other's ids."""
        with contextlib.closing(self._connect()) as conn:
            with conn:
                row = conn.execute(
                    "SELECT * FROM experiments WHERE id = ? AND tenant = ?", (exp_id, tenant)
                ).fetchone()
        if row is None:
            raise ExperimentNotFound(f"no experiment with id {exp_id!r}")
        return _row_to_experiment(row)

    def list(self, status: Status | None = None, *, tenant: str = "default") -> list[Experiment]:
        """List `tenant`'s experiments, optionally filtered to one status, oldest first."""
        with contextlib.closing(self._connect()) as conn:
            with conn:
                if status is None:
                    rows = conn.execute(
                        "SELECT * FROM experiments WHERE tenant = ? ORDER BY created_at", (tenant,)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM experiments WHERE tenant = ? AND status = ? ORDER BY created_at",
                        (tenant, status.value),
                    ).fetchall()
        return [_row_to_experiment(r) for r in rows]

    # --- lifecycle mutations ---------------------------------------------- #

    def set_status(
        self, exp_id: str, status: Status, *, force: bool = False, error: str | None = None,
        tenant: str = "default",
    ) -> Experiment:
        """Flip an experiment's status, enforcing `legal_transition`.

        Raises `IllegalTransition` (with the attempted old/new status in the
        message) if the move is not legal. A forced `DONE -> READY` discards
        the prior `result` / `provenance` / `error` (a fresh re-run). `error`
        is stored alongside a transition to `FAILED` (ignored otherwise). Any
        transition INTO `READY` clears a stale `error` left over from a prior
        `FAILED` run (`FAILED -> READY` retry) - contract: `error` is
        populated on `FAILED` only, so a retried experiment must not still
        carry the old, now-contradictory error message.
        """
        with contextlib.closing(self._connect()) as conn:
            with conn:
                row = conn.execute(
                    "SELECT * FROM experiments WHERE id = ? AND tenant = ?", (exp_id, tenant)
                ).fetchone()
                if row is None:
                    raise ExperimentNotFound(f"no experiment with id {exp_id!r}")
                current = Status(row["status"])
                if not legal_transition(current, status, force=force):
                    raise IllegalTransition(
                        f"illegal status transition for {exp_id!r}: "
                        f"{current.value} -> {status.value} (force={force})"
                    )
                now = _now_iso()
                reset_result = force and current == Status.DONE and status == Status.READY
                if reset_result:
                    conn.execute(
                        "UPDATE experiments SET status = ?, updated_at = ?, "
                        "result = NULL, provenance = NULL, error = NULL WHERE id = ? AND tenant = ?",
                        (status.value, now, exp_id, tenant),
                    )
                elif status == Status.FAILED:
                    conn.execute(
                        "UPDATE experiments SET status = ?, updated_at = ?, error = ? "
                        "WHERE id = ? AND tenant = ?",
                        (status.value, now, error, exp_id, tenant),
                    )
                elif status == Status.READY:
                    conn.execute(
                        "UPDATE experiments SET status = ?, updated_at = ?, error = NULL "
                        "WHERE id = ? AND tenant = ?",
                        (status.value, now, exp_id, tenant),
                    )
                else:
                    conn.execute(
                        "UPDATE experiments SET status = ?, updated_at = ? WHERE id = ? AND tenant = ?",
                        (status.value, now, exp_id, tenant),
                    )
                row = conn.execute(
                    "SELECT * FROM experiments WHERE id = ? AND tenant = ?", (exp_id, tenant)
                ).fetchone()
        return _row_to_experiment(row)

    def save_result(
        self, exp_id: str, result: dict[str, Any], provenance: dict[str, Any], *, tenant: str = "default"
    ) -> Experiment:
        """Push a processed result and mark `tenant`'s experiment `DONE`.

        Only legal from `PROCESSING` (via `legal_transition`); raises
        `IllegalTransition` otherwise, so a stray push cannot silently
        clobber a `DRAFT`/`READY`/`FAILED` experiment's result.
        """
        with contextlib.closing(self._connect()) as conn:
            with conn:
                row = conn.execute(
                    "SELECT * FROM experiments WHERE id = ? AND tenant = ?", (exp_id, tenant)
                ).fetchone()
                if row is None:
                    raise ExperimentNotFound(f"no experiment with id {exp_id!r}")
                current = Status(row["status"])
                if not legal_transition(current, Status.DONE):
                    raise IllegalTransition(
                        f"cannot save a result for {exp_id!r} from status {current.value} "
                        f"(must be {Status.PROCESSING.value})"
                    )
                now = _now_iso()
                conn.execute(
                    "UPDATE experiments SET status = ?, updated_at = ?, result = ?, provenance = ?, error = NULL "
                    "WHERE id = ? AND tenant = ?",
                    (Status.DONE.value, now, _dumps(result), _dumps(provenance), exp_id, tenant),
                )
                row = conn.execute(
                    "SELECT * FROM experiments WHERE id = ? AND tenant = ?", (exp_id, tenant)
                ).fetchone()
        return _row_to_experiment(row)
