"""Kalos portal — the campaign store: many named, closed optimization loops
per tenant.

See docs/CAMPAIGN_LOOP.md for the full contract this module implements. A
*campaign* is one optimization target plus a growing base dataset
(`base_rows`) and a list of `pending` runs — recipes started from a proposed
batch, each either *awaiting* a measured outcome (`result: null`) or
*measured* (`result` filled), waiting to be folded into `base_rows` on the
next re-analyze. Each campaign carries a stable `id` and a human `name`, and a
tenant may hold many of them at once (docs/CAMPAIGN_LOOP.md, "Campaign
identity").

Deliberately kept torch-free: this module only persists data. `_analyze`
(`kalos.portal.analysis`) is the science; `kalos.portal.campaign_routes`
calls it, not this store, so importing `kalos.portal.campaign` never pays the
torch/botorch import tax and the store stays trivially unit-testable.

Persistence is a SQLite table `campaigns(tenant, campaign_id, state,
updated_at)` in `<KALOS_STATE_DIR>/portal.db` (default `~/.kalos`), **one row
per (tenant, campaign_id)** — a tenant may own many campaigns
(docs/CAMPAIGN_LOOP.md, "Campaign identity"; supersedes the one-row-per-tenant
layout of docs/HARDENING.md Phase 1b). Every method takes a `tenant`
(defaulting to `"default"`, the open-mode tenant), so two tenants can never
see or overwrite each other's campaigns. A single lock serializes
read-modify-write cycles, and each write runs in a SQLite transaction, so a
reader never observes a partial state. The generation-token check that makes
re-analyze transactional is unchanged in spirit — it still lives inside each
campaign's own `state` blob, now scoped to that one (tenant, campaign_id) row
instead of the tenant's sole row.

A campaign is never hard-deleted. Two lifecycle operations bound the growth a
tenant's campaign list would otherwise have (docs/CAMPAIGN_LOOP.md,
"Archiving" and "Replacing a campaign's data"):

  - `archive()` / `unarchive()` flip an `archived` flag inside the state blob.
    An archived campaign disappears from `list_campaigns()` and from EVERY
    default resolution, but stays fully readable by explicit id and refuses
    further writes until unarchived. No row is dropped.
  - `replace()` swaps one campaign's uploaded base dataset for a corrected
    one, in place, keeping the campaign's `id`. It REFUSES rather than
    discard anything a scientist measured in the lab.

A store opened against a `portal.db` written before campaign identity existed
(the old `campaigns(tenant PRIMARY KEY, state, updated_at)` layout) migrates
it forward once, at `__init__` — see `_migrate_legacy_schema`.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("kalos.portal")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS campaigns (
    tenant TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    state TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (tenant, campaign_id)
)
"""

_DEFAULT_NAME = "Untitled campaign"


class CampaignError(ValueError):
    """A client-facing campaign operation error: unknown run id, non-finite
    result value, unknown campaign id, no active campaign, a write against an
    archived campaign, or a refused `replace`. The message is authored here
    (never a raw parser/library error), so it is always safe for a route
    handler to echo back to the caller as-is."""


def _finite_number(value: Any) -> float | None:
    """`value` as a float if it is a real, finite number (excluding bool,
    which is technically an `int` subclass but never a measured outcome),
    else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _best_measured(base_rows: list[dict[str, Any]], target: str) -> float | None:
    """The best (max) MEASURED target value over `base_rows`, or None when no
    row carries a finite numeric target — never a prediction (docs/CAMPAIGN_LOOP.md,
    "Honesty constraints"). Because base rows only ever grow, this is
    non-decreasing round over round, which is what the progress trajectory plots."""
    values = [v for row in base_rows if (v := _finite_number(row.get(target))) is not None]
    return max(values) if values else None


def _is_archived(state: dict[str, Any]) -> bool:
    """Whether a campaign state is archived (docs/CAMPAIGN_LOOP.md,
    "Archiving").

    A missing key reads as ACTIVE, which is the only correct default: every
    campaign written before archiving existed - including every row the legacy
    migration forward-ports - has no `archived` key at all, and silently
    treating those as archived would make a tenant's whole history vanish from
    the left rail on upgrade.
    """
    return bool(state.get("archived", False))


def _measured_count(state: dict[str, Any]) -> int:
    """How many measured lab outcomes a campaign already holds: pending runs
    carrying a real `result`, plus base rows already folded in from one
    (`base_row_meta` entries that are not None).

    This is the quantity `replace()` refuses on - it is exactly "the data a
    scientist logged from the lab", as opposed to the uploaded run sheet,
    which is the only thing a replace is allowed to correct.
    """
    pending = sum(1 for run in state.get("pending", []) if run.get("result") is not None)
    folded = sum(1 for meta in (state.get("base_row_meta") or []) if meta is not None)
    return pending + folded


def _reject_if_archived(state: dict[str, Any], campaign_id: str, action: str) -> None:
    """Raise `CampaignError` if `state` belongs to an archived campaign.

    Every WRITE path calls this. Archiving is not just a display filter: an
    archived campaign is a closed record, so it accepts no new pending runs,
    no new results, no fold, and no replace until it is explicitly unarchived
    (docs/CAMPAIGN_LOOP.md, "Archiving"). Reads are deliberately NOT gated -
    the whole point of archiving instead of deleting is that the data stays
    queryable.
    """
    if _is_archived(state):
        raise CampaignError(f"campaign {campaign_id!r} is archived; unarchive it before {action}")


def _day_bucket(ts: float) -> str:
    """The UTC calendar day of a unix timestamp, as an ISO date string
    (`YYYY-MM-DD`).

    Every timestamp this module buckets (`created_at`/`measured_at`) is a
    unix float with no recorded timezone, so a client-local bucketing would
    make the exact same data disagree across two viewers' machines. UTC is
    picked and fixed here, once, so the activity series is reproducible
    regardless of where it is read (docs/CAMPAIGN_LOOP.md, "Activity series").
    """
    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()


def _activity_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """The `{days, stats, note}` activity payload for one campaign's state.

    A "run" here is any pending run OR folded base row that carries a real
    `measured_at` — i.e. every recipe that was actually measured, whether or
    not it has since been folded into `base_rows` by a re-analyze. Awaiting
    (never-measured) runs are not counted.

    KNOWN LIMITATION (docs/CAMPAIGN_LOOP.md, "Activity series"): the
    originally uploaded base dataset (`seed()`) has no per-row timestamps —
    only rows that passed through the campaign loop
    (`POST /api/campaign/start` -> measured -> folded) carry a `measured_at`.
    This series is therefore honest but partial: it covers campaign-loop
    activity only, never the original upload. The `note` field says so
    explicitly, so a client can caption the heatmap truthfully.
    """
    measured_ats: list[float] = []
    for run in state.get("pending", []):
        ts = run.get("measured_at")
        if ts is not None:
            measured_ats.append(ts)
    for meta in state.get("base_row_meta") or []:
        if meta is not None and meta.get("measured_at") is not None:
            measured_ats.append(meta["measured_at"])

    counts: dict[str, int] = {}
    for ts in measured_ats:
        day = _day_bucket(ts)
        counts[day] = counts.get(day, 0) + 1
    days = [{"date": d, "runs": n} for d, n in sorted(counts.items())]

    if not days:
        stats = {"days_running": 0, "days_with_runs": 0, "longest_pause_days": 0}
    else:
        dates = [date.fromisoformat(str(d["date"])) for d in days]
        longest_pause = 0
        for earlier, later in zip(dates, dates[1:]):
            longest_pause = max(longest_pause, (later - earlier).days - 1)
        stats = {
            "days_running": (dates[-1] - dates[0]).days + 1,
            "days_with_runs": len(dates),
            "longest_pause_days": longest_pause,
        }

    return {
        "days": days,
        "stats": stats,
        "note": (
            "covers only runs started via POST /api/campaign/start and later "
            "measured; the originally uploaded base dataset has no per-row "
            "timestamps and is not represented here. Days are bucketed in UTC."
        ),
    }


def _campaign_brief(state: dict[str, Any]) -> dict[str, Any]:
    """The lightweight per-campaign summary `GET /api/campaigns` (list) returns
    — enough for a left-rail entry, not the full `pending`/`history` detail
    `summary()` returns for one campaign."""
    base_rows: list[dict[str, Any]] = state.get("base_rows", [])
    pending: list[dict[str, Any]] = state.get("pending", [])
    target = state.get("target", "")
    return {
        "id": state.get("id"),
        "name": state.get("name", _DEFAULT_NAME),
        "target": target,
        "n_base": len(base_rows),
        "best": _best_measured(base_rows, target),
        "round": state.get("round", 0),
        "n_awaiting": sum(1 for run in pending if run.get("result") is None),
        "archived": _is_archived(state),
        "updated_at": state.get("updated_at"),
    }


class CampaignStore:
    """Persists many campaigns PER TENANT as rows in `<state_dir>/portal.db`.

    Every read and write is guarded by a single lock and runs in a SQLite
    transaction, so a concurrent reader never observes a partial state. Each
    public method takes a `tenant` (default `"default"`, the open-mode
    tenant) and an optional `campaign_id`; rows are keyed by
    `(tenant, campaign_id)`, so no two tenants share a campaign
    (docs/HARDENING.md, Phase 1b), and a tenant's campaigns never collide with
    each other.

    When `campaign_id` is omitted, methods resolve it to the tenant's
    most-recently-updated ACTIVE (un-archived) campaign
    (docs/CAMPAIGN_LOOP.md, "Campaign identity") — this is what lets every
    pre-existing `/api/campaign*` call site (which never passed an id) keep
    working unchanged: right after a fresh `seed()`, that new campaign IS the
    most recent, so the old single-campaign call pattern still reaches it.
    `archive`/`unarchive`/`replace` are the exceptions: they take a REQUIRED
    `campaign_id`, because "whichever campaign happens to be current" is not a
    safe target for an operation that rewrites or retires one.
    """

    def __init__(self, state_dir: Path | None = None) -> None:
        self._dir = Path(state_dir) if state_dir is not None else Path(
            os.environ.get("KALOS_STATE_DIR", Path.home() / ".kalos")
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "portal.db"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._migrate_legacy_schema()
        with self._conn:
            self._conn.execute(_SCHEMA)

    # --- migration ------------------------------------------------------------ #

    def _migrate_legacy_schema(self) -> None:
        """One-time forward migration from the pre-identity layout
        (`campaigns(tenant PRIMARY KEY, state, updated_at)`, one unnamed
        campaign per tenant) to the current `(tenant, campaign_id)` layout.

        Runs unconditionally at every `__init__` but is a no-op unless the
        legacy table shape is actually found (`PRAGMA table_info` has no
        `campaign_id` column) — a fresh db or an already-migrated db returns
        immediately. Each legacy row is assigned a fresh `id` (and a generic
        `name`, since the old layout never had one) exactly ONCE: that id is
        written into the row's own `state` blob and the new table, so every
        later `__init__` against the same file sees the new schema already in
        place and never re-mints it — this is what makes the id STABLE across
        restarts, not just present after the first one.

        Genuinely atomic: the RENAME, the CREATE of the new table, every
        INSERT, and the final DROP all run inside one explicit `BEGIN
        IMMEDIATE` / `COMMIT` transaction, with a `ROLLBACK` (and the
        triggering exception re-raised) on any failure. This is NOT the same
        as `with self._conn:` — Python's sqlite3 module (legacy/default
        transaction control, still the default on 3.12) only opens an
        IMPLICIT transaction right before the first DML statement (INSERT/
        UPDATE/DELETE/REPLACE); DDL (ALTER TABLE/CREATE TABLE/DROP TABLE) runs
        in autocommit and is durable the instant it executes, so a `with
        self._conn:` block that starts with DDL does not actually cover it.
        SQLite itself fully supports transactional DDL — this was a Python
        transaction-control gap, not a SQLite one. The explicit BEGIN here
        moves the connection out of autocommit BEFORE the first DDL statement
        runs, so a crash or exception at ANY point (mid-loop, mid-DROP, an
        `INSERT` failure, anything) leaves the on-disk db exactly as it was
        before this method was called — never a half-migrated state with the
        legacy table gone and the new one only partially populated.

        This explicit BEGIN/COMMIT/ROLLBACK is scoped to this one method: the
        connection is always handed back in its normal (out-of-transaction)
        state before returning or raising, so every other caller of `with
        self._conn:` elsewhere in this class (`_write_locked`, most notably —
        the generation-token write behind `plan_fold`/`commit_fold`) keeps
        relying on the same implicit-transaction-per-DML behavior it always
        has, unaffected by this method ever having run.
        """
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(campaigns)").fetchall()}
        if not cols or "campaign_id" in cols:
            return  # no table yet, or already the current schema

        log.info("legacy campaigns schema detected (no campaign_id column); migrating")
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute("ALTER TABLE campaigns RENAME TO campaigns_legacy")
            self._conn.execute(_SCHEMA)
            legacy_rows = self._conn.execute(
                "SELECT tenant, state, updated_at FROM campaigns_legacy"
            ).fetchall()
            migrated = 0
            skipped = 0
            for tenant, state_json, updated_at in legacy_rows:
                try:
                    state = json.loads(state_json)
                except ValueError:  # a corrupt legacy row must not break startup
                    log.warning(
                        "skipping corrupt legacy campaign row for tenant %r: "
                        "state is not valid JSON",
                        tenant,
                    )
                    skipped += 1
                    continue
                if not isinstance(state, dict):
                    log.warning(
                        "skipping corrupt legacy campaign row for tenant %r: "
                        "state JSON decoded to %s, not an object",
                        tenant,
                        type(state).__name__,
                    )
                    skipped += 1
                    continue
                campaign_id = uuid.uuid4().hex
                state["id"] = campaign_id
                state.setdefault("name", _DEFAULT_NAME)
                state.setdefault("base_row_meta", [None] * len(state.get("base_rows", [])))
                self._conn.execute(
                    "INSERT INTO campaigns (tenant, campaign_id, state, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (tenant, campaign_id, json.dumps(state), updated_at),
                )
                migrated += 1
            self._conn.execute("DROP TABLE campaigns_legacy")
        except Exception:
            self._conn.rollback()
            log.exception(
                "legacy campaign schema migration failed and was rolled back; "
                "no data was changed on disk, migration will retry on next startup"
            )
            raise
        else:
            self._conn.commit()
            log.info(
                "legacy campaign schema migration complete: read %d row(s), "
                "migrated %d, skipped %d corrupt row(s)",
                len(legacy_rows),
                migrated,
                skipped,
            )

    # --- persistence -------------------------------------------------------- #

    def _read_locked(self, tenant: str, campaign_id: str) -> dict[str, Any] | None:
        """Read one campaign row. Caller must hold `self._lock`."""
        row = self._conn.execute(
            "SELECT state FROM campaigns WHERE tenant = ? AND campaign_id = ?",
            (tenant, campaign_id),
        ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
        except ValueError:  # a corrupt row must not break the API
            return None
        return data if isinstance(data, dict) else None

    def _resolve_default_locked(self, tenant: str) -> str | None:
        """The tenant's most-recently-updated ACTIVE campaign id, or None if
        the tenant owns no un-archived campaigns. Caller must hold
        `self._lock`.

        Archived campaigns are skipped here, not just in `list_campaigns` -
        archiving has to remove a campaign from EVERY implicit resolution
        path (docs/CAMPAIGN_LOOP.md, "Archiving"), or an id-less
        `start`/`result`/`reanalyze` could still silently land on the campaign
        the scientist just put away. A tenant whose only campaigns are all
        archived resolves to None, exactly like a tenant that has never
        uploaded anything: the id-less caller gets "no active campaign"
        rather than a surprise.

        The filtering happens in Python rather than SQL because `archived`
        lives inside the row's JSON `state` blob - no schema change, no second
        migration path to get wrong, and a tenant's campaign count is small
        enough that scanning its rows costs nothing.
        """
        rows = self._conn.execute(
            "SELECT campaign_id, state FROM campaigns WHERE tenant = ? ORDER BY updated_at DESC",
            (tenant,),
        ).fetchall()
        for campaign_id, state_json in rows:
            try:
                state = json.loads(state_json)
            except ValueError:  # a corrupt row must not break resolution
                continue
            if isinstance(state, dict) and not _is_archived(state):
                return str(campaign_id)
        return None

    def _write_locked(self, tenant: str, campaign_id: str, state: dict[str, Any]) -> None:
        """Persist `state` for `(tenant, campaign_id)` in one transaction,
        stamping it with a fresh `generation` token. Caller must hold
        `self._lock`.

        Every write mints a new `generation` (a random token), so any
        intervening write to THIS SAME campaign — a `set_result`, another
        `start` — is detectable: a re-analyze captures the generation it
        planned against (`plan_fold`) and refuses to commit (`commit_fold`)
        if it changed underneath. This is what stops a concurrent mutation
        from silently destroying just-folded measured data (see
        docs/CAMPAIGN_LOOP.md, "Re-analyze = the loop closing"). A `seed()` of
        a DIFFERENT campaign never touches this row at all, so it cannot
        perturb this generation — campaigns are now fully isolated from each
        other, not just from other tenants.
        """
        state["generation"] = uuid.uuid4().hex
        with self._conn:
            self._conn.execute(
                "INSERT INTO campaigns (tenant, campaign_id, state, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(tenant, campaign_id) DO UPDATE SET state = excluded.state, "
                "updated_at = excluded.updated_at",
                (tenant, campaign_id, json.dumps(state), time.time()),
            )

    # --- public API ----------------------------------------------------------- #

    def seed(
        self,
        df: pd.DataFrame,
        target: str,
        features: list[str],
        *,
        tenant: str = "default",
        name: str | None = None,
    ) -> str:
        """Start a NEW campaign for `tenant` and return its id.

        Called after every successful `/api/run` upload: a new upload is a
        new base dataset, so it becomes its OWN campaign rather than
        overwriting an existing one — this is what lets a tenant accumulate
        many named, independently addressable campaigns
        (docs/CAMPAIGN_LOOP.md, "Campaign identity"). `target`/`features`
        come from the analysis result (`result["target"]`,
        `result["proposal_features"]`), never guessed from column names —
        see docs/CAMPAIGN_LOOP.md, "Seeding". `name` defaults to the uploaded
        file's name when not given; a caller with no name to offer gets
        `"Untitled campaign"` rather than a blank string.

        Every OTHER campaign method defaults `campaign_id` to the tenant's
        most-recently-updated campaign when omitted — right after this call,
        that is always the campaign just seeded, so a caller that never
        passes `campaign_id` (the pre-identity call pattern) keeps landing on
        the freshest upload, unchanged.

        `features` must be the FULL modeled feature list, not the abridged
        `proposal_features` the UI displays: it is the schema every started
        recipe is validated against, so a short list here would let
        partially-specified recipes through.
        """
        from kalos.portal.serialization import _json_safe_records  # local: avoids a cycle at import time

        with self._lock:
            campaign_id = uuid.uuid4().hex
            base_rows = _json_safe_records(df)
            now = time.time()
            state: dict[str, Any] = {
                "id": campaign_id,
                "name": (name or "").strip() or _DEFAULT_NAME,
                "target": target,
                "features": list(features),
                "base_rows": base_rows,
                # Lineage (docs/CAMPAIGN_LOOP.md, "Lineage"): parallel to
                # base_rows, one entry per row. None = an originally uploaded
                # row with no traceable run id or timestamp (the honest
                # limitation); a dict = a row folded in from a measured
                # pending run, carrying that run's id/created_at/measured_at.
                # Never fed to `_analyze` — kept out of the feature/target
                # DataFrame entirely so it can never be mistaken for a
                # modeled column.
                "base_row_meta": [None] * len(base_rows),
                "pending": [],
                "round": 0,
                # Progress trajectory: best-so-far at each round, starting at
                # round 0 (the uploaded base). Each reanalyze appends a point.
                "history": [{"round": 0, "best": _best_measured(base_rows, target), "n_base": len(base_rows)}],
                "updated_at": now,
            }
            self._write_locked(tenant, campaign_id, state)
            return campaign_id

    def list_campaigns(
        self, *, tenant: str = "default", include_archived: bool = False
    ) -> list[dict[str, Any]]:
        """Every ACTIVE campaign for `tenant`, most-recently-updated first -
        the left-rail listing. `{id, name, target, n_base, best, round,
        n_awaiting, archived, updated_at}` per campaign; empty list if the
        tenant has never uploaded anything. (Named `list_campaigns`, not
        `list`, so it cannot shadow the builtin `list[...]` type used in every
        other method's annotations on this class.)

        `include_archived=True` returns archived campaigns too, in the same
        most-recently-updated order and flagged by their `archived` field -
        the "show everything" view. Archived campaigns are never deleted, so
        this always returns the tenant's complete history
        (docs/CAMPAIGN_LOOP.md, "Archiving").
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT state FROM campaigns WHERE tenant = ? ORDER BY updated_at DESC",
                (tenant,),
            ).fetchall()
        briefs: list[dict[str, Any]] = []
        for (state_json,) in rows:
            try:
                state = json.loads(state_json)
            except ValueError:
                continue
            if not isinstance(state, dict):
                continue
            if _is_archived(state) and not include_archived:
                continue
            briefs.append(_campaign_brief(state))
        return briefs

    def get(self, *, tenant: str = "default", campaign_id: str | None = None) -> dict[str, Any] | None:
        """The raw state of one campaign, or None if it does not exist (or
        `campaign_id` is omitted and `tenant` owns no ACTIVE campaigns).

        An explicit `campaign_id` reaches an archived campaign just fine -
        archiving hides, it never hides data (docs/CAMPAIGN_LOOP.md,
        "Archiving")."""
        with self._lock:
            cid = campaign_id or self._resolve_default_locked(tenant)
            if cid is None:
                return None
            return self._read_locked(tenant, cid)

    def summary(self, *, tenant: str = "default", campaign_id: str | None = None) -> dict[str, Any]:
        """The `GET /api/campaign` response: `{"has_campaign": False}` if
        `campaign_id` does not resolve to anything (omitted with no campaigns
        yet, or an unknown id), else the full summary (docs/CAMPAIGN_LOOP.md,
        "GET /api/campaign response").

        `best` is `max(target over base_rows)` — the best MEASURED value so
        far, never a prediction (see "Honesty constraints" in the design
        doc) — and is None when there is no base data yet or the target
        column carries no finite numeric value.

        An ARCHIVED campaign still answers here in full when addressed by an
        explicit `campaign_id` - archiving hides a campaign, it never makes
        its data unreadable (docs/CAMPAIGN_LOOP.md, "Archiving") - and says so
        via `archived: true`. It is only unreachable when `campaign_id` is
        omitted, because default resolution skips archived campaigns.
        """
        with self._lock:
            cid = campaign_id or self._resolve_default_locked(tenant)
            state = self._read_locked(tenant, cid) if cid is not None else None
        if state is None:
            return {"has_campaign": False}

        target = state["target"]
        base_rows: list[dict[str, Any]] = state["base_rows"]
        best = _best_measured(base_rows, target)

        pending = [{**run, "awaiting": run["result"] is None} for run in state["pending"]]
        n_awaiting = sum(1 for run in pending if run["awaiting"])
        folded_rows = [
            {"run_id": meta["run_id"], "created_at": meta["created_at"], "measured_at": meta["measured_at"]}
            for meta in (state.get("base_row_meta") or [])
            if meta is not None
        ]
        return {
            "has_campaign": True,
            "id": state.get("id", cid),
            "name": state.get("name", _DEFAULT_NAME),
            "archived": _is_archived(state),
            # Audit trail of every `replace()` this campaign has been through
            # (docs/CAMPAIGN_LOOP.md, "Replacing a campaign's data"): empty
            # for a campaign whose uploaded base has never been corrected.
            "revisions": state.get("revisions") or [],
            "target": target,
            "features": state["features"],
            "n_base": len(base_rows),
            "best": best,
            "round": state["round"],
            # Older campaigns (seeded before history existed) fall back to a
            # single current point so the frontend always has something to plot.
            "history": state.get("history")
            or [{"round": state["round"], "best": best, "n_base": len(base_rows)}],
            "pending": pending,
            "n_awaiting": n_awaiting,
            "n_measured": len(pending) - n_awaiting,
            # Lineage (docs/CAMPAIGN_LOOP.md, "Lineage"): every base row this
            # module can trace back to the pending run it was folded from.
            # Original upload rows carry no such row here — see the module
            # docstring on `base_row_meta` for why that is the honest answer,
            # not a gap to paper over. Each `pending` run above also carries
            # its own `parent_ids`/`parent_row_count` (set at `start()` time).
            "lineage": {"folded_rows": folded_rows},
        }

    def replace(
        self,
        campaign_id: str,
        df: pd.DataFrame,
        target: str,
        features: list[str],
        *,
        tenant: str = "default",
        name: str | None = None,
    ) -> str:
        """Swap ONE campaign's uploaded base dataset for a corrected one, in
        place, keeping its `id`. Returns that same id.

        This is the corrected-run-sheet path (docs/CAMPAIGN_LOOP.md,
        "Replacing a campaign's data"): a scientist uploads a sheet, spots a
        typo, and re-uploads the fixed file as
        `POST /api/run?campaign_id=<id>`. Without it, the corrected upload
        would seed a SECOND campaign and leave the typo'd one in the left rail
        forever.

        What it replaces: `base_rows`, `base_row_meta` (reset to all-None -
        the new rows are an upload, so none of them is traceable to a run),
        `features`, `name`, and `history` (reset to a single round-0 point
        over the new base). What it keeps: `id`, `target`, `round` (always 0
        here - see below), `archived`, and every pending run.

        What it REFUSES, rather than silently discard:

        - **Any measured result.** If the campaign holds a measured pending
          run, a folded base row, or any completed round, this raises. Those
          are outcomes a scientist recorded from the lab; a corrected upload
          is a data-entry fix and has no business deleting them. Worse, they
          are not even reconcilable: a folded row descends from a proposal the
          GP conditioned on the OLD base, and `history`/`best` describe a
          dataset that would no longer exist. The honest move is to refuse and
          say so - upload the correction as a new campaign and `archive()`
          this one, which loses nothing.
        - **A different `target`.** A campaign IS one optimization target plus
          a growing dataset; changing the target makes it a different
          experiment, not a corrected one. This also keeps any preserved
          pending run coherent, since its eventual result is folded in under
          this campaign's target.
        - **An archived campaign.** Unarchive it first; an archived campaign
          is a closed record.

        Pending runs are deliberately PRESERVED (they can only be awaiting
        ones, since a measured one is refused above): a started run is
        physically in the lab, so the record that it happened must survive a
        correction to the sheet it was proposed from. Their
        `parent_ids`/`parent_row_count` are left untouched, still describing
        the base as it stood when they were started - history, not a live
        pointer.

        Each replacement appends an audit entry to `revisions`
        (`{replaced_at, name, n_base}` of the base it superseded), so the fact
        that a correction happened survives even though the superseded rows do
        not.

        Like every other write, this runs under `self._lock` and mints a fresh
        `generation`, so it can never sneak past the `plan_fold`/`commit_fold`
        guard.
        """
        from kalos.portal.serialization import _json_safe_records  # local: avoids a cycle at import time

        with self._lock:
            state = self._read_locked(tenant, campaign_id)
            if state is None:
                raise CampaignError(f"no campaign with id {campaign_id!r}")
            _reject_if_archived(state, campaign_id, "replacing its data")
            if state.get("target") != target:
                raise CampaignError(
                    f"campaign {campaign_id!r} optimizes {state.get('target')!r}, but the "
                    f"uploaded file analyzes {target!r}; a different target is a different "
                    "experiment - upload it as a new campaign instead of replacing this one"
                )
            measured = _measured_count(state)
            round_ = state.get("round", 0)
            if measured or round_:
                raise CampaignError(
                    f"campaign {campaign_id!r} already holds measured results "
                    f"({measured} measured run(s), round {round_}); replacing its data would "
                    "discard outcomes logged from the lab - upload this file as a new "
                    "campaign and archive this one instead"
                )

            now = time.time()
            base_rows = _json_safe_records(df)
            revisions: list[dict[str, Any]] = state.setdefault("revisions", [])
            revisions.append(
                {
                    "replaced_at": now,
                    "name": state.get("name", _DEFAULT_NAME),
                    "n_base": len(state.get("base_rows", [])),
                }
            )
            state["name"] = (name or "").strip() or state.get("name") or _DEFAULT_NAME
            state["features"] = list(features)
            state["base_rows"] = base_rows
            state["base_row_meta"] = [None] * len(base_rows)
            state["history"] = [
                {"round": 0, "best": _best_measured(base_rows, target), "n_base": len(base_rows)}
            ]
            state["updated_at"] = now
            self._write_locked(tenant, campaign_id, state)
            return campaign_id

    def archive(self, campaign_id: str, *, tenant: str = "default") -> dict[str, Any]:
        """Hide one campaign from the left rail and from every default
        resolution, WITHOUT deleting any of its data. Returns its brief.

        Nothing is ever hard-deleted here (docs/CAMPAIGN_LOOP.md,
        "Archiving"): the row stays exactly where it was, still readable by
        explicit `campaign_id` through `get`/`summary`/`activity`, and still
        listable via `list_campaigns(include_archived=True)`. The record that
        a run happened survives, which is the whole point for a regulated lab.
        What changes is that the campaign stops competing for attention: it
        drops out of `list_campaigns()` and can never be picked by
        `_resolve_default_locked`, and it refuses further writes until
        `unarchive`.

        `campaign_id` is REQUIRED (no default resolution): archiving the
        "current" campaign by accident is exactly the mistake worth designing
        out. Archiving an already-archived campaign is a no-op success.

        Raises `CampaignError` if the id is unknown for this tenant.
        """
        return self._set_archived(campaign_id, True, tenant=tenant)

    def unarchive(self, campaign_id: str, *, tenant: str = "default") -> dict[str, Any]:
        """Bring an archived campaign back into the left rail. Returns its
        brief.

        Like every mutation, this bumps `updated_at`, so an unarchived
        campaign becomes the tenant's most-recently-updated one and therefore
        its default again - which is what a caller reaching for it explicitly
        means. Unarchiving an already-active campaign is a no-op success.
        """
        return self._set_archived(campaign_id, False, tenant=tenant)

    def _set_archived(self, campaign_id: str, archived: bool, *, tenant: str) -> dict[str, Any]:
        """The shared body of `archive`/`unarchive`: flip the flag under the
        store lock and persist, which also mints a fresh `generation` - so an
        archive landing mid-re-analysis is caught by the exact same
        `plan_fold`/`commit_fold` guard every other concurrent write is."""
        with self._lock:
            state = self._read_locked(tenant, campaign_id)
            if state is None:
                raise CampaignError(f"no campaign with id {campaign_id!r}")
            state["archived"] = archived
            state["updated_at"] = time.time()
            self._write_locked(tenant, campaign_id, state)
            return _campaign_brief(state)

    def start(
        self, recipes: list[dict[str, Any]], *, tenant: str = "default", campaign_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Append each proposed recipe as a pending, awaiting run to one
        campaign. Returns the appended runs (each with its assigned `id`)."""
        with self._lock:
            cid = campaign_id or self._resolve_default_locked(tenant)
            if cid is None:
                raise CampaignError("no active campaign; upload a run sheet first")
            state = self._read_locked(tenant, cid)
            if state is None:
                raise CampaignError(f"no campaign with id {cid!r}")
            _reject_if_archived(state, cid, "starting new runs on it")
            now = time.time()
            # Lineage (docs/CAMPAIGN_LOOP.md, "Lineage"): the proposals being
            # started here were computed by the last re-analyze/seed over
            # EXACTLY this campaign's current base_rows — nothing mutates
            # base_rows between that fit and this call. `parent_ids` is every
            # base row this module can name (i.e. one folded from a prior
            # measured run); `parent_row_count` is the TRUE total, including
            # originally uploaded rows with no traceable id. Reporting both,
            # honestly, rather than inventing ids for the untraceable rows —
            # a proposal genuinely descends from all of them, not just the
            # named subset (see the module docstring on `base_row_meta`).
            parent_ids = [
                meta["run_id"] for meta in (state.get("base_row_meta") or []) if meta is not None
            ]
            parent_row_count = len(state["base_rows"])
            appended: list[dict[str, Any]] = []
            for recipe in recipes:
                # Validate the recipe mapping HERE, at the point of the bad
                # input, so a malformed body 400s cleanly at /api/campaign/start
                # rather than surfacing as an unhandled 500 rounds later inside
                # fold_and_snapshot's `{**recipe, target: result}` expansion.
                recipe_map = recipe.get("recipe")
                if not isinstance(recipe_map, dict) or not recipe_map:
                    raise CampaignError(
                        "each recipe must include a non-empty 'recipe' mapping of feature -> value"
                    )
                run = {
                    "id": uuid.uuid4().hex,
                    "recipe": recipe_map,
                    "pred": recipe.get("pred"),
                    "std": recipe.get("std"),
                    "mode": recipe.get("mode"),
                    "reason": recipe.get("reason"),
                    "result": None,
                    "created_at": now,
                    "measured_at": None,
                    "parent_ids": parent_ids,
                    "parent_row_count": parent_row_count,
                }
                state["pending"].append(run)
                appended.append(run)
            state["updated_at"] = now
            self._write_locked(tenant, cid, state)
            return appended

    def set_result(
        self, run_id: str, value: float, *, tenant: str = "default", campaign_id: str | None = None
    ) -> dict[str, Any]:
        """Set the measured outcome on a pending run in one campaign. Raises
        `CampaignError` (never silently ignored) if `campaign_id`/`run_id` is
        unknown or `value` is not a finite number — a NaN/inf result must
        never be foldable into the base dataset on the next re-analyze."""
        if not math.isfinite(value):
            raise CampaignError(f"result value must be a finite number; got {value!r}")
        with self._lock:
            cid = campaign_id or self._resolve_default_locked(tenant)
            if cid is None:
                raise CampaignError(f"no active campaign; unknown run id {run_id!r}")
            state = self._read_locked(tenant, cid)
            if state is None:
                raise CampaignError(f"no campaign with id {cid!r}")
            _reject_if_archived(state, cid, "logging a result on it")
            for run in state["pending"]:
                if run["id"] == run_id:
                    now = time.time()
                    run["result"] = value
                    run["measured_at"] = now
                    state["updated_at"] = now
                    self._write_locked(tenant, cid, state)
                    return run
            raise CampaignError(f"unknown pending run id {run_id!r}")

    def plan_fold(
        self, *, tenant: str = "default", campaign_id: str | None = None
    ) -> tuple[pd.DataFrame, str, str | None, str]:
        """Compute the folded base dataset for one campaign WITHOUT
        persisting anything.

        Returns `(DataFrame(folded_base_rows), target, generation, campaign_id)`
        where the DataFrame is `base_rows` plus every measured pending run
        folded in, `generation` is the token of the campaign state this plan
        was built from, and `campaign_id` is the id this plan actually
        resolved to (the caller MUST pass this exact id back to
        `commit_fold`, not let it re-resolve "the default campaign" a second
        time — if a fresh `seed()` created a newer campaign in the meantime,
        that would silently re-resolve to the WRONG campaign and either
        commit nothing or, worse, race a different campaign's generation).
        The caller (`kalos.portal.campaign_routes`) runs `_analyze` on the
        DataFrame and, only if that succeeds, calls
        `commit_fold(generation, campaign_id=campaign_id)` to make the fold
        durable.

        `generation` is read with `.get()` (not `[]`): a state written before
        the generation token existed has no such key, and the very first
        re-analyze after that upgrade must not crash. It flows back as `None`,
        which `commit_fold` compares by equality just like any other token — and
        the migration is self-healing, because `commit_fold`'s own write mints a
        real generation from then on.

        Nothing is mutated or written here, so a `_analyze` failure — or a
        concurrent write to this SAME campaign — leaves it exactly as it was:
        no lost round, no half-folded base (docs/CAMPAIGN_LOOP.md,
        "Re-analyze = the loop closing"). A concurrent `seed()` creates an
        UNRELATED campaign now (docs/CAMPAIGN_LOOP.md, "Campaign identity"),
        so it can no longer perturb this one at all — a strictly stronger
        guarantee than before identity existed. Raises `CampaignError` if
        `campaign_id` does not resolve to an existing campaign, or there is
        no measured run to fold (re-analyzing with nothing new would only
        inflate `round` and the progress trajectory).
        """
        with self._lock:
            cid = campaign_id or self._resolve_default_locked(tenant)
            if cid is None:
                raise CampaignError("no active campaign; upload a run sheet first")
            state = self._read_locked(tenant, cid)
            if state is None:
                raise CampaignError(f"no campaign with id {cid!r}")
            _reject_if_archived(state, cid, "re-analyzing it")
            target = state["target"]
            folded = list(state["base_rows"])
            measured = 0
            for run in state["pending"]:
                if run["result"] is not None:
                    folded.append({**run["recipe"], target: run["result"]})
                    measured += 1
            if measured == 0:
                raise CampaignError("no measured runs to fold; log at least one result first")
            return pd.DataFrame(folded), target, state.get("generation"), cid

    def commit_fold(
        self, generation: str | None, *, tenant: str = "default", campaign_id: str | None = None
    ) -> dict[str, Any]:
        """Make the fold planned by `plan_fold` durable, but ONLY if the
        campaign has not changed since (its `generation` still matches).

        `campaign_id` should be the exact id `plan_fold` returned (see its
        docstring on why re-resolving "the default campaign" here would be
        wrong); it defaults to the tenant's most-recent campaign only for
        callers with a single campaign in play (e.g. direct store-level
        tests), matching the pre-identity call pattern.

        Folds every measured pending run into `base_rows` (carrying its `id`,
        `created_at`, `measured_at` forward into `base_row_meta` — see
        `base_row_meta` on `seed()` — never into the modeled DataFrame
        itself), drops them from `pending` (awaiting runs stay), increments
        `round`, appends the progress-trajectory point, and persists. Returns
        the updated state.

        Raises `CampaignError` if the campaign was reseeded or otherwise
        written underneath the in-flight re-analysis (`generation` mismatch):
        the analysis the caller just computed describes a base dataset that no
        longer exists, so committing it — and letting the caller overwrite
        `/api/latest` with it — would silently destroy the newer data. The
        caller turns this into a "please retry" 409 and leaves `/api/latest`
        untouched. Also raises if the campaign was ARCHIVED mid-analysis, with
        a distinct message - the retry the generic wording invites would never
        succeed (docs/CAMPAIGN_LOOP.md, "Archiving").

        A concurrent `replace()` cannot reach this branch at all: `plan_fold`
        only succeeds when at least one run is measured, and `replace()`
        refuses outright on a campaign holding any measured result, so the two
        are mutually exclusive by construction rather than by timing (see
        `replace`).
        """
        with self._lock:
            cid = campaign_id or self._resolve_default_locked(tenant)
            if cid is None:
                raise CampaignError("no active campaign; nothing to commit")
            state = self._read_locked(tenant, cid)
            if state is None:
                raise CampaignError(f"no campaign with id {cid!r}; nothing to commit")
            # Checked BEFORE the generation token, even though archiving also
            # bumps the generation and would trip the check below anyway: the
            # generic "please re-analyze again" message would be actively
            # misleading here, because retrying will keep failing until the
            # campaign is unarchived.
            if _is_archived(state):
                raise CampaignError(
                    f"campaign {cid!r} was archived during re-analysis; nothing was "
                    "committed - unarchive it and re-analyze again"
                )
            if state.get("generation") != generation:
                raise CampaignError(
                    "the campaign changed during re-analysis (a new write landed); "
                    "nothing was committed — please re-analyze again"
                )
            target = state["target"]
            base_row_meta: list[dict[str, Any] | None] = state.setdefault(
                "base_row_meta", [None] * len(state["base_rows"])
            )
            still_pending: list[dict[str, Any]] = []
            for run in state["pending"]:
                if run["result"] is not None:
                    state["base_rows"].append({**run["recipe"], target: run["result"]})
                    base_row_meta.append(
                        {
                            "run_id": run["id"],
                            "created_at": run["created_at"],
                            "measured_at": run["measured_at"],
                        }
                    )
                else:
                    still_pending.append(run)
            state["pending"] = still_pending
            state["round"] += 1
            # Record the new best-so-far for the progress trajectory. base_rows
            # only grew, so this point is >= the previous one.
            history = state.setdefault("history", [])
            history.append(
                {
                    "round": state["round"],
                    "best": _best_measured(state["base_rows"], target),
                    "n_base": len(state["base_rows"]),
                }
            )
            state["updated_at"] = time.time()
            self._write_locked(tenant, cid, state)
            return state

    def activity(self, *, tenant: str = "default", campaign_id: str | None = None) -> dict[str, Any]:
        """The `GET /api/campaign/activity` response for one campaign:
        `{"has_campaign": False}` if `campaign_id` does not resolve, else
        `{has_campaign, id, days, stats, note}` — see `_activity_from_state`
        for the day-bucketing rule, the derived stats, and the documented
        coverage limitation."""
        with self._lock:
            cid = campaign_id or self._resolve_default_locked(tenant)
            state = self._read_locked(tenant, cid) if cid is not None else None
        if state is None:
            return {"has_campaign": False}
        return {"has_campaign": True, "id": state.get("id", cid), **_activity_from_state(state)}


# --- module-level singleton, mirroring kalos.portal.experiments.get_store --- #
_STORE: CampaignStore | None = None


def get_campaign_store() -> CampaignStore:
    """The module-level `CampaignStore` at the default `~/.kalos/portal.db`.

    Tests override this via `app.dependency_overrides[get_campaign_store]`
    (same pattern as `kalos.portal.experiments.get_store`), pointing at a
    fresh `CampaignStore(tmp_path)` so they never touch the real
    `~/.kalos/portal.db`.
    """
    global _STORE
    if _STORE is None:
        _STORE = CampaignStore()
    return _STORE
