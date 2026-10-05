"""Kalos portal — M2: `/api/experiments*`, thin wrappers over the store + the
Singleton runner (docs/M2_INTEGRATION.md, "Portal API additions").
"""
from __future__ import annotations

import dataclasses
import hmac
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from kalos.portal.auth import (
    READ,
    RUNNER,
    WRITE,
    AuthConfigError,
    Authenticator,
    Principal,
    get_authenticator,
    require_principal,
    require_scope,
)
from kalos.portal.serialization import _json_safe_records
from kalos.portal.uploads import _ERR_PARSE, UploadRejected, _parse_upload
from kalos.runner.adapter import LocalStoreAdapter
from kalos.runner.singleton import run_one, run_ready
from kalos.store import ExperimentNotFound, IllegalTransition, SqliteStore, Status

log = logging.getLogger("kalos.portal")

router = APIRouter()

# Module-level singleton store + injectable lock path, resolved lazily through
# FastAPI dependencies so tests can override both via `app.dependency_overrides`
# and never touch the real `~/.kalos/experiments.db` or `~/.kalos/runner.lock`.
_STORE: SqliteStore | None = None
_RUNNER_LOCK_PATH: str | Path | None = None


def get_store() -> SqliteStore:
    """The module-level `SqliteStore` at the default `~/.kalos/experiments.db`.

    Tests MUST override this via `app.dependency_overrides[get_store]` to
    point at a `tmp_path` store.
    """
    global _STORE
    if _STORE is None:
        _STORE = SqliteStore()
    return _STORE


def get_lock_path() -> str | Path | None:
    """The Singleton runner lock path. `None` defers to the runner's own
    default (`~/.kalos/runner.lock`); tests override this via
    `app.dependency_overrides[get_lock_path]` to point at a `tmp_path` lock.
    """
    return _RUNNER_LOCK_PATH


class _PatchStatusBody(BaseModel):
    """Body for `PATCH /api/experiments/{id}`. Mirrors the `set_status` shape
    `HttpBackendAdapter` already sends (`kalos/runner/adapter.py`)."""

    status: str
    force: bool = False
    error: str | None = None


@router.get("/api/experiments")
def list_experiments(
    status: str | None = None,
    store: SqliteStore = Depends(get_store),
    principal: Principal = Depends(require_scope(READ)),
) -> list[dict[str, Any]]:
    """`{id, name, status, updated_at}` for every experiment, oldest first.

    Optional `?status=` filters to one status (e.g. `READY`, the filter
    `HttpBackendAdapter.list_ready` sends, `kalos/runner/adapter.py`);
    omitted, the default and unchanged behavior, returns every experiment.
    """
    status_filter: Status | None = None
    if status is not None:
        try:
            status_filter = Status(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"unknown status {status!r}")
    return [
        {"id": e.id, "name": e.name, "status": e.status.value, "updated_at": e.updated_at}
        for e in store.list(status=status_filter, tenant=principal.tenant)
    ]


@router.post("/api/experiments")
async def create_experiment(
    file: UploadFile = File(...),
    name: str = Form(...),
    target: str = Form(default=""),
    outcomes: str = Form(default=""),
    anonymize: bool = Form(default=False),
    store: SqliteStore = Depends(get_store),
    principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    """Create an experiment (DRAFT) from an uploaded run sheet.

    Reuses `_parse_upload` - the exact CSV/xlsx parsing path `/api/run` already
    uses - to turn the upload into a DataFrame, then stores it as the
    `{columns, rows}` payload the Experiment contract documents. `target`,
    `outcomes` (comma-separated column names), and `anonymize` become `config`,
    matching `/api/run`'s multipart Form style.
    """
    raw = await file.read()
    try:
        df = await run_in_threadpool(_parse_upload, raw)
    except UploadRejected as rej:
        log.warning("experiment create rejected: %s", rej)
        return JSONResponse({"error": str(rej)}, status_code=400)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, UnicodeError):
        log.exception("failed to parse an uploaded run sheet for experiment create")
        return JSONResponse({"error": _ERR_PARSE}, status_code=400)
    except Exception:  # noqa: BLE001 - normalized to a generic 400, same as /api/run
        log.exception("failed to parse an uploaded run sheet for experiment create")
        return JSONResponse({"error": _ERR_PARSE}, status_code=400)

    config: dict[str, Any] = {"target": target or None, "anonymize": anonymize}
    if outcomes.strip():
        config["outcomes"] = [c.strip() for c in outcomes.split(",") if c.strip()]
    payload = {"columns": [str(c) for c in df.columns], "rows": _json_safe_records(df)}
    exp = store.create(name, payload, config, tenant=principal.tenant)
    return JSONResponse(exp.to_dict(), status_code=201)


@router.get("/api/experiments/{exp_id}")
def get_experiment(
    exp_id: str,
    store: SqliteStore = Depends(get_store),
    principal: Principal = Depends(require_scope(READ)),
) -> dict[str, Any]:
    """The full experiment, including `result`. 404 if it does not exist for
    this tenant (a cross-tenant id reads as not-found)."""
    try:
        return store.get(exp_id, tenant=principal.tenant).to_dict()
    except ExperimentNotFound:
        raise HTTPException(status_code=404, detail=f"no experiment with id {exp_id!r}")


@router.patch("/api/experiments/{exp_id}")
def patch_experiment_status(
    exp_id: str,
    body: _PatchStatusBody,
    store: SqliteStore = Depends(get_store),
    principal: Principal = Depends(require_principal),
) -> dict[str, Any]:
    """Set an experiment's status - this is how the front end flips the READY
    flag, and how a remote runner (`HttpBackendAdapter.set_status`) claims
    and fails experiments.

    Scope is checked per transition (least privilege): the `READY` flip needs
    `write`, the runner transitions need `runner`, and a principal with
    neither is refused (403) before the body is even looked at. So a polling
    runner's principal needs only `read` + `runner`, never `write`.

    A CLIENT-FACING ALLOWLIST, enforced here (not just `legal_transition`):
    the only status a client may PATCH to is `READY` - allowed from `DRAFT`
    and `FAILED` (retry), and from `DONE` only with `force=true` (re-queue).
    Any target of `PROCESSING`/`DONE`/`FAILED` is rejected with a 409 before
    it ever reaches the store, because those are set by the runner
    (`kalos/runner/singleton.py`), not by a client request - without this, a
    client could walk an experiment DRAFT -> READY -> PROCESSING -> DONE via
    PATCH alone, reaching DONE with an empty `result`. A request whose
    experiment is currently `PROCESSING` is rejected too, even when the
    target is `READY` - that edge is legal at the store layer ONLY for the
    Singleton's own orphan recovery (`reclaim_stale`), never for a client.
    An illegal transition that survives the allowlist (e.g. `DONE -> READY`
    without `force`) is still a 409, not a crash.

    A principal holding the `runner` scope (a provisioned token only, never
    open mode) may ALSO PATCH to `PROCESSING` and `FAILED` - the two runner
    transitions `legal_transition` then confines to `READY -> PROCESSING` and
    `PROCESSING -> FAILED`. `DONE` stays unreachable through PATCH for
    everyone: it is only ever set by the result push, which carries the result.
    """
    if not (principal.has(WRITE) or principal.has(RUNNER)):
        raise HTTPException(status_code=403, detail=f"requires '{WRITE}' scope")
    try:
        new_status = Status(body.status)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"unknown status {body.status!r}")

    runner_transition = principal.has(RUNNER) and new_status in (Status.PROCESSING, Status.FAILED)
    if new_status != Status.READY and not runner_transition:
        raise HTTPException(
            status_code=409,
            detail=f"status {new_status.value!r} is set by the runner, not directly settable",
        )
    if new_status == Status.READY and not principal.has(WRITE):
        raise HTTPException(status_code=403, detail=f"requires '{WRITE}' scope")

    try:
        current = store.get(exp_id, tenant=principal.tenant)
    except ExperimentNotFound:
        raise HTTPException(status_code=404, detail=f"no experiment with id {exp_id!r}")

    if current.status == Status.PROCESSING and not runner_transition:
        raise HTTPException(
            status_code=409,
            detail=(
                f"experiment {exp_id!r} is currently PROCESSING; status is set by "
                "the runner, not directly settable"
            ),
        )

    try:
        exp = store.set_status(
            exp_id, new_status, force=body.force, error=body.error, tenant=principal.tenant
        )
    except ExperimentNotFound:
        raise HTTPException(status_code=404, detail=f"no experiment with id {exp_id!r}")
    except IllegalTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return exp.to_dict()


class _PushResultBody(BaseModel):
    """Body for `POST /api/experiments/{id}/result` - the result-push
    endpoint `HttpBackendAdapter.push_result` targets (see
    `kalos/runner/adapter.py`)."""

    result: dict[str, Any]
    provenance: dict[str, Any]


def _result_push_tenant(authorization: str | None, auth: Authenticator) -> str:
    """Authorize a `/result` push and return the tenant it may write to
    (SECURITY - see the module section header below).

    Without a gate, `/result` accepted an unauthenticated `{result,
    provenance}` from anyone and marked a `PROCESSING` experiment `DONE`,
    letting an attacker race a fabricated result in ahead of the genuine one.

    Two credentials are accepted, both read AT REQUEST TIME (not import time),
    so tests can monkeypatch/env-override them per-test:
      - the legacy machine-channel secret `KALOS_RUNNER_TOKEN`, as
        `Authorization: Bearer <token>` matching EXACTLY (constant-time
        compare via `hmac.compare_digest`, so a wrong guess cannot be timed
        byte-by-byte). It carries no identity, so it writes to the `default`
        tenant only.
      - a provisioned `KALOS_AUTH_TOKENS` principal holding the `runner`
        scope, which writes to that principal's tenant. A principal without
        `runner` (e.g. a plain read+write client token) -> 403.
    With neither configured the endpoint is disabled entirely: 404. The local
    M2 loop (`LocalStoreAdapter`) never calls this endpoint - it writes to
    the store directly - so only a remote runner needs it, and it must be
    opted into explicitly. Otherwise a missing or wrong bearer -> 401.
    Never logs the token.
    """
    try:
        enforces = auth.enforces()
    except AuthConfigError:
        log.exception("kalos auth: token configuration is invalid")
        raise HTTPException(status_code=500, detail="authentication is misconfigured") from None
    legacy_token = os.environ.get("KALOS_RUNNER_TOKEN")
    if not legacy_token and not enforces:
        raise HTTPException(status_code=404, detail="not found")

    if legacy_token:
        provided = None
        if authorization and authorization.startswith("Bearer "):
            provided = authorization[len("Bearer ") :]
        # Compare BYTES: `compare_digest` raises TypeError on a non-ASCII str,
        # which would let any caller turn a wrong guess into a 500.
        if provided and hmac.compare_digest(provided.encode("utf-8"), legacy_token.encode("utf-8")):
            return "default"
    if not enforces:
        raise HTTPException(status_code=401, detail="unauthorized")
    principal = require_principal(authorization, auth)  # 401 on a missing/unknown token
    if not principal.has(RUNNER):
        raise HTTPException(status_code=403, detail=f"requires '{RUNNER}' scope")
    return principal.tenant


@router.post("/api/experiments/{exp_id}/result")
def push_experiment_result(
    exp_id: str,
    body: _PushResultBody,
    authorization: str | None = Header(default=None),
    auth: Authenticator = Depends(get_authenticator),
    store: SqliteStore = Depends(get_store),
) -> dict[str, Any]:
    """Ingest a processed result + provenance and mark the experiment DONE
    (`SqliteStore.save_result`) - only legal from `PROCESSING`. This is the
    push endpoint the Singleton's `HttpBackendAdapter` targets; the LOCAL
    Singleton path (`LocalStoreAdapter`, the M2 default) calls
    `store.save_result` directly and never goes through this HTTP endpoint.

    Gated and off by default - see `_result_push_tenant`: with neither
    `KALOS_RUNNER_TOKEN` nor auth tokens configured -> 404, missing/wrong
    bearer -> 401, a principal without the `runner` scope -> 403. The
    experiment is looked up on the tenant that credential grants, so a
    cross-tenant id reads as not-found (404). Also rejects an
    experiment that already has a non-null `result` with 409 ("result
    already present"), independent of the `PROCESSING`-only rule already
    enforced by `save_result` - an integrity/idempotency guard so a stray or
    racing push can never silently overwrite a genuine result.
    """
    tenant = _result_push_tenant(authorization, auth)

    try:
        current = store.get(exp_id, tenant=tenant)
    except ExperimentNotFound:
        raise HTTPException(status_code=404, detail=f"no experiment with id {exp_id!r}")
    if current.result is not None:
        raise HTTPException(status_code=409, detail="result already present")

    try:
        exp = store.save_result(exp_id, body.result, body.provenance, tenant=tenant)
    except ExperimentNotFound:
        raise HTTPException(status_code=404, detail=f"no experiment with id {exp_id!r}")
    except IllegalTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return exp.to_dict()


@router.post("/api/experiments/{exp_id}/run")
def run_experiment(
    exp_id: str,
    force: bool = False,
    store: SqliteStore = Depends(get_store),
    lock_path: str | Path | None = Depends(get_lock_path),
    principal: Principal = Depends(require_scope(WRITE)),
) -> dict[str, Any]:
    """Run one of the caller's experiments now (`singleton.run_one`). The run
    always completes: a zero-variance target (or any other analysis failure)
    lands the experiment on FAILED with the actionable message at HTTP 200 - the
    run itself did not error, the experiment did.
    """
    adapter = LocalStoreAdapter(store, tenant=principal.tenant)
    try:
        run_one(adapter, exp_id, force=force, lock_path=lock_path)
    except ExperimentNotFound:
        raise HTTPException(status_code=404, detail=f"no experiment with id {exp_id!r}")
    except (ValueError, RuntimeError) as exc:
        # Already DONE without force, or the Singleton lock is held elsewhere.
        raise HTTPException(status_code=409, detail=str(exc))
    return store.get(exp_id, tenant=principal.tenant).to_dict()


@router.post("/api/experiments/run-ready")
def run_all_ready(
    store: SqliteStore = Depends(get_store),
    lock_path: str | Path | None = Depends(get_lock_path),
    principal: Principal = Depends(require_scope(WRITE)),
) -> list[dict[str, Any]]:
    """Run every one of the caller's READY experiments now (`singleton.run_ready`)
    and return a per-experiment summary.

    Synchronous by design for M2 (single-tenant local, deterministic) -
    see docs/M2_INTEGRATION.md, "Keep the run endpoints synchronous for M2".
    """
    adapter = LocalStoreAdapter(store, tenant=principal.tenant)
    results = run_ready(adapter, lock_path=lock_path)
    return [dataclasses.asdict(r) for r in results]
