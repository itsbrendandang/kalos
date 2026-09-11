"""Kalos portal — `/api/campaign*`: the closed optimization loop (M3,
docs/CAMPAIGN_LOOP.md). Thin wrappers over `kalos.portal.campaign.CampaignStore`,
mirroring the router pattern in `kalos.portal.experiments`.

Auth posture: the mutating endpoints (`start`, `result`, `reanalyze`) require
the `write` scope via `kalos.portal.auth` (docs/HARDENING.md, Phase 1). In open
mode (no tokens configured) the anonymous principal holds read+write, so the
local dev/pilot loop is unchanged; once tokens are provisioned these endpoints
enforce them. `GET /api/campaign` stays open for now (read gating is a
follow-up). Per-tenant isolation of the campaign store is the next slice.
"""
from __future__ import annotations

import functools
import logging
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from kalos.portal.busy import RETRY_AFTER_SECONDS, AnalysisBusy, run_exclusively

from kalos.portal.auth import READ, WRITE, Principal, require_scope
from kalos.portal.campaign import CampaignError, CampaignStore, get_campaign_store

log = logging.getLogger("kalos.portal")

router = APIRouter()


class _StartBody(BaseModel):
    """Body for `POST /api/campaign/start` — the proposed batch the caller
    wants to start running, one entry per `{recipe, pred, std, mode, reason}`
    (the shape `_annotate`'s proposals already carry)."""

    recipes: list[dict[str, Any]]


class _ResultBody(BaseModel):
    """Body for `POST /api/campaign/result`."""

    id: str
    value: float


@router.get("/api/campaign")
def get_campaign(
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(READ)),
) -> dict[str, Any]:
    """The caller's tenant's campaign summary for the `/decide` view, or
    `{"has_campaign": false}` before any upload has ever seeded one.

    Joins in the campaign stamp `/api/latest` was published with so the summary
    can report `analysis_in_sync` — whether the analysis `/results` renders and
    this campaign provably describe the same run sheet. The join lives here
    rather than in `CampaignStore` so the store stays free of any `_LATEST`
    knowledge; `_load_latest` is imported inside the function for the same
    circular-import reason as `reanalyze_campaign` below.
    """
    from kalos.portal.app import _load_latest

    latest = _load_latest(principal.tenant) or {}
    return store.summary(
        tenant=principal.tenant,
        analysis_generation=latest.get("campaign_generation"),
    )


@router.post("/api/campaign/start")
def start_campaign(
    body: _StartBody,
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    """Append each proposed recipe as a pending, awaiting run. Requires `write`."""
    try:
        started = store.start(body.recipes, tenant=principal.tenant)
    except CampaignError as exc:
        # Authored by CampaignStore, not a raw library error — safe to echo.
        log.warning("campaign start rejected: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"started": started})


@router.post("/api/campaign/result")
def log_campaign_result(
    body: _ResultBody,
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    """Set the measured outcome on a pending run. Requires `write`. 400 on an
    unknown id or a non-finite value — an unmeasured or bad-value run must never
    silently become foldable into the base dataset."""
    try:
        run = store.set_result(body.id, body.value, tenant=principal.tenant)
    except CampaignError as exc:
        log.warning("campaign result rejected: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(run)


@router.post("/api/campaign/reanalyze")
async def reanalyze_campaign(
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    """Fold every measured pending run into the base dataset, re-run the
    engine (`_analyze`) on the grown dataset, and persist it as the new
    `/api/latest` — the loop closing (docs/CAMPAIGN_LOOP.md, "Re-analyze =
    the loop closing").

    Transactional: the fold is committed to `campaign.json` only after
    `_analyze` succeeds, and only if the campaign's `generation` is unchanged
    (`plan_fold` → `_analyze` → `commit_fold`). A failed analysis or a
    concurrent upload that reseeds mid-analysis leaves `campaign.json`
    untouched and returns a retry (409).

    `/api/latest` is a SEPARATE resource (`_LATEST`, its own lock) with no
    shared transaction, so its final write can only be guarded best-effort:
    after committing, this re-checks the campaign generation right before
    `_save_latest` and skips the write if a fresh upload reseeded in the
    meantime (its own `_save_latest` already wrote the newer analysis). This
    eliminates the multi-second race across `_analyze` and closes the
    upload-lands-after-commit case; a sub-millisecond window between the
    re-check and `_save_latest` remains (fully sealing it would need an
    ordered generation stamp on `_LATEST` itself — see docs/CAMPAIGN_LOOP.md).

    `_analyze`/`_save_latest` are imported from `kalos.portal.app` INSIDE
    this function, not at module load: `app.py` imports and mounts this
    router, so a module-level import here would be circular. `_analyze` is
    CPU-bound (a GP fit + acquisition optimization), so it runs via
    `run_in_threadpool`, matching how `/api/run` offloads it in `app.py`.
    """
    from kalos.domains import BIOPROCESS_PROFILE
    from kalos.portal.app import _analyze, _load_latest, _save_latest

    tenant = principal.tenant
    # Transactional re-analyze (docs/CAMPAIGN_LOOP.md): plan the fold in memory,
    # run the (slow, thread-offloaded) analysis, and only THEN commit. All store
    # and /api/latest access is scoped to this tenant.
    try:
        df, target, generation = store.plan_fold(tenant=tenant)
    except CampaignError as exc:
        log.warning("reanalyze rejected: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)

    # Runs still in the incubator. They have no outcome to fold, so they never
    # reach `df`; passed separately they keep the next batch from re-proposing an
    # experiment already underway.
    awaiting = store.awaiting_recipes(tenant=tenant)

    if df.empty:
        log.warning("reanalyze rejected: campaign has no base rows to analyze")
        return JSONResponse({"error": "the campaign has no data to analyze"}, status_code=400)

    try:
        # Same single analysis slot /api/run holds (kalos/portal/busy.py): a
        # reanalyze racing an upload - or another reanalyze - gets an honest
        # 503 instead of contending for the CPU with an unfinished fit.
        result = await run_in_threadpool(
            run_exclusively(
                functools.partial(
                    _analyze, df, target, profile=BIOPROCESS_PROFILE, pending=awaiting
                )
            )
        )
    except AnalysisBusy as busy:
        return JSONResponse(
            {"error": str(busy)},
            status_code=503,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )
    except Exception:  # noqa: BLE001 - normalized to a generic 400, same as /api/run
        # Full traceback logged server-side; the client only ever sees a
        # generic message, matching /api/run's catch-all in kalos/portal/app.py.
        # Nothing was committed (plan_fold does not mutate), so the campaign is
        # untouched and a retry is meaningful.
        log.exception("failed to re-analyze the campaign")
        return JSONResponse({"error": "could not analyze the campaign data"}, status_code=400)

    try:
        state = store.commit_fold(generation, tenant=tenant)
    except CampaignError as exc:
        # The campaign was reseeded/rewritten while _analyze ran: `result`
        # describes a base dataset that no longer exists. Do NOT _save_latest —
        # that would overwrite /api/latest with stale numbers over the fresh
        # upload. Leave the campaign as-is and ask the caller to retry.
        log.warning("reanalyze not committed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=409)

    # Best-effort guard on the SEPARATE /api/latest resource: if a fresh upload
    # reseeded the campaign in the small window since commit_fold wrote it, that
    # upload's own _save_latest has already published the newer analysis — our
    # `result` is now stale, so skip the write rather than clobber it. (campaign
    # and _LATEST have independent locks taken in opposite orders by the upload
    # path, so they cannot be spanned by one lock without risking deadlock; this
    # re-check is the safe ceiling — see this function's docstring.)
    current = store.get(tenant=tenant)
    if current is None or current.get("generation") != state["generation"]:
        log.warning("reanalyze committed but a fresh upload landed; skipping stale /api/latest write")
        return JSONResponse(
            {"error": "the campaign changed during re-analysis (a new upload landed); please re-analyze again"},
            status_code=409,
        )

    _save_latest(
        result,
        f"campaign round {state['round']}",
        tenant=tenant,
        # Tie the published analysis to the campaign it describes: the fold we
        # just committed. `summary()` compares this against the live generation
        # to report `analysis_in_sync`.
        campaign_generation=state["generation"],
    )
    # Return the SAME shape GET /api/latest returns: _save_latest stamps the
    # persisted state with `dataset` and `updated`, so read it back rather than
    # returning the bare `result` (which lacks those two fields the frontend's
    # PopulatedResult type expects).
    saved = _load_latest(tenant) or result
    return JSONResponse(
        {"analysis": {"has_data": True, **saved}, "campaign": store.summary(tenant=tenant)}
    )
