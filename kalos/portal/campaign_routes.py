"""Kalos portal — `/api/campaign*`: many named, closed optimization loops per
tenant (M3, docs/CAMPAIGN_LOOP.md). Thin wrappers over
`kalos.portal.campaign.CampaignStore`, mirroring the router pattern in
`kalos.portal.experiments`.

Every route takes an optional `campaign_id` query parameter. Omitted, it
resolves to the caller's tenant's most-recently-updated ACTIVE campaign
(docs/CAMPAIGN_LOOP.md, "Campaign identity") — this is exactly how every
pre-existing call site (none of which ever passed an id) keeps working
unchanged. `GET /api/campaigns` (plural, no id) lists every campaign a tenant
owns, for a client that wants to address one explicitly. The two exceptions
are `archive`/`unarchive`, where `campaign_id` is REQUIRED: retiring or
restoring "whichever campaign happens to be current" is not a safe default.

Auth posture: the mutating endpoints (`start`, `result`, `reanalyze`,
`archive`, `unarchive`) require
the `write` scope via `kalos.portal.auth` (docs/HARDENING.md, Phase 1). In open
mode (no tokens configured) the anonymous principal holds read+write, so the
local dev/pilot loop is unchanged; once tokens are provisioned these endpoints
enforce them. `GET /api/campaign`/`GET /api/campaigns`/`GET /api/campaign/activity`
stay open for now (read gating is a follow-up). Per-tenant isolation of the
campaign store is already in place (docs/HARDENING.md, Phase 1b).
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

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


@router.get("/api/campaigns")
def list_campaigns(
    include_archived: bool = False,
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(READ)),
) -> list[dict[str, Any]]:
    """Every ACTIVE campaign the caller's tenant owns, most-recently-updated
    first - the left-rail listing (docs/CAMPAIGN_LOOP.md, "Campaign
    identity"). Empty list if the tenant has never uploaded anything.

    `?include_archived=true` adds the archived ones back in, flagged by their
    `archived` field - nothing is ever hard-deleted, so this is always the
    tenant's complete history (docs/CAMPAIGN_LOOP.md, "Archiving")."""
    return store.list_campaigns(tenant=principal.tenant, include_archived=include_archived)


@router.get("/api/campaign")
def get_campaign(
    campaign_id: str | None = None,
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(READ)),
) -> dict[str, Any]:
    """One campaign's summary for the `/decide` view — the tenant's
    most-recently-updated campaign if `campaign_id` is omitted, else that
    specific campaign. `{"has_campaign": false}` if nothing resolves (no
    campaign_id and the tenant has never uploaded anything, or an unknown
    id)."""
    return store.summary(tenant=principal.tenant, campaign_id=campaign_id)


@router.get("/api/campaign/activity")
def get_campaign_activity(
    campaign_id: str | None = None,
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(READ)),
) -> dict[str, Any]:
    """Per-day runs-measured series + derived stats for one campaign — the
    activity heatmap (docs/CAMPAIGN_LOOP.md, "Activity series"). Same
    campaign_id resolution as `GET /api/campaign`."""
    return store.activity(tenant=principal.tenant, campaign_id=campaign_id)


@router.post("/api/campaign/archive")
def archive_campaign(
    campaign_id: str,
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    """Hide one campaign from the left rail and from every default
    resolution, keeping all of its data queryable by id
    (docs/CAMPAIGN_LOOP.md, "Archiving"). Requires `write`. 400 on an unknown
    campaign id.

    `campaign_id` is REQUIRED here, unlike every other route on this router:
    archiving is the one operation where falling back to "whichever campaign
    is current" could retire the wrong one on a caller that simply forgot the
    parameter. FastAPI 422s a request that omits it.

    Nothing is deleted. An archived campaign still answers
    `GET /api/campaign?campaign_id=...` and `GET /api/campaign/activity`, and
    still appears in `GET /api/campaigns?include_archived=true`. It refuses
    writes (`start`/`result`/`reanalyze`/replace) until `unarchive`.
    """
    try:
        brief = store.archive(campaign_id, tenant=principal.tenant)
    except CampaignError as exc:
        log.warning("campaign archive rejected: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(brief)


@router.post("/api/campaign/unarchive")
def unarchive_campaign(
    campaign_id: str,
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    """Bring an archived campaign back into the left rail and re-enable its
    writes. Requires `write`. 400 on an unknown campaign id. `campaign_id` is
    REQUIRED - an archived campaign can never be the default, so there is
    nothing to resolve to."""
    try:
        brief = store.unarchive(campaign_id, tenant=principal.tenant)
    except CampaignError as exc:
        log.warning("campaign unarchive rejected: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(brief)


@router.post("/api/campaign/start")
def start_campaign(
    body: _StartBody,
    campaign_id: str | None = None,
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    """Append each proposed recipe as a pending, awaiting run to one
    campaign (default: the tenant's most-recently-updated). Requires
    `write`."""
    try:
        started = store.start(body.recipes, tenant=principal.tenant, campaign_id=campaign_id)
    except CampaignError as exc:
        # Authored by CampaignStore, not a raw library error — safe to echo.
        log.warning("campaign start rejected: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"started": started})


@router.post("/api/campaign/result")
def log_campaign_result(
    body: _ResultBody,
    campaign_id: str | None = None,
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    """Set the measured outcome on a pending run in one campaign (default:
    the tenant's most-recently-updated). Requires `write`. 400 on an unknown
    campaign/run id or a non-finite value — an unmeasured or bad-value run
    must never silently become foldable into the base dataset."""
    try:
        run = store.set_result(body.id, body.value, tenant=principal.tenant, campaign_id=campaign_id)
    except CampaignError as exc:
        log.warning("campaign result rejected: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(run)


@router.post("/api/campaign/reanalyze")
async def reanalyze_campaign(
    campaign_id: str | None = None,
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    """Fold every measured pending run into one campaign's base dataset,
    re-run the engine (`_analyze`) on the grown dataset, and persist it as
    the new `/api/latest` — the loop closing (docs/CAMPAIGN_LOOP.md,
    "Re-analyze = the loop closing").

    Transactional: the fold is committed to that campaign's row only after
    `_analyze` succeeds, and only if the campaign's `generation` is unchanged
    (`plan_fold` → `_analyze` → `commit_fold`). A failed analysis, or a
    concurrent `start`/`result` write landing on this SAME campaign
    mid-analysis, leaves it untouched and returns a retry (409). A fresh
    `/api/run` upload no longer risks this campaign at all — it seeds an
    UNRELATED campaign now (docs/CAMPAIGN_LOOP.md, "Campaign identity") — but
    the resolved `campaign_id` from `plan_fold` is still threaded through to
    `commit_fold` explicitly rather than re-resolved, so a fresh upload
    becoming the tenant's new "most recent" campaign in between can never
    cause this call to silently commit against the wrong campaign.

    `/api/latest` is a SEPARATE resource (`_LATEST`, its own lock) with no
    shared transaction, so its final write can only be guarded best-effort:
    after committing, this re-checks the campaign generation right before
    `_save_latest` and skips the write if a concurrent write landed in the
    meantime. This eliminates the multi-second race across `_analyze`; a
    sub-millisecond window between the re-check and `_save_latest` remains
    (fully sealing it would need an ordered generation stamp on `_LATEST`
    itself — see docs/CAMPAIGN_LOOP.md).

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
    # and /api/latest access is scoped to this tenant AND this resolved campaign.
    try:
        df, target, generation, cid = store.plan_fold(tenant=tenant, campaign_id=campaign_id)
    except CampaignError as exc:
        log.warning("reanalyze rejected: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)

    if df.empty:
        log.warning("reanalyze rejected: campaign has no base rows to analyze")
        return JSONResponse({"error": "the campaign has no data to analyze"}, status_code=400)

    try:
        result = await run_in_threadpool(_analyze, df, target, profile=BIOPROCESS_PROFILE)
    except Exception:  # noqa: BLE001 - normalized to a generic 400, same as /api/run
        # Full traceback logged server-side; the client only ever sees a
        # generic message, matching /api/run's catch-all in kalos/portal/app.py.
        # Nothing was committed (plan_fold does not mutate), so the campaign is
        # untouched and a retry is meaningful.
        log.exception("failed to re-analyze the campaign")
        return JSONResponse({"error": "could not analyze the campaign data"}, status_code=400)

    try:
        # `campaign_id=cid`: the EXACT campaign plan_fold resolved against, not
        # `campaign_id` (the possibly-omitted request param) — see the
        # docstring above and `CampaignStore.plan_fold`.
        state = store.commit_fold(generation, tenant=tenant, campaign_id=cid)
    except CampaignError as exc:
        # The campaign was rewritten while _analyze ran: `result` describes a
        # base dataset that no longer exists. Do NOT _save_latest — that would
        # overwrite /api/latest with stale numbers over the newer write. Leave
        # the campaign as-is and ask the caller to retry.
        log.warning("reanalyze not committed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=409)

    # Best-effort guard on the SEPARATE /api/latest resource: if a concurrent
    # write landed on this campaign in the small window since commit_fold wrote
    # it, skip the write rather than clobber whatever landed after. (campaign
    # and _LATEST have independent locks taken in opposite orders by the upload
    # path, so they cannot be spanned by one lock without risking deadlock; this
    # re-check is the safe ceiling — see this function's docstring.)
    current = store.get(tenant=tenant, campaign_id=cid)
    if current is None or current.get("generation") != state["generation"]:
        log.warning("reanalyze committed but a concurrent write landed; skipping stale /api/latest write")
        return JSONResponse(
            {"error": "the campaign changed during re-analysis (a concurrent write landed); please re-analyze again"},
            status_code=409,
        )

    _save_latest(result, f"campaign round {state['round']}", tenant=tenant, campaign_id=cid)
    # Return the SAME shape GET /api/latest returns: _save_latest stamps the
    # persisted state with `dataset` and `updated`, so read it back rather than
    # returning the bare `result` (which lacks those two fields the frontend's
    # PopulatedResult type expects).
    saved = _load_latest(tenant) or result
    return JSONResponse(
        {"analysis": {"has_data": True, **saved}, "campaign": store.summary(tenant=tenant, campaign_id=cid)}
    )
