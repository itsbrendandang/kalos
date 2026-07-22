"""Kalos portal — `/api/campaign*`: the closed optimization loop (M3,
docs/CAMPAIGN_LOOP.md). Thin wrappers over `kalos.portal.campaign.CampaignStore`,
mirroring the router pattern in `kalos.portal.experiments`.

Auth posture (deliberate): these endpoints are UNAUTHENTICATED, the same as
`/api/run` and `/api/latest` — the campaign loop is the browser-facing surface
`/decide` drives, and the portal is a single-local-user tool bound to localhost.
This is intentionally *not* gated like the machine-to-machine
`/api/experiments/{id}/result` runner channel (which uses a runner token). If
this portal is ever exposed beyond localhost, `/api/campaign/result` — which
injects `base_rows` that become ground truth for the surrogate — must be gated
too; until then the trust boundary is the loopback interface, matching the rest
of the browser-facing API.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

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
def get_campaign(store: CampaignStore = Depends(get_campaign_store)) -> dict[str, Any]:
    """The current campaign summary for the `/decide` view, or
    `{"has_campaign": false}` before any upload has ever seeded one."""
    return store.summary()


@router.post("/api/campaign/start")
def start_campaign(
    body: _StartBody, store: CampaignStore = Depends(get_campaign_store)
) -> JSONResponse:
    """Append each proposed recipe as a pending, awaiting run."""
    try:
        started = store.start(body.recipes)
    except CampaignError as exc:
        # Authored by CampaignStore, not a raw library error — safe to echo.
        log.warning("campaign start rejected: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"started": started})


@router.post("/api/campaign/result")
def log_campaign_result(
    body: _ResultBody, store: CampaignStore = Depends(get_campaign_store)
) -> JSONResponse:
    """Set the measured outcome on a pending run. 400 on an unknown id or a
    non-finite value — an unmeasured or bad-value run must never silently
    become foldable into the base dataset."""
    try:
        run = store.set_result(body.id, body.value)
    except CampaignError as exc:
        log.warning("campaign result rejected: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(run)


@router.post("/api/campaign/reanalyze")
async def reanalyze_campaign(
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Fold every measured pending run into the base dataset, re-run the
    engine (`_analyze`) on the grown dataset, and persist it as the new
    `/api/latest` — the loop closing (docs/CAMPAIGN_LOOP.md, "Re-analyze =
    the loop closing").

    `_analyze`/`_save_latest` are imported from `kalos.portal.app` INSIDE
    this function, not at module load: `app.py` imports and mounts this
    router, so a module-level import here would be circular. `_analyze` is
    CPU-bound (a GP fit + acquisition optimization), so it runs via
    `run_in_threadpool`, matching how `/api/run` offloads it in `app.py`.
    """
    from kalos.domains import BIOPROCESS_PROFILE
    from kalos.portal.app import _analyze, _load_latest, _save_latest

    # Transactional re-analyze (docs/CAMPAIGN_LOOP.md): plan the fold in memory,
    # run the (slow, thread-offloaded) analysis, and only THEN commit — so an
    # analysis failure or a concurrent upload never leaves a half-folded
    # campaign or clobbers /api/latest with stale numbers.
    try:
        df, target, generation = store.plan_fold()
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
        state = store.commit_fold(generation)
    except CampaignError as exc:
        # The campaign was reseeded/rewritten while _analyze ran: `result`
        # describes a base dataset that no longer exists. Do NOT _save_latest —
        # that would overwrite /api/latest with stale numbers over the fresh
        # upload. Leave the campaign as-is and ask the caller to retry.
        log.warning("reanalyze not committed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=409)

    _save_latest(result, f"campaign round {state['round']}")
    # Return the SAME shape GET /api/latest returns: _save_latest stamps the
    # persisted state with `dataset` and `updated`, so read it back rather than
    # returning the bare `result` (which lacks those two fields the frontend's
    # PopulatedResult type expects).
    saved = _load_latest() or result
    return JSONResponse({"analysis": {"has_data": True, **saved}, "campaign": store.summary()})
