"""Kalos Engine portal — a small web view that runs the real engine.

A FastAPI app that, on request, runs the BoTorch optimization (single- and
multi-objective) on a synthetic bioprocess surface and returns the results, plus
a single page that charts them. It is a *viewer over the live engine*, not a
mock: every number comes from an actual BoTorch fit + acquisition.

Run:  python -m kalos.portal   (then open http://127.0.0.1:8050)
Needs the portal extra:  pip install -e ".[portal]"

This module wires up the FastAPI app, CORS, the `/` HTML route, and the
legacy `/api/run|latest|single|multi` routes. The rest of the portal lives in
sibling modules:
  - `kalos.portal.uploads` - the untrusted-input boundary (`_parse_upload`,
    `UploadRejected`, the upload size/shape caps).
  - `kalos.portal.analysis` - the science (`_analyze` and its helpers).
  - `kalos.portal.serialization` - JSON-safe serialization helpers.
  - `kalos.portal.experiments` - the M2 `/api/experiments*` surface.
Names historically imported `from kalos.portal.app import ...` are re-exported
below so existing call sites and tests keep working unchanged.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from kalos.portal.busy import RETRY_AFTER_SECONDS, AnalysisBusy, run_exclusively

from kalos.domains import BIOPROCESS_PROFILE, GENERIC_PROFILE, ColumnRoles
from kalos.portal.analysis import _analyze, _annotate
from kalos.portal.auth import READ, WRITE, Principal, get_authenticator, require_scope
from kalos.portal.config import (
    cors_config,
    is_local_client,
    log_security_posture,
    open_access_explicitly_allowed,
)
from kalos.portal.campaign_routes import router as _campaign_router
from kalos.portal.experiments import get_lock_path, get_store
from kalos.portal.experiments import router as _experiments_router
from kalos.portal.uploads import (
    MAX_COLUMNS,
    MAX_CSV_ROWS,
    MAX_UPLOAD_BYTES,
    MAX_XLSX_CELLS,
    UploadRejected,
    _ERR_FIT,
    _ERR_PARSE,
    _ERR_TOO_MANY_FIT_ROWS,
    _parse_upload,
)
from kalos.providers import provider_status
from kalos.store import SqliteStore

log = logging.getLogger("kalos.portal")

# --- CPU thread budget ------------------------------------------------------- #
# The upload handler offloads its CPU-bound body (pandas parse + ~7 GP fits +
# optimize_acqf) to a worker thread so concurrent uploads do not serialize on the
# event loop. Cap the torch intra-op thread count so several concurrent fits do
# not oversubscribe the CPU and thrash. Override with KALOS_TORCH_THREADS.
# Empty-string-tolerant like the sizing knobs in uploads.py (see the comment
# there): a compose/k8s template passing `KALOS_TORCH_THREADS=` must mean
# "use the default", not crash the import.
#
# KALOS_TORCH_THREADS=0 means DO NOT PIN - leave torch's own default. Not
# hypothetical tuning surface: on the linux-aarch64 torch build inside the
# reference container, the 4-thread pin took the demo analyze from 47s
# (unpinned) to over 600s - a >12x pathological slowdown from the pin
# interacting with that build's OpenBLAS threading, measured 2026-09-10 on
# the first real compose deployment. The 4-thread default is kept for bare
# metal (where it was tuned and behaves); containers, which already isolate
# CPU, should set 0 - deploy/docker-compose.yaml now does.
_TORCH_THREADS = int(
    os.environ.get("KALOS_TORCH_THREADS", "").strip() or str(min(4, os.cpu_count() or 1))
)
_torch_threads_configured = False


def _ensure_torch_threads() -> None:
    """Apply `_TORCH_THREADS` on first use. Deferred (not applied at import
    time) so `import kalos.portal.app` stays torch-free until a fit actually
    runs; idempotent, so it is cheap to call before every fit path."""
    global _torch_threads_configured
    if _torch_threads_configured:
        return
    if _TORCH_THREADS <= 0:
        # 0 (or negative) = do not pin at all; torch keeps its own default.
        # See the KALOS_TORCH_THREADS comment above for the measured reason.
        _torch_threads_configured = True
        return
    import torch

    torch.set_num_threads(_TORCH_THREADS)
    _torch_threads_configured = True

app = FastAPI(title="Kalos Engine API")

# CORS policy: an explicit allowlist when `KALOS_CORS_ORIGINS` is set (production),
# else the permissive localhost default (dev/pilot). See kalos.portal.config.
# Refuse REMOTE callers while running unauthenticated.
#
# With no tokens provisioned every request resolves to an anonymous read+write
# principal on the `default` tenant (kalos.portal.auth), so an open portal that is
# reachable off-box publishes uploads, analyses and campaign mutation to anyone who
# can route to it. Until now that was guarded by a startup warning only.
#
# This is enforced on the PEER ADDRESS rather than on the bind host on purpose. A
# bind-time check lives in `kalos/portal/__main__.py`, but it can be sidestepped
# entirely by launching `uvicorn kalos.portal.app:app --host 0.0.0.0`, which is
# exactly what a real deployment does. The peer address is the actual threat model
# and cannot be avoided by choosing a different entrypoint.
#
# Local development is untouched: loopback callers are always served, so open mode
# still works on a bench machine and in tests. Auth is re-read per request
# (deliberately, for rotation without a restart), so provisioning tokens lifts this
# immediately with no restart either.
@app.middleware("http")
async def _refuse_open_remote_access(request, call_next):
    if not is_local_client(request.client.host if request.client else None):
        if not get_authenticator().enforces() and not open_access_explicitly_allowed():
            log.warning(
                "refused a remote request to an unauthenticated portal from %s %s",
                request.client.host if request.client else "unknown",
                request.url.path,
            )
            # 503 rather than 401: the caller has no credential to supply and
            # nothing they can send will help. This is a server posture problem,
            # and saying so points the operator at the fix instead of sending the
            # client hunting for a token that does not exist.
            return JSONResponse(
                {
                    "error": (
                        "This kalos portal is running without authentication and "
                        "therefore only serves local requests. The operator should "
                        "set KALOS_AUTH_TOKENS_FILE to enable remote access."
                    )
                },
                status_code=503,
            )
    return await call_next(request)


app.add_middleware(CORSMiddleware, **cors_config())

# Log the effective security posture (auth + CORS) once at import/startup.
log_security_posture(auth_enforced=get_authenticator().enforces())

_HTML = (Path(__file__).parent / "index.html").read_text()
BOUNDS = np.array([[0, 0, 0], [1, 1, 1]], float)
TITER_OPT = np.array([0.7, 0.3, 0.5])
PURITY_OPT = np.array([0.2, 0.8, 0.4])  # different recipe -> titer/purity trade off


# Synthetic but honest demo objectives in real units: titer 0..22 mg/L (peaks at
# TITER_OPT), purity 0..100 % (peaks at a different recipe, so they trade off).
def _titer(X: np.ndarray) -> np.ndarray:
    d = np.sum((np.atleast_2d(X) - TITER_OPT) ** 2, axis=1)
    return 22.0 * np.exp(-3.0 * d)


def _purity(X: np.ndarray) -> np.ndarray:
    d = np.sum((np.atleast_2d(X) - PURITY_OPT) ** 2, axis=1)
    return 100.0 * np.exp(-3.0 * d)


def _objectives(X: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(X)
    return np.stack([_titer(x), _purity(x)], axis=-1)


# --- persistence of the most-recently analyzed real dataset, PER TENANT ------ #
# The Overview reads /api/latest so the landing page reflects the LAST dataset a
# tenant actually uploaded. Keyed by tenant (docs/HARDENING.md, Phase 1b) so two
# tenants never see each other's analysis. In-memory is the source of truth; a
# per-tenant JSON file is best-effort so it survives a portal restart.
_STATE_DIR = Path(os.environ.get("KALOS_STATE_DIR", Path.home() / ".kalos"))
_LATEST_DIR = _STATE_DIR / "latest"
_LATEST: dict[str, dict] = {}  # tenant -> latest analysis state
# `_analyze` runs in a worker thread (see `run_uploaded`), so `_save_latest`/
# `_load_latest` touch the `_LATEST` map and the per-tenant files from multiple
# threads. This lock serializes the read-modify-write + file write so two
# overlapping uploads cannot interleave a half-written entry or file.
_LATEST_LOCK = threading.Lock()


def _latest_path(tenant: str) -> Path:
    """Per-tenant best-effort cache file. The tenant is sanitized to a safe
    filename so a token-provisioned tenant id can never traverse the path."""
    safe = "".join(c if (c.isalnum() or c in "_.-") else "_" for c in tenant) or "default"
    return _LATEST_DIR / f"{safe}.json"


def _load_latest(tenant: str = "default") -> dict | None:
    with _LATEST_LOCK:
        if tenant not in _LATEST:
            path = _latest_path(tenant)
            if not path.exists():
                return None
            try:
                _LATEST[tenant] = json.loads(path.read_text())
            except (OSError, ValueError):  # a corrupt cache must not break the API
                return None
        return _LATEST.get(tenant)


def _save_latest(
    result: dict, dataset: str, *, tenant: str = "default", campaign_generation: str | None = None
) -> None:
    """Publish `result` as this tenant's `/api/latest`.

    `campaign_generation` is the campaign this analysis describes. The campaign
    and `_LATEST` are separate resources with independent locks and independent
    persistence, so nothing else ties them together: without the stamp the two
    can describe different run sheets while both look valid, which is how
    `/results` and `/decide` came to report different bests off the same portal.
    `CampaignStore.summary()` compares this stamp against the live campaign to
    report `analysis_in_sync` (docs/CAMPAIGN_LOOP.md, "Analysis/campaign
    coherence"). `None` means the analysis is not tied to any campaign, which
    reads as out-of-sync rather than as fine.
    """
    with _LATEST_LOCK:
        state = {
            **result,
            "dataset": dataset,
            "updated": time.time(),
            "campaign_generation": campaign_generation,
        }
        _LATEST[tenant] = state
        try:
            path = _latest_path(tenant)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state))
        except (OSError, TypeError, ValueError):
            # Disk persistence is best-effort; in-memory still serves this run and a
            # serialization hiccup must never fail the upload.
            pass


@app.get("/api/latest")
def latest(principal: Principal = Depends(require_scope(READ))) -> dict:
    """The most recent real (uploaded) analysis for the caller's tenant, for the
    Overview. `has_data` is False until the first successful /api/run so the home
    can show an upload prompt instead of pretending there is data."""
    data = _load_latest(principal.tenant)
    if not data:
        return {"has_data": False}
    return {"has_data": True, **data}


@app.get("/api/providers")
def providers(principal: Principal = Depends(require_scope(READ))) -> dict:
    """Credential status for every external-provider slot (kalos.providers) -
    which ones are live, which are keyless fallbacks, and what each unlocks.
    Never leaks a credential value, only env var names."""
    return {"providers": provider_status()}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe for a deploy healthcheck (deploy/Dockerfile.engine).

    Unauthenticated BY DESIGN, and safe to leave that way: no
    `Depends(require_scope(...))`, the same pattern `/` above already uses.
    The response is fixed to exactly `{"status": "ok"}` - no tenant data, no
    config echo, no store contents, nothing beyond "this process is up and
    answering HTTP requests". `engine_version` is deliberately NOT included:
    verified by reading every route in this module that `kalos.__version__`
    is not exposed anywhere unauthenticated today (not `/`'s index.html, not
    any other public route), so this stays the minimal liveness fact rather
    than becoming the first place the version leaves the process without a
    token.

    Must never touch the database - that is what `/readyz` is for. A
    liveness probe that can block on a locked/slow SQLite file would make an
    orchestrator restart a perfectly healthy process because the DB, not the
    process, was briefly unavailable.
    """
    return {"status": "ok"}


@app.get("/readyz")
def readyz(store: SqliteStore = Depends(get_store)) -> JSONResponse:
    """Readiness probe: unlike `/healthz`, this DOES touch the database - on
    purpose, to answer "can this instance actually serve requests that need
    the store", not just "is the process up".

    Read-only: `store.list()` is a plain SELECT, never a write, so a
    readiness check can never itself be the thing that corrupts or contends
    for the store it is checking. Degrades to a 503 with a `reason` on any
    failure (missing/corrupt/locked database file) instead of raising, so an
    orchestrator's readiness probe always gets a normal HTTP response rather
    than a stack trace.
    """
    try:
        store.list()
    except Exception as exc:  # noqa: BLE001 - any store failure means "not ready", not a 500
        return JSONResponse({"status": "unavailable", "reason": str(exc)}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.get("/fonts/Satoshi-Variable.woff2")
def satoshi_font() -> FileResponse:
    """Serve the one brand typeface the portal page needs.

    Satoshi is the kalos brand face (docs/DESIGN.md) and is not on Google Fonts, so
    it ships vendored beside index.html under the Fontshare license in
    `fonts/SATOSHI-LICENSE.txt`. Served as a single explicit route rather than
    a StaticFiles mount so the portal never exposes a browsable directory.
    Immutable + long max-age: the filename changes if the font ever does.
    """
    return FileResponse(
        Path(__file__).parent / "fonts" / "Satoshi-Variable.woff2",
        media_type="font/woff2",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/single")
def run_single(rounds: int = 6, q: int = 2) -> dict:
    from kalos.core.optimize import propose
    from kalos.core.surrogate import Surrogate

    _ensure_torch_threads()
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (6, 3))
    y = _titer(X)
    traj = [{"round": 0, "best": float(y.max()), "n": int(len(y))}]
    proposals: list = []
    for r in range(rounds):
        s = Surrogate().fit(X, y, bounds=BOUNDS)
        nxt = propose(s, BOUNDS, q=q)
        X = np.vstack([X, nxt])
        y = np.concatenate([y, _titer(nxt)])
        traj.append({"round": r + 1, "best": float(y.max()), "n": int(len(y))})
    # annotate the last proposed batch with predicted value + uncertainty + why
    incumbent = float(y[:-q].max())  # best before this batch was measured
    mean, std = s.posterior(nxt)
    proposals = _annotate(nxt, mean, std, incumbent)
    bi = int(y.argmax())
    return {
        "trajectory": traj,
        "proposals": proposals,
        "best_recipe": np.round(X[bi], 3).tolist(),
        "best_value": float(y[bi]),
        "optimum": TITER_OPT.tolist(),
    }


@app.get("/api/multi")
def run_multi(rounds: int = 5, q: int = 2) -> dict:
    from kalos.core.multiobjective import MultiObjectiveSurrogate, propose_multiobjective
    from kalos.core.surrogate import DEVICE, DTYPE

    _ensure_torch_threads()
    rounds = max(1, rounds)  # at least one round, so last_batch is always bound
    q = max(1, q)
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (8, 3))
    Y = _objectives(X)
    traj: list = []
    proposals: list = []
    last_batch = np.empty((0, X.shape[1]))
    for r in range(rounds):
        s = MultiObjectiveSurrogate().fit(X, Y, bounds=BOUNDS)
        nxt = propose_multiobjective(s, BOUNDS, q=q)
        X = np.vstack([X, nxt])
        Y = np.vstack([Y, _objectives(nxt)])
        _, py = s.pareto()
        traj.append(
            {"round": r + 1, "pareto": int(len(py)), "best_titer": float(Y[:, 0].max()), "best_purity": float(Y[:, 1].max())}
        )
        last_batch = nxt
    s = MultiObjectiveSurrogate().fit(X, Y, bounds=BOUNDS)
    assert s.model is not None  # fit() always sets the ModelListGP
    # predicted titer + purity (with uncertainty) for each proposed experiment
    import torch
    post = s.model.posterior(torch.as_tensor(last_batch, dtype=DTYPE, device=DEVICE))
    mY = post.mean.detach().cpu().numpy().reshape(len(last_batch), -1)
    sdY = post.variance.clamp_min(1e-12).sqrt().detach().cpu().numpy().reshape(len(last_batch), -1)
    sd_mean = sdY.mean(axis=1)
    sd_thr = float(np.quantile(sd_mean, 2 / 3)) if len(sd_mean) > 2 else float(sd_mean.max())
    proposals = [{
        "vals": np.round(last_batch[i], 3).tolist(),
        "pred_titer": round(float(mY[i, 0]), 1), "pred_purity": round(float(mY[i, 1]), 0),
        "mode": "explore" if sd_mean[i] >= sd_thr else "exploit",
        "reason": "fills a gap on the titer/purity frontier",
    } for i in range(len(last_batch))]
    _, py = s.pareto()
    order = np.argsort(py[:, 0])
    return {
        "trajectory": traj,
        "pareto": [{"titer": float(py[i, 0]), "purity": float(py[i, 1])} for i in order],
        "all": [{"titer": float(Y[i, 0]), "purity": float(Y[i, 1])} for i in range(len(Y))],
        "proposals": proposals,
    }


def _decide_roles(df: pd.DataFrame, target: str | None) -> tuple[ColumnRoles | None, dict]:
    """Decide column roles for `roles=auto` from a normalization plan.

    `propose_plan` uses the configured provider (`KALOS_LLM_PROVIDER`):
    TypeSafe's per-column typed judgments, an LLM, or - with no provider
    configured, or on any provider failure - the deterministic offline plan.
    The plan only ever sees the identity-screened payload.

    Returns `(roles, decision)`. `roles` is `None` when the plan names no
    target or cannot be built, and the caller then infers roles from the
    bioprocess profile exactly as without `roles=auto`; `decision` says which
    happened and carries each column's role and rationale (column names and
    probabilities only, never cell values). An explicit `target` form field
    overrides the plan's target.
    """
    from dataclasses import replace

    from kalos.normalize import plan_to_roles, propose_plan

    try:
        plan = propose_plan(df)
    except ValueError as err:
        # The offline fallback rejects a sheet it reads as having two outcome
        # columns; that is a reason to infer roles the old way, not to fail.
        log.warning("roles=auto: no usable plan (%s); inferring roles", type(err).__name__)
        return None, {"applied": False, "created_by": None, "model": None, "columns": [],
                      "reason": "no consistent plan; roles inferred from the bioprocess profile"}
    roles = plan_to_roles(plan)
    if roles is not None and target:
        roles = replace(
            roles,
            target=target,
            features=tuple(c for c in roles.features if c != target),
            ids=tuple(c for c in roles.ids if c != target),
            groups=None if roles.groups == target else roles.groups,
        )
    decision = {
        "applied": roles is not None,
        "created_by": plan.created_by,
        "model": plan.model,
        "columns": [{"column": c.raw_name, "role": c.role, "note": c.note} for c in plan.columns],
    }
    if roles is None:
        decision["reason"] = "the plan named no target; roles inferred from the bioprocess profile"
    return roles, decision


def _run_uploaded_sync(
    raw: bytes, target: str | None, anonymize: bool, filename: str, roles_json: str = "",
    *, tenant: str = "default",
) -> dict:
    """The CPU-bound body of an upload: parse -> analyze -> persist -> result.

    This is the heavy, blocking work (pandas parse, ~7 GP fits, optimize_acqf) and
    runs in a worker thread (see `run_uploaded`), NOT on the asyncio event loop, so
    concurrent uploads do not serialize and GET /api/latest never hangs behind a
    fit. It raises `UploadRejected` / `ValueError` / `FitError` / parser errors;
    the async wrapper maps each to the right 400 envelope. Kept fully synchronous
    so it is trivially unit-testable in isolation.

    When `roles_json` is a non-empty JSON object it is parsed into a `ColumnRoles`
    schema and the analysis runs domain-neutrally (the generic profile) instead of
    inferring bioprocess roles from column names. The literal `"auto"` asks
    `kalos.normalize.propose_plan` to decide the roles (TypeSafe typed judgments
    when `KALOS_LLM_PROVIDER=typesafe` and a key is set; see `_decide_roles`).
    Omitted, the bioprocess profile infers roles exactly as before.
    """
    _ensure_torch_threads()
    df = _parse_upload(raw)
    if roles_json.strip().lower() == "auto":
        roles, decision = _decide_roles(df, target)
        if roles is not None:
            result = _analyze(df, target, anonymize=anonymize, roles=roles, profile=GENERIC_PROFILE)
        else:
            result = _analyze(df, target, anonymize=anonymize, profile=BIOPROCESS_PROFILE)
        result["role_decision"] = decision
    elif roles_json.strip():
        roles = ColumnRoles.from_dict(json.loads(roles_json))
        result = _analyze(df, target, anonymize=anonymize, roles=roles, profile=GENERIC_PROFILE)
    else:
        result = _analyze(df, target, anonymize=anonymize, profile=BIOPROCESS_PROFILE)
    generation: str | None = None
    try:
        # A fresh upload starts a fresh campaign (docs/CAMPAIGN_LOOP.md,
        # "Seeding") for THIS tenant. Best-effort: a seeding failure must never
        # break the upload response the client is waiting on.
        from kalos.portal.campaign import get_campaign_store

        generation = get_campaign_store().seed(
            df, result["target"], result["proposal_features"], tenant=tenant
        )
    except Exception:  # noqa: BLE001 - seeding must never fail the upload
        log.exception("failed to seed the campaign from an uploaded run sheet")
    # Seed BEFORE publishing so the analysis can carry the generation it was
    # seeded alongside. The two resources take their locks sequentially (never
    # nested), so the order is free of deadlock either way. If seeding failed,
    # the stamp is None and the campaign reports itself out of sync with
    # /api/latest instead of the two silently describing different run sheets.
    _save_latest(result, filename, tenant=tenant, campaign_generation=generation)
    return result


@app.post("/api/run")
async def run_uploaded(
    file: UploadFile = File(...),
    target: str = Form(default=""),
    anonymize: bool = Form(default=False),
    roles: str = Form(default=""),
    _principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    from kalos.core.surrogate import FitError  # deferred: only needed to match the except below

    raw = await file.read()
    filename = file.filename or "uploaded dataset"
    tenant = _principal.tenant
    try:
        # Offload the CPU-bound parse + fit + save to a worker thread so this
        # single-worker service does not block the event loop (and every other
        # request, including GET /api/latest) while a GP fit runs - under the
        # single analysis slot (kalos/portal/busy.py): a second upload while
        # one is fitting gets an honest 503, and a disconnected client's
        # orphaned fit keeps the slot until it truly finishes.
        result = await run_in_threadpool(
            run_exclusively(
                functools.partial(
                    _run_uploaded_sync, raw, target or None, anonymize, filename, roles, tenant=tenant
                )
            )
        )
        return JSONResponse(result)
    except AnalysisBusy as busy:
        return JSONResponse(
            {"error": str(busy)},
            status_code=503,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )
    except UploadRejected as rej:
        # A guard tripped: the message is already generic and safe to return.
        log.warning("upload rejected: %s", rej)
        return JSONResponse({"error": str(rej)}, status_code=400)
    except FitError:
        # A numerically-hard-but-valid file that defeated the jittered fit retry.
        # Give it a DISTINCT message so it does not masquerade as a parse failure;
        # the exception text itself is already generic, but we do not echo it.
        log.warning("upload rejected: model could not be fit (ill-conditioned rows)")
        return JSONResponse({"error": _ERR_FIT}, status_code=400)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, UnicodeError):
        # Log the full traceback server-side for debugging; return a generic
        # message so no parser detail, column name, cell value, or stack trace
        # ever reaches the (unauthenticated) client.
        log.exception("failed to parse or analyze an uploaded file")
        return JSONResponse({"error": _ERR_PARSE}, status_code=400)
    except Exception:  # noqa: BLE001 - normalized to a generic 400 below
        # Catch-all so a fit-time failure that is NOT a plain ValueError still
        # returns the {error} JSON envelope the frontend parses, never FastAPI's
        # default text/plain HTTP 500. This covers torch.linalg.LinAlgError from a
        # GP fit and an AssertionError from the leakage guard, among others. As
        # above, the full traceback is logged server-side and only the generic
        # message reaches the (unauthenticated) client.
        log.exception("failed to parse or analyze an uploaded file")
        return JSONResponse({"error": _ERR_PARSE}, status_code=400)


# --- M2.2: /api/experiments - thin wrappers over the store + Singleton runner  #
# (docs/M2_INTEGRATION.md, "Portal API additions"). The routes themselves live
# in `kalos.portal.experiments`; mounted here so they are served by this app.
app.include_router(_experiments_router)

# --- campaign loop: /api/campaign* (docs/CAMPAIGN_LOOP.md) - the closed
# optimization loop: propose -> run -> log outcome -> re-propose. Routes live
# in `kalos.portal.campaign_routes`; mounted here so they are served by this
# app, same pattern as the experiments router above.
app.include_router(_campaign_router)

__all__ = [
    "app",
    "get_store",
    "get_lock_path",
    "_analyze",
    "UploadRejected",
    "MAX_COLUMNS",
    "MAX_CSV_ROWS",
    "MAX_UPLOAD_BYTES",
    "MAX_XLSX_CELLS",
    "_ERR_FIT",
    "_ERR_PARSE",
    "_ERR_TOO_MANY_FIT_ROWS",
]
