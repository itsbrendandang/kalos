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

import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from kalos.portal.analysis import _analyze, _annotate
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

log = logging.getLogger("kalos.portal")

# --- CPU thread budget ------------------------------------------------------- #
# The upload handler offloads its CPU-bound body (pandas parse + ~7 GP fits +
# optimize_acqf) to a worker thread so concurrent uploads do not serialize on the
# event loop. Cap the torch intra-op thread count so several concurrent fits do
# not oversubscribe the CPU and thrash. Override with KALOS_TORCH_THREADS.
_TORCH_THREADS = int(os.environ.get("KALOS_TORCH_THREADS", str(min(4, os.cpu_count() or 1))))
_torch_threads_configured = False


def _ensure_torch_threads() -> None:
    """Apply `_TORCH_THREADS` on first use. Deferred (not applied at import
    time) so `import kalos.portal.app` stays torch-free until a fit actually
    runs; idempotent, so it is cheap to call before every fit path."""
    global _torch_threads_configured
    if _torch_threads_configured:
        return
    import torch

    torch.set_num_threads(max(1, _TORCH_THREADS))
    _torch_threads_configured = True

app = FastAPI(title="Bioqore Engine API")

# Allow the kalos-web Next.js app (dev + any localhost) to call the engine.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# --- persistence of the most-recently analyzed real dataset ------------------ #
# The Overview reads /api/latest so the landing page reflects the LAST dataset a
# user actually uploaded, not the synthetic demo objective. In-memory is the
# source of truth; the JSON file is best-effort so it survives a portal restart.
_STATE_DIR = Path(os.environ.get("KALOS_STATE_DIR", Path.home() / ".kalos"))
_LATEST_PATH = _STATE_DIR / "latest_analysis.json"
_LATEST: dict | None = None
# `_analyze` now runs in a worker thread (see `run_uploaded`), so `_save_latest`
# and `_load_latest` touch the `_LATEST` global and the state file from multiple
# threads concurrently. This lock serializes the read-modify-write + file write
# so two overlapping uploads cannot interleave a half-written global or file.
_LATEST_LOCK = threading.Lock()


def _load_latest() -> dict | None:
    global _LATEST
    with _LATEST_LOCK:
        if _LATEST is None and _LATEST_PATH.exists():
            try:
                _LATEST = json.loads(_LATEST_PATH.read_text())
            except (OSError, ValueError):  # a corrupt cache must not break the API
                _LATEST = None
        return _LATEST


def _save_latest(result: dict, dataset: str) -> None:
    global _LATEST
    with _LATEST_LOCK:
        _LATEST = {**result, "dataset": dataset, "updated": time.time()}
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            _LATEST_PATH.write_text(json.dumps(_LATEST))
        except (OSError, TypeError, ValueError):
            # Disk persistence is best-effort; in-memory still serves this run and a
            # serialization hiccup must never fail the upload.
            pass


@app.get("/api/latest")
def latest() -> dict:
    """The most recent real (uploaded) analysis, for the Overview. `has_data` is
    False until the first successful /api/run so the home can show an upload
    prompt instead of pretending there is data."""
    data = _load_latest()
    if not data:
        return {"has_data": False}
    return {"has_data": True, **data}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML


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
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (8, 3))
    Y = _objectives(X)
    traj: list = []
    proposals: list = []
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


def _run_uploaded_sync(raw: bytes, target: str | None, anonymize: bool, filename: str) -> dict:
    """The CPU-bound body of an upload: parse -> analyze -> persist -> result.

    This is the heavy, blocking work (pandas parse, ~7 GP fits, optimize_acqf) and
    runs in a worker thread (see `run_uploaded`), NOT on the asyncio event loop, so
    concurrent uploads do not serialize and GET /api/latest never hangs behind a
    fit. It raises `UploadRejected` / `ValueError` / `FitError` / parser errors;
    the async wrapper maps each to the right 400 envelope. Kept fully synchronous
    so it is trivially unit-testable in isolation.
    """
    _ensure_torch_threads()
    df = _parse_upload(raw)
    result = _analyze(df, target, anonymize=anonymize)
    _save_latest(result, filename)
    return result


@app.post("/api/run")
async def run_uploaded(
    file: UploadFile = File(...),
    target: str = Form(default=""),
    anonymize: bool = Form(default=False),
) -> JSONResponse:
    from kalos.core.surrogate import FitError  # deferred: only needed to match the except below

    raw = await file.read()
    filename = file.filename or "uploaded dataset"
    try:
        # Offload the CPU-bound parse + fit + save to a worker thread so this
        # single-worker service does not block the event loop (and every other
        # request, including GET /api/latest) while a GP fit runs.
        result = await run_in_threadpool(
            _run_uploaded_sync, raw, target or None, anonymize, filename
        )
        return JSONResponse(result)
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
