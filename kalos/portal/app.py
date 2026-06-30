"""Kalos Engine portal — a small web view that runs the real engine.

A FastAPI app that, on request, runs the BoTorch optimization (single- and
multi-objective) on a synthetic bioprocess surface and returns the results, plus
a single page that charts them. It is a *viewer over the live engine*, not a
mock: every number comes from an actual BoTorch fit + acquisition.

Run:  python -m kalos.portal   (then open http://127.0.0.1:8050)
Needs the portal extra:  pip install -e ".[portal]"
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from scipy.stats import spearmanr

from kalos.core.evaluation import grouped_folds
from kalos.core.multiobjective import MultiObjectiveSurrogate, propose_multiobjective
from kalos.core.optimize import propose
from kalos.core.surrogate import DEVICE, DTYPE, Surrogate

app = FastAPI(title="Kalos Engine API")

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


def _annotate(batch: np.ndarray, mean, std, best: float, cols=None) -> list:
    """Attach predicted value, uncertainty, and an explore/exploit rationale to
    each proposed experiment. Explore = high model uncertainty (chosen to learn);
    exploit = high predicted value (chosen to win). Current human-in-the-loop BO
    research says a recommendation must carry exactly this."""
    mean = np.asarray(mean, float).reshape(-1)
    std = np.asarray(std, float).reshape(-1)
    thr = float(np.quantile(std, 2 / 3)) if len(std) > 2 else float(std.max() if len(std) else 0.0)
    eps = 0.02 * max(abs(best), 1e-9)
    rows = []
    for i in range(len(batch)):
        m, sd = float(mean[i]), float(std[i])
        gain = m - best
        if gain > eps:
            mode, reason = "exploit", f"predicted high (+{gain:.3g} vs best)"
        elif sd >= thr:
            mode, reason = "explore", f"reduce model uncertainty here (±{sd:.3g})"
        else:
            mode, reason = "explore", f"diversifies the batch (predicted {m:.3g}, ±{sd:.3g})"
        row = {"pred": round(m, 3), "std": round(sd, 3), "mode": mode, "reason": reason}
        row["vals"] = (np.round(batch[i], 3).tolist() if cols is None
                       else [round(float(batch[i][j]), 3) for j in cols])
        rows.append(row)
    return rows


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML


@app.get("/api/single")
def run_single(rounds: int = 6, q: int = 2) -> dict:
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


# --- run the engine on an uploaded data file --------------------------------- #
# Outcome-like columns are candidate TARGETS and are excluded from input features
# so the model never "predicts" titer from another measured output (leakage).
_OUTCOME_HINT = re.compile(r"titer|titre|yield|conc|purity|lipase|biomass|od\d|product|response|output|score|kda|activity|titer", re.I)
_TARGET_PREF = re.compile(r"titer|titre|lipase|yield", re.I)
_ID_HINT = re.compile(r"^(id|name|sample.*|well|index|run|experiment|round|date|time|medium|strain|recipe|batch|campaign|group|lot|notes?)$", re.I)
_GROUP_HINT = re.compile(r"medium|strain|recipe|batch|campaign|group|lot", re.I)


def _numeric_cols(df: pd.DataFrame) -> list:
    # Numeric if >=80% of NON-BLANK cells parse as numbers. Blanks are treated as
    # "absent" (filled with 0 later), so a sparse component column still counts.
    out = []
    for c in df.columns:
        s = df[c]
        nonblank = s.notna() & (s.astype(str).str.strip() != "")
        if nonblank.sum() < 3:
            continue
        if pd.to_numeric(s[nonblank], errors="coerce").notna().mean() >= 0.8:
            out.append(c)
    return out


def _analyze(df: pd.DataFrame, target: str | None = None) -> dict:
    """Run the engine on an arbitrary run sheet: pick the target (the value to
    maximize), use the process INPUTS as features (other measured outputs are
    excluded to avoid leakage), then honest grouped-CV, signed drivers, and a
    proposed next batch."""
    df = df.dropna(axis=1, how="all")
    num = _numeric_cols(df)
    if not num:
        raise ValueError("no numeric columns found")
    outcomes = [c for c in num if _OUTCOME_HINT.search(str(c))]
    if target is None or target not in df.columns:
        target = next((c for c in outcomes if _TARGET_PREF.search(str(c))), outcomes[0] if outcomes else num[-1])
    feats = []
    for c in num:
        if c == target or _ID_HINT.match(str(c).strip()) or _OUTCOME_HINT.search(str(c)):
            continue  # drop the target, ids, and OTHER measured outputs (anti-leakage)
        col = pd.to_numeric(df[c], errors="coerce")
        if col.std(skipna=True) and col.std() > 1e-9:
            feats.append(c)
    if len(feats) < 1:
        raise ValueError("no varying process-input columns found (only outputs/ids?)")
    candidate_targets = outcomes or [target]

    y_all = pd.to_numeric(df[target], errors="coerce")
    keep = y_all.notna()
    X = df.loc[keep, feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    y = y_all[keep].to_numpy(float)
    if len(y) < 6:
        raise ValueError(f"need at least 6 rows with a numeric {target!r}; got {len(y)}")
    bounds = np.vstack([X.min(0), X.max(0)])

    gcol = next((c for c in df.columns if _GROUP_HINT.search(str(c))), None)
    groups = df.loc[keep, gcol].astype(str).tolist() if gcol else [hash(tuple(np.round(r, 6))) for r in X]

    # honest out-of-fold predictions + rank correlation
    oof_a: list = []
    oof_p: list = []
    for tr, te in grouped_folds(groups, 5):
        if len(tr) < max(4, len(feats)) or len(te) < 1:
            continue
        s = Surrogate().fit(X[tr], y[tr], bounds=bounds)
        m, _ = s.posterior(X[te])
        oof_a += y[te].tolist()
        oof_p += m.tolist()
    rho = float(spearmanr(oof_p, oof_a).statistic) if len(oof_a) > 3 and np.std(oof_a) > 0 else float("nan")

    # signed drivers
    drv = []
    for c in feats:
        r = spearmanr(pd.to_numeric(df.loc[keep, c], errors="coerce").fillna(0.0), y).statistic
        drv.append([str(c), 0.0 if r != r else float(r)])
    drv.sort(key=lambda d: -abs(d[1]))
    drv = drv[:8]

    # proposed next batch, with predicted target + uncertainty + a why per row
    s = Surrogate().fit(X, y, bounds=bounds)
    batch = propose(s, bounds, q=5)
    show = [d[0] for d in drv[:4]]
    show_idx = [feats.index(f) for f in show]
    p_mean, p_std = s.posterior(batch)

    return {
        "n": int(keep.sum()), "d": len(feats), "target": str(target), "group_col": gcol,
        "targets": [str(c) for c in candidate_targets], "features": [str(c) for c in feats],
        "cv_spearman": None if rho != rho else round(rho, 3),
        "best": round(float(y.max()), 4),
        "drivers": [{"name": c, "rho": round(r, 3)} for c, r in drv],
        "proposal_features": show,
        "proposals": _annotate(batch, p_mean, p_std, float(y.max()), cols=show_idx),
        "oof": [[round(a, 4), round(p, 4)] for a, p in zip(oof_a, oof_p)],
    }


@app.post("/api/run")
async def run_uploaded(file: UploadFile = File(...), target: str = Form(default="")) -> JSONResponse:
    raw = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            text = raw.decode("utf-8", errors="replace")
            head = text[:4000]
            sep = "\t" if (name.endswith(".tsv") or head.count("\t") > head.count(",")) else ","
            df = pd.read_csv(io.StringIO(text), sep=sep)
        return JSONResponse(_analyze(df, target or None))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)
