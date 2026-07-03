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
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from scipy.stats import spearmanr

from kalos import __version__ as ENGINE_VERSION
from kalos.core.conformal import q_from_residuals
from kalos.core.evaluation import grouped_cv_report
from kalos.core.splits import row_hash_groups
from kalos.core.multiobjective import MultiObjectiveSurrogate, propose_multiobjective
from kalos.core.optimize import propose
from kalos.core.surrogate import DEVICE, DTYPE, Surrogate
from kalos.data.anonymizer import _hash
from kalos.portal.validate import column_provenance, provenance_dicts

log = logging.getLogger("kalos.portal")

app = FastAPI(title="Kalos Engine API")

# --- upload safety limits ---------------------------------------------------- #
# This engine ingests untrusted run sheets from external clients, so the raw
# upload and its parsed shape are capped to bound memory and blunt zip-bomb
# expansion. Sizes are configurable via env; the CSV/xlsx caps are constants.
_MAX_UPLOAD_MB = float(os.environ.get("KALOS_MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = int(_MAX_UPLOAD_MB * 1024 * 1024)
MAX_CSV_ROWS = 100_000       # rows read from a CSV/TSV upload
MAX_COLUMNS = 512            # columns allowed in any upload (CSV or xlsx)
MAX_XLSX_CELLS = 2_000_000   # rows * cols ceiling for a parsed xlsx (zip-bomb guard)
_ZIP_MAGIC = b"PK\x03\x04"   # xlsx/xls-as-zip start-of-file marker

# Generic, non-leaking messages. We never echo the parser error, a column name,
# or a cell value back to an unauthenticated caller.
_ERR_PARSE = "Could not parse the uploaded file. Check it is a CSV or Excel run-sheet."
_ERR_TOO_LARGE = "The uploaded file is too large."
_ERR_TOO_MANY_COLUMNS = "The uploaded file has too many columns."


class UploadRejected(ValueError):
    """A client upload failed a safety guard. Carries a generic, safe message."""

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


# --- persistence of the most-recently analyzed real dataset ------------------ #
# The Overview reads /api/latest so the landing page reflects the LAST dataset a
# user actually uploaded, not the synthetic demo objective. In-memory is the
# source of truth; the JSON file is best-effort so it survives a portal restart.
_STATE_DIR = Path(os.environ.get("KALOS_STATE_DIR", Path.home() / ".kalos"))
_LATEST_PATH = _STATE_DIR / "latest_analysis.json"
_LATEST: dict | None = None


def _load_latest() -> dict | None:
    global _LATEST
    if _LATEST is None and _LATEST_PATH.exists():
        try:
            _LATEST = json.loads(_LATEST_PATH.read_text())
        except (OSError, ValueError):  # a corrupt cache must not break the API
            _LATEST = None
    return _LATEST


def _save_latest(result: dict, dataset: str) -> None:
    global _LATEST
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


# Deterministic seed for the analyze path. The same upload -> the same GP fit and
# the same proposed batch, which matters for client reproducibility and audit.
ANALYZE_SEED = 1234


def _seed_everything(seed: int = ANALYZE_SEED) -> None:
    """Seed torch + numpy so one upload yields one deterministic set of proposals."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Suffix duplicate column labels (`X`, `X.1`, ...) so every column is a Series.

    A run sheet with a repeated header would otherwise make `df[label]` return a
    2-D frame and break the analysis. pandas already does this on CSV read; we do
    it here too so a directly-built frame (or an xlsx with duplicate headers) is
    handled identically, and the duplicate stays visible in the provenance report.
    """
    if not df.columns.duplicated().any():
        return df
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for col in df.columns:
        key = str(col)
        if key in seen:
            seen[key] += 1
            new_cols.append(f"{key}.{seen[key]}")
        else:
            seen[key] = 0
            new_cols.append(key)
    out = df.copy()
    out.columns = new_cols
    return out


def _analyze(
    df: pd.DataFrame, target: str | None = None, *, anonymize: bool = False
) -> dict:
    """Run the engine on an arbitrary run sheet: pick the target (the value to
    maximize), use the process INPUTS as features (other measured outputs are
    excluded to avoid leakage), then honest grouped-CV, signed drivers, and a
    proposed next batch.

    Deterministic: seeds torch + numpy up front so the same sheet gives the same
    proposals. Returns a per-column `provenance` report (what was kept/dropped and
    why) plus `seed`, `timestamp`, and `engine_version` for audit. When
    `anonymize` is True, identifier-type column names are replaced with stable
    pseudonyms in the response (the owner UI keeps real names when False)."""
    _seed_everything()
    df = _dedupe_columns(df.dropna(axis=1, how="all"))
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
    X_raw = df.loc[keep, feats].apply(pd.to_numeric, errors="coerce")   # NaN preserved (for grouping)
    X = X_raw.fillna(0.0).to_numpy(float)                                # zero-filled (for the GP)
    y = y_all[keep].to_numpy(float)
    if len(y) < 6:
        raise ValueError(f"need at least 6 rows with a numeric {target!r}; got {len(y)}")
    bounds = np.vstack([X.min(0), X.max(0)])

    gcol = next((c for c in df.columns if _GROUP_HINT.search(str(c))), None)
    if gcol:
        groups = df.loc[keep, gcol].astype(str).tolist()
    else:
        # group on the RAW values (NaN preserved) so rows missing different
        # components are not merged into one replicate group by the zero-fill —
        # via the one leakage-checked, deterministic grouper.
        groups = row_hash_groups(X_raw)

    # honest grouped cross-validation: pooled out-of-fold predictions + a
    # group-level bootstrap CI, all through the single leakage-checked splitter.
    rep = grouped_cv_report(X, y, groups=groups, n_splits=5, bounds=bounds)
    rho = rep["spearman"]
    oof_a, oof_p = rep["oof_actual"], rep["oof_pred"]

    # distribution-free +/- band from the pooled out-of-fold residuals (approximate
    # coverage under grouped CV). Honest alternative to the surrogate's own std,
    # which is often overconfident on small bioprocess datasets.
    resid = np.asarray(oof_a, float) - np.asarray(oof_p, float)
    conformal_q = round(q_from_residuals(resid, alpha=0.1), 4) if len(resid) else None

    # Honest reliability verdict: only what this path can actually assess. The
    # spearman floor mirrors GatesConfig.min_spearman (kalos/core/gates.py); we do
    # NOT assert feasibility or calibration gates, which are not measured here.
    ci95 = None if rho != rho else [round(rep["ci95"][0], 3), round(rep["ci95"][1], 3)]
    reliability = {
        "spearman": None if rho != rho else round(rho, 3),
        "ci95": ci95,
        "spearman_floor": 0.20,
        "clears_floor": bool(rho == rho and rho >= 0.20),
        "ci_excludes_zero": bool(ci95 is not None and ci95[0] > 0),
        "unmodeled": ["feasibility probability", "calibration (ECE)", "scale-up transfer"],
    }

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

    # per-column provenance: what was kept as a feature, used as the target, or
    # dropped (id / other output / constant / sparse), so the client is never left
    # guessing about a silently dropped column. Mirrors the selection logic above.
    provenance = provenance_dicts(
        column_provenance(
            df,
            target=str(target),
            features=[str(c) for c in feats],
            numeric_cols=[str(c) for c in num],
            id_hint=_ID_HINT,
            outcome_hint=_OUTCOME_HINT,
        )
    )

    result = {
        "n": int(keep.sum()), "d": len(feats), "target": str(target), "group_col": gcol,
        "targets": [str(c) for c in candidate_targets], "features": [str(c) for c in feats],
        "cv_spearman": None if rho != rho else round(rho, 3),
        "cv_ci95": ci95,
        "cv_n_groups": rep["n_groups"],
        "conformal_q": conformal_q,
        "reliability": reliability,
        "best": round(float(y.max()), 4),
        "drivers": [{"name": c, "rho": round(r, 3)} for c, r in drv],
        "proposal_features": show,
        "proposals": _annotate(batch, p_mean, p_std, float(y.max()), cols=show_idx),
        "oof": [[round(a, 4), round(p, 4)] for a, p in zip(oof_a, oof_p)],
        "provenance": provenance,
        "seed": ANALYZE_SEED,
        "timestamp": int(time.time()),
        "engine_version": ENGINE_VERSION,
    }
    if anonymize:
        result = _anonymize_result(result)
    return result


def _anonymize_result(result: dict) -> dict:
    """Replace identifier-type column names in the response with stable pseudonyms.

    Only identifier-like columns (matched by `_ID_HINT`) are renamed; process
    features and the target keep their real names because the authenticated owner
    UI legitimately shows them (a driver like "Methanol"). The mapping is stable
    (same name -> same pseudonym) via the anonymizer's irreversible hash, so an
    anonymized report is still internally consistent across fields.
    """
    def alias(name: str) -> str:
        return f"col_{_hash(name)[:8]}" if _ID_HINT.match(str(name).strip()) else name

    out = dict(result)
    if out.get("group_col"):
        out["group_col"] = alias(out["group_col"])
    out["provenance"] = [
        {**row, "name": alias(row["name"])} for row in out.get("provenance", [])
    ]
    return out


def _parse_upload(raw: bytes) -> pd.DataFrame:
    """Turn raw upload bytes into a bounded dataframe, or raise `UploadRejected`.

    Guards, in order:
      1. byte-size cap (default 25 MB, env `KALOS_MAX_UPLOAD_MB`);
      2. filetype sniff by MAGIC BYTES, not extension: a zip header (`PK\\x03\\x04`)
         is treated as xlsx/xls, anything else as UTF-8 text/CSV;
      3. shape caps: CSV/TSV rows and a column ceiling, and an xlsx cell-count
         (rows * cols) ceiling to blunt zip-bomb expansion.
    All rejection messages are generic (no parser text, column, or cell echoed).
    """
    if len(raw) > MAX_UPLOAD_BYTES:
        raise UploadRejected(_ERR_TOO_LARGE)

    if raw[:4] == _ZIP_MAGIC:
        # xlsx/xls: read once, then enforce the parsed cell-count ceiling. A zip
        # header on a corrupt or non-xlsx zip (BadZipFile, openpyxl's
        # InvalidFileException, ValueError) is a client error, not a server one.
        try:
            df = pd.read_excel(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001 - normalized to a generic 400 below
            log.warning("rejected an unreadable xlsx upload: %s", type(exc).__name__)
            raise UploadRejected(_ERR_PARSE) from exc
        rows, cols = df.shape
        if cols > MAX_COLUMNS:
            raise UploadRejected(_ERR_TOO_MANY_COLUMNS)
        if rows * cols > MAX_XLSX_CELLS:
            raise UploadRejected(_ERR_TOO_LARGE)
        return df

    # Otherwise treat as text/CSV. A binary blob that is neither a zip nor valid
    # tabular text will not yield usable numeric columns and is rejected downstream
    # with the same generic parse message.
    text = raw.decode("utf-8", errors="replace")
    if text.strip() == "":
        raise UploadRejected(_ERR_PARSE)
    head = text[:4000]
    sep = "\t" if head.count("\t") > head.count(",") else ","
    # cap columns first (cheap, from the header) before reading the full body
    ncols = pd.read_csv(io.StringIO(text), sep=sep, nrows=0).shape[1]
    if ncols > MAX_COLUMNS:
        raise UploadRejected(_ERR_TOO_MANY_COLUMNS)
    return pd.read_csv(io.StringIO(text), sep=sep, nrows=MAX_CSV_ROWS)


@app.post("/api/run")
async def run_uploaded(
    file: UploadFile = File(...),
    target: str = Form(default=""),
    anonymize: bool = Form(default=False),
) -> JSONResponse:
    raw = await file.read()
    try:
        df = _parse_upload(raw)
        result = _analyze(df, target or None, anonymize=anonymize)
        _save_latest(result, file.filename or "uploaded dataset")
        return JSONResponse(result)
    except UploadRejected as rej:
        # A guard tripped: the message is already generic and safe to return.
        log.warning("upload rejected: %s", rej)
        return JSONResponse({"error": str(rej)}, status_code=400)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, UnicodeError):
        # Log the full traceback server-side for debugging; return a generic
        # message so no parser detail, column name, cell value, or stack trace
        # ever reaches the (unauthenticated) client.
        log.exception("failed to parse or analyze an uploaded file")
        return JSONResponse({"error": _ERR_PARSE}, status_code=400)
