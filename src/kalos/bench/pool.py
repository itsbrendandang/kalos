"""Pool-based retrospective benchmark on a REAL run sheet.

The synthetic benchmark (`objectives.py`) measures the machinery on known
surfaces. This one asks the retrospective question on actual data: given the pool
of experiments a client already ran, does the BO loop find the best recipes in
FEWER picks than random selection? The candidate set is the real rows; a strategy
"runs" a candidate by revealing its measured titer (already noisy - this is the
real assay), and we track the best titer found so far.

The data is never committed here (it is client data); this module only takes a
DataFrame, so the code stays reproducible and data-free. See docs/BENCHMARK.md for the
result on the combined media DoE.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from kalos.core.feasibility import FeasibilityClassifier, feasible_labels
from kalos.core.replicates import aggregate_replicates
from kalos.core.surrogate import FitError, Surrogate

_norm_cdf = norm.cdf
_norm_pdf = norm.pdf

# Columns that are OTHER measured outputs or identifiers, never model inputs
# (anti-leakage): matches the portal's intent so the pool uses process inputs only.
_OUTPUT_HINT = re.compile(r"titer|titre|yield|conc|purity|size|kda|expression|od\d|biomass", re.I)
_ID_HINT = re.compile(r"^(sample|well|plate|position|revvity|experiment|medium|strain|name|.*plate.*)$", re.I)


def pool_from_frame(
    df: pd.DataFrame, target: str, *, aggregate: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (X, y, feature_names): process-input features and the target titer.

    Features are numeric, varying columns that are not the target, not another
    measured output, and not an identifier. Missing feature cells are zero-filled
    (as the engine does); rows without a numeric target are dropped.

    `aggregate=True` collapses replicate rows (identical feature vectors, see
    `kalos.core.replicates.aggregate_replicates`) down to one row per distinct
    recipe with the mean target, giving a reproducible per-recipe objective
    instead of one row per (noisy) assay read. `feats` is unaffected either
    way; only `X` and `y` are reduced. Default `False` keeps one row per input
    row, the current behavior.
    """
    y_all = pd.to_numeric(df[target], errors="coerce")
    keep = y_all.notna()
    feats: List[str] = []
    for c in df.columns:
        if c == target or _OUTPUT_HINT.search(str(c)) or _ID_HINT.match(str(c).strip()):
            continue
        col = pd.to_numeric(df.loc[keep, c], errors="coerce")
        if col.notna().mean() >= 0.5 and float(col.std(skipna=True) or 0.0) > 1e-9:
            feats.append(c)
    if not feats:
        raise ValueError("no varying numeric process-input features found")
    X = df.loc[keep, feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    y = y_all[keep].to_numpy(float)
    if aggregate:
        X, y, _, _ = aggregate_replicates(X, y)
    return X, y, feats


def _normalize(X: np.ndarray) -> np.ndarray:
    lo, hi = X.min(0), X.max(0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (X - lo) / span


def run_pool_one(
    X: np.ndarray, y: np.ndarray, strategy: str, *, n_init: int, budget: int, seed: int,
    noise: float | None = None,
) -> np.ndarray:
    """One retrospective trial; return best-titer-found after each pick.

    BO fits on the revealed (noisy, real) titers, proposes a point in the
    normalized design box, and snaps to the nearest not-yet-run pool candidate.

    `noise`, when given, is passed through to every `Surrogate.fit(...)` call
    in the BO branches as a fixed observation-noise variance (see
    `kalos.core.surrogate.Surrogate.fit`), e.g. an assay noise floor from
    `kalos.core.replicates.estimate_noise_floor`. Default `None` infers noise
    as before and leaves the bo/random trajectories unchanged.
    """
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n = len(y)
    budget = min(budget, n - n_init)
    Xn = _normalize(X)
    bounds = np.vstack([Xn.min(0), Xn.max(0)])

    order = rng.permutation(n)
    evaluated = list(order[:n_init])
    remaining = set(order[n_init:])
    best = float(y[evaluated].max())
    traj: List[float] = list(np.maximum.accumulate(y[evaluated]))

    for _ in range(budget):
        if not remaining:
            break
        if strategy in ("bo", "bo_feas", "bo_feas_clean"):
            try:
                # Proper pool-based BO: score every remaining candidate by analytic
                # Expected Improvement from the surrogate posterior and pick the
                # argmax. This gives the model its fair shot (no propose-then-snap
                # handicap) and is the standard way to run BO on a fixed pool.
                evaluated_arr = np.asarray(evaluated)
                if strategy == "bo_feas_clean":
                    # Fit the GP only on feasible (producer, y > 0) evaluated points,
                    # so the surrogate never has to fit the zero-inflated spike. Falls
                    # back to fitting on all evaluated points when too few feasible
                    # points have been observed yet (same behavior as "bo").
                    feas_mask = y[evaluated_arr] > 0
                    if int(feas_mask.sum()) >= 3:
                        gp_idx = evaluated_arr[feas_mask]
                    else:
                        gp_idx = evaluated_arr
                else:
                    gp_idx = evaluated_arr
                s = Surrogate().fit(Xn[gp_idx], y[gp_idx], bounds=bounds, noise=noise)
                rem = np.array(sorted(remaining))
                mu, sd = s.posterior(Xn[rem])
                mu = np.asarray(mu, float).reshape(-1)
                sd = np.maximum(np.asarray(sd, float).reshape(-1), 1e-9)
                best_y = float(y[gp_idx].max())
                z = (mu - best_y) / sd
                ei = (mu - best_y) * _norm_cdf(z) + sd * _norm_pdf(z)
                if strategy in ("bo_feas", "bo_feas_clean"):
                    # Gate EI by predicted P(feasible), trained on ALL evaluated
                    # points' labels (the classifier's whole job is telling feasible
                    # from infeasible, regardless of which points the GP itself used).
                    fc = FeasibilityClassifier().fit(Xn[evaluated_arr], feasible_labels(y[evaluated_arr]))
                    p_feasible = fc.predict_proba(Xn[rem])
                    ei = ei * p_feasible
                pick = int(rem[int(np.argmax(ei))])
            except FitError:
                pick = int(rng.choice(sorted(remaining)))
        else:  # "random": naive baseline - pick any un-run candidate
            pick = int(rng.choice(sorted(remaining)))
        remaining.discard(pick)
        evaluated.append(pick)
        best = max(best, float(y[pick]))
        traj.append(best)

    return np.asarray(traj, dtype=float)


def run_pool(
    X: np.ndarray, y: np.ndarray, *, strategies: Iterable[str] = ("bo", "random"),
    n_init: int = 8, budget: int = 40, seeds: Iterable[int] = range(20),
    noise: float | None = None,
) -> Dict[str, dict]:
    """Run each strategy's retrospective trial across `seeds`.

    `noise`, when given, is forwarded to `run_pool_one` as a fixed
    observation-noise variance for the BO strategies. Default `None` leaves
    bo/random behavior unchanged.
    """
    seeds = list(seeds)
    out: Dict[str, dict] = {}
    L = None
    for strat in strategies:
        trajs = [
            run_pool_one(X, y, strat, n_init=n_init, budget=budget, seed=s, noise=noise) for s in seeds
        ]
        L = min(len(t) for t in trajs)
        arr = np.vstack([t[:L] for t in trajs])
        out[strat] = {"mean": arr.mean(0), "std": arr.std(0), "final": arr[:, -1]}
    out["_meta"] = {"pool_max": float(y.max()), "pool_mean": float(y.mean()), "n": len(y), "steps": L}
    return out
