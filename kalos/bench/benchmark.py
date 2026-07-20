"""The closed-loop race: kalos BO vs Latin Hypercube vs random.

Every strategy starts from the SAME seeded space-filling init design (a fair
race), then spends a fixed budget of one-at-a-time experiments. The surrogate
only ever sees NOISY observations, like a real assay; simple regret is judged on
the noiseless truth at the queried points. Results are averaged over seeds.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from scipy.stats import qmc

from kalos.core.optimize import propose
from kalos.core.surrogate import FitError, Surrogate

from .objectives import MixedObjective, Objective

STRATEGIES = ("bo", "lhs", "random")
MIXED_STRATEGIES = ("bo", "random")


def _lhs(n: int, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lower, upper = bounds[0], bounds[1]
    sampler = qmc.LatinHypercube(d=len(lower), seed=rng)
    u = sampler.random(n)
    return lower + u * (upper - lower)


def _uniform(n: int, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lower, upper = bounds[0], bounds[1]
    return rng.uniform(lower, upper, size=(n, len(lower)))


def run_one(
    obj: Objective,
    strategy: str,
    *,
    budget: int = 15,
    n_init: int = 5,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Run one closed-loop trial; return best-true-so-far after each evaluation.

    The returned trajectory has length `n_init + budget`: the best noiseless value
    found so far after each experiment (the init points first, then each proposed
    experiment). Lower final simple regret (`obj.optimum - trajectory[-1]`) is
    better and reaching a given value in fewer experiments is the whole point.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    import torch  # local import: keep the module import light

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X = _lhs(n_init, obj.bounds, rng)
    y_true, y_obs = obj.evaluate(X, noise, rng)

    # One-shot LHS design for the static baseline: a full space-filling plan drawn
    # up front (what a scientist without a model would run), evaluated in order.
    lhs_pool = _lhs(n_init + budget, obj.bounds, np.random.default_rng(seed + 10_000))

    traj: List[float] = list(np.maximum.accumulate(y_true))
    best = float(traj[-1])

    for step in range(budget):
        if strategy == "bo":
            try:
                s = Surrogate().fit(X, y_obs, bounds=obj.bounds)
                x_next = propose(s, obj.bounds, q=1)[0]
            except FitError:
                # A degenerate fit falls back to a random draw rather than crashing;
                # this is the honest worst case for BO, not a free pass.
                x_next = _uniform(1, obj.bounds, rng)[0]
        elif strategy == "random":
            x_next = _uniform(1, obj.bounds, rng)[0]
        else:  # "lhs": take the next point of the pre-drawn space-filling design
            x_next = lhs_pool[n_init + step]

        yt, yo = obj.evaluate(x_next, noise, rng)
        X = np.vstack([X, x_next])
        y_obs = np.append(y_obs, yo)
        best = max(best, float(yt[0]))
        traj.append(best)

    return np.asarray(traj, dtype=float)


def _sample_mixed(
    n: int, obj: MixedObjective, rng: np.random.Generator
) -> np.ndarray:
    """Draw `n` random mixed points: LHS over the continuous dims, uniform integer
    codes over each categorical dim."""
    lower, upper = obj.cont_bounds[0], obj.cont_bounds[1]
    if obj.n_cont:
        u = qmc.LatinHypercube(d=obj.n_cont, seed=rng).random(n)
        cont = lower + u * (upper - lower)
    else:
        cont = np.empty((n, 0))
    cats = np.column_stack([
        rng.integers(0, k, size=n).astype(float) for k in obj.cat_cardinalities
    ]) if obj.cat_cardinalities else np.empty((n, 0))
    return np.hstack([cont, cats])


def run_mixed_one(
    obj: MixedObjective,
    strategy: str,
    *,
    budget: int = 15,
    n_init: int = 6,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """One closed-loop mixed (continuous + categorical) trial; returns the best
    true-so-far trajectory (length `n_init + budget`).

    `bo` fits the mixed GP and proposes with `optimize_acqf_mixed`; `random`
    draws uniform mixed points. A degenerate fit falls back to a random draw
    rather than crashing (the honest worst case, not a free pass)."""
    if strategy not in MIXED_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {MIXED_STRATEGIES}")
    import torch  # local import: keep the module import light

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    bounds = obj.bounds()
    cat_dims = obj.cat_dims
    cat_cardinalities = list(obj.cat_cardinalities)

    X = _sample_mixed(n_init, obj, rng)
    y_true, y_obs = obj.evaluate(X, noise, rng)
    traj: List[float] = list(np.maximum.accumulate(y_true))

    for _ in range(budget):
        if strategy == "bo":
            try:
                s = Surrogate().fit(X, y_obs, bounds=bounds, cat_dims=cat_dims)
                x_next = propose(
                    s, bounds, q=1, cat_dims=cat_dims, cat_cardinalities=cat_cardinalities
                )[0]
            except FitError:
                x_next = _sample_mixed(1, obj, rng)[0]
        else:  # "random"
            x_next = _sample_mixed(1, obj, rng)[0]

        yt, yo = obj.evaluate(x_next, noise, rng)
        X = np.vstack([X, x_next])
        y_obs = np.append(y_obs, yo)
        traj.append(max(traj[-1], float(yt[0])))

    return np.asarray(traj, dtype=float)


def run_benchmark(
    obj: Objective,
    *,
    strategies: Iterable[str] = STRATEGIES,
    budget: int = 15,
    n_init: int = 5,
    noise: float = 0.0,
    seeds: Iterable[int] = range(10),
) -> Dict[str, dict]:
    """Race the strategies over many seeds; return per-strategy mean/std curves."""
    seeds = list(seeds)
    out: Dict[str, dict] = {}
    for strat in strategies:
        trajs = np.vstack([
            run_one(obj, strat, budget=budget, n_init=n_init, noise=noise, seed=s)
            for s in seeds
        ])
        out[strat] = {
            "mean": trajs.mean(axis=0),
            "std": trajs.std(axis=0),
            "final": trajs[:, -1],  # best value each seed reached
        }
    return out


def summarize(result: Dict[str, dict], obj: Objective, n_init: int = 5) -> dict:
    """Reduce a benchmark result to the numbers that answer the question.

    Returns, per strategy: final simple regret (optimum - mean best, lower better).
    Plus `bo_beats_lhs`: how many fewer experiments BO needs to reach the value the
    static LHS design ends at (positive = BO is faster; None if BO never reaches it).
    """
    per = {}
    for strat, r in result.items():
        final_regret = float(obj.optimum - r["mean"][-1])
        per[strat] = {
            "final_best_mean": float(r["mean"][-1]),
            "final_best_std": float(r["final"].std()),
            "final_simple_regret": final_regret,
        }

    speedup = None
    if "bo" in result and "lhs" in result:
        lhs_final = result["lhs"]["mean"][-1]
        bo_curve = result["bo"]["mean"]
        reached = np.where(bo_curve >= lhs_final)[0]
        if reached.size:
            # experiments past the shared init that BO needed to match LHS's ending
            bo_experiments = int(reached[0]) - n_init
            lhs_experiments = len(bo_curve) - 1 - n_init
            speedup = max(0, lhs_experiments - bo_experiments)
    return {"per_strategy": per, "bo_fewer_experiments_than_lhs": speedup}
