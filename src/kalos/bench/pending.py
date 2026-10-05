"""How often does a mid-round re-proposal repeat work that is already running?

A campaign round is not atomic: recipes are proposed, started, and measured at
different times, so the loop is routinely re-analyzed while some of the batch is
still in the incubator. Those runs have no outcome yet, so they cannot join the
fit — and if the acquisition is not told about them separately, it treats their
region of the design space as unexplored and proposes them again.

This measures that directly. For each seed: fit a GP on a small design, propose a
batch (the recipes that "go into the incubator"), then re-propose twice under a
different optimizer seed — once blind, the way the loop behaved before, and once
with the first batch passed as `pending`. Counting how many of the new recipes
land within `DUPLICATE_RADIUS` of an in-flight one gives the wasted fraction of
the round, and the difference between the two counts is what the pending block
buys.

Reproducible from committed code, like the rest of `kalos.bench`:

    python -m kalos.bench --pending
"""
from __future__ import annotations

import numpy as np
import torch

from kalos.core.optimize import propose
from kalos.core.surrogate import Surrogate

# A re-proposal within this distance of an in-flight recipe (in the unit design
# box) is the same experiment for any practical purpose: the scientist would be
# preparing the same formulation twice. Deliberately tight — a looser radius
# would flatter the result by counting merely-nearby recipes as duplicates.
DUPLICATE_RADIUS = 0.05


def _fit(seed: int, d: int, n_init: int, noise: float) -> tuple[Surrogate, np.ndarray]:
    """A GP over a `d`-factor unit design with one broad interior optimum.

    A smooth single-optimum surface is the conservative choice here: it is the
    case where a blind re-proposal is LEAST likely to collide, because the
    acquisition's own q-batch diversity already spreads a single batch out. Any
    duplication measured on it is not an artifact of a pathological landscape.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_init, d))
    optimum = np.linspace(0.3, 0.7, d)
    y = -np.sum((X - optimum) ** 2, axis=1) + noise * rng.standard_normal(n_init)
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    return Surrogate().fit(X, y, bounds=bounds), bounds


def _n_duplicated(batch: np.ndarray, in_flight: np.ndarray) -> int:
    nearest = np.min(np.linalg.norm(batch[:, None, :] - in_flight[None, :, :], axis=-1), axis=1)
    return int((nearest < DUPLICATE_RADIUS).sum())


def run_pending_duplication(
    *,
    d: int = 4,
    q: int = 5,
    n_init: int = 12,
    noise: float = 0.05,
    seeds=range(10),
) -> dict:
    """Duplicated-recipe counts per round, blind versus pending-aware.

    Returns the per-seed counts and their means. Both re-proposals in a seed are
    run from the SAME optimizer seed, so the only difference between them is the
    pending block.
    """
    blind: list[int] = []
    aware: list[int] = []
    for seed in seeds:
        surrogate, bounds = _fit(seed, d, n_init, noise)
        torch.manual_seed(seed)
        np.random.seed(seed)
        in_flight = propose(surrogate, bounds, q=q)

        torch.manual_seed(seed + 99)
        np.random.seed(seed + 99)
        blind.append(_n_duplicated(propose(surrogate, bounds, q=q), in_flight))

        torch.manual_seed(seed + 99)
        np.random.seed(seed + 99)
        aware.append(_n_duplicated(propose(surrogate, bounds, q=q, pending=in_flight), in_flight))

    return {
        "d": d,
        "q": q,
        "n_init": n_init,
        "noise": noise,
        "n_seeds": len(blind),
        "duplicate_radius": DUPLICATE_RADIUS,
        "blind": blind,
        "pending_aware": aware,
        "blind_mean": float(np.mean(blind)) if blind else float("nan"),
        "pending_aware_mean": float(np.mean(aware)) if aware else float("nan"),
        "blind_worst": max(blind) if blind else 0,
    }


__all__ = ["DUPLICATE_RADIUS", "run_pending_duplication"]
