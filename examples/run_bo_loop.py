#!/usr/bin/env python3
"""Demo: a BoTorch closed loop on a synthetic bioprocess-like objective.

Proves the platform core runs end to end on the real stack: fit a GP surrogate,
propose the next batch by qLogEI, "measure" it, repeat. Best-so-far should climb.

Run:  python examples/run_bo_loop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing

import numpy as np  # noqa: E402

from kalos.core.optimize import propose  # noqa: E402
from kalos.core.surrogate import Surrogate  # noqa: E402

OPT = np.array([0.7, 0.3, 0.5])  # the (unknown) optimum of the synthetic surface


def objective(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Synthetic titer surface to MAXIMIZE (peak at OPT), with light noise."""
    x = np.atleast_2d(np.asarray(X, float))
    rng = np.random.default_rng(seed)
    return -np.sum((x - OPT) ** 2, axis=1) + 0.02 * rng.normal(size=len(x))


def main() -> int:
    rng = np.random.default_rng(0)
    d = 3
    bounds = np.array([[0, 0, 0], [1, 1, 1]], float)
    X = rng.uniform(0, 1, (6, d))
    y = objective(X)
    print(f"start: best {y.max():.3f} from {len(y)} random runs")
    for r in range(6):
        s = Surrogate().fit(X, y, bounds=bounds)  # normalize to the fixed design box
        nxt = propose(s, bounds, q=2)
        yn = objective(nxt, seed=r + 1)
        X = np.vstack([X, nxt])
        y = np.concatenate([y, yn])
        print(f"  round {r + 1}: proposed {nxt.shape[0]}  best {y.max():.3f}  (n={len(y)})")
    best = X[y.argmax()]
    print(f"best recipe: {best.round(3)}  (true optimum {OPT})  |  gap {np.linalg.norm(best - OPT):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
