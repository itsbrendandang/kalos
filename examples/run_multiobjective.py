#!/usr/bin/env python3
"""Demo: multi-objective BO (titer AND purity) with qLogNEHVI.

Two objectives that peak at different recipes, so they trade off. The loop grows
the Pareto front — the set of recipes where you can't raise one objective without
lowering the other. Run:  python examples/run_multiobjective.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

import numpy as np  # noqa: E402

from kalos.core.multiobjective import MultiObjectiveSurrogate, propose_multiobjective  # noqa: E402

TITER_OPT = np.array([0.7, 0.3, 0.5])
PURITY_OPT = np.array([0.2, 0.8, 0.4])  # different recipe -> the two objectives tension


def objectives(X: np.ndarray) -> np.ndarray:
    """Two objectives to MAXIMIZE: (titer, purity), peaking at different recipes."""
    x = np.atleast_2d(np.asarray(X, float))
    titer = -np.sum((x - TITER_OPT) ** 2, axis=1)
    purity = -np.sum((x - PURITY_OPT) ** 2, axis=1)
    return np.stack([titer, purity], axis=-1)


def main() -> int:
    rng = np.random.default_rng(0)
    d = 3
    bounds = np.array([[0, 0, 0], [1, 1, 1]], float)
    X = rng.uniform(0, 1, (8, d))
    Y = objectives(X)
    print(f"start: {len(Y)} random runs")
    for r in range(5):
        s = MultiObjectiveSurrogate().fit(X, Y, bounds=bounds)
        nxt = propose_multiobjective(s, bounds, q=2)
        X = np.vstack([X, nxt])
        Y = np.vstack([Y, objectives(nxt)])
        _, py = s.pareto()
        print(
            f"  round {r + 1}: n={len(Y)}  Pareto front size {len(py)}  "
            f"best titer {Y[:, 0].max():.3f}  best purity {Y[:, 1].max():.3f}"
        )
    _, py = MultiObjectiveSurrogate().fit(X, Y, bounds=bounds).pareto()
    print("\nfinal Pareto front (titer, purity) — the achievable tradeoffs:")
    for t, p in sorted(py.tolist(), reverse=True):
        print(f"  titer {t:7.3f}   purity {p:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
