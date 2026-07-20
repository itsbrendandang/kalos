"""kalos/bench: the closed-loop optimizer must beat space-filling when the signal
is clean, trials must be reproducible, and the synthetic optima must be real."""
from __future__ import annotations

import numpy as np

from kalos.bench import (
    ackley,
    gaussian_bump,
    mixed_bump,
    run_benchmark,
    run_mixed_one,
    run_one,
    summarize,
)


def test_run_one_is_reproducible_and_monotone():
    obj = gaussian_bump(3)
    # LHS is fully deterministic given the seed; two runs must match exactly.
    a = run_one(obj, "lhs", budget=4, n_init=4, noise=0.0, seed=1)
    b = run_one(obj, "lhs", budget=4, n_init=4, noise=0.0, seed=1)
    assert np.allclose(a, b)
    assert a.shape == (8,)  # n_init + budget
    assert np.all(np.diff(a) >= -1e-9)  # best-so-far never goes down


def test_bo_runs_and_returns_a_monotone_trajectory():
    obj = ackley(3)
    t = run_one(obj, "bo", budget=6, n_init=5, noise=0.1, seed=0)
    assert t.shape == (11,)
    assert np.all(np.isfinite(t))
    assert np.all(np.diff(t) >= -1e-9)


def test_bo_beats_space_filling_when_signal_is_clean():
    # On a smooth, noiseless surface the model advantage must show up: BO ends with
    # strictly lower simple regret than both LHS and random on average.
    obj = gaussian_bump(4)
    res = run_benchmark(obj, budget=12, n_init=5, noise=0.0, seeds=range(4))
    s = summarize(res, obj, n_init=5)
    bo = s["per_strategy"]["bo"]["final_simple_regret"]
    assert bo < s["per_strategy"]["lhs"]["final_simple_regret"]
    assert bo < s["per_strategy"]["random"]["final_simple_regret"]


def test_objectives_expose_a_reachable_optimum():
    for obj in (gaussian_bump(3), ackley(3)):
        best = np.full((1, obj.dim), 0.6) if "bump" in obj.name else np.zeros((1, obj.dim))
        y_true, _ = obj.evaluate(best, noise=0.0)
        assert abs(float(y_true[0]) - obj.optimum) < 1e-6


def test_mixed_objective_optimum_needs_the_right_level():
    # The optimum is only reached on the best categorical level; the same
    # continuous point on a worse level leaves the bonus on the table.
    obj = mixed_bump(n_cont=2, n_levels=3)
    best = np.array([[0.6, 0.6, 2.0]])  # bump center + best level (code 2)
    worse = np.array([[0.6, 0.6, 0.0]])  # bump center, wrong level
    assert abs(float(obj.evaluate(best, noise=0.0)[0][0]) - obj.optimum) < 1e-6
    assert float(obj.evaluate(worse, noise=0.0)[0][0]) < obj.optimum - 2.0


def test_mixed_bo_beats_random():
    # The mixed loop must both find the continuous peak and pick the right level,
    # so it ends with strictly lower simple regret than random mixed sampling.
    obj = mixed_bump(n_cont=2, n_levels=3)
    bo = np.mean([
        obj.optimum - run_mixed_one(obj, "bo", budget=12, n_init=6, noise=0.05, seed=s)[-1]
        for s in range(3)
    ])
    rnd = np.mean([
        obj.optimum - run_mixed_one(obj, "random", budget=12, n_init=6, noise=0.05, seed=s)[-1]
        for s in range(3)
    ])
    assert bo < rnd


def test_mixed_run_is_reproducible_and_monotone():
    obj = mixed_bump(n_cont=2, n_levels=3)
    a = run_mixed_one(obj, "random", budget=5, n_init=6, noise=0.0, seed=1)
    b = run_mixed_one(obj, "random", budget=5, n_init=6, noise=0.0, seed=1)
    assert np.allclose(a, b)
    assert a.shape == (11,)  # n_init + budget
    assert np.all(np.diff(a) >= -1e-9)  # best-so-far never decreases
