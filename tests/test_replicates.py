"""Replicate aggregation, assay noise-floor estimation, and the optional
fixed-noise GP path: does collapsing replicate rows to a recipe-mean objective
and telling the GP the known assay noise actually work end to end."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kalos.bench.pool import pool_from_frame, run_pool
from kalos.core.replicates import aggregate_replicates, estimate_noise_floor, noise_report
from kalos.core.surrogate import Surrogate


def test_aggregate_replicates_known_duplicates():
    # Recipes A and B each run 3x, C is a singleton, in a shuffled arrival
    # order; first-occurrence order should be A, C, B, and within-group
    # variance/n_reps/means should match hand-computed values.
    X = np.array([
        [1.0, 2.0],  # A
        [1.0, 2.0],  # A
        [5.0, 5.0],  # C (singleton)
        [3.0, 4.0],  # B
        [1.0, 2.0],  # A
        [3.0, 4.0],  # B
        [3.0, 4.0],  # B
    ])
    y = np.array([10.0, 12.0, 7.0, 20.0, 11.0, 22.0, 24.0])

    X_unique, y_mean, y_var, n_reps = aggregate_replicates(X, y)

    assert X_unique.shape == (3, 2)
    np.testing.assert_allclose(X_unique, [[1.0, 2.0], [5.0, 5.0], [3.0, 4.0]])
    np.testing.assert_allclose(n_reps, [3, 1, 3])
    np.testing.assert_allclose(y_mean, [(10 + 12 + 11) / 3, 7.0, (20 + 22 + 24) / 3])
    assert y_var[1] == 0.0  # singleton -> 0.0 variance
    np.testing.assert_allclose(y_var[0], np.var([10.0, 12.0, 11.0], ddof=1))
    np.testing.assert_allclose(y_var[2], np.var([20.0, 22.0, 24.0], ddof=1))


def test_aggregate_replicates_rounding_merges_near_duplicates():
    X = np.array([[1.000000, 2.0], [1.0000001, 2.0], [9.0, 9.0]])
    y = np.array([1.0, 3.0, 100.0])
    X_unique, y_mean, y_var, n_reps = aggregate_replicates(X, y, decimals=6)
    assert X_unique.shape == (2, 2)
    assert n_reps.tolist() == [2, 1]
    np.testing.assert_allclose(y_mean, [2.0, 100.0])


def test_estimate_noise_floor_recovers_injected_variance():
    rng = np.random.default_rng(0)
    n_recipes = 30
    n_reps = 6
    sigma2 = 0.25
    recipes = rng.uniform(0, 10, size=(n_recipes, 3))
    true_mean = recipes.sum(axis=1)

    X_rows = np.repeat(recipes, n_reps, axis=0)
    noise = rng.normal(0.0, np.sqrt(sigma2), size=n_recipes * n_reps)
    y_rows = np.repeat(true_mean, n_reps) + noise

    est = estimate_noise_floor(X_rows, y_rows)
    assert np.isfinite(est)
    # Pooled variance estimate over 30 groups x 6 reps should be within ~30%
    # of the injected noise variance.
    assert abs(est - sigma2) / sigma2 < 0.3


def test_estimate_noise_floor_nan_when_all_rows_unique():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(10, 2))
    y = rng.normal(size=10)
    assert np.isnan(estimate_noise_floor(X, y))


def test_noise_report_keys_and_icc_ballpark():
    rng = np.random.default_rng(2)
    n_recipes = 20
    n_reps = 5
    noise_sigma2 = 1.0
    signal_sigma2 = 9.0  # true recipe-to-recipe variance, roughly

    recipe_means = rng.normal(0.0, np.sqrt(signal_sigma2), size=n_recipes)
    recipes = rng.uniform(0, 1, size=(n_recipes, 2))
    X = np.repeat(recipes, n_reps, axis=0)
    y = np.repeat(recipe_means, n_reps) + rng.normal(0, np.sqrt(noise_sigma2), n_recipes * n_reps)

    report = noise_report(X, y)
    assert set(report.keys()) == {"n_rows", "n_recipes", "n_replicated", "noise_var", "signal_var", "icc"}
    assert report["n_rows"] == n_recipes * n_reps
    assert report["n_recipes"] == n_recipes
    assert report["n_replicated"] == n_recipes
    # True ICC = signal / (signal + noise) = 9 / 10 = 0.9; allow generous slack
    # for finite-sample estimation error.
    assert 0.6 < report["icc"] < 1.0


def test_surrogate_fit_with_fixed_noise_end_to_end():
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 1, (24, 3))
    y = X @ np.array([1.0, -0.5, 0.2]) + rng.normal(0, 0.05, 24)
    bounds = np.array([[0, 0, 0], [1, 1, 1]], float)

    s = Surrogate().fit(X, y, bounds=bounds, noise=0.05 ** 2)
    mean, std = s.posterior(X)
    assert mean.shape == (24,)
    assert std.shape == (24,)
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()
    assert (std > 0).all()


def test_surrogate_fit_with_per_point_noise_array():
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, (16, 2))
    y = X.sum(1) + rng.normal(0, 0.1, 16)
    bounds = np.array([[0, 0], [1, 1]], float)
    noise = rng.uniform(0.005, 0.02, 16)

    s = Surrogate().fit(X, y, bounds=bounds, noise=noise)
    mean, std = s.posterior(X)
    assert mean.shape == (16,) and std.shape == (16,)
    assert np.isfinite(mean).all() and np.isfinite(std).all()


def test_run_pool_with_fixed_noise_is_finite_and_monotone():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(60, 4))
    center = np.full(4, 0.5)
    sigma2 = 0.05 ** 2
    y = 10.0 * np.exp(-3.0 * ((X - center) ** 2).sum(1)) + rng.normal(0, np.sqrt(sigma2), 60)

    res = run_pool(X, y, strategies=("bo",), n_init=8, budget=15, seeds=range(3), noise=sigma2)
    traj = res["bo"]["mean"]
    expected_len = res["_meta"]["steps"]
    assert len(traj) == expected_len
    assert np.all(np.isfinite(traj))
    assert np.all(np.diff(traj) >= -1e-12)  # best-so-far is monotone non-decreasing


def test_pool_from_frame_aggregate_reduces_to_unique_recipes_with_means():
    df = pd.DataFrame({
        "Glycerol": [0.1, 0.1, 0.2, 0.2, 0.2, 0.3],
        "NaCl": [1.0, 1.0, 2.0, 2.0, 2.0, 3.0],
        "Titer_g_L": [5.0, 7.0, 10.0, 12.0, 14.0, 20.0],
    })

    X, y, feats = pool_from_frame(df, target="Titer_g_L", aggregate=True)

    assert feats == ["Glycerol", "NaCl"]
    assert X.shape == (3, 2)
    assert len(y) == 3
    # Order follows first occurrence: (0.1,1.0), (0.2,2.0), (0.3,3.0)
    np.testing.assert_allclose(X, [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]])
    np.testing.assert_allclose(y, [6.0, 12.0, 20.0])


def test_pool_from_frame_default_does_not_aggregate():
    df = pd.DataFrame({
        "Glycerol": [0.1, 0.1, 0.2],
        "NaCl": [1.0, 1.0, 2.0],
        "Titer_g_L": [5.0, 7.0, 10.0],
    })
    X, y, feats = pool_from_frame(df, target="Titer_g_L")
    assert X.shape == (3, 2)
    assert len(y) == 3
