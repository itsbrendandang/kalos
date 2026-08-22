"""kalos/core/feasibility: the binary producer/non-producer gate for BO on
zero-inflated titers. Covers the classifier's shape/range contract, its
cold-start fallback (must never crash on a single-class fit), the strict
`feasible_labels` threshold semantics, and a comparative pool check that
feasibility-gated BO should not lose to plain BO under zero-inflation."""
from __future__ import annotations

import numpy as np

from kalos.bench.pool import run_pool
from kalos.core.feasibility import FeasibilityClassifier, feasibility_cv_auc, feasible_labels


def test_predict_proba_shape_and_range():
    rng = np.random.default_rng(0)
    X0 = rng.normal(-2.0, 1.0, size=(20, 2))
    X1 = rng.normal(2.0, 1.0, size=(20, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 20 + [1] * 20)
    clf = FeasibilityClassifier().fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (40,)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_feasibility_cv_auc_separable_blobs_high_auc():
    rng = np.random.default_rng(1)
    X0 = rng.normal(-3.0, 0.5, size=(60, 2))
    X1 = rng.normal(3.0, 0.5, size=(60, 2))
    X = np.vstack([X0, X1])
    y = np.array([0.0] * 60 + [1.0] * 60)  # feasible iff y > 0
    auc = feasibility_cv_auc(X, y, threshold=0.0, n_splits=5, seed=0)
    assert auc > 0.8


def test_cold_start_all_one_class_returns_ones_no_exception():
    rng = np.random.default_rng(2)
    X = rng.normal(0.0, 1.0, size=(10, 3))
    y_binary = np.ones(10, dtype=int)  # all-feasible, single class
    clf = FeasibilityClassifier().fit(X, y_binary)
    p = clf.predict_proba(X)
    assert np.array_equal(p, np.ones(10))


def test_feasible_labels_strict_threshold_boundary():
    y = np.array([-1.0, 0.0, 0.5, 1.0])
    labels = feasible_labels(y, threshold=0.0)
    # strict >: exactly-zero is NOT feasible
    assert list(labels) == [0, 0, 1, 1]

    y2 = np.array([0.05, 0.1, 0.15, 0.2])
    labels2 = feasible_labels(y2, threshold=0.1)
    # strict >: exactly-at-threshold (0.1) is NOT feasible
    assert list(labels2) == [0, 0, 1, 1]


def test_zero_inflated_pool_feasibility_gating_does_not_hurt():
    # Build a smooth positive response surface, then zero out ~25% of rows at
    # random to simulate zero-inflated non-producers (independent of the
    # underlying surface value) - the same pathology BENCHMARK.md documents on
    # the real media DoE data.
    rng = np.random.default_rng(42)
    n = 100
    X = rng.uniform(0, 1, size=(n, 4))
    center = np.full(4, 0.5)
    y = 10.0 * np.exp(-3.0 * ((X - center) ** 2).sum(1)) + rng.normal(0, 0.05, n)
    y = np.clip(y, 0.0, None)
    zero_mask = rng.random(n) < 0.25
    y[zero_mask] = 0.0

    seeds = range(8)
    strategies = ("bo", "bo_feas", "bo_feas_clean", "random")
    res = run_pool(X, y, strategies=strategies, n_init=8, budget=22, seeds=seeds)

    expected_len = res["_meta"]["steps"]
    for strat in strategies:
        mean_traj = res[strat]["mean"]
        assert len(mean_traj) == expected_len
        assert np.all(np.isfinite(mean_traj))
        # best-found-so-far must be monotone non-decreasing
        assert np.all(np.diff(mean_traj) >= -1e-12)

    # Control: gating on feasibility should not hurt, and should help, under
    # zero-inflation - this is the hypothesis under test, not a claim of a large
    # effect size, so we use a small numeric tolerance rather than a strict >.
    tol = 1e-6
    bo_final = res["bo"]["mean"][-1]
    assert res["bo_feas"]["mean"][-1] >= bo_final - tol
    assert res["bo_feas_clean"]["mean"][-1] >= bo_final - tol


def test_cv_auc_declines_instead_of_raising_when_groups_are_fewer_than_folds():
    """The minority-class cap does not imply the group cap.

    A replicated sheet can hold plenty of non-producing ROWS across very few
    RECIPES: here 6 non-producers spread over 4 recipes, so the old guard picked
    `n_splits = min(5, 6) = 5` and handed sklearn more folds than there are
    groups, which raises `ValueError`. That value feeds the fail-closed promotion
    gate, and an exception is not a closed gate - it is a crash that skips the
    verdict. Unmeasurable must come back as `nan`, which the gate then blocks on.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 3))
    y = np.array([0.0] * 6 + [1.0] * 3 + [2.0] * 3)
    groups = np.repeat(np.arange(4), 3)  # 4 recipes, 3 replicates each
    auc = feasibility_cv_auc(X, y, groups=groups, n_splits=5, seed=0)
    assert np.isnan(auc)


def test_cv_auc_still_scores_when_there_are_enough_groups():
    """The group cap must not silently disable a measurable AUC: with enough
    recipes the same replicated shape still returns a real number."""
    rng = np.random.default_rng(4)
    n_recipes = 12
    centers = np.array([-3.0] * (n_recipes // 2) + [3.0] * (n_recipes // 2))
    X = np.repeat(centers, 3).reshape(-1, 1) + rng.normal(0.0, 0.3, size=(n_recipes * 3, 1))
    y = np.repeat((centers > 0).astype(float), 3)
    groups = np.repeat(np.arange(n_recipes), 3)
    auc = feasibility_cv_auc(X, y, groups=groups, n_splits=5, seed=0)
    assert np.isfinite(auc) and auc > 0.8
