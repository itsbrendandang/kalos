"""kalos/core/feasibility: the binary producer/non-producer gate for BO on
zero-inflated titers. Covers the classifier's shape/range contract, its
cold-start fallback (must never crash on a single-class fit), the strict
`feasible_labels` threshold semantics, a comparative pool check that
feasibility-gated BO should not lose to plain BO under zero-inflation, and
`feasibility_cv_report` - the AUC/Brier/ECE report the promotion gate
(kalos/core/gates.py's `check_gates`) now actually reads, with
`feasibility_cv_auc` kept as a thin wrapper around its `auc` field."""
from __future__ import annotations

import numpy as np

from kalos.bench.pool import run_pool
from kalos.core.feasibility import (
    FeasibilityClassifier,
    feasibility_cv_auc,
    feasibility_cv_report,
    feasible_labels,
)


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


# --- feasibility_cv_report: the report the promotion gate reads ------------- #


def test_feasibility_cv_report_finite_metrics_on_separable_blobs():
    """Same separable-blobs shape as the AUC test above: with two well-separated
    classes, all three gate metrics - AUC, Brier, ECE - must come back finite
    and `evaluable` must be True, not just the AUC field alone."""
    rng = np.random.default_rng(1)
    X0 = rng.normal(-3.0, 0.5, size=(60, 2))
    X1 = rng.normal(3.0, 0.5, size=(60, 2))
    X = np.vstack([X0, X1])
    y = np.array([0.0] * 60 + [1.0] * 60)  # feasible iff y > 0
    rep = feasibility_cv_report(X, y, threshold=0.0, n_splits=5, seed=0)
    assert rep["evaluable"] is True
    assert np.isfinite(rep["auc"]) and rep["auc"] > 0.8
    assert np.isfinite(rep["brier"]) and 0.0 <= rep["brier"] <= 1.0
    assert np.isfinite(rep["ece"]) and 0.0 <= rep["ece"] <= 1.0
    # pooled OOF counts describe what the metrics were actually computed over
    assert rep["n"] == rep["n_feasible"] + rep["n_infeasible"]
    assert rep["n"] > 0


def test_feasibility_cv_report_all_nan_and_unevaluable_on_single_class():
    """A single-class label set makes AUC/Brier/ECE all undefined - none of the
    three should quietly report a number that looks measured."""
    rng = np.random.default_rng(2)
    X = rng.normal(0.0, 1.0, size=(20, 3))
    y = np.ones(20)  # all-feasible, single class after thresholding
    rep = feasibility_cv_report(X, y, threshold=0.0, n_splits=5, seed=0)
    assert rep["evaluable"] is False
    assert np.isnan(rep["auc"])
    assert np.isnan(rep["brier"])
    assert np.isnan(rep["ece"])
    # falls back to the full input's label counts (no CV ran)
    assert rep["n"] == 20
    assert rep["n_feasible"] == 20
    assert rep["n_infeasible"] == 0


def test_feasibility_cv_auc_delegates_to_the_report():
    """`feasibility_cv_auc` must not duplicate the CV loop - it is the report's
    `auc` field, on the same data and seed, every time."""
    rng = np.random.default_rng(1)
    X0 = rng.normal(-3.0, 0.5, size=(60, 2))
    X1 = rng.normal(3.0, 0.5, size=(60, 2))
    X = np.vstack([X0, X1])
    y = np.array([0.0] * 60 + [1.0] * 60)
    auc = feasibility_cv_auc(X, y, threshold=0.0, n_splits=5, seed=0)
    rep = feasibility_cv_report(X, y, threshold=0.0, n_splits=5, seed=0)
    assert auc == rep["auc"]


def test_feasibility_cv_report_group_cap_declines_instead_of_raising():
    """Mirrors `test_cv_auc_declines_instead_of_raising_when_groups_are_fewer_
    than_folds` above: the group cap that keeps `feasibility_cv_auc` from
    raising `ValueError` on a replicated sheet must carry over to the report,
    since `feasibility_cv_auc` now delegates to it."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12, 3))
    y = np.array([0.0] * 6 + [1.0] * 3 + [2.0] * 3)
    groups = np.repeat(np.arange(4), 3)  # 4 recipes, 3 replicates each
    rep = feasibility_cv_report(X, y, groups=groups, n_splits=5, seed=0)
    assert rep["evaluable"] is False
    assert np.isnan(rep["auc"]) and np.isnan(rep["brier"]) and np.isnan(rep["ece"])
