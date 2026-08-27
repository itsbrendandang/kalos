"""Response shapes read off the GP posterior: find what Spearman cannot see, and
decline to claim a peak the model does not actually resolve.

This replaces a gradient-boosted-tree approach that worked but was disqualified:
xgboost and torch each carry their own OpenMP runtime and the second to enter a
parallel region segfaults the process, which no `available=False` guard can
defend against because a SIGSEGV is not catchable. Reading the shapes off the GP
instead is better on the merits regardless of that crash - the shapes describe
the same posterior that produces the proposals rather than a second model, and
the posterior's own standard deviation gates the interior-optimum claim, which a
point-predicting tree could not do at all.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("botorch")

from kalos.core.gp_shape import (  # noqa: E402
    NEGLIGIBLE_RELEVANCE_SHARE,
    PEAK_SD_MULTIPLE,
    ard_lengthscales,
    gp_shape_report,
)
from kalos.core.surrogate import Surrogate  # noqa: E402

_NAMES = ["pH", "Methanol", "noise"]


def _fit(X: np.ndarray, y: np.ndarray) -> Surrogate:
    return Surrogate().fit(X, y, bounds=np.vstack([X.min(axis=0), X.max(axis=0)]))


def _interior_optimum_data(n: int = 60, seed: int = 0):
    """Titer peaking at pH 7.0, a monotonic methanol term, and a noise column.
    The truth is known, so every assertion below is checkable."""
    rng = np.random.default_rng(seed)
    pH = rng.uniform(5.5, 8.5, n)
    methanol = rng.uniform(0.0, 4.0, n)
    noise = rng.uniform(0.0, 1.0, n)
    y = -1.2 * (pH - 7.0) ** 2 + 0.8 * methanol + rng.normal(0.0, 0.15, n)
    return np.column_stack([pH, methanol, noise]), y


def _feature(report, name):
    return next(f for f in report.features if f.name == name)


# --- the headline case ------------------------------------------------------ #


def test_interior_optimum_found_where_spearman_is_blind():
    X, y = _interior_optimum_data()
    report = gp_shape_report(_fit(X, y), X, y, feature_names=_NAMES)
    assert report.available, report.reason

    ph = _feature(report, "pH")
    # Spearman genuinely finds nothing: a peak makes the rank correlation on
    # either side cancel.
    assert abs(ph.spearman_rho) < 0.20
    # The GP names the shape and locates the optimum in original units.
    assert ph.shape == "interior_optimum"
    assert ph.optimum_at == pytest.approx(7.0, abs=0.35)
    assert ph.missed_by_spearman is True
    # And it says how well resolved the peak is, which is the whole advantage
    # over a point-predicting model.
    assert ph.peak_gain is not None and ph.peak_gain > 0
    assert ph.peak_separation_sd is not None
    assert ph.peak_separation_sd >= PEAK_SD_MULTIPLE


def test_ard_ranks_the_real_driver_above_noise():
    """Relevance comes free from the fitted kernel: a short lengthscale means the
    response moves fast along that axis."""
    X, y = _interior_optimum_data()
    report = gp_shape_report(_fit(X, y), X, y, feature_names=_NAMES)
    ph, meth, noise = (_feature(report, n) for n in _NAMES)

    assert ph.relevance > noise.relevance
    assert meth.relevance > noise.relevance
    # shorter lengthscale <-> higher relevance, the inverse relation
    assert ph.lengthscale < noise.lengthscale
    # shares are normalized
    assert sum(f.relevance for f in report.features) == pytest.approx(1.0, abs=1e-3)
    # and the report is ordered most-relevant first
    assert [f.name for f in report.features][0] == "pH"


def test_monotonic_feature_is_monotonic_and_not_flagged():
    """Spearman handles a monotonic feature correctly, so nothing to flag. A flag
    that fires on everything conveys nothing."""
    X, y = _interior_optimum_data()
    meth = _feature(gp_shape_report(_fit(X, y), X, y, feature_names=_NAMES), "Methanol")
    assert meth.shape == "monotonic_up"
    assert meth.spearman_rho > 0.5
    assert meth.missed_by_spearman is False


def test_irrelevant_feature_is_flat_not_a_trend():
    """A shape read off a feature the GP ignores describes the fit, not the
    process. With a very long lengthscale the profile is a near-flat line whose
    direction is arbitrary, and "monotonic_up" would read as a real trend."""
    X, y = _interior_optimum_data()
    noise = _feature(gp_shape_report(_fit(X, y), X, y, feature_names=_NAMES), "noise")
    assert noise.shape == "flat"
    assert noise.relevance < NEGLIGIBLE_RELEVANCE_SHARE / 3 or noise.optimum_at is None
    assert noise.missed_by_spearman is False


# --- the uncertainty gate: declining to claim a peak ------------------------ #


def test_no_interior_claim_on_a_purely_monotonic_response():
    """A clean monotonic response must never produce an interior optimum, however
    the fitted mean wobbles."""
    rng = np.random.default_rng(3)
    n = 60
    a = rng.uniform(0, 10, n)
    b = rng.uniform(0, 1, n)
    y = 2.0 * a + rng.normal(0, 0.2, n)
    X = np.column_stack([a, b])
    report = gp_shape_report(_fit(X, y), X, y, feature_names=["a", "b"])
    assert report.available, report.reason
    assert _feature(report, "a").shape == "monotonic_up"


def test_pure_noise_target_yields_no_interior_optima():
    """With an unpredictable target the GP has nothing to resolve, so no feature
    should come back with a confident interior peak."""
    rng = np.random.default_rng(4)
    n = 60
    X = rng.uniform(0, 1, (n, 3))
    y = rng.normal(0, 1, n)

    # Posterior separation ALONE is not sufficient, and this is the case that
    # proves it: near the training data the posterior sd is tiny, so a meaningless
    # fitted wiggle scores a large separation ratio. Without the skill gate the
    # report confidently claimed interior optima on pure noise.
    ungated = gp_shape_report(_fit(X, y), X, y, feature_names=_NAMES)
    assert ungated.available
    assert any(
        "predicts held-out runs" in u for u in ungated.unmodeled
    ), "must disclose that no skill measure was supplied"

    # Supplying the engine's own out-of-fold score - which on noise is at or below
    # zero - correctly refuses the whole report.
    report = gp_shape_report(_fit(X, y), X, y, feature_names=_NAMES, cv_spearman=0.02)
    assert report.available is False
    assert report.features == ()
    assert report.reason is not None and "out-of-fold" in report.reason


def test_peak_separation_is_reported_so_a_stricter_bar_can_be_applied():
    X, y = _interior_optimum_data()
    ph = _feature(gp_shape_report(_fit(X, y), X, y, feature_names=_NAMES), "pH")
    # gain in target units, separation in combined posterior sds, both present
    assert ph.peak_gain is not None
    assert ph.peak_separation_sd is not None
    assert ph.peak_separation_sd > 0


# --- contract: it describes the deployed model and never crashes ------------ #


def test_never_fits_anything_itself():
    """It must describe the model that makes the proposals, not a fresh one. An
    unfitted surrogate is therefore refused rather than quietly fitted."""
    report = gp_shape_report(Surrogate(), np.zeros((6, 2)), np.zeros(6), feature_names=["a", "b"])
    assert report.available is False
    assert report.reason is not None and "not fitted" in report.reason


def test_declines_on_mismatched_feature_names():
    X, y = _interior_optimum_data()
    report = gp_shape_report(_fit(X, y), X, y, feature_names=["only", "two"])
    assert report.available is False
    assert report.reason is not None


def test_declines_on_empty_input():
    report = gp_shape_report(Surrogate(), np.zeros((0, 2)), np.zeros(0), feature_names=["a", "b"])
    assert report.available is False


def test_constant_feature_is_flat_not_an_error():
    """A column with no variation has no axis to sweep; it must not raise."""
    rng = np.random.default_rng(6)
    n = 40
    a = rng.uniform(0, 10, n)
    const = np.full(n, 3.0)
    y = 2.0 * a + rng.normal(0, 0.2, n)
    X = np.column_stack([a, const])
    report = gp_shape_report(_fit(X, y), X, y, feature_names=["a", "const"])
    assert report.available, report.reason
    assert _feature(report, "const").shape == "flat"


def test_report_states_where_the_sweep_was_taken():
    """Holding the other features at the column medians would put the sweep on a
    recipe nobody ran; the incumbent is a real one, and the report says so rather
    than leaving the reader to assume."""
    X, y = _interior_optimum_data()
    report = gp_shape_report(_fit(X, y), X, y, feature_names=_NAMES)
    assert report.swept_at == "incumbent"


def test_unmodeled_names_the_limits_including_causation():
    X, y = _interior_optimum_data()
    report = gp_shape_report(_fit(X, y), X, y, feature_names=_NAMES)
    joined = " ".join(report.unmodeled).lower()
    assert "caus" in joined
    assert "incumbent" in joined


def test_deterministic_and_json_safe():
    """Two reads of one fitted model agree, and the report survives the strict
    JSON the portal serializes with."""
    X, y = _interior_optimum_data()
    s = _fit(X, y)
    first = gp_shape_report(s, X, y, feature_names=_NAMES).to_dict()
    second = gp_shape_report(s, X, y, feature_names=_NAMES).to_dict()
    assert first == second
    json.dumps(first, allow_nan=False)


def test_ard_lengthscales_returns_none_when_misaligned():
    """Guard against reporting a lengthscale vector that does not line up with the
    feature list, which would attribute relevance to the wrong column."""
    X, y = _interior_optimum_data()
    s = _fit(X, y)
    assert ard_lengthscales(s, X.shape[1]) is not None
    assert ard_lengthscales(s, X.shape[1] + 5) is None
