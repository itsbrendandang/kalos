"""Nonlinear drivers: find what Spearman structurally cannot see, and refuse to
invent anything when the model has not earned the right to speak.

`kalos/core/drivers.py` reports Spearman rho per feature. Spearman is univariate
and MONOTONIC, so two things are invisible to it no matter how strong they are:

  - an INTERIOR OPTIMUM. A titer that peaks at pH 7.0 and falls off both sides
    gives a rank correlation near zero, so the engine reports "no signal" for the
    single most important variable on the sheet. Bioprocess optima are almost
    always interior, so this is not a corner case.
  - an INTERACTION. pH mattering only at high temperature cannot be expressed as
    a per-feature rank statistic at all.

The negative tests here matter as much as the positive ones. This module runs on
datasets as small as a few dozen rows, where boosted trees will happily carve
structure out of pure noise. A module that reports importances it cannot justify
would be worse than not having one, so the gate tests below are load-bearing.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("xgboost")  # optional extra; on macOS also needs libomp

from kalos.core.nonlinear_drivers import (  # noqa: E402
    CV_SCORE_FLOOR,
    MIN_ROWS,
    nonlinear_drivers,
)

_NAMES = ["pH", "Methanol", "noise"]


def _interior_optimum_sheet(n: int = 120, seed: int = 0):
    """Titer peaking at pH 7.0, plus a genuinely monotonic methanol term and a
    pure-noise column. The truth is known, so every assertion is checkable."""
    rng = np.random.default_rng(seed)
    pH = rng.uniform(5.5, 8.5, n)
    methanol = rng.uniform(0.0, 4.0, n)
    noise = rng.uniform(0.0, 1.0, n)
    y = -1.2 * (pH - 7.0) ** 2 + 0.8 * methanol + rng.normal(0.0, 0.15, n)
    return np.column_stack([pH, methanol, noise]), y


def _driver(report, name):
    return next(d for d in report.drivers if d.name == name)


# --- the headline case: the whole justification for this module ------------- #


def test_interior_optimum_is_found_where_spearman_is_blind():
    """The case the module exists for, asserted end to end."""
    from scipy.stats import spearmanr

    X, y = _interior_optimum_sheet()
    report = nonlinear_drivers(X, y, feature_names=_NAMES)
    assert report.available, report.reason

    ph = _driver(report, "pH")

    # 1. Spearman genuinely mischaracterizes pH. It does not merely understate
    #    the effect - it reports a weak NEGATIVE monotonic trend for a variable
    #    whose true response is a peak, so acting on rho pushes pH the wrong way.
    rho_direct, _ = spearmanr(X[:, 0], y)
    assert abs(rho_direct) < 0.35, "the blind spot must be real for this test to mean anything"
    assert ph.spearman_rho == pytest.approx(rho_direct, abs=1e-6)

    # 2. XGBoost sees it, names the shape, and locates the optimum in the
    #    feature's ORIGINAL units so a scientist can act on the number.
    assert ph.shape == "interior_optimum"
    assert ph.optimum_at == pytest.approx(7.0, abs=0.35)

    # 3. And it is flagged, because that is what the client needs to notice.
    assert ph.missed_by_spearman is True


def test_monotonic_feature_is_reported_as_monotonic_and_not_flagged():
    """Spearman handles a monotonic feature correctly, so there is nothing to
    flag - the flag must not fire on everything or it means nothing."""
    X, y = _interior_optimum_sheet()
    methanol = _driver(X_report := nonlinear_drivers(X, y, feature_names=_NAMES), "Methanol")
    assert X_report.available
    assert methanol.shape == "monotonic_up"
    assert methanol.spearman_rho > 0.5
    assert methanol.missed_by_spearman is False


def test_pure_noise_feature_is_flat_and_unimportant():
    """Boosted trees always carve some structure out of noise. Reporting its
    direction as a trend would be a description of overfitting."""
    X, y = _interior_optimum_sheet()
    noise = _driver(nonlinear_drivers(X, y, feature_names=_NAMES), "noise")
    assert noise.shape == "flat"
    assert noise.missed_by_spearman is False
    assert noise.importance < 0.15


def test_interaction_is_surfaced():
    rng = np.random.default_rng(1)
    n = 160
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    x3 = rng.uniform(-1, 1, n)
    y = 3.0 * x1 * x2 + rng.normal(0, 0.1, n)  # pure interaction, no main effects
    report = nonlinear_drivers(np.column_stack([x1, x2, x3]), y, feature_names=["x1", "x2", "x3"])
    assert report.available, report.reason
    assert report.interactions, "an interaction-only response must surface an interaction"
    top = report.interactions[0]
    assert set(top.pair) == {"x1", "x2"}, f"expected x1/x2 strongest, got {top.pair}"


# --- the gates: refusing to speak is a feature ------------------------------ #


def test_noise_target_does_not_clear_the_gate_and_reports_nothing():
    """The most important negative test. With an unpredictable target the
    importances are noise, so the module must refuse rather than rank them."""
    rng = np.random.default_rng(2)
    n = 120
    X = rng.uniform(0, 1, (n, 3))
    y = rng.normal(0, 1, n)  # independent of X

    report = nonlinear_drivers(X, y, feature_names=_NAMES)
    assert report.available is False
    assert report.drivers == ()
    assert report.interactions == ()
    assert report.reason is not None
    assert "floor" in report.reason.lower()
    # the failed score is still reported, so the refusal is auditable
    assert report.cv_score is not None and report.cv_score < CV_SCORE_FLOOR


def test_too_few_rows_refuses_with_a_reason():
    rng = np.random.default_rng(3)
    n = MIN_ROWS - 1
    X = rng.uniform(0, 1, (n, 2))
    y = X[:, 0] * 2.0
    report = nonlinear_drivers(X, y, feature_names=["a", "b"])
    assert report.available is False
    assert report.drivers == ()
    assert report.reason is not None and str(MIN_ROWS) in report.reason


def test_unmodeled_states_what_this_does_not_establish():
    """The codebase's convention is to name its own limits (see
    `reliability.unmodeled` in kalos/portal/analysis.py). Association-not-
    causation is the one a client is most likely to over-read."""
    X, y = _interior_optimum_sheet()
    report = nonlinear_drivers(X, y, feature_names=_NAMES)
    joined = " ".join(report.unmodeled).lower()
    assert report.unmodeled
    assert "caus" in joined, "must state these are associations, not causal"


# --- contract: determinism, serialization, grouping ------------------------- #


def test_deterministic_across_runs():
    """Same sheet, same drivers. A client-facing report that shifts between
    identical runs is not auditable."""
    X, y = _interior_optimum_sheet()
    first = nonlinear_drivers(X, y, feature_names=_NAMES)
    second = nonlinear_drivers(X, y, feature_names=_NAMES)
    assert first.to_dict() == second.to_dict()


def test_report_survives_strict_json():
    """The portal serializes with allow_nan=False, so a NaN or inf anywhere in
    the report would turn a good analysis into a 400 (see
    kalos/validation/report.py for the precedent this repeats)."""
    X, y = _interior_optimum_sheet()
    payload = nonlinear_drivers(X, y, feature_names=_NAMES).to_dict()
    json.dumps(payload, allow_nan=False)
    for d in payload["drivers"]:
        assert type(d["importance"]) is float
        assert type(d["missed_by_spearman"]) is bool
        assert d["optimum_at"] is None or type(d["optimum_at"]) is float


def test_replicate_groups_do_not_straddle_folds():
    """Replicates of one recipe must stay on one side of every split, or the
    out-of-fold score this report is gated on is inflated by leakage."""
    rng = np.random.default_rng(4)
    n_recipes, reps = 40, 3
    base = rng.uniform(5.5, 8.5, n_recipes)
    pH = np.repeat(base, reps)
    methanol = np.repeat(rng.uniform(0, 4, n_recipes), reps)
    groups = np.repeat(np.arange(n_recipes), reps)
    y = -1.2 * (pH - 7.0) ** 2 + 0.8 * methanol + rng.normal(0, 0.1, len(pH))

    report = nonlinear_drivers(
        np.column_stack([pH, methanol]), y, feature_names=["pH", "Methanol"], groups=groups
    )
    assert report.available, report.reason
    # the interior optimum must still be recovered under grouping
    assert _driver(report, "pH").shape == "interior_optimum"


def test_importance_spread_is_reported():
    """A single importance number on a small sheet is noise. The fold-to-fold
    spread is what tells a scientist whether it is established."""
    X, y = _interior_optimum_sheet()
    report = nonlinear_drivers(X, y, feature_names=_NAMES)
    assert all(d.importance_sd >= 0.0 for d in report.drivers)
    assert sum(d.importance for d in report.drivers) == pytest.approx(1.0, abs=1e-6)
