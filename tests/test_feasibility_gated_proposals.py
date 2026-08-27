"""PART 1 of the #1 backlog item: feasibility-gated PRODUCTION proposals.

BENCHMARK.md proved plain BO loses to random search on zero-inflated titers
(~21% non-producers), and that feasibility-gated EI fixes it - but, until this
change, only inside `kalos/bench/pool.py`'s discrete candidate-pool re-ranking.
The production path (`kalos.portal.analysis._analyze` -> `kalos.core.optimize.
propose`, using continuous `optimize_acqf`) never gated at all.

This file covers three layers:
  - the acquisition MECHANISM (`_FeasibilityGate` / `_FeasibilityGatedLogNEI`
    in `kalos.core.optimize`): gradient flow, and that an unfitted (cold-start
    fallback) classifier is defensively ignored rather than gating on a no-op.
  - `_analyze`'s GATE POLICY: gated only when the classifier actually fit, the
    sheet is zero-inflated enough, and its CV AUC clears the promotion floor;
    every outcome is reported, never silent; a producer-only or too-small
    sheet's proposals are byte-identical to the pre-gating acquisition.
  - THE MONEY TEST: gating the PRODUCTION continuous-acquisition path moves
    the proposed batch away from the infeasible region (mirrors the existing
    `bo_feas` comparative claim in tests/test_feasibility.py, but exercised
    through `_analyze` -> `propose`, not the discrete bench pool).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("botorch")

import torch  # noqa: E402

import kalos.portal.analysis as analysis_mod  # noqa: E402
from kalos.core.feasibility import FeasibilityClassifier, feasible_labels  # noqa: E402
from kalos.core.optimize import (  # noqa: E402
    _FeasibilityGate,
    _FeasibilityGatedLogNEI,
    propose,
)
from kalos.core.surrogate import Surrogate  # noqa: E402
from kalos.portal.analysis import GATE_MIN_N_INFEASIBLE, _analyze  # noqa: E402

from botorch.acquisition.logei import qLogNoisyExpectedImprovement  # noqa: E402


def _zero_inflated_sheet(seed: int = 42, n: int = 100) -> pd.DataFrame:
    """Real signal (`titer` rises with both features) plus ~20-25% structured
    non-producers whose infeasibility depends on `Feat1` being large - the
    same pathology BENCHMARK.md documents on the real media DoE (infeasibility
    correlated with a feature, not pure random zeroing), so the classifier has
    something learnable to gate on."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    titer = 5.0 + 0.4 * x1 + 0.2 * x2 + rng.normal(0, 0.3, n)
    infeasible = (x1 > 7.0) & (rng.random(n) < 0.85)
    titer = np.where(infeasible, 0.0, titer)
    return pd.DataFrame({
        "Feat1": x1.round(3), "Feat2": x2.round(3), "product_titer": titer.round(3),
    })


def _producer_only_sheet(seed: int = 0, n: int = 40) -> pd.DataFrame:
    """No zero-inflation at all - every row is a producer."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    titer = 5.0 + 0.4 * x1 + 0.2 * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "Feat1": x1.round(3), "Feat2": x2.round(3), "product_titer": titer.round(3),
    })


def _classifier_fallback_sheet(seed: int = 1, n: int = 40) -> pd.DataFrame:
    """Exactly 2 non-producers - below `FeasibilityClassifier`'s own 3-minority-
    example floor (see its docstring), so a classifier fit on this sheet MUST
    land in the cold-start fallback, independent of GATE POLICY's own
    `GATE_MIN_N_INFEASIBLE=5` floor (this sheet fails an even earlier bar)."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    titer = 5.0 + 0.4 * x1 + 0.2 * x2 + rng.normal(0, 0.3, n)
    titer[:2] = 0.0
    return pd.DataFrame({
        "Feat1": x1.round(3), "Feat2": x2.round(3), "product_titer": titer.round(3),
    })


# --- the acquisition mechanism: kalos.core.optimize -------------------------- #


def test_gate_gradient_flows_through_the_wrapped_acquisition():
    """`optimize_acqf`'s gradient-based L-BFGS restarts need a real gradient
    through the gated score, not merely a forward pass that runs without
    error - this is the "verify gradient flow" requirement directly."""
    rng = np.random.default_rng(0)
    n, d = 30, 2
    X = rng.random((n, d))
    y = -np.sum((X - 0.5) ** 2, axis=1) + 0.05 * rng.standard_normal(n)
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    s = Surrogate().fit(X, y, bounds=bounds)
    y_bin = (rng.random(n) > 0.3).astype(int)
    clf = FeasibilityClassifier().fit(X, y_bin)
    assert clf.fitted

    mean, scale, coef, intercept = clf.torch_gate_params()
    gate = _FeasibilityGate(mean, scale, coef, intercept)
    base = qLogNoisyExpectedImprovement(s.model, X_baseline=s._X, prune_baseline=True)
    acq = _FeasibilityGatedLogNEI(base, gate)

    # batch_shape=2, q=3, d=2 - a realistic optimize_acqf candidate tensor.
    Xc = torch.tensor(rng.random((2, 3, d)), dtype=torch.double, requires_grad=True)
    val = acq(Xc)
    assert val.shape == (2,)
    assert torch.isfinite(val).all()
    val.sum().backward()
    assert Xc.grad is not None
    assert torch.isfinite(Xc.grad).all()
    assert torch.any(Xc.grad != 0.0)  # a real (non-degenerate) gradient, not all-zero


def test_unfitted_classifier_is_defensively_ignored():
    """`propose` checks `.fitted` itself before gating (see its docstring):
    passing a cold-start-fallback classifier must be a no-op, not a gate on a
    uniform P(feasible)=1 that changes nothing while claiming to gate."""
    rng = np.random.default_rng(0)
    n, d = 20, 2
    X = rng.random((n, d))
    y = -np.sum((X - 0.5) ** 2, axis=1)
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    s = Surrogate().fit(X, y, bounds=bounds)
    clf = FeasibilityClassifier().fit(X, np.ones(n, dtype=int))  # single class -> fallback
    assert not clf.fitted

    gated = propose(s, bounds, q=2, seed=0, feasibility_classifier=clf)
    ungated = propose(s, bounds, q=2, seed=0)
    assert np.array_equal(gated, ungated)


# --- _analyze(): GATE POLICY --------------------------------------------------- #


def test_zero_inflated_sheet_gates_with_auc_and_per_row_p_feasible():
    out = _analyze(_zero_inflated_sheet())
    gating = out["proposal_gating"]
    assert gating["gated"] is True
    assert gating["n_infeasible"] >= GATE_MIN_N_INFEASIBLE
    assert isinstance(gating["feasibility_auc"], float) and gating["feasibility_auc"] > 0.0
    assert gating["threshold"] == 0.0
    assert isinstance(gating["reason"], str) and gating["reason"]
    # every proposal row carries a real (non-null) p_feasible when gated
    for p in out["proposals"]:
        assert p["p_feasible"] is not None
        assert 0.0 <= p["p_feasible"] <= 1.0


def test_producer_only_sheet_does_not_gate():
    df = _producer_only_sheet()
    out = _analyze(df)
    gating = out["proposal_gating"]
    assert gating["gated"] is False
    assert "not zero-inflated enough" in gating["reason"]
    for p in out["proposals"]:
        assert p["p_feasible"] is None


def test_producer_only_sheet_calls_propose_with_no_gating_or_constraint(monkeypatch):
    """No-regression guarantee, verified precisely rather than by trying to
    replay global RNG state (which `_analyze` also consumes for CV bootstraps
    before ever reaching `propose`, so re-seeding and calling `propose` a
    second time from outside would NOT reproduce the same draws - that is not
    what "byte-identical" can mean here).

    Instead: spy on `kalos.core.optimize.propose` and assert `_analyze` calls
    it with the three new keyword arguments at their `None` defaults on a
    sheet GATE POLICY declines to gate - i.e. the EXACT call shape the
    pre-gating code made, which never had these parameters at all. Combined
    with `test_propose_default_kwargs_match_omitting_the_new_parameters_
    entirely` below (which proves passing `None` explicitly is byte-identical
    to never passing the parameter), this is the full no-regression chain:
    the sheet's batch is produced by a call that behaves exactly like the
    historical one."""
    import kalos.core.optimize as optimize_mod

    captured: dict = {}
    original = optimize_mod.propose

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(optimize_mod, "propose", _spy)
    out = _analyze(_producer_only_sheet())
    assert out["proposal_gating"]["gated"] is False
    assert captured.get("feasibility_classifier") is None
    assert captured.get("constraint_surrogate") is None
    assert captured.get("constraint_floor") is None


def test_propose_default_kwargs_match_omitting_the_new_parameters_entirely():
    """Mechanism-level half of the no-regression guarantee: passing the three
    new keyword arguments at their documented `None` defaults must be
    byte-identical to a call that never mentions them at all - proving
    `propose`'s signature extension changes nothing for a caller that does
    not opt in (the historical contract every existing caller relies on)."""
    rng = np.random.default_rng(3)
    n, d = 25, 3
    X = rng.random((n, d))
    y = -np.sum((X - 0.4) ** 2, axis=1) + 0.05 * rng.standard_normal(n)
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    s = Surrogate().fit(X, y, bounds=bounds)
    a = propose(s, bounds, q=4, seed=9)
    b = propose(
        s, bounds, q=4, seed=9,
        feasibility_classifier=None, constraint_surrogate=None, constraint_floor=None,
    )
    assert np.array_equal(a, b)


def test_classifier_fallback_sheet_does_not_gate_and_never_crashes():
    out = _analyze(_classifier_fallback_sheet())
    gating = out["proposal_gating"]
    assert gating["gated"] is False
    assert "cold-start fallback" in gating["reason"]
    assert len(out["proposals"]) >= 1
    for p in out["proposals"]:
        assert p["p_feasible"] is None


def test_proposal_gating_block_is_json_serializable():
    out = _analyze(_zero_inflated_sheet())
    json.dumps(out["proposal_gating"])
    json.dumps(out["proposals"])
    out2 = _analyze(_producer_only_sheet())
    json.dumps(out2["proposal_gating"])
    json.dumps(out2["proposals"])


# --- THE MONEY TEST: gating the production path moves the batch ------------- #


def test_gated_production_batch_has_higher_mean_p_feasible_than_ungated(monkeypatch):
    """Mirrors tests/test_feasibility.py's `bo_feas` comparative claim, but
    through the PRODUCTION `_analyze` -> `propose` path this backlog item
    actually wires up, not the discrete `kalos.bench.pool` harness.

    Both calls run the identical `_analyze` pipeline on the identical sheet
    (same seed, via `ANALYZE_SEED`) - the ONLY difference is GATE POLICY
    condition (b) (zero-inflation), forced off by monkeypatching
    `GATE_MIN_N_INFEASIBLE` sky-high. That isolates gating as the sole cause
    of any difference between the two batches, rather than comparing two
    differently-constructed fits."""
    df = _zero_inflated_sheet()

    gated_out = _analyze(df)
    assert gated_out["proposal_gating"]["gated"] is True
    mean_gated = float(np.mean([p["p_feasible"] for p in gated_out["proposals"]]))

    monkeypatch.setattr(analysis_mod, "GATE_MIN_N_INFEASIBLE", 10_000)
    ungated_out = _analyze(df)
    assert ungated_out["proposal_gating"]["gated"] is False

    # Score the ungated batch with the SAME classifier _analyze fit
    # internally (same features, same labels) to ask "how feasible are the
    # points an ungated acquisition proposed".
    feats = gated_out["features"]
    X = df[feats].to_numpy(float)
    y = df["product_titer"].to_numpy(float)
    clf = FeasibilityClassifier().fit(X, feasible_labels(y))
    ungated_vals = np.array([p["vals"] for p in ungated_out["proposals"]])
    mean_ungated = float(np.mean(clf.predict_proba(ungated_vals)))

    assert mean_gated > mean_ungated


def test_gated_vs_ungated_best_found_so_far_not_worse_on_zero_inflated_pool():
    """THE BENCH TIE-IN. Retrospective best-found-so-far trial through the
    PRODUCTION continuous acquisition (`propose`, snap-to-nearest-pool-
    candidate - the standard way to run a continuous acquisition against a
    fixed pool, the same idea as `kalos.bench.pool.run_pool_one`'s `bo`/
    `bo_feas` strategies, but exercising `propose`'s own gate rather than the
    discrete EI-times-P(feasible) re-ranking that module already covers).

    Pool design: a narrow 2-D response peak (so the search is genuinely hard
    - a small pool subset does not already contain a near-optimal point by
    luck) with structured, feature-dependent infeasibility (`x1 > 0.7`,
    zeroed with 85% probability) FAR from the peak (`x1 ~= 0.3`) - so a
    gated run avoiding the infeasible region never trades away the true
    optimum to do so, matching this file's other zero-inflated sheet (real
    signal, infeasibility that depends on a feature, not pure random
    zeroing - the BENCHMARK.md pathology).

    Control, not a large-effect claim (same tolerance style as
    tests/test_feasibility.py's `bo_feas` check, `tol=1e-6`): gating the
    production path must not HURT best-found-so-far under zero-inflation."""
    rng = np.random.default_rng(42)
    n = 150
    X = rng.uniform(0, 1, size=(n, 2))
    y = 10.0 * np.exp(-25.0 * ((X[:, 0] - 0.3) ** 2 + (X[:, 1] - 0.5) ** 2))
    y = y + rng.normal(0, 0.05, n)
    y = np.clip(y, 0.0, None)
    infeasible = (X[:, 0] > 0.7) & (rng.random(n) < 0.85)
    y = np.where(infeasible, 0.0, y)

    def _trial(gate: bool, seed: int) -> float:
        local_rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        order = local_rng.permutation(n)
        n_init, budget = 6, 10
        evaluated = list(order[:n_init])
        remaining = set(order[n_init:])
        best = float(y[evaluated].max())
        for _ in range(budget):
            if not remaining:
                break
            ev = np.asarray(evaluated)
            bounds = np.vstack([X.min(0), X.max(0)])
            s = Surrogate().fit(X[ev], y[ev], bounds=bounds)
            clf = None
            if gate:
                clf = FeasibilityClassifier().fit(X[ev], feasible_labels(y[ev]))
            cand = propose(s, bounds, q=1, seed=seed, feasibility_classifier=clf)[0]
            rem = np.array(sorted(remaining))
            dists = np.sum((X[rem] - cand) ** 2, axis=1)
            pick = int(rem[int(np.argmin(dists))])
            remaining.discard(pick)
            evaluated.append(pick)
            best = max(best, float(y[pick]))
        return best

    seeds = range(8)
    tol = 1e-6
    gated_finals = [_trial(True, s) for s in seeds]
    ungated_finals = [_trial(False, s) for s in seeds]
    assert np.mean(gated_finals) >= np.mean(ungated_finals) - tol
