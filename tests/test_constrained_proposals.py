"""PART 2 of the #1 backlog item: constrained single-objective proposals -
maximize the target subject to a second fitted outcome clearing a floor
(e.g. `purity_pct >= 95.0`).

Mechanism: `kalos.core.optimize.propose`'s `constraint_surrogate` +
`constraint_floor` arguments compose the target and constraint `Surrogate`s
into a `ModelListGP` and hand BoTorch's NATIVE `constraints=` argument on
`qLogNoisyExpectedImprovement` a callable satisfied where
`constraint_floor - constraint_value <= 0`.

Policy: `kalos.portal.analysis._analyze`'s `constraint` parameter decides
WHETHER that mechanism gets used on a given upload - column present, enough
rows, the constraint model actually fits - reported honestly and falling back
to an unconstrained proposal (never crashing) when it cannot be applied.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("botorch")

from kalos.core.optimize import propose  # noqa: E402
from kalos.core.surrogate import Surrogate  # noqa: E402
from kalos.portal.analysis import CONSTRAINT_MIN_ROWS, _analyze  # noqa: E402


def _anti_correlated_design(seed: int = 11, n: int = 60):
    """The target REWARDS a high `Feat1`; the constraint outcome (purity)
    FALLS as `Feat1` rises - the real tension a constrained-optimization sheet
    has (a process that makes more product but at lower purity), so honoring
    a purity floor should visibly redirect the proposed batch toward lower
    `Feat1`, not leave it unchanged."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 10.0, n)
    x2 = rng.uniform(0.0, 10.0, n)
    titer = 2.0 * x1 + 0.3 * x2 + rng.normal(0, 0.3, n)
    purity = 99.0 - 1.2 * x1 + rng.normal(0, 0.3, n)
    X = np.column_stack([x1, x2])
    bounds = np.vstack([X.min(0), X.max(0)])
    return X, titer, purity, bounds


def _sheet(seed: int = 11, n: int = 60) -> pd.DataFrame:
    X, titer, purity, _ = _anti_correlated_design(seed=seed, n=n)
    return pd.DataFrame({
        "Feat1": X[:, 0].round(3),
        "Feat2": X[:, 1].round(3),
        "purity_pct": purity.round(2),
        "product_titer": titer.round(3),
    })


# --- propose(): the acquisition mechanism ------------------------------------ #


def test_constraint_requires_both_surrogate_and_floor():
    """A floor with no model, or a model with no floor, cannot be turned into
    a constraint - `propose` must refuse rather than silently ignoring one
    half of the pair."""
    X, titer, purity, bounds = _anti_correlated_design()
    s = Surrogate().fit(X, titer, bounds=bounds)
    cs = Surrogate().fit(X, purity, bounds=bounds)
    with pytest.raises(ValueError):
        propose(s, bounds, q=2, constraint_surrogate=cs)
    with pytest.raises(ValueError):
        propose(s, bounds, q=2, constraint_floor=95.0)


def test_constrained_batch_has_higher_predicted_purity_than_unconstrained():
    """THE MONEY CLAIM for PART 2, at the mechanism level: on a sheet where
    high target correlates with LOW purity, constraining the acquisition on
    `purity >= floor` must visibly move the proposed batch toward higher
    predicted purity - a floor that changes nothing would mean the
    `constraints=` wiring is a no-op, not a real constraint."""
    X, titer, purity, bounds = _anti_correlated_design()
    s = Surrogate().fit(X, titer, bounds=bounds)
    cs = Surrogate().fit(X, purity, bounds=bounds)

    unconstrained = propose(s, bounds, q=5, seed=0)
    # Near the achievable purity ceiling and above the pool mean (~89), so
    # honoring it actually costs something against the (purity-reducing)
    # target - a floor no candidate could ever violate would make this a
    # vacuous pass.
    floor = 95.0
    constrained = propose(
        s, bounds, q=5, seed=0, constraint_surrogate=cs, constraint_floor=floor
    )

    pred_purity_unconstrained, _ = cs.posterior(unconstrained)
    pred_purity_constrained, _ = cs.posterior(constrained)
    assert pred_purity_constrained.mean() > pred_purity_unconstrained.mean()


def test_gating_and_constraint_compose():
    """Both PART 1 and PART 2 can be active on the same call - a smoke test
    that composing them does not raise and returns a valid batch (the
    per-part behavior is covered on its own elsewhere)."""
    from kalos.core.feasibility import FeasibilityClassifier

    X, titer, purity, bounds = _anti_correlated_design()
    s = Surrogate().fit(X, titer, bounds=bounds)
    cs = Surrogate().fit(X, purity, bounds=bounds)
    y_bin = (np.random.default_rng(0).random(len(titer)) > 0.3).astype(int)
    clf = FeasibilityClassifier().fit(X, y_bin)
    assert clf.fitted

    batch = propose(
        s, bounds, q=3, seed=0,
        feasibility_classifier=clf, constraint_surrogate=cs, constraint_floor=95.0,
    )
    assert batch.shape == (3, 2)
    assert np.isfinite(batch).all()


# --- _analyze(): the policy that decides whether to apply the constraint ---- #


def test_analyze_applies_constraint_and_reports_it():
    out = _analyze(_sheet(), constraint={"column": "purity_pct", "floor": 95.0})
    c = out["constraint"]
    assert c["applied"] is True
    assert c["column"] == "purity_pct"
    assert c["floor"] == 95.0
    assert c["n_rows_with_value"] == 60
    assert c["reason"] is None
    for p in out["proposals"]:
        assert "pred_purity_pct" in p
        assert set(p["pred_purity_pct"]) == {"mean", "std"}
        assert isinstance(p["pred_purity_pct"]["mean"], float)


def test_analyze_no_constraint_requested_reports_absence():
    out = _analyze(_sheet())
    c = out["constraint"]
    assert c["applied"] is False
    assert c["column"] is None
    assert c["floor"] is None
    assert c["n_rows_with_value"] == 0
    assert c["reason"] == "no constraint requested"
    for p in out["proposals"]:
        assert "pred_purity_pct" not in p


def test_analyze_constraint_column_missing_falls_back_unconstrained():
    out = _analyze(_sheet(), constraint={"column": "does_not_exist", "floor": 95.0})
    c = out["constraint"]
    assert c["applied"] is False
    assert "not found" in c["reason"]
    assert len(out["proposals"]) >= 1  # never crashes; still proposes
    for p in out["proposals"]:
        assert "pred_does_not_exist" not in p


def test_analyze_constraint_column_too_sparse_falls_back_unconstrained():
    df = _sheet()
    # Blank out all but 3 purity readings - below CONSTRAINT_MIN_ROWS (6).
    df.loc[3:, "purity_pct"] = np.nan
    out = _analyze(df, constraint={"column": "purity_pct", "floor": 95.0})
    c = out["constraint"]
    assert c["applied"] is False
    assert c["n_rows_with_value"] == 3
    assert c["n_rows_with_value"] < CONSTRAINT_MIN_ROWS
    assert "need at least" in c["reason"]
    assert len(out["proposals"]) >= 1  # never crashes; still proposes unconstrained


def test_constraint_column_never_appears_in_features():
    """The excluded-outcome leakage rule still holds with a constraint in
    play. `purity_pct` matches `profile.outcome_hint` and would be excluded
    from features on its own even with no constraint requested - asserted
    directly here (not merely assumed) as the spec requires."""
    out = _analyze(_sheet(), constraint={"column": "purity_pct", "floor": 95.0})
    assert "purity_pct" not in out["features"]
    assert "purity_pct" not in out["categorical_features"]


def test_constraint_column_never_appears_in_features_even_if_declared_a_role_feature():
    """A DEFENSIVE case beyond the natural outcome_hint exclusion: an explicit
    `ColumnRoles.features` list bypasses `outcome_hint` entirely (see
    `_resolve_columns`'s declared-mode branch), so a caller who both declares
    the constraint column as a feature AND asks to constrain on it must not
    leak that outcome into the model it is meant to be held out from."""
    from kalos.domains import ColumnRoles

    df = _sheet()
    roles = ColumnRoles(
        target="product_titer",
        features=("Feat1", "Feat2", "purity_pct"),  # deliberately includes it
    )
    out = _analyze(df, roles=roles, constraint={"column": "purity_pct", "floor": 95.0})
    assert "purity_pct" not in out["features"]


def test_constraint_response_is_json_serializable():
    out = _analyze(_sheet(), constraint={"column": "purity_pct", "floor": 95.0})
    json.dumps(out["constraint"])
    json.dumps(out["proposals"])
    out2 = _analyze(_sheet())
    json.dumps(out2["constraint"])
    json.dumps(out2["proposals"])
