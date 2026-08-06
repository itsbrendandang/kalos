"""Core platform tests: BoTorch surrogate + acquisition, gates, evaluation, embedder."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import RBFKernel, ScaleKernel

from kalos import (
    GatesConfig,
    MultiObjectiveSurrogate,
    Surrogate,
    ard_main_effects,
    check_gates,
    grouped_cv_spearman,
    propose,
    propose_multiobjective,
)
from kalos.features import KmerEmbedder


def test_surrogate_fits_signal():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (24, 3))
    y = X @ np.array([1.0, -0.5, 0.2])
    s = Surrogate().fit(X, y, bounds=np.array([[0, 0, 0], [1, 1, 1]], float))
    mean, std = s.posterior(X)
    assert mean.shape == (24,) and (std > 0).all()
    assert np.corrcoef(mean, y)[0, 1] > 0.8


def test_propose_returns_points_within_bounds():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (12, 2))
    y = -((X[:, 0] - 0.7) ** 2) - (X[:, 1] - 0.3) ** 2
    s = Surrogate().fit(X, y, bounds=np.array([[0, 0], [1, 1]], float))
    nxt = propose(s, np.array([[0, 0], [1, 1]], float), q=3)
    assert nxt.shape == (3, 2)
    assert (nxt >= 0).all() and (nxt <= 1).all()


def test_gates_fail_closed():
    good = {"surrogate_spearman": 0.5, "feasibility_auc": 0.8, "ece": 0.05, "brier": 0.1}
    assert check_gates(good).passed
    assert not check_gates({**good, "feasibility_auc": float("nan")}).passed  # unmeasured -> blocked
    assert not check_gates({k: v for k, v in good.items() if k != "ece"}).passed  # missing -> blocked
    assert not check_gates({**good, "surrogate_spearman": 0.05}).passed  # weak rank -> blocked
    assert check_gates(good, GatesConfig(enabled=False)).passed


def test_kmer_embedder_deterministic():
    e = KmerEmbedder(k=3, dim=32)
    a, b = e.embed("MKWVTFISLLFLF"), e.embed("MKWVTFISLLFLF")
    assert a.shape == (32,) and np.allclose(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-9


def test_grouped_cv_recovers_signal():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (40, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.05 * rng.normal(size=40)
    assert grouped_cv_spearman(X, y, n_splits=5) > 0.6


def test_fit_rejects_bad_input():
    s = Surrogate()
    b3 = np.array([[0, 0, 0], [1, 1, 1]], float)
    b2 = np.array([[0, 0], [1, 1]], float)
    with pytest.raises(ValueError):
        s.fit(np.empty((0, 3)), np.array([]), bounds=b3)  # empty
    with pytest.raises(ValueError):
        s.fit([[1.0, np.nan]], [1.0], bounds=b2)  # NaN in X
    with pytest.raises(ValueError):
        s.fit([[1.0, 2.0], [3.0, 4.0]], [1.0, np.nan], bounds=b2)  # NaN in y


def test_propose_before_fit_raises():
    with pytest.raises(AssertionError):
        propose(Surrogate(), np.array([[0, 0], [1, 1]], float), q=2)


def test_fit_with_design_bounds_normalizes_to_box():
    rng = np.random.default_rng(3)
    X = rng.uniform(0.2, 0.8, (20, 2))  # data covers only a sub-box of [0,1]^2
    y = X[:, 0] - 0.5 * X[:, 1]
    bounds = np.array([[0, 0], [1, 1]], float)
    s = Surrogate().fit(X, y, bounds=bounds)  # normalize to the design box, not the data
    nxt = propose(s, bounds, q=2)
    assert nxt.shape == (2, 2) and (nxt >= 0).all() and (nxt <= 1).all()


def test_multiobjective_proposes_and_pareto():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (12, 3))
    Y = np.stack([-((X[:, 0] - 0.7) ** 2), -((X[:, 0] - 0.2) ** 2)], axis=-1)  # tensioned objectives
    bounds = np.array([[0, 0, 0], [1, 1, 1]], float)
    s = MultiObjectiveSurrogate().fit(X, Y, bounds=bounds)
    nxt = propose_multiobjective(s, bounds, q=2)
    assert nxt.shape == (2, 3) and (nxt >= 0).all() and (nxt <= 1).all()
    px, py = s.pareto()
    assert py.shape[1] == 2 and len(py) >= 1


def test_multiobjective_degenerate_feature_never_proposes_out_of_range():
    # Regression for the Culture_Volume ~= 33,000,000 blow-up on the multi-objective
    # path: a constant feature (lower == upper) must not NaN-poison the fit or let a
    # proposal escape the observed range, mirroring the single-objective guard.
    rng = np.random.default_rng(2)
    n = 16
    methanol = rng.uniform(0, 4, n)
    culture_volume = np.full(n, 1000.0)  # constant column -> zero-width bound
    X = np.column_stack([methanol, culture_volume])
    Y = np.stack([1.5 * methanol, -((methanol - 2.0) ** 2)], axis=-1)
    bounds = np.vstack([X.min(0), X.max(0)])
    s = MultiObjectiveSurrogate().fit(X, Y, bounds=bounds)
    nxt = propose_multiobjective(s, bounds, q=3)
    assert np.isfinite(nxt).all()  # no NaN blow-up from the degenerate normalization box
    # the constant feature (col 1) stays pinned at ~1000, never explodes
    assert np.all(np.abs(nxt[:, 1] - 1000.0) <= 1e-3)
    assert np.all(nxt[:, 0] >= methanol.min() - 1e-6)
    assert np.all(nxt[:, 0] <= methanol.max() + 1e-6)


@pytest.mark.skipif(
    not os.environ.get("KALOS_TEST_ESM"),
    reason="set KALOS_TEST_ESM=1 to run the ESM-2 model download + embed test",
)
def test_esm2_embeds():
    pytest.importorskip("transformers")
    from kalos.features import ESM2Embedder

    e = ESM2Embedder()
    v = e.embed("MKWVTFISLLFLFSSAYSRGVF")
    assert v.shape == (e.dim,) and np.isfinite(v).all()


def _wrap_model(model, d: int) -> Surrogate:
    """Wrap an already-constructed botorch model into a bare `Surrogate` shell,
    bypassing `fit()` (which does not expose a `covar_module`/`cat_dims`
    override). Only `.model` and `._X`'s dimension count are used by
    `ard_main_effects`, so a dummy `._X` of the right shape is enough."""
    s = Surrogate()
    s.model = model
    s._X = torch.zeros(1, d, dtype=torch.double)
    return s


def test_ard_main_effects_on_fitted_model():
    rng = np.random.default_rng(3)
    n = 30
    x0 = rng.uniform(0, 1, n)
    x1 = rng.uniform(0, 1, n)
    y = 5.0 * x0 + 0.001 * x1 + rng.normal(0, 0.01, n)  # x0 dominates by a lot
    X = np.column_stack([x0, x1])
    s = Surrogate().fit(X, y, bounds=np.array([[0, 0], [1, 1]], float))

    out = ard_main_effects(s, ["dominant", "negligible"])
    assert out["available"] is True and out["reason"] is None
    effects = out["effects"]
    assert [e["name"] for e in effects] == ["dominant", "negligible"]  # correct dim mapping
    assert effects[0]["relative_importance"] > effects[1]["relative_importance"]
    total = sum(e["relative_importance"] for e in effects)
    assert abs(total - 1.0) < 1e-6  # normalized to sum to 1
    assert all(e["lengthscale"] > 0 for e in effects)


def test_ard_main_effects_rejects_feature_name_mismatch():
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, (20, 2))
    y = X[:, 0] - X[:, 1]
    s = Surrogate().fit(X, y, bounds=np.array([[0, 0], [1, 1]], float))
    with pytest.raises(AssertionError):
        ard_main_effects(s, ["only_one_name"])  # 1 name for a 2-D fit


def test_ard_main_effects_before_fit_raises():
    with pytest.raises(AssertionError):
        ard_main_effects(Surrogate(), ["a", "b"])


def test_ard_main_effects_reports_unavailable_for_non_ard_kernel():
    d = 3
    X = torch.rand(10, d, dtype=torch.double)
    y = X.sum(dim=-1, keepdim=True)
    covar = ScaleKernel(RBFKernel())  # no ard_num_dims -> one shared lengthscale
    model = SingleTaskGP(
        X, y, covar_module=covar,
        input_transform=Normalize(d=d), outcome_transform=Standardize(m=1),
    )
    s = _wrap_model(model, d)

    out = ard_main_effects(s, ["a", "b", "c"])
    assert out["available"] is False and out["effects"] is None
    assert "not ARD" in out["reason"]


def test_ard_main_effects_reports_unavailable_for_mixed_kernel():
    d = 3
    X = torch.rand(10, d, dtype=torch.double)
    X[:, 2] = (X[:, 2] * 3).floor()  # a fake 3-level categorical dim
    y = X[:, :1] - X[:, 1:2]
    model = MixedSingleTaskGP(
        X, y, cat_dims=[2],
        input_transform=Normalize(d=d, indices=[0, 1]),
        outcome_transform=Standardize(m=1),
    )
    s = _wrap_model(model, d)

    out = ard_main_effects(s, ["a", "b", "c"])
    assert out["available"] is False and out["effects"] is None
    assert "composite" in out["reason"]
