"""Core platform tests: BoTorch surrogate + acquisition, gates, evaluation, embedder."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from kalos import (
    GatesConfig,
    MultiObjectiveSurrogate,
    Surrogate,
    bootstrap_spearman,
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


def test_multiobjective_models_do_not_share_an_input_transform():
    """`MultiObjectiveSurrogate.fit` builds one GP per objective in a
    `ModelListGP`. Passing the SAME `Normalize` instance to every model is
    benign while bounds are fixed, but it is one `learn_bounds=True` away from
    cross-coupling the objectives through shared transform state - sharing a
    stateful `torch.nn.Module` across models is a latent aliasing hazard even
    when today's fit never mutates it. Each model must own its own instance."""
    rng = np.random.default_rng(1)
    n, d = 14, 2
    X = rng.uniform(0, 1, (n, d))
    Y = np.stack([X[:, 0] - 0.5 * X[:, 1], 0.5 * X[:, 1] - X[:, 0]], axis=-1)
    bounds = np.array([[0, 0], [1, 1]], float)
    s = MultiObjectiveSurrogate().fit(X, Y, bounds=bounds)
    transforms = list(s.model.models)
    assert len({id(m.input_transform) for m in transforms}) == len(transforms)


def test_multiobjective_fresh_normalize_leaves_the_fit_unchanged():
    """The per-model `Normalize` is a pure isolation change, not a behavior
    change: the bounds are identical across objectives either way, so fitting
    the same data twice must still give the same posterior mean."""
    rng = np.random.default_rng(2)
    n, d = 14, 2
    X = rng.uniform(0, 1, (n, d))
    Y = np.stack([X[:, 0] - 0.5 * X[:, 1], 0.5 * X[:, 1] - X[:, 0]], axis=-1)
    bounds = np.array([[0, 0], [1, 1]], float)
    s1 = MultiObjectiveSurrogate().fit(X, Y, bounds=bounds)
    s2 = MultiObjectiveSurrogate().fit(X, Y, bounds=bounds)
    Xt = torch.as_tensor(X, dtype=torch.float64)
    p1 = s1.model.posterior(Xt).mean.detach().numpy()
    p2 = s2.model.posterior(Xt).mean.detach().numpy()
    assert np.allclose(p1, p2, atol=1e-6)


def test_bootstrap_spearman_degenerate_draws_are_not_zero_anchored():
    """A near-constant column makes some bootstrap resamples degenerate: with
    only two non-zero rows out of ten, roughly 1 in 10 resamples misses both
    and the column goes constant for that draw, so Spearman's rho is undefined
    (nan) for it. Coercing that nan to 0.0 (the old behavior) is a fabricated
    "this draw found no association" data point, not a neutral default - it
    pulls the whole bootstrap distribution toward zero and narrows the CI
    dishonestly for exactly the noisiest features. The fix computes mean/std/CI
    over the valid draws only, so a real association is not zero-anchored."""
    rng = np.random.default_rng(0)
    n = 10
    x = np.array([0.0] * 8 + [1.0, 2.0])  # near-constant: most mass at one value
    y = np.arange(n, dtype=float) + rng.normal(0, 0.01, n)
    out = bootstrap_spearman(x.reshape(-1, 1), y, B=200, random_state=0, feature_names=["x"])
    assert np.isfinite(out["mean"]).all()
    assert np.isfinite(out["std"]).all()
    assert np.isfinite(out["lo"]).all() and np.isfinite(out["hi"]).all()
    assert out["lo"][0] > 0  # a real positive association; the CI must not include 0


def test_bootstrap_spearman_all_degenerate_column_falls_back_to_zero():
    """Every resample of a truly constant column is degenerate - there is no
    valid draw anywhere to compute a mean/std/CI from. That reproduces the
    pre-fix safe output (all zeros) rather than leaking a nan into a
    strict-JSON API response, which would 500 the request."""
    x = np.zeros(10)
    y = np.arange(10, dtype=float)
    out = bootstrap_spearman(x.reshape(-1, 1), y, B=50, random_state=0, feature_names=["x"])
    assert out["mean"][0] == 0.0
    assert out["std"][0] == 0.0
    assert out["lo"][0] == 0.0
    assert out["hi"][0] == 0.0


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
