"""Core platform tests: BoTorch surrogate + acquisition, gates, evaluation, embedder."""
from __future__ import annotations

import os

import numpy as np
import pytest

from voyager import GatesConfig, Surrogate, check_gates, grouped_cv_spearman, propose
from voyager.features import KmerEmbedder


def test_surrogate_fits_signal():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (24, 3))
    y = X @ np.array([1.0, -0.5, 0.2])
    s = Surrogate().fit(X, y)
    mean, std = s.posterior(X)
    assert mean.shape == (24,) and (std > 0).all()
    assert np.corrcoef(mean, y)[0, 1] > 0.8


def test_propose_returns_points_within_bounds():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (12, 2))
    y = -((X[:, 0] - 0.7) ** 2) - (X[:, 1] - 0.3) ** 2
    s = Surrogate().fit(X, y)
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


@pytest.mark.skipif(
    not os.environ.get("VOYAGER_TEST_ESM"),
    reason="set VOYAGER_TEST_ESM=1 to run the ESM-2 model download + embed test",
)
def test_esm2_embeds():
    pytest.importorskip("transformers")
    from voyager.features import ESM2Embedder

    e = ESM2Embedder()
    v = e.embed("MKWVTFISLLFLFSSAYSRGVF")
    assert v.shape == (e.dim,) and np.isfinite(v).all()
