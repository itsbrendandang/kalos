"""kalos.diagnostics renders headless (matplotlib Agg). Smoke tests."""
import numpy as np
import pytest

pytest.importorskip("matplotlib")
from matplotlib.figure import Figure

from kalos.diagnostics import (
    parity,
    cv_forest,
    parallel_coordinates,
    pca_scatter,
    partial_dependence,
)


def _data(n=30, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 1.5 - X[:, 1] + rng.normal(scale=0.2, size=n)
    return X, y


def test_parity_returns_figure():
    X, y = _data()
    pred = y + np.random.default_rng(1).normal(scale=0.1, size=len(y))
    fig = parity(y, pred, std=np.full(len(y), 0.2))
    assert isinstance(fig, Figure)


def test_cv_forest_returns_figure():
    fig = cv_forest([("GP", 0.37, 0.11, 0.60), ("mean baseline", 0.0, 0.0, 0.0)])
    assert isinstance(fig, Figure)


def test_parallel_coordinates_returns_figure():
    X, y = _data()
    fig = parallel_coordinates(X, y, [f"f{i}" for i in range(X.shape[1])])
    assert isinstance(fig, Figure)


def test_pca_scatter_returns_figure():
    X, y = _data()
    fig = pca_scatter(X, y)
    assert isinstance(fig, Figure)


def test_partial_dependence_returns_figure():
    X, y = _data()

    def predict(G):  # returns (mean, std)
        return G[:, 0] * 1.5, np.full(len(G), 0.3)

    fig = partial_dependence(predict, X, 0, "f0")
    assert isinstance(fig, Figure)
