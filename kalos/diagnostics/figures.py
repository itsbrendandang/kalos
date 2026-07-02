"""Diagnostic figures for the small-data BO regime (matplotlib, headless).

These render on the Python engine side (never in the JS dashboards) and are the
figures that make the model's honesty legible: a parity plot with posterior error
bars, the group-bootstrap CI of the CV metric against baselines, a parallel-
coordinates view of the design box, a PCA of the runs, and 1-D partial dependence
with an uncertainty band. Import is opt-in (`import kalos.diagnostics`) so the core
package never requires matplotlib.

Every function takes plain arrays and returns a matplotlib Figure; call `.savefig`.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")  # the engine renders to files, not a screen
import matplotlib.pyplot as plt  # noqa: E402

_ACCENT = "#2D4FCB"   # cobalt
_WARN = "#B4531A"      # clay (used only when a CI touches/crosses zero)


def _span(a: np.ndarray) -> float:
    return float(a.max() - a.min()) or 1.0


def _newax(ax, figsize=(5, 4)):
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def parity(actual, pred, std=None, *, ax=None, title="Predicted vs observed (out-of-fold)"):
    """Out-of-fold predicted vs observed with a y=x reference and optional GP
    posterior error bars. At tiny n this exposes whether a few high points drive
    the correlation and whether the error bars actually cover the residuals."""
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    fig, ax = _newax(ax)
    if std is not None:
        ax.errorbar(actual, pred, yerr=np.asarray(std, float), fmt="o", ms=4,
                    color=_ACCENT, ecolor="#9aa3b2", elinewidth=0.8, capsize=2, alpha=0.85)
    else:
        ax.scatter(actual, pred, s=22, color=_ACCENT, alpha=0.85)
    lo = float(min(actual.min(), pred.min()))
    hi = float(max(actual.max(), pred.max()))
    pad = 0.05 * ((hi - lo) or 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#888", lw=1, label="y = x")
    ax.set_xlabel("observed")
    ax.set_ylabel("predicted (out-of-fold)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def cv_forest(results: Sequence, *, ax=None, title="Grouped-CV Spearman (95% bootstrap CI)"):
    """Forest plot of point estimate + CI per model. `results` is an iterable of
    (name, point, lo, hi). The zero line makes 'does it beat chance?' unmissable,
    and any CI touching zero is drawn in the warn tone."""
    results = list(results)
    fig, ax = _newax(ax, figsize=(5.6, 0.6 * len(results) + 1.3))
    ys = np.arange(len(results))[::-1]
    for y, (name, point, lo, hi) in zip(ys, results):
        good = lo is not None and lo > 0
        color = _ACCENT if good else _WARN
        ax.plot([lo, hi], [y, y], color=color, lw=2.4, solid_capstyle="round")
        ax.plot(point, y, "o", color=color, ms=7)
        ax.text(hi, y + 0.14, f"  {point:.2f} [{lo:.2f}, {hi:.2f}]", va="bottom", fontsize=8, color="#333")
    ax.axvline(0, color="#888", ls="--", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in results], fontsize=9)
    ax.set_xlabel("Spearman rank correlation")
    ax.set_title(title)
    left = min(-0.1, min(r[2] for r in results) - 0.05)
    ax.set_xlim(left, 1.0)
    ax.set_ylim(-0.7, len(results) - 1 + 0.7)  # headroom so value labels never clip the title
    fig.tight_layout()
    return fig


def parallel_coordinates(X, y, feature_names, *, max_features=10, title="Design space (colored by outcome)"):
    """Min-max-normalized parallel coordinates of the most-varied inputs, each line
    a run colored by its outcome. Shows which factor combinations track high titer
    and where the runs actually sit in the box (coverage gaps)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    idx = np.sort(np.argsort(X.std(0))[::-1][:max_features])
    Xs = X[:, idx]
    names = [feature_names[i] for i in idx]
    lo, hi = Xs.min(0), Xs.max(0)
    Z = (Xs - lo) / np.where(hi > lo, hi - lo, 1.0)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(names)), 4))
    cmap = plt.cm.viridis
    ynorm = (y - y.min()) / _span(y)
    xs = np.arange(len(names))
    for r in np.argsort(y):  # draw high-outcome runs last (on top)
        ax.plot(xs, Z[r], color=cmap(ynorm[r]), lw=0.8, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["min", "max"], fontsize=8)
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(y.min(), y.max()))
    fig.colorbar(sm, ax=ax, label="outcome", fraction=0.04, pad=0.02)
    fig.tight_layout()
    return fig


def pca_scatter(X, y, groups=None, *, title="Run landscape (PCA)"):
    """PCA (not UMAP — unstable at this n) of the runs colored by outcome. Reveals
    cross-campaign overlap: the honest visual test of whether pooling across
    campaigns/products is even plausible before building multi-task."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    Z = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    fig, ax = _newax(None, figsize=(5, 4.2))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=28, edgecolor="white", linewidth=0.4)
    fig.colorbar(sc, ax=ax, label="outcome", fraction=0.045, pad=0.02)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.text(0.02, 0.98, "PCA (UMAP not trustworthy at this n)", transform=ax.transAxes,
            va="top", fontsize=7, color="#888")
    fig.tight_layout()
    return fig


def partial_dependence(predict_fn: Callable, X, feature_index: int, feature_name: str,
                       *, grid: int = 40, title: Optional[str] = None):
    """1-D partial dependence of the surrogate mean over one feature, with a
    posterior-std band if `predict_fn` returns (mean, std). At small n the
    flat-line-with-huge-band failure mode becomes visually obvious."""
    X = np.asarray(X, float)
    xs = np.linspace(X[:, feature_index].min(), X[:, feature_index].max(), grid)
    means = np.empty(grid)
    stds = np.zeros(grid)
    have_std = False
    base = X.mean(0, keepdims=True).repeat(len(X), 0)
    for k, v in enumerate(xs):
        G = base.copy()
        G[:, feature_index] = v
        out = predict_fn(G)
        if isinstance(out, tuple):
            m, s = out
            have_std = True
            stds[k] = float(np.mean(s))
        else:
            m = out
        means[k] = float(np.mean(m))
    fig, ax = _newax(None)
    ax.plot(xs, means, color=_ACCENT, lw=2)
    if have_std:
        ax.fill_between(xs, means - stds, means + stds, color=_ACCENT, alpha=0.15, label="±1 posterior sd")
        ax.legend(frameon=False, fontsize=8)
    ax.plot(X[:, feature_index], np.full(len(X), means.min()), "|", color="#aaa", ms=8)  # rug
    ax.set_xlabel(feature_name)
    ax.set_ylabel("partial dependence (surrogate mean)")
    ax.set_title(title or f"Partial dependence: {feature_name}")
    fig.tight_layout()
    return fig


__all__ = ["parity", "cv_forest", "parallel_coordinates", "pca_scatter", "partial_dependence"]
