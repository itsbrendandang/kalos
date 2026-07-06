"""Synthetic bioprocess-like objectives with KNOWN optima.

Every objective is a MAXIMIZATION problem (titer / yield up is good), exposes its
true optimum so simple regret can be measured, and takes an observation-noise
argument so the benchmark can sweep the regime where a model's advantage over
random sampling appears or disappears. That noise sweep is the honest link to
real data: on the real media DoE the held-out signal is weak (grouped-CV
Spearman ~0.37-0.52), and the noise level is a large part of why.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class Objective:
    name: str
    dim: int
    bounds: np.ndarray  # (2, d): [lower_row, upper_row]
    optimum: float  # max of the noiseless surface
    scale: float  # typical output spread, so a noise fraction is meaningful
    _f: Callable[[np.ndarray], np.ndarray]

    def evaluate(
        self, X, noise: float = 0.0, rng: np.random.Generator | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (y_true, y_obs): the noiseless truth and a noisy observation.

        `noise` is a FRACTION of the objective's output scale, so `noise=0.15`
        means Gaussian measurement noise with sigma = 15% of the surface's spread
        - comparable across objectives with different units.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y_true = np.asarray(self._f(X), dtype=float).reshape(-1)
        if noise <= 0.0:
            return y_true, y_true.copy()
        if rng is None:
            rng = np.random.default_rng()
        sigma = noise * self.scale
        y_obs = y_true + rng.normal(0.0, sigma, size=y_true.shape)
        return y_true, y_obs


def gaussian_bump(dim: int = 4) -> Objective:
    """Smooth unimodal 'titer' surface: one Gaussian peak. Max = 10 at the center.

    The easy case. A model should help a little, but even random sampling does
    fine because there is a single wide basin of attraction.
    """
    center = np.full(dim, 0.6)
    amp, beta = 10.0, 4.0

    def f(X: np.ndarray) -> np.ndarray:
        d2 = ((X - center) ** 2).sum(axis=1)
        return amp * np.exp(-beta * d2)

    bounds = np.vstack([np.zeros(dim), np.ones(dim)])
    return Objective(f"gaussian_bump_{dim}d", dim, bounds, amp, amp, f)


def ackley(dim: int = 4) -> Objective:
    """Negated Ackley on [-2, 2]^d: rugged and multimodal. Max = 0 at the origin.

    The hard case. The surface has structure worth learning, so a model-based
    optimizer should beat space-filling by a wider margin here than on the bump -
    this is where BO is supposed to earn its keep.
    """
    a, b, c = 20.0, 0.2, 2.0 * np.pi

    def f(X: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        s1 = (X ** 2).sum(axis=1)
        s2 = np.cos(c * X).sum(axis=1)
        ack = -a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + np.e
        return -ack  # negate: maximization, optimum 0 at the origin

    bounds = np.vstack([np.full(dim, -2.0), np.full(dim, 2.0)])
    # Typical spread of -Ackley over [-2,2]^d is a few units; 6 is a fair scale.
    return Objective(f"ackley_{dim}d", dim, bounds, 0.0, 6.0, f)
