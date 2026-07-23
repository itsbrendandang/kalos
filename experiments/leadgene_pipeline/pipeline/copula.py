"""Gaussian copula: models the joint DEPENDENCE structure between variables
independently of their marginal distributions.

  1. Each column's real values -> normal scores via its own empirical CDF
     (rank-based, nonparametric -- no assumption of Gaussian marginals).
  2. Correlation matrix estimated on those normal scores (shrunk toward
     identity if needed to stay positive-definite).
  3. Sample new points from N(0, R), map back through each column's inverse
     empirical CDF -> synthetic rows that preserve the real data's
     feature<->target dependency structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _to_normal_scores(x: np.ndarray) -> np.ndarray:
    n = len(x)
    ranks = pd.Series(x).rank(method="average").to_numpy()
    u = np.clip((ranks - 0.5) / n, 1e-6, 1 - 1e-6)
    return norm.ppf(u)


def _from_normal_scores(z: np.ndarray, real_values: np.ndarray) -> np.ndarray:
    u = np.clip(norm.cdf(z), 0.0, 1.0)
    sorted_vals = np.sort(real_values)
    n = len(sorted_vals)
    probs = (np.arange(1, n + 1) - 0.5) / n
    return np.interp(u, probs, sorted_vals)


def _shrink_to_pd(R: np.ndarray, max_tries: int = 12) -> np.ndarray:
    shrink = 0.05
    Rs = R.copy()
    for _ in range(max_tries):
        try:
            np.linalg.cholesky(Rs)
            return Rs
        except np.linalg.LinAlgError:
            Rs = (1 - shrink) * R + shrink * np.eye(R.shape[0])
            shrink = min(shrink * 2, 0.95)
    return np.eye(R.shape[0])


class GaussianCopula:
    """Fit on real rows; sample() draws new synthetic rows over the same columns."""

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.real_values_: dict[str, np.ndarray] = {}
        self.R_: np.ndarray | None = None
        self.L_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "GaussianCopula":
        imputed = df[self.cols].apply(pd.to_numeric, errors="coerce")
        imputed = imputed.fillna(imputed.median())
        Z = np.column_stack([_to_normal_scores(imputed[c].to_numpy()) for c in self.cols])
        for c in self.cols:
            self.real_values_[c] = imputed[c].to_numpy()
        R = np.corrcoef(Z, rowvar=False)
        R = np.nan_to_num(R, nan=0.0)
        np.fill_diagonal(R, 1.0)
        self.R_ = _shrink_to_pd(R)
        self.L_ = np.linalg.cholesky(self.R_)
        return self

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n, len(self.cols))) @ self.L_.T
        out = {c: _from_normal_scores(Z[:, j], self.real_values_[c]) for j, c in enumerate(self.cols)}
        return pd.DataFrame(out)
