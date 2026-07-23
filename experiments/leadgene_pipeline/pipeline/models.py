"""Regressor classes for the productivity-ranking task.

Every model implements the same small interface: fit(X, y) -> self,
predict_mean_std(X) -> (mean, std). A model with no native uncertainty (the
plain gradient-boosted point estimate) reports std=0 rather than faking one
-- callers can rely on std meaning something whenever it's nonzero.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import BayesianRidge

from .copula import GaussianCopula
from .preprocess import build_preprocessor


class ConfidenceRegressor(ABC):
    """Shared fit/predict_mean_std interface for every model in this pipeline."""

    def __init__(self, cols: list[str], cfg_raw: dict):
        self.cols = cols
        self.cfg_raw = cfg_raw

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ConfidenceRegressor":
        ...

    @abstractmethod
    def predict_mean_std(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_mean_std(X)[0]

    def score_client(self, client_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict on an external cohort that may be missing some of this
        model's selected columns entirely -- fill with NaN so the fitted
        imputer handles them, rather than erroring."""
        scored = client_df.copy()
        for c in self.cols:
            if c not in scored.columns:
                scored[c] = np.nan
        return self.predict_mean_std(scored)


class GradientBoostingPointModel(ConfidenceRegressor):
    """The pipeline's baseline productivity regressor: one
    HistGradientBoostingRegressor, no native uncertainty (std reported as 0)."""

    def __init__(self, cols, cfg_raw, seed: int):
        super().__init__(cols, cfg_raw)
        self.seed = seed
        self.pre = None
        self.reg = None

    def fit(self, X, y):
        self.pre = build_preprocessor(self.cols, [], self.cfg_raw)
        Xt = self.pre.fit_transform(X[self.cols])
        self.reg = HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, learning_rate=0.08, random_state=self.seed).fit(Xt, y)
        return self

    def predict_mean_std(self, X):
        mean = self.reg.predict(self.pre.transform(X[self.cols]))
        return mean, np.zeros_like(mean)


class BootstrapEnsembleRegressor(ConfidenceRegressor):
    """Bagged HistGradientBoostingRegressor members; mean/std across members
    is the epistemic-uncertainty estimate -- the same idea EnsembleGate uses
    for the classifier, applied here to the regression side of the pipeline."""

    def __init__(self, cols, cfg_raw, seed: int, n_members: int = 25):
        super().__init__(cols, cfg_raw)
        self.seed = seed
        self.n_members = n_members
        self.members: list[tuple] = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.members = []
        for m in range(self.n_members):
            idx = rng.integers(0, n, size=n)
            pre = build_preprocessor(self.cols, [], self.cfg_raw)
            Xt = pre.fit_transform(X.iloc[idx][self.cols])
            reg = HistGradientBoostingRegressor(
                max_depth=3, max_iter=200, learning_rate=0.08, random_state=self.seed + m)
            reg.fit(Xt, y[idx])
            self.members.append((pre, reg))
        return self

    def predict_mean_std(self, X):
        preds = np.stack([reg.predict(pre.transform(X[self.cols])) for pre, reg in self.members])
        return preds.mean(axis=0), preds.std(axis=0)


class GaussianProcessModel(ConfidenceRegressor):
    """Closed-form Bayesian regressor: mean + variance, no sampling required.
    Degrades gracefully off-distribution -- reverts to the prior with wide
    uncertainty instead of a tree's silently-wrong constant leaf value."""

    def __init__(self, cols, cfg_raw, seed: int):
        super().__init__(cols, cfg_raw)
        self.seed = seed
        self.pre = None
        self.gp = None

    def fit(self, X, y):
        self.pre = build_preprocessor(self.cols, [], self.cfg_raw)
        Xt = self.pre.fit_transform(X[self.cols])
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-2)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=3, random_state=self.seed)
        self.gp.fit(Xt, y)
        return self

    def predict_mean_std(self, X):
        return self.gp.predict(self.pre.transform(X[self.cols]), return_std=True)


class BayesianRidgeModel(ConfidenceRegressor):
    """Linear Bayesian model with evidence-based automatic regularization --
    the alpha/lambda precisions are optimized via type-II maximum likelihood,
    so it self-regularizes harder as n shrinks rather than needing a
    manually tuned penalty."""

    def __init__(self, cols, cfg_raw, seed: int = 0):
        super().__init__(cols, cfg_raw)
        self.seed = seed
        self.pre = None
        self.reg = None

    def fit(self, X, y):
        self.pre = build_preprocessor(self.cols, [], self.cfg_raw)
        Xt = self.pre.fit_transform(X[self.cols])
        self.reg = BayesianRidge().fit(Xt, y)
        return self

    def predict_mean_std(self, X):
        return self.reg.predict(self.pre.transform(X[self.cols]), return_std=True)


class CopulaAugmentedModel(ConfidenceRegressor):
    """Fits a Gaussian copula over (features, target), draws n_synthetic
    rows, and trains a point regressor on the synthetic rows only."""

    def __init__(self, cols, cfg_raw, seed: int, target_col: str, n_synthetic: int = 150):
        super().__init__(cols, cfg_raw)
        self.seed = seed
        self.target_col = target_col
        self.n_synthetic = n_synthetic
        self.inner: GradientBoostingPointModel | None = None

    def fit(self, X, y):
        df = X.copy()
        df[self.target_col] = y
        copula = GaussianCopula(self.cols + [self.target_col]).fit(df)
        synth = copula.sample(self.n_synthetic, self.seed)
        self.inner = GradientBoostingPointModel(self.cols, self.cfg_raw, self.seed)
        self.inner.fit(synth, synth[self.target_col].to_numpy())
        return self

    def predict_mean_std(self, X):
        return self.inner.predict_mean_std(X)


class HierarchicalShrinkageModel(ConfidenceRegressor):
    """Empirical-Bayes pooling: a reference model trained on a large cohort
    (e.g. the moat) supplies a prior prediction; a small Bayesian-Ridge
    correction is fit on the target cohort's RESIDUAL against that prior.
    The target cohort borrows strength from the reference instead of being
    trained standalone (fails at small n) or force-pooled at equal weight
    (dilutes the reference).

    `reference` is a pre-fitted BayesianRidgeModel, shared read-only across
    every CV fold -- it never sees the correction cohort's rows, so there is
    no leakage from reusing one fit across folds.
    """

    def __init__(self, reference: BayesianRidgeModel, correction_cols, cfg_raw, seed: int = 0):
        super().__init__(correction_cols, cfg_raw)
        self.reference = reference
        self.correction = BayesianRidgeModel(correction_cols, cfg_raw, seed)

    def fit(self, X, y):
        ref_mean, _ = self.reference.score_client(X)
        residual = y - ref_mean
        self.correction.fit(X, residual)
        return self

    def predict_mean_std(self, X):
        ref_mean, _ = self.reference.score_client(X)
        corr_mean, corr_std = self.correction.score_client(X)
        return ref_mean + corr_mean, corr_std
