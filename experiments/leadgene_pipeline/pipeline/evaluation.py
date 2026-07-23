"""Cross-validation + bootstrap confidence intervals: turns a model class
into a trust metric instead of a bare point estimate.

A method whose CV Spearman's 95% bootstrap CI includes 0 is not
distinguishable from noise at this sample size, regardless of how the point
estimate looks -- the asymptotic p-value alone is unreliable down at n=10
(a Spearman of exactly -1.0 with p=0.0 is a degenerate small-n artifact, not
a real, significant inverse relationship).

The CV primitives (`_out_of_fold_predictions`, `bootstrap_ci`) and the
`MethodResult` container are ported unchanged from Leadgene_Clone_Picker's
clone_ranking/evaluation.py. The original `CrossValidator.evaluate` combined
CV, final-fit, and client scoring in one call; here train (CV + final fit,
`fit_method`) and predict (client scoring, done in pipeline.py) are split
across the two CLI steps -- same computations, same numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .data import TrainingPool
from .features import FeatureSelector


@dataclass
class MethodResult:
    name: str
    n_train: int
    n_unique_groups: int
    n_features: int
    cv_spearman: float
    cv_p: float
    ci_low: float
    ci_high: float
    verdict: str
    client_well_ids: np.ndarray
    client_mean: np.ndarray
    client_std: np.ndarray
    collapsed_on_client: bool = False

    @property
    def weight(self) -> float:
        """Zero out a method that (a) shows no real CV signal, or (b) has
        collapsed to a near-constant prediction on the client cohort -- real
        in-distribution skill that provides zero differentiation for these
        specific wells is not something a blend should trust."""
        return 0.0 if self.collapsed_on_client else max(0.0, self.cv_spearman)


@dataclass
class FittedMethod:
    """Train-time output for one method: the model fit on all training rows,
    its selected columns, and the CV trust metrics. Persisted in the artifact;
    the client-cohort scoring (client_mean/std, collapse, tiers) is deferred to
    predict time in pipeline.py."""
    name: str
    model: object
    cols: list[str]
    n_train: int
    n_unique_groups: int
    n_features: int
    cv_spearman: float
    cv_p: float
    ci_low: float
    ci_high: float
    verdict: str


class CrossValidator:
    """Runs group-aware CV for a model factory over a TrainingPool, then
    fits a final model on all of it (client scoring happens later, at predict)."""

    def __init__(self, feature_selector: FeatureSelector, sensor_cols: list[str],
                 seed: int = 0, n_boot: int = 2000, use_loo: bool = False):
        self.feature_selector = feature_selector
        self.sensor_cols = sensor_cols
        self.seed = seed
        self.n_boot = n_boot
        self.use_loo = use_loo

    def _splitter(self, n_groups: int):
        return LeaveOneGroupOut() if self.use_loo else GroupKFold(n_splits=min(5, n_groups))

    def _out_of_fold_predictions(self, pool: TrainingPool, model_factory) -> np.ndarray:
        y = pool.y()
        groups = pool.groups()
        splitter = self._splitter(len(np.unique(groups)))
        preds = np.zeros(len(y))
        for tr, te in splitter.split(pool.df, y, groups):
            cols = self.feature_selector.select(pool.df.iloc[tr], self.sensor_cols)
            model = model_factory(cols, self.seed)
            model.fit(pool.df.iloc[tr], y[tr])
            preds[te] = model.predict(pool.df.iloc[te])
        return preds

    def bootstrap_ci(self, preds: np.ndarray, y: np.ndarray) -> dict:
        rho, p = spearmanr(preds, y)
        rng = np.random.default_rng(self.seed)
        n = len(preds)
        boots = np.empty(self.n_boot)
        for b in range(self.n_boot):
            idx = rng.integers(0, n, size=n)
            r, _ = spearmanr(preds[idx], y[idx])
            boots[b] = 0.0 if np.isnan(r) else r
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = ("USABLE (CI excludes 0)" if lo > 0 else
                   "NOT VALIDATED (bootstrap CI on Spearman includes 0)")
        return {"cv_spearman": float(rho), "cv_p": float(p),
                "ci_low": float(lo), "ci_high": float(hi), "verdict": verdict}

    def fit_method(self, name: str, pool: TrainingPool, model_factory) -> FittedMethod:
        """Group-aware CV for the trust metrics, then a final fit on all rows.
        Identical to the CV + final-fit half of the original `evaluate`."""
        preds = self._out_of_fold_predictions(pool, model_factory)
        stats = self.bootstrap_ci(preds, pool.y())
        cols = self.feature_selector.select(pool.df, self.sensor_cols)
        final_model = model_factory(cols, self.seed).fit(pool.df, pool.y())
        return FittedMethod(
            name=name, model=final_model, cols=cols, n_train=pool.n_rows,
            n_unique_groups=pool.n_unique_groups, n_features=len(cols), **stats,
        )

    def fit_hierarchical(self, name: str, reference_pool: TrainingPool,
                         correction_pool: TrainingPool) -> FittedMethod:
        """Empirical-Bayes reference+correction fit, ported from Leadgene's
        evaluate_hierarchical: one BayesianRidge reference (fit on the reference
        pool, reused read-only), and a correction fit on the correction pool's
        residual against that prior. CV is group-aware over the correction pool;
        no leakage because the two pools share no groups."""
        from .models import BayesianRidgeModel, HierarchicalShrinkageModel
        cfg_raw = self.feature_selector.cfg_raw
        ref_cols = self.feature_selector.select(reference_pool.df, self.sensor_cols)
        reference = BayesianRidgeModel(ref_cols, cfg_raw, self.seed)
        reference.fit(reference_pool.df, reference_pool.y())

        corr_cols = self.feature_selector.select(correction_pool.df, self.sensor_cols)
        y, groups = correction_pool.y(), correction_pool.groups()
        splitter = self._splitter(len(np.unique(groups)))
        preds = np.zeros(len(y))
        for tr, te in splitter.split(correction_pool.df, y, groups):
            model = HierarchicalShrinkageModel(reference, corr_cols, cfg_raw, self.seed)
            model.fit(correction_pool.df.iloc[tr], y[tr])
            preds[te] = model.predict(correction_pool.df.iloc[te])
        stats = self.bootstrap_ci(preds, y)

        final_model = HierarchicalShrinkageModel(reference, corr_cols, cfg_raw, self.seed)
        final_model.fit(correction_pool.df, y)
        return FittedMethod(
            name=name, model=final_model, cols=corr_cols, n_train=correction_pool.n_rows,
            n_unique_groups=correction_pool.n_unique_groups, n_features=len(corr_cols), **stats,
        )


def permutation_importance_spearman(model, X, y, cols, seed: int, n_repeats: int = 10):
    """Drop in titer-ranking Spearman when each feature is shuffled. Ported from
    Leadgene's driver_and_contrast_figure.permutation_importance_spearman.

    NOTE: this is IN-SAMPLE (model was fit on X) -- it reports which features the
    fitted model leans on, not a held-out importance. Labelled honestly downstream;
    see TODO.md for the out-of-fold version.
    """
    rng = np.random.default_rng(seed)
    base_pred = model.predict(X)
    base_rho, _ = spearmanr(base_pred, y)
    base_rho = 0.0 if np.isnan(base_rho) else base_rho  # constant preds -> undefined rho
    rows = []
    for col in cols:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            rho, _ = spearmanr(model.predict(Xp), y)
            drops.append(base_rho - (0.0 if np.isnan(rho) else rho))
        rows.append({"feature": col, "importance": float(np.mean(drops))})
    rows.sort(key=lambda r: -r["importance"])
    return rows
