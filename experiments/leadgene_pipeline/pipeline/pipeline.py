"""Train and predict orchestration.

The source's ClonePickingPipeline.run() did CV, final-fit, client-scoring, and
blending in one pass. Here the same computations are split across the two CLI
steps:

  TrainPipeline   -- build the pooled TrainingPool, CV each model for its trust
                     weight, fit each model on all training rows, persist.
  PredictPipeline -- load the fitted models, score the cohort, compute the
                     collapse check + confidence tiers, and blend.

No printing / disk I/O here except Report.save(); the CLI drives output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import artifact
from .blend import Blender
from .config import BASE_MODELS
from .data import TrainingPool, feature_columns, load_predict_cohort
from .evaluation import (CrossValidator, FittedMethod, MethodResult,
                         permutation_importance_spearman)
from .features import FeatureSelector
from .models import (BayesianRidgeModel, BootstrapEnsembleRegressor,
                     CopulaAugmentedModel, GaussianProcessModel,
                     GradientBoostingPointModel)

MIN_UNIQUE_GROUPS = 2

# A model has "collapsed" on the cohort when its predictions are near-constant,
# i.e. it no longer differentiates the wells. Measured relative to the
# predictions' own magnitude (coefficient of variation) so the test is
# invariant to titer units -- an absolute epsilon silently under-detects
# collapse on a raw-titer target where predictions are O(10-1000).
COLLAPSE_REL_TOL = 1e-3   # spread < 0.1% of the mean prediction -> collapsed
COLLAPSE_ABS_FLOOR = 1e-9  # predictions centered on ~0: fall back to absolute spread


def _collapsed_on_cohort(client_mean: np.ndarray) -> bool:
    """True when the cohort predictions are near-constant (the model no longer
    differentiates the wells). Measured relative to the predictions' own
    magnitude so the test is invariant to titer units; falls back to an
    absolute spread when the predictions are centered on ~0."""
    spread = float(np.std(client_mean))
    scale = float(np.abs(np.mean(client_mean)))
    if scale > COLLAPSE_ABS_FLOOR:
        return spread / scale < COLLAPSE_REL_TOL
    return spread < COLLAPSE_ABS_FLOOR


def _model_factories(cfg: dict, n_synthetic: int) -> dict:
    return {
        "point_gb": lambda cols, seed: GradientBoostingPointModel(cols, cfg, seed),
        "bootstrap_ensemble": lambda cols, seed: BootstrapEnsembleRegressor(cols, cfg, seed),
        "gaussian_process": lambda cols, seed: GaussianProcessModel(cols, cfg, seed),
        "bayesian_ridge": lambda cols, seed: BayesianRidgeModel(cols, cfg, seed),
        "copula_augmented": lambda cols, seed: CopulaAugmentedModel(
            cols, cfg, seed, TrainingPool.TARGET, n_synthetic),
    }


class TrainPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 0))
        cv = cfg.get("cv", {})
        self.n_boot = int(cv.get("n_boot", 2000))
        self.n_synthetic = int(cv.get("n_synthetic", 150))
        self.use_loo = bool(cv.get("use_loo", False))

    def run(self) -> tuple[list[FittedMethod], list[str], dict]:
        pool = TrainingPool.from_dir(self.cfg["data"]["train_dir"], self.cfg)
        feature_cols = feature_columns(pool.df, self.cfg)
        if not feature_cols:
            raise ValueError("no numeric feature columns found in the training files "
                             f"under {self.cfg['data']['train_dir']}")
        if not pool.is_trainable(MIN_UNIQUE_GROUPS):
            raise ValueError(
                f"training pool has only {pool.n_unique_groups} unique group(s); "
                f"need >= {MIN_UNIQUE_GROUPS} distinct groups to fit a ranker")

        selector = FeatureSelector(self.cfg, base_cap=int(self.cfg["preprocess"]["max_numeric_features"]))
        cv = CrossValidator(selector, feature_cols, seed=self.seed,
                            n_boot=self.n_boot, use_loo=self.use_loo)
        factories = _model_factories(self.cfg, self.n_synthetic)

        methods = [cv.fit_method(name, pool, factories[name])
                   for name in self.cfg["models"] if name in BASE_MODELS]
        if "hierarchical" in self.cfg["models"]:
            methods.append(self._fit_hierarchical(cv, pool))

        reference = self._build_reference(methods, pool)
        return methods, feature_cols, reference

    def _fit_hierarchical(self, cv: CrossValidator, pool: TrainingPool) -> FittedMethod:
        cols = self.cfg["columns"]
        source_col = cols.get("source_col", "source")
        ref_val = cols["reference_source"]
        if source_col not in pool.df.columns:
            raise ValueError(f"hierarchical needs columns.source_col {source_col!r} in the training data")
        ref_mask = pool.df[source_col].astype(str) == str(ref_val)
        if not ref_mask.any() or ref_mask.all():
            raise ValueError(f"reference_source {ref_val!r} must select some (not all) training rows")
        reference_pool = TrainingPool(pool.df[ref_mask], "reference")
        correction_pool = TrainingPool(pool.df[~ref_mask], "correction")
        return cv.fit_hierarchical("hierarchical", reference_pool, correction_pool)

    def _build_reference(self, methods: list[FittedMethod], pool: TrainingPool) -> dict:
        """The reference model (config `reference_model`) whose fit + CV Spearman +
        permutation importance drive the ranking / driver figures."""
        name = self.cfg["reference_model"]
        fm = next((m for m in methods if m.name == name), methods[0])
        importance = permutation_importance_spearman(fm.model, pool.df, pool.y(), fm.cols, self.seed)
        return {
            "model_name": fm.name,
            "cv_spearman": fm.cv_spearman, "cv_p": fm.cv_p,
            "ci_low": fm.ci_low, "ci_high": fm.ci_high, "verdict": fm.verdict,
            "importance": importance,
            "titer_mean": float(np.mean(pool.y())),
            "feature_cols": list(fm.cols),
            "train_features": pool.df[fm.cols].reset_index(drop=True),
        }

    def save(self, path: str | Path, methods: list[FittedMethod], feature_cols: list[str],
             reference: dict) -> Path:
        return artifact.save(path, seed=self.seed, config=self.cfg,
                             feature_cols=feature_cols, methods=methods, reference=reference)


@dataclass
class Report:
    results: list[MethodResult]
    blend_table: pd.DataFrame
    weights: dict
    note: str
    cohort: pd.DataFrame | None = None      # prediction rows (features), for driver/novelty
    reference: dict | None = None           # reference-model info persisted at train time

    def reference_result(self) -> MethodResult | None:
        """The MethodResult of the model designated as the reference (drives figures)."""
        if not self.reference:
            return None
        name = self.reference.get("model_name")
        return next((r for r in self.results if r.name == name), None)

    def predictions_table(self) -> pd.DataFrame:
        """Blended predicted titer + rank + confidence tier per row, plus each
        model's individual predicted titer and uncertainty."""
        table = self.blend_table.rename(columns={"blended_score": "predicted_titer"}).copy()
        table = table.drop(columns=["n_methods_agreeing_top10"], errors="ignore")
        # Run-level honesty flag carried in the primary output: True only when at
        # least one method earned real (validated, non-collapsed) weight. False
        # means the blend fell back to an equal-weight consensus of unvalidated
        # models -- the ranking is not statistically distinguishable from noise.
        table["blend_validated"] = any(r.weight > 0 for r in self.results)
        for r in self.results:
            mean = pd.Series(r.client_mean, index=r.client_well_ids)
            std = pd.Series(r.client_std, index=r.client_well_ids)
            table[f"{r.name}_pred"] = np.round(mean.reindex(table["well_id"]).to_numpy(), 4)
            table[f"{r.name}_std"] = np.round(std.reindex(table["well_id"]).to_numpy(), 4)
        return table

    def manifest(self) -> dict:
        return {
            "final_blend": {"weights": {k: round(v, 3) for k, v in self.weights.items()},
                            "note": self.note},
            "methods": {
                r.name: {"n_train": r.n_train, "n_unique_groups": r.n_unique_groups,
                         "n_features": r.n_features, "cv_spearman": round(r.cv_spearman, 3),
                         "cv_p": round(r.cv_p, 4),
                         "cv_spearman_ci95": [round(r.ci_low, 3), round(r.ci_high, 3)],
                         "verdict": r.verdict, "collapsed_on_client": r.collapsed_on_client,
                         "weight": round(self.weights.get(r.name, 0.0), 3)}
                for r in self.results
            },
        }

    def save(self, output_csv: str | Path, manifest_json: str | Path | None) -> dict[str, Path]:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.predictions_table().to_csv(output_csv, index=False)
        written = {"predictions": output_csv}
        if manifest_json:
            manifest_json = Path(manifest_json)
            manifest_json.parent.mkdir(parents=True, exist_ok=True)
            manifest_json.write_text(json.dumps(self.manifest(), indent=2))
            written["manifest"] = manifest_json
        return written


class PredictPipeline:
    def __init__(self, bundle: dict, cfg: dict):
        self.bundle = bundle
        self.cfg = cfg

    def run(self) -> Report:
        cohort = load_predict_cohort(self.cfg)
        if cohort.empty:
            raise ValueError("prediction cohort is empty (check data.predict_csv)")

        results = []
        for fm in self.bundle["methods"]:
            client_mean, client_std = fm.model.score_client(cohort)
            collapsed = _collapsed_on_cohort(client_mean)
            results.append(MethodResult(
                name=fm.name, n_train=fm.n_train, n_unique_groups=fm.n_unique_groups,
                n_features=fm.n_features, cv_spearman=fm.cv_spearman, cv_p=fm.cv_p,
                ci_low=fm.ci_low, ci_high=fm.ci_high, verdict=fm.verdict,
                client_well_ids=cohort["well_id"].to_numpy(),
                client_mean=client_mean, client_std=client_std, collapsed_on_client=collapsed,
            ))

        blender = Blender(results)
        return Report(results=results, blend_table=blender.blended_table(),
                      weights=blender.weights(), note=blender.note(),
                      cohort=cohort, reference=self.bundle.get("reference"))
