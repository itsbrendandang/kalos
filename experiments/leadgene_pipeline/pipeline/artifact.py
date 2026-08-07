"""Persist / restore the trained-model bundle via joblib.

The bundle holds the fitted models (each carrying its own fitted preprocessor
and selected columns), their CV trust metrics, the resolved feature columns, and
a config snapshot. Predict-time scoring refits nothing -- it loads this bundle,
so there is no train/serve skew.
"""
from __future__ import annotations

from pathlib import Path

import joblib

from .evaluation import FittedMethod

# v2 adds the `reference` block (reference-model CV Spearman, permutation
# importance, training titer mean + feature frame) that drives the figures.
ARTIFACT_VERSION = 2


def save(path: str | Path, *, seed: int, config: dict, feature_cols: list[str],
         methods: list[FittedMethod], reference: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": ARTIFACT_VERSION,
        "seed": seed,
        "config": config,
        "feature_cols": feature_cols,
        "methods": methods,
        "reference": reference,
    }
    joblib.dump(bundle, path)
    return path


def load(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model artifact not found: {path} (run `train` first)")
    bundle = joblib.load(path)
    if bundle.get("version") != ARTIFACT_VERSION:
        raise ValueError(
            f"artifact version {bundle.get('version')} != expected {ARTIFACT_VERSION}; "
            f"retrain with `python -m pipeline train` to regenerate {path}")
    return bundle
