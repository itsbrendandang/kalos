"""Configuration loading and validation.

The YAML config is the single source of truth. It is loaded as a plain dict so
the ported model code (which reads cfg["preprocess"][...] / cfg["features"][...]
directly) works unchanged.
"""
from __future__ import annotations

from pathlib import Path

import yaml

# Base regressors fit via the uniform model factory.
BASE_MODELS = {"point_gb", "bootstrap_ensemble", "gaussian_process",
               "bayesian_ridge", "copula_augmented"}
# `hierarchical` is an empirical-Bayes reference+correction model wired via a
# separate optional path (needs a configured reference source), not the factory.
VALID_MODELS = BASE_MODELS | {"hierarchical"}


class ConfigError(ValueError):
    """Raised when the configuration is malformed or self-inconsistent."""


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ConfigError(f"config must be a YAML mapping, got {type(cfg).__name__}")
    _fill_defaults(cfg)
    _validate(cfg)
    return cfg


def _fill_defaults(cfg: dict) -> None:
    cfg.setdefault("seed", 0)
    cfg.setdefault("features", {})
    cfg["features"].setdefault("exclude", [])
    cfg["features"].setdefault("exclude_duration_dependent", False)
    cfg.setdefault("models", sorted(VALID_MODELS))
    cfg.setdefault("reference_model", "point_gb")
    cfg.setdefault("outputs", {})
    cfg["outputs"].setdefault("analysis", True)
    cfg["outputs"].setdefault("visualize", True)
    cfg["outputs"].setdefault("n_top", 5)
    cfg["outputs"].setdefault("top_features", 7)
    cfg.setdefault("preprocess", {})
    pp = cfg["preprocess"]
    pp.setdefault("numeric_impute", "median")
    pp.setdefault("categorical_impute", "most_frequent")
    pp.setdefault("correlation_prune_threshold", 0.85)
    pp.setdefault("drop_low_variance", True)
    pp.setdefault("max_numeric_features", 25)


def _validate(cfg: dict) -> None:
    for key in ("data", "columns"):
        if key not in cfg:
            raise ConfigError(f"config missing required top-level key: {key!r}")

    if "target" not in cfg["columns"]:
        raise ConfigError("columns.target is required (the titer column to predict)")

    bad = [m for m in cfg["models"] if m not in VALID_MODELS]
    if bad:
        raise ConfigError(f"unknown model(s) {bad}; valid: {sorted(VALID_MODELS)}")
    if not cfg["models"]:
        raise ConfigError("models must list at least one model")

    ref = cfg["reference_model"]
    if ref not in BASE_MODELS:
        raise ConfigError(f"reference_model {ref!r} must be one of {sorted(BASE_MODELS)}")
    if ref not in cfg["models"]:
        raise ConfigError(f"reference_model {ref!r} must also appear in models {cfg['models']}")

    # `hierarchical` needs a reference source to split reference vs correction rows.
    if "hierarchical" in cfg["models"] and not cfg["columns"].get("reference_source"):
        raise ConfigError("models includes 'hierarchical' but columns.reference_source is not set "
                           "(the source_col value whose rows form the reference pool)")

    include_sources = cfg.get("data", {}).get("include_sources")
    if include_sources is not None and not include_sources:
        raise ConfigError("data.include_sources is set but empty; omit it to include everything")
