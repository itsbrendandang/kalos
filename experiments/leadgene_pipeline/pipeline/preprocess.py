"""Preprocessing + feature reduction, ported unchanged from Leadgene_Clone_Picker
(clone_select/preprocess/transform.py and the reduction helpers in
clone_select/preprocess/assemble.py). Only the pieces clone_ranking depends on
are kept.

  numeric:     SimpleImputer(median) -> StandardScaler
  categorical: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')

Fit once, frozen at inference: the fitted ColumnTransformer is carried inside each
model artifact so predict applies exactly the transform it was trained with.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str], cfg) -> ColumnTransformer:
    pp = cfg["preprocess"]
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy=pp.get("numeric_impute", "median"))),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy=pp.get("categorical_impute", "most_frequent"))),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical, categorical_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def reduce_numeric_features(train_df: pd.DataFrame, numeric_cols: list[str], cfg) -> list[str]:
    """Dimensionality control on the TRAIN slice only.

    Drops near-zero-variance columns, greedily prunes one of any highly-correlated
    pair (|Pearson r| > threshold, dropping the higher-mean-correlation column),
    then caps at max_numeric_features by variance. Fit on train; applied as a
    column list to all splits.
    """
    pp = cfg["preprocess"]
    numeric_cols = duration_robust_filter(numeric_cols, cfg)  # cross-run comparability guard
    X = train_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    keep = list(numeric_cols)

    if pp.get("drop_low_variance", True):
        var = X.var(numeric_only=True)
        keep = [c for c in keep if var.get(c, 0.0) and var.get(c, 0.0) > 1e-12]

    if len(keep) > 1:
        corr = X[keep].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        thr = float(pp.get("correlation_prune_threshold", 0.85))
        to_drop = set()
        mean_corr = corr.mean()
        for col in upper.columns:
            for row in upper.index:
                if upper.loc[row, col] > thr:
                    drop = col if mean_corr[col] >= mean_corr[row] else row
                    to_drop.add(drop)
        keep = [c for c in keep if c not in to_drop]

    cap = int(pp.get("max_numeric_features", 25))
    if len(keep) > cap:
        var = X[keep].var().sort_values(ascending=False)
        keep = var.head(cap).index.tolist()
    return keep


# Feature bases whose value scales with run length / sampling and is therefore
# NOT comparable across runs of different duration (the cross-cohort artifact).
DURATION_DEPENDENT_DEFAULT = {
    "n_points", "n_missing", "auc",
    "t_at_max", "t_at_min", "t_to_half_peak_recovery", "t_max_rolling_mean_change",
    "n_peaks", "n_troughs", "t_first_peak", "t_last_peak", "t_first_trough", "t_last_trough",
}


def feature_base_name(col: str) -> str:
    """Strip channel prefix (DO_/pH_) and volume suffix (_24w) -> bare feature name."""
    b = col
    for pre in ("DO_", "pH_"):
        if b.startswith(pre):
            b = b[len(pre):]
            break
    return re.sub(r"_[0-9]+w$", "", b)


def duration_robust_filter(cols, cfg) -> list:
    """Drop duration/sampling-dependent features when enabled in config."""
    if not cfg.get("features", {}).get("exclude_duration_dependent", False):
        return list(cols)
    bad = set(cfg["features"].get("duration_dependent_features") or DURATION_DEPENDENT_DEFAULT)
    return [c for c in cols if feature_base_name(c) not in bad]


def cohort_novelty(reference_df: pd.DataFrame, cohort_df: pd.DataFrame,
                   cols: list[str], z_thresh: float = 3.0) -> dict:
    """Flag features where the prediction cohort sits off the training distribution.

    Returns {feature: cohort_median_z} for features whose cohort median is more than
    z_thresh training-SDs from the training mean -- a systematic cohort-wide offset
    meaning the model is extrapolating on that axis. Ported from
    Leadgene_Clone_Picker/clone_select/preprocess/assemble.py:cohort_novelty.
    """
    flagged: dict[str, float] = {}
    for c in cols:
        if c not in reference_df.columns or c not in cohort_df.columns:
            continue
        mu = reference_df[c].mean()
        sd = reference_df[c].std()
        if not sd or np.isnan(sd):
            continue
        z = float((cohort_df[c].median() - mu) / sd)
        if abs(z) > z_thresh:
            flagged[c] = round(z, 2)
    return dict(sorted(flagged.items(), key=lambda kv: -abs(kv[1])))
