"""Feature selection: correlation pruning + variance filtering, with a cap
that scales with training-set size instead of a fixed constant regardless
of n. A fixed cap of 25 features on a 10-row training set (2.5 features per
sample) is a textbook overfitting setup -- it's very plausibly why the
plain passage-only model's leave-one-clone-out Spearman came out at exactly
-1.0 (the classic small-n/high-dimensionality noise-fitting signature)."""
from __future__ import annotations

import copy

import pandas as pd

from .preprocess import reduce_numeric_features


class FeatureSelector:
    def __init__(self, cfg_raw: dict, base_cap: int = 25, floor: int = 3, dynamic: bool = True):
        self.cfg_raw = cfg_raw
        self.base_cap = base_cap
        self.floor = floor
        self.dynamic = dynamic

    def cap_for(self, n_train: int) -> int:
        if not self.dynamic:
            return self.base_cap
        return max(self.floor, min(self.base_cap, n_train // 4))

    def select(self, train_df: pd.DataFrame, sensor_cols: list[str]) -> list[str]:
        cap = self.cap_for(len(train_df))
        cfg2 = copy.deepcopy(self.cfg_raw)
        cfg2["preprocess"]["max_numeric_features"] = cap
        return reduce_numeric_features(train_df, sensor_cols, cfg2)
