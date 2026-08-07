"""Data access layer.

Training data is a *folder* of featurized CSVs (one row per well, a `titer`
target column, everything else numeric treated as a feature). Files may have
different feature columns -- they are unioned by name and the gaps are imputed
downstream, so heterogeneous "shapes" pool cleanly. The prediction cohort is a
single CSV with the same feature columns and no titer.

The modeling contract is unchanged: a TrainingPool exposes y() / groups() /
n_rows / n_unique_groups and a raw-titer TARGET.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Every numeric column that isn't a designated meta column. String columns
    (ids, group labels, source names) are naturally excluded by the numeric test."""
    cols = cfg["columns"]
    meta = {cols["target"], cols.get("id_col"), cols.get("group_col"),
            TrainingPool.TARGET, TrainingPool.GROUP, "well_id"}
    meta |= set(cfg.get("features", {}).get("exclude") or [])
    return [c for c in df.columns
            if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


def load_predict_cohort(cfg: dict) -> pd.DataFrame:
    """The single prediction CSV. The id column is renamed to 'well_id' so the
    ported evaluation/blend code (which references 'well_id') is unchanged."""
    cols = cfg["columns"]
    df = pd.read_csv(cfg["data"]["predict_csv"])
    id_col = cols.get("id_col")
    if id_col and id_col in df.columns:
        df = df.rename(columns={id_col: "well_id"})
    elif "well_id" not in df.columns:
        df["well_id"] = [f"row{i}" for i in range(len(df))]
    # well_id is a row key: downstream scoring/blend/propose index by it, and a
    # duplicate silently misaligns per-well predictions and the diversity batch.
    dupes = df["well_id"][df["well_id"].duplicated()].unique()
    if len(dupes):
        raise ValueError(f"prediction cohort has duplicate well_id(s): {list(dupes)[:5]}")
    return df.reset_index(drop=True)


class TrainingPool:
    """A labeled training set: features + a raw-titer TARGET + a group key,
    pooled from every CSV in the training directory."""

    TARGET = "target"
    GROUP = "_group"

    def __init__(self, df: pd.DataFrame, name: str):
        self.df = df.reset_index(drop=True)
        self.name = name

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_unique_groups(self) -> int:
        return int(self.df[self.GROUP].nunique())

    def y(self) -> np.ndarray:
        return self.df[self.TARGET].to_numpy(dtype=float)

    def groups(self) -> np.ndarray:
        return self.df[self.GROUP].to_numpy()

    def is_trainable(self, min_unique_groups: int = 2) -> bool:
        """Fewer than 2 distinct labeled groups means zero inter-group variation
        to learn from -- a data problem, not something any model can fix."""
        return self.n_unique_groups >= min_unique_groups

    @classmethod
    def from_dir(cls, train_dir: str | Path, cfg: dict, name: str = "pooled") -> "TrainingPool":
        cols = cfg["columns"]
        target = cols["target"]
        group_col = cols.get("group_col")
        id_col = cols.get("id_col")
        train_dir = Path(train_dir)

        files = sorted(train_dir.glob("*.csv"))
        if not files:
            raise ValueError(f"no *.csv training files found in {train_dir}")

        parts = []
        for path in files:
            df = pd.read_csv(path)
            if target not in df.columns:
                continue  # not a training file (no titer)
            df = df.dropna(subset=[target]).copy()
            if df.empty:
                continue
            df[cls.TARGET] = df[target].astype(float)
            # Group key for leak-free CV; namespaced by file so identically-named
            # groups in different files are never merged. Falls back to the id
            # column, then to a per-row id (each well its own group).
            if group_col and group_col in df.columns:
                grp = df[group_col].astype(str)
            elif id_col and id_col in df.columns:
                grp = df[id_col].astype(str)
            else:
                grp = pd.Series(np.arange(len(df)).astype(str), index=df.index)
            df[cls.GROUP] = path.stem + "::" + grp.to_numpy()
            parts.append(df)

        if not parts:
            raise ValueError(
                f"no training rows with a non-null '{target}' column in {train_dir}")
        combined = pd.concat(parts, ignore_index=True, sort=False)

        # Optional: restrict training to rows from specific sources (e.g. the same
        # measurement/duration class as the cohort being ranked -- MP passage titer
        # and SF fed-batch titer are the same units at very different scales/durations,
        # and pooling both would let a model just learn "which source", not signal).
        include_sources = cfg.get("data", {}).get("include_sources")
        if include_sources:
            source_col = cols.get("source_col", "source")
            if source_col not in combined.columns:
                raise ValueError(f"data.include_sources is set but source_col {source_col!r} "
                                 f"is not a column in the training data")
            combined = combined.loc[combined[source_col].astype(str).isin(
                [str(s) for s in include_sources])].reset_index(drop=True)
            if combined.empty:
                raise ValueError(f"data.include_sources={include_sources!r} matched no rows "
                                 f"via source_col {source_col!r}")

        return cls(combined, name)
