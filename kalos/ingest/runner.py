"""Decoupled closed loop — propose the next batch from the ingested feed.

No synchronous lab call. One step: read the feed (anonymized) -> fit the BoTorch
surrogate -> propose the next batch within bounds -> write it to the sink. The
experiments run outside this process; "repeat" is calling step() again once new
rows have landed. State lives in the feed, so the loop is async and crash-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from kalos.core.optimize import propose
from kalos.core.surrogate import Surrogate

from .feed import DataFeed, ProposalSink


@dataclass
class Problem:
    """What to optimize: the target column, the input names, and their bounds."""
    target: str
    feature_names: List[str]
    bounds: Dict[str, Tuple[float, float]]


def propose_next_batch(feed: DataFeed, problem: Problem, batch: int = 3, seed: int = 0) -> pd.DataFrame:
    """Fit on the current feed and return the top `batch` recipes by qLogEI."""
    df = feed.read()
    if df.empty or problem.target not in df.columns:
        raise ValueError(f"feed has no usable rows for target {problem.target!r}")
    feats = [f for f in problem.feature_names if f in df.columns and f in problem.bounds]
    if not feats:
        raise ValueError("no declared features survived anonymization / bounds")

    y = pd.to_numeric(df[problem.target], errors="coerce")
    keep = y.notna()
    X = df.loc[keep, feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    y = y[keep].to_numpy(float)
    bounds = np.array([[problem.bounds[f][0] for f in feats], [problem.bounds[f][1] for f in feats]], float)

    s = Surrogate().fit(X, y, bounds=bounds)
    batch_arr = propose(s, bounds, q=batch)
    return pd.DataFrame(batch_arr, columns=feats)


class IngestionLoop:
    """Thin orchestrator over a DataFeed + ProposalSink. State lives in the feed."""

    def __init__(self, feed: DataFeed, problem: Problem, sink: ProposalSink, batch: int = 3):
        self.feed = feed
        self.problem = problem
        self.sink = sink
        self.batch = batch

    def step(self, round_id: str, seed: int = 0) -> Tuple[pd.DataFrame, Path]:
        proposals = propose_next_batch(self.feed, self.problem, self.batch, seed)
        return proposals, self.sink.emit(proposals, round_id)


__all__ = ["Problem", "propose_next_batch", "IngestionLoop"]
