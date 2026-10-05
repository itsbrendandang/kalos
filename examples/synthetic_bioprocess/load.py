#!/usr/bin/env python3
"""One-call loader for the synthetic bioprocess fixture.

`kalos/bench` otherwise has only abstract math surfaces (`objectives.py`) -
its pool benchmark (`kalos.bench.pool.pool_from_frame`) needs a real run
sheet, and the real pool benchmarks all require client data that cannot be
committed. This fixture is the non-proprietary stand-in: 160 seeded rows of a
synthetic bioprocess with a known optimum, so `kalos.bench.pool` has
something to run in CI and in a fresh clone with zero setup.

`load_frame()` is a plain `pd.read_csv`. `load_pool()` is the one call this
module exists for: it hands `pool_from_frame` the frame and the known target
column name, returning `(X, y, feature_names)` ready for
`kalos.bench.pool.run_pool_one` (or any other pool-benchmark entry point)
with no further wiring.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kalos.bench.pool import pool_from_frame  # noqa: E402

CSV_PATH = Path(__file__).resolve().parent / "synthetic_bioprocess.csv"
TARGET = "titer_g_L"


def load_frame() -> pd.DataFrame:
    """Return the fixture exactly as shipped on disk (id + categorical included)."""
    return pd.read_csv(CSV_PATH)


def load_pool() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """`(X, y, feature_names)`: the fixture ready for a pool benchmark.

    `pool_from_frame` already drops `run_id` and `medium_base` on its own -
    both coerce to all-NaN under `pd.to_numeric` (the id is text like "R001",
    the categorical is a medium label), so they fail its own varying-numeric
    feature test without this module needing to name them.
    """
    return pool_from_frame(load_frame(), TARGET)


if __name__ == "__main__":
    X, y, feats = load_pool()
    print(f"{X.shape[0]} rows x {len(feats)} feature(s) -> {TARGET}")
    print(f"features: {feats}")
    print(f"non-producers: {(y <= 0).mean():.0%}  best titer: {y.max():.3f}")
