#!/usr/bin/env python3
"""Reproduce the real-data pool retrospective from docs/BENCHMARK.md.

The honest question: on the real media DoE, does the optimizer reach the best
recipe in fewer experiments than random selection? docs/BENCHMARK.md's resolved
answer is "yes, on the objective that matters" - the replicate-averaged
(reproducible) titer, with the measured assay noise floor fed to the surrogate -
but "no" on single measurements, which reward lucky noise spikes.

This script makes those numbers reproducible from committed code. It:
  1. loads the proprietary media sheet from KALOS_MEDIA_DATA (nothing vendored),
  2. builds the reproducible per-recipe objective via `pool_from_frame(aggregate=True)`,
  3. estimates the assay noise floor from the raw replicates,
  4. races BO / feasibility-gated BO / random over many seeds with that floor fed in,
  5. prints the best-found trajectory and the normalized speed (area under the
     best-found curve) for each strategy.

Also prints the single-measurement (un-aggregated) race for contrast - the
metric under which BO "loses", which is the artifact, not the target.

Run:  KALOS_MEDIA_DATA=/path/to/combined.tsv python examples/benchmark_media_pool.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kalos.bench.pool import pool_from_frame, run_pool  # noqa: E402
from kalos.core.replicates import estimate_noise_floor, noise_report  # noqa: E402

TARGET = "Lipase_g.L"
STRATEGIES = ("bo", "bo_feas_clean", "random")
SEEDS = range(40)
N_INIT = 8
CHECKPOINTS = (5, 8, 11, 14, 22)


def _load(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)
    if "Sample Name" in df:  # drop complex-medium controls, as the analysis does
        df = df[~df["Sample Name"].astype(str).str.startswith("BMMY")]
    return df.reset_index(drop=True)


def _speed(mean_curve: np.ndarray, pool_max: float) -> float:
    """Normalized area under the best-found curve: the mean of best-found/pool_max
    across picks, in [0, 1]. Higher = reaches the pool's best recipe sooner."""
    if pool_max <= 0:
        return float("nan")
    return float(np.mean(mean_curve / pool_max))


def _race(X: np.ndarray, y: np.ndarray, *, noise: float | None, label: str) -> None:
    budget = len(y) - N_INIT
    res = run_pool(X, y, strategies=STRATEGIES, n_init=N_INIT, budget=budget, seeds=SEEDS, noise=noise)
    pool_max = res["_meta"]["pool_max"]
    steps = res["_meta"]["steps"]
    print(f"\n{label}  (pool: {res['_meta']['n']} recipes, max {pool_max:.4f}, "
          f"{len(list(SEEDS))} seeds)")
    header = "  picks  " + "  ".join(f"{s:>14}" for s in STRATEGIES)
    print(header)
    for c in CHECKPOINTS:
        if c >= steps:
            continue
        row = "  ".join(f"{res[s]['mean'][c]:14.4f}" for s in STRATEGIES)
        print(f"  {c:5d}  {row}")
    speeds = {s: _speed(res[s]["mean"], pool_max) for s in STRATEGIES}
    print("  speed (norm. AUC): " + "  ".join(f"{s}={speeds[s]:.3f}" for s in STRATEGIES))
    best = max(speeds, key=lambda s: speeds[s])
    print(f"  -> fastest: {best}")


def main() -> int:
    raw = os.environ.get("KALOS_MEDIA_DATA", "").strip()
    path = Path(raw).expanduser() if raw else None
    if path is None or not path.is_file():
        print("set KALOS_MEDIA_DATA to a merged media+titer TSV (no client data is committed).")
        return 1
    df = _load(path)

    # raw (one row per assay read) and reproducible (one row per recipe) views
    X_raw, y_raw, _ = pool_from_frame(df, TARGET, aggregate=False)
    X_rep, y_rep, _ = pool_from_frame(df, TARGET, aggregate=True)

    rep = noise_report(X_raw, y_raw)
    floor = estimate_noise_floor(X_raw, y_raw)
    print("=" * 78)
    print(f"KALOS real-data pool retrospective  ({path.name})")
    print(f"  {rep['n_rows']} assay reads -> {rep['n_recipes']} distinct recipes "
          f"({rep['n_replicated']} replicated)")
    print(f"  ICC {rep['icc']:.2f}  | assay-noise sd {np.sqrt(rep['noise_var']):.4f} "
          f"| between-recipe signal sd {np.sqrt(rep['signal_var']):.4f}")
    print(f"  best single measurement {y_raw.max():.4f}  vs  best reproducible recipe {y_rep.max():.4f}")
    print("=" * 78)

    # The objective that matters: reproducible titer + measured noise floor fed in.
    _race(X_rep, y_rep, noise=floor, label="REPRODUCIBLE objective (replicate-averaged, noise floor fed)")
    # The artifact: single measurements reward noise spikes, so undirected search "wins".
    _race(X_raw, y_raw, noise=None, label="SINGLE-MEASUREMENT objective (the artifact, for contrast)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
