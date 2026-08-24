#!/usr/bin/env python3
"""Generate a synthetic bioprocess dataset to test the drop-and-run flow.

Ported verbatim from the owner's earlier engine (voyager-brain-rebuild,
samples/make_synthetic.py) into kalos as a non-proprietary benchmark fixture.
Every number here is generated from a fixed seed, not measured - there is no
client or proprietary data anywhere in this file or the CSV it writes, so
both are safe to commit.

Known structure (so the engine has real, recoverable signal):
- continuous process inputs (pH, temp, feeds, DO, ...) sampled by Latin Hypercube;
- a smooth titer response with a true optimum (Gaussian bumps + saturation);
- a feasibility gate: out-of-band conditions kill the culture -> titer = 0
  (~30% non-producers), so the feasibility model has negatives to learn;
- plus an id column and a categorical, to check the brain skips non-features.

Seeded -> reproducible. This is a TEST fixture for the pipeline mechanics, not
training data for a real model (synthetic outcomes must never stand in for
measured ones).

Run:  python examples/synthetic_bioprocess/make_synthetic.py
      -> examples/synthetic_bioprocess/synthetic_bioprocess.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

N = 160
SPECS = {  # input -> (low, high)
    "pH": (6.0, 7.6),
    "temperature_C": (30.0, 39.0),
    "glucose_gL": (2.0, 12.0),
    "glutamine_mM": (1.0, 8.0),
    "feed_rate_mLh": (0.1, 2.0),
    "DO_pct": (20.0, 80.0),
    "agitation_rpm": (100.0, 400.0),
    "inoculum_1e6": (0.2, 2.0),
    "antifoam_mgL": (0.0, 50.0),
    "osmolality_mOsm": (280.0, 360.0),
}


def _bump(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def main() -> None:
    rng = np.random.default_rng(42)
    names = list(SPECS)
    lows = np.array([SPECS[k][0] for k in names])
    highs = np.array([SPECS[k][1] for k in names])
    X = qmc.scale(qmc.LatinHypercube(d=len(names), seed=7).random(N), lows, highs)
    df = pd.DataFrame(X, columns=names)

    # smooth titer response with a true optimum (~pH 7.0, 36.5C, 52% DO, high glucose)
    titer = (
        3.2
        * _bump(df.pH, 7.0, 0.35)
        * _bump(df.temperature_C, 36.5, 1.6)
        * _bump(df.DO_pct, 52.0, 14.0)
        * (df.glucose_gL / (df.glucose_gL + 3.0))
        * (0.6 + 0.4 * _bump(df.glutamine_mM, 5.0, 2.0))
        * (0.7 + 0.3 * _bump(df.feed_rate_mLh, 1.2, 0.5))
    ).to_numpy()

    # feasibility gate -> non-producers (zeros, ~30%)
    viable = (
        df.pH.between(6.25, 7.45)
        & (df.temperature_C < 38.4)
        & (df.osmolality_mOsm < 352)
        & (df.DO_pct > 24)
    ).to_numpy()
    titer = np.where(viable, titer * rng.normal(1.0, 0.12, N), 0.0)
    titer = np.clip(titer, 0.0, None)

    df.insert(0, "run_id", [f"R{i + 1:03d}" for i in range(N)])     # id (should be ignored)
    df["medium_base"] = rng.choice(["CD-CHO", "BalanCD", "ProCHO"], N)  # categorical (ignored)
    df["titer_g_L"] = np.round(titer, 3)                            # <- the target to pick
    df = df.round(3)

    out = Path(__file__).parent / "synthetic_bioprocess.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows, {(df.titer_g_L == 0).mean():.0%} non-producers, "
          f"best titer {df.titer_g_L.max():.3f})")


if __name__ == "__main__":
    main()
