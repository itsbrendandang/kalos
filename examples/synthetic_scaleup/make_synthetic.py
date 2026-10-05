#!/usr/bin/env python3
"""Generate a synthetic multi-scale bioprocess dataset for the Scale-Up
Readout.

Promoted verbatim from `fabricate_harder_synthetic` in
`tests/test_scale_v1_harder_synthetic.py` (this repo's own "harder than the
real dataset" fixture) so the demo dataset and the test fixture that proved
the v1 candidates can learn a rank crossing are the SAME code, not two copies
that can drift apart. See that test module's docstring for the full backstory
on why this dataset exists and how it was used in the v1 promotion decision.

FULLY SYNTHETIC. Every number here comes from a fixed seed, not a measurement
- there is no client or proprietary data anywhere in this file or the CSV it
writes, so both are safe to commit. It is a SIMULATION of a scale effect
(a planted recipe x scale interaction), not real process data: do not treat
any number in it as a physical measurement.

Known structure (so the readout has something real to find):
- nine scales spanning 1 L to 5000 L, ten runs per scale;
- `ph_setpoint` and `temperature_C` as the process (recipe) inputs;
- `agitation_rpm` and `airflow_L_per_min` as physics inputs (these are the
  exact column names `kalos.scale.transfer.ScaleFeatureConfig` defaults to,
  so the fixture plugs into the scale-up transfer model with zero renaming);
- a titer response whose optimal pH shifts linearly (in log10-scale-
  normalized units) from 6.6 at 1 L to 7.4 at 5000 L - a swing wide enough,
  against a ph_setpoint sampling range of [6.4, 7.6], to flip which end of
  the range wins between the smallest and largest scale.

Seeded -> reproducible.

Run:  python examples/synthetic_scaleup/make_synthetic.py
      -> examples/synthetic_scaleup/synthetic_scaleup.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260825
SCALES: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 5000.0)
N_PER_SCALE = 10
PROCESS_COLUMNS = ["ph_setpoint", "temperature_C"]
TARGET_COLUMN = "titer_g_per_L"

# The optimal pH shifts linearly (in log10-scale-normalized units) from 6.6
# at the smallest scale to 7.4 at the largest - a swing wide enough, against
# a ph_setpoint sampling range of [6.4, 7.6], to flip which end of the range
# wins between the two extremes.
_OPTIMAL_PH_AT_SMALLEST_SCALE = 6.6
_OPTIMAL_PH_AT_LARGEST_SCALE = 7.4


def make_synthetic_scaleup(
    scales: tuple[float, ...] = SCALES, n_per_scale: int = N_PER_SCALE, seed: int = SEED
) -> pd.DataFrame:
    """Fabricate the dataset described in this module's docstring.
    Deterministic for a fixed `(scales, n_per_scale, seed)`."""
    rng = np.random.default_rng(seed)
    log_scales = np.log10(np.asarray(scales, dtype=float))
    lmin, lmax = log_scales.min(), log_scales.max()

    rows = []
    for s, log_s in zip(scales, log_scales):
        l_norm = (log_s - lmin) / (lmax - lmin)
        optimal_ph = _OPTIMAL_PH_AT_SMALLEST_SCALE + (
            _OPTIMAL_PH_AT_LARGEST_SCALE - _OPTIMAL_PH_AT_SMALLEST_SCALE
        ) * l_norm
        for _ in range(n_per_scale):
            ph = float(rng.uniform(6.4, 7.6))
            temp = float(rng.uniform(36.0, 38.0))  # nuisance feature, no interaction
            rpm = float(rng.uniform(80, 250))
            airflow = float(max(s * 0.05, 0.05) * rng.uniform(0.8, 1.2))
            # Genuine recipe x scale interaction: titer peaks near the
            # scale-dependent optimal pH (a quadratic penalty for being off
            # it), plus a mild overall scale trend and observation noise.
            titer = 6.0 - 4.0 * (ph - optimal_ph) ** 2 + 0.3 * l_norm + float(rng.normal(0, 0.15))
            rows.append(
                {
                    "scale_L": s,
                    "agitation_rpm": rpm,
                    "airflow_L_per_min": airflow,
                    "ph_setpoint": ph,
                    "temperature_C": temp,
                    "titer_g_per_L": titer,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = make_synthetic_scaleup()
    out = Path(__file__).parent / "synthetic_scaleup.csv"
    df.to_csv(out, index=False)
    print(
        f"wrote {out}  ({len(df)} rows, {df['scale_L'].nunique()} scales, "
        f"{TARGET_COLUMN} range [{df[TARGET_COLUMN].min():.3f}, {df[TARGET_COLUMN].max():.3f}])"
    )


if __name__ == "__main__":
    main()
