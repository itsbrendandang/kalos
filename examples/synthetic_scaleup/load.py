#!/usr/bin/env python3
"""One-call loader for the synthetic scale-up fixture.

The Scale-Up Readout needs a multi-scale run sheet with a planted, learnable
rank crossing to demo against, and the real `mab-scaleup-synthetic` dataset
cannot be committed (client data) and does not have a strong enough
interaction to make the demo's point on its own (see
`tests/test_scale_v1_harder_synthetic.py`'s module docstring). This fixture
is the non-proprietary stand-in: 90 seeded rows across 9 scales, ready to
hand straight to `kalos.scale.transfer.ScaleUpTransferModel` or
`kalos.scale.evaluation.leave_one_scale_out_report` with no further wiring.

`load_frame()` is a plain `pd.read_csv`. `load_features()` builds the
process + physics-proxy feature matrix via
`kalos.scale.transfer.build_feature_matrix` under the default
`ScaleFeatureConfig` (the fixture's column names - `scale_L`,
`agitation_rpm`, `airflow_L_per_min` - already match that config's
defaults, so no renaming is needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from kalos.scale.transfer import DEFAULT_SCALE_FEATURE_CONFIG, build_feature_matrix  # noqa: E402

CSV_PATH = Path(__file__).resolve().parent / "synthetic_scaleup.csv"
PROCESS_COLUMNS = ["ph_setpoint", "temperature_C"]
TARGET = "titer_g_per_L"


def load_frame() -> pd.DataFrame:
    """Return the fixture exactly as shipped on disk."""
    return pd.read_csv(CSV_PATH)


def load_features() -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """`(X, y, feature_names)`: the fixture's process + physics-proxy
    feature matrix and target, ready for `ScaleUpTransferModel` or
    `leave_one_scale_out_report`."""
    df = load_frame()
    X, names = build_feature_matrix(df, PROCESS_COLUMNS, DEFAULT_SCALE_FEATURE_CONFIG)
    y = df[TARGET].to_numpy(dtype=float)
    return X, y, names


if __name__ == "__main__":
    df = load_frame()
    X, y, names = load_features()
    print(f"{len(df)} rows across {df['scale_L'].nunique()} scales -> {TARGET}")
    print(f"features: {names}")
    print(f"titer range: [{y.min():.3f}, {y.max():.3f}]")
