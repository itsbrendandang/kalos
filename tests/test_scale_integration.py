"""Slow integration test against the real `mab-scaleup-synthetic` dataset
(itsbrendandang/kalos-data, `datasets/mab-scaleup-synthetic/`).

Guarded by the `KALOS_DATA` env var so the suite stays green on any machine
without that private data repo cloned: `KALOS_DATA` must point at the ROOT of
a `kalos-data` checkout (the directory containing
`datasets/mab-scaleup-synthetic/batches.csv`). No path is hardcoded here -
where the checkout actually lives is an environment detail, not something
committed code should know.

DERIVED SENSOR COLUMNS. `batches.csv` has no `agitation_rpm` /
`airflow_L_per_min` columns of its own - those live in each batch's sensor
time series (`sensors/RUN-####_sensors.csv`). This test derives a single
per-batch value for each as the TIME-AVERAGE over the full run (a simple,
reproducible choice - not a claim that it is the best possible summary; a
richer summary, e.g. a steady-state-window average, is a v1 question).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.scale.evaluation import leave_one_scale_out_report

_DATASET_REL = Path("datasets/mab-scaleup-synthetic")


def _dataset_dir() -> Path | None:
    root = os.environ.get("KALOS_DATA")
    if not root:
        return None
    candidate = Path(root) / _DATASET_REL
    return candidate if (candidate / "batches.csv").exists() else None


pytestmark = pytest.mark.skipif(
    _dataset_dir() is None,
    reason=(
        "KALOS_DATA env var not set (or dataset not found under it) - skipping "
        "the real-data integration test; set KALOS_DATA to the root of an "
        "itsbrendandang/kalos-data checkout to run it"
    ),
)

# Upstream culture SETPOINTS (the "recipe" an operator holds fixed across
# scale) - deliberately excludes every column DATA.md flags as an outcome,
# a calculated/circular column, or run bookkeeping.
_PROCESS_COLUMNS = ["ph_setpoint", "dissolved_oxygen_pct", "temperature_C"]
_TARGET_COLUMN = "titer_g_per_L"


def _load_real_dataset() -> pd.DataFrame:
    dataset_dir = _dataset_dir()
    assert dataset_dir is not None  # guarded by pytestmark above
    batches = pd.read_csv(dataset_dir / "batches.csv")

    agitation_means = []
    airflow_means = []
    for batch_id in batches["batch_id"]:
        sensors = pd.read_csv(dataset_dir / "sensors" / f"{batch_id}_sensors.csv")
        agitation_means.append(sensors["agitation_rpm"].mean())
        airflow_means.append(sensors["airflow_L_per_min"].mean())
    batches["agitation_rpm"] = agitation_means
    batches["airflow_L_per_min"] = airflow_means
    return batches


def test_real_dataset_leave_one_scale_out_report_runs_and_is_sane():
    df = _load_real_dataset()
    report = leave_one_scale_out_report(df, _TARGET_COLUMN, _PROCESS_COLUMNS)

    assert report["n_scales"] == df["scale_L"].nunique()
    assert report["n_rows_used"] + report["n_rows_dropped"] == len(df)
    assert len(report["per_scale"]) == report["n_scales"]

    # The dataset spans 0.01-2000 L, so the smallest and largest scale must
    # each be a pure extrapolation bucket, and there must be interpolation
    # buckets in between - if this ever fails it means the fixture/dataset
    # shape changed underneath the test, not a modeling result to tune around.
    assert report["by_direction"]["extrapolate_up"] is not None
    assert report["by_direction"]["extrapolate_down"] is not None
    assert report["by_direction"]["interpolate"] is not None

    assert np.isfinite(report["overall"]["mae"])
    assert np.isfinite(report["overall"]["naive_mean_mae"])
    # Deliberately NOT asserting the GP beats the naive baseline here - see
    # kalos/scale/evaluation.py's module docstring and this package's final
    # report for the measured (and honestly reported) comparison.
