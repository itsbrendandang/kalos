"""Real-dataset tests for the v1 report's two mandatory checks: the
rank-crossing check (is there a genuine, statistically detectable
recipe-by-scale interaction in the real data?) and whether any v1
candidate's extrapolate-up ranking beats v0 by more than noise at n=5.

Guarded by `KALOS_DATA` exactly like `tests/test_scale_integration.py` (see
that file's docstring for the env var contract) - skips on any machine
without the private `kalos-data` checkout.

WHY THESE ASSERTIONS ARE LOOSE/QUALITATIVE RATHER THAN PINNED TO TODAY'S
EXACT DIGITS. A GP fit is deterministic on ONE machine (see
`test_scale_v1_evaluation.py`'s determinism tests) but L-BFGS-based
hyperparameter optimization can differ in its last few significant figures
across BLAS/LAPACK builds - this repo's CHANGELOG already records loosening
a different pinned-number anchor for exactly this reason. So these tests
lock in the DIRECTIONAL, decision-relevant finding (not significant; no
candidate clears the noise floor) rather than exact MAE/Spearman digits -
the precise numbers measured on THIS run are in the v1 report, not
hardcoded here as a brittle regression anchor.
"""
from __future__ import annotations

import os
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f as f_dist
from scipy.stats import spearmanr

from kalos.scale.candidates import MultiFidelitySurrogate, PhysicsMeanSurrogate
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
        "the real-data v1 checks; set KALOS_DATA to the root of an "
        "itsbrendandang/kalos-data checkout to run them"
    ),
)

_PROCESS_COLUMNS = ["ph_setpoint", "dissolved_oxygen_pct", "temperature_C"]
_TARGET_COLUMN = "titer_g_per_L"


def _load_real_dataset() -> pd.DataFrame:
    dataset_dir = _dataset_dir()
    assert dataset_dir is not None  # guarded by pytestmark above
    batches = pd.read_csv(dataset_dir / "batches.csv")
    agitation_means, airflow_means = [], []
    for batch_id in batches["batch_id"]:
        sensors = pd.read_csv(dataset_dir / "sensors" / f"{batch_id}_sensors.csv")
        agitation_means.append(sensors["agitation_rpm"].mean())
        airflow_means.append(sensors["airflow_L_per_min"].mean())
    batches["agitation_rpm"] = agitation_means
    batches["airflow_L_per_min"] = airflow_means
    return batches


# ---------------------------------------------------------------------------
# Mandatory rank-crossing check (real data), two independent angles:
#   1. a pooled OLS F-test for feature-by-log(scale) interaction across all
#      rows (more power than any one held-out bucket, n=55).
#   2. the EXACT permutation null distribution of Spearman's rho at n=5 (the
#      extrapolate-up bucket's actual size), checking whether v0's own
#      headline -0.7 is even distinguishable from pure chance.
# ---------------------------------------------------------------------------


def _interaction_f_test(df: pd.DataFrame) -> tuple[float, float]:
    """F-test for whether adding feature x log10(scale) interaction terms to
    an OLS of titer on [log_scale, features] reduces residual sum of squares
    more than chance. Returns (F, p). Plain `numpy.linalg.lstsq` rather than
    `statsmodels` (not a project dependency) - this is a standard nested-
    model F-test, no library-specific machinery needed."""
    y = df[_TARGET_COLUMN].to_numpy(dtype=float)
    log_scale = np.log10(df["scale_L"].to_numpy(dtype=float))
    centered = {f: df[f].to_numpy(dtype=float) - df[f].mean() for f in _PROCESS_COLUMNS}

    def rss(cols: list[np.ndarray]) -> tuple[float, int]:
        X = np.column_stack([np.ones(len(y))] + cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        return float(np.sum(resid**2)), X.shape[1]

    reduced_cols = [log_scale] + [centered[f] for f in _PROCESS_COLUMNS]
    rss_r, p_r = rss(reduced_cols)
    full_cols = reduced_cols + [centered[f] * log_scale for f in _PROCESS_COLUMNS]
    rss_f, p_f = rss(full_cols)

    n = len(y)
    df1, df2 = p_f - p_r, n - p_f
    f_stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
    p_value = float(1.0 - f_dist.cdf(f_stat, df1, df2))
    return float(f_stat), p_value


def test_real_data_no_significant_feature_by_scale_interaction():
    """Pooled across all 55 rows (more power than any single n=5 bucket),
    adding process-feature x log(scale) interaction terms to a titer OLS
    does NOT significantly reduce residual error. This is the rank-crossing
    check the v1 report leads with: the real dataset shows no statistically
    detectable evidence that the best recipe changes with scale."""
    df = _load_real_dataset()
    f_stat, p_value = _interaction_f_test(df)
    assert np.isfinite(f_stat)
    assert p_value > 0.05, (
        f"unexpected: found a significant feature-by-scale interaction (F={f_stat:.3f}, "
        f"p={p_value:.4f}) - the v1 report's rank-crossing finding assumed there was none"
    )


def _exact_spearman_null_n5() -> np.ndarray:
    """Every one of the 5! = 120 possible rank permutations' Spearman rho
    against a fixed reference ranking - the EXACT (not asymptotic) null
    distribution of rho at n=5, used because the asymptotic normal
    approximation `scipy.stats.spearmanr`'s p-value relies on is not
    trustworthy at n=5."""
    base = np.arange(5)
    return np.array([spearmanr(base, p).statistic for p in permutations(range(5))])


def _exact_two_sided_p(rho_obs: float, null: np.ndarray) -> float:
    return float(np.mean(np.abs(null) >= abs(rho_obs) - 1e-9))


def test_real_data_extrapolate_up_spearman_not_distinguishable_from_null():
    """v0's own headline extrapolate-up number (train <=1000 L, predict
    2000 L) is measured on n=5 - too few points for a rank correlation to be
    a measurement. This locks that in with an exact (not asymptotic)
    permutation p-value, independent of which candidate is being judged."""
    df = _load_real_dataset()
    report = leave_one_scale_out_report(df, _TARGET_COLUMN, _PROCESS_COLUMNS)
    bucket = report["by_direction"]["extrapolate_up"]
    assert bucket["n"] == 5
    null = _exact_spearman_null_n5()
    p_value = _exact_two_sided_p(bucket["spearman"], null)
    assert p_value > 0.05, (
        f"unexpected: v0's extrapolate-up spearman={bucket['spearman']} was significant "
        f"(exact p={p_value:.4f}) at n=5 - the v1 report assumed it was noise"
    )


# ---------------------------------------------------------------------------
# The three-way comparison itself: no candidate's extrapolate-up ranking is
# a measured, non-noise win over v0. See `kalos/scale/transfer.py`'s "V1
# CANDIDATES" docstring section for the decision this backs.
# ---------------------------------------------------------------------------


def test_no_candidate_extrapolate_up_spearman_clears_the_noise_floor():
    df = _load_real_dataset()
    null = _exact_spearman_null_n5()
    factories = {
        "v0": None,
        "physics_mean": lambda names: PhysicsMeanSurrogate(names),
        "multi_fidelity": lambda names: MultiFidelitySurrogate(names),
    }
    for label, factory in factories.items():
        report = leave_one_scale_out_report(
            df, _TARGET_COLUMN, _PROCESS_COLUMNS, model_factory=factory, model_label=label
        )
        bucket = report["by_direction"]["extrapolate_up"]
        assert bucket["n"] == 5, label
        p_value = _exact_two_sided_p(bucket["spearman"], null)
        assert p_value > 0.05, (
            f"{label}: extrapolate-up spearman={bucket['spearman']} was significant "
            f"(exact p={p_value:.4f}) at n=5 - would contradict the v1 report's verdict "
            "that no candidate clears the noise floor on the real dataset"
        )
