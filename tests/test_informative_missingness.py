"""Tests for `check_informative_missingness` (MNAR: missingness vs. the target).

`check_missingness` already reports HOW blank a column or row is. This check
asks a different question: whether blankness itself correlates with the
outcome, the failure mode documented on the owner's real Cytena clone-funnel
data (voyager-brain-rebuild/docs/ml-review.md, finding P0.1) where per-row
missing-fraction vs. titer had Spearman rho = -0.835 - "no measurement" was
the human culling decision, not benign absence, and zero-filling before a GP
fit would have quietly encoded that decision into the features.

Every fixture here is hand-built (no RNG) so the exact rho and n are known in
advance and the tests are fully deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kalos.domains import BIOPROCESS_PROFILE
from kalos.validation.checks import check_informative_missingness
from kalos.validation.runner import validate_frame


def _mnar_frame() -> pd.DataFrame:
    """24 rows, blanks concentrated on the low-target half (rows 0-11 = titer 0-11).

    9 of the 12 low-target rows are blank in `feat`, only 1 of the 12
    high-target rows (titer 20-31) is - the same shape as the real MNAR
    signal: low performers go unmeasured far more often than high performers.
    """
    n = 24
    titer = list(range(0, 12)) + list(range(20, 32))
    feat = list(np.arange(n, dtype=float))
    other = list(np.arange(n, dtype=float) * 2)
    df = pd.DataFrame({"feat": feat, "other": other, "titer": titer})
    df.loc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 12], "feat"] = np.nan
    return df


def _benign_frame() -> pd.DataFrame:
    """Same shape and same TOTAL blank count as `_mnar_frame`, but every third
    row is blank regardless of target rank - missingness uncorrelated with
    the outcome."""
    n = 24
    titer = list(range(0, 12)) + list(range(20, 32))
    feat = list(np.arange(n, dtype=float))
    other = list(np.arange(n, dtype=float) * 2)
    df = pd.DataFrame({"feat": feat, "other": other, "titer": titer})
    df.loc[[0, 3, 6, 9, 12, 15, 18, 21], "feat"] = np.nan
    return df


# --- the core MNAR-vs-benign distinction ------------------------------------- #


def test_row_level_mnar_missingness_triggers_warning():
    findings = check_informative_missingness(_mnar_frame(), "titer")
    row_level = [f for f in findings if f.column is None]
    assert len(row_level) == 1
    f = row_level[0]
    assert f.severity == "warning"
    assert f.check == "informative_missingness"
    assert f.detail["rho"] < -0.35  # confidently negative, well past the warn threshold
    assert f.detail["n"] == 24


def test_column_level_mnar_missingness_names_the_sparse_column():
    findings = check_informative_missingness(_mnar_frame(), "titer")
    col_level = [f for f in findings if f.column == "feat"]
    assert len(col_level) == 1
    assert col_level[0].severity == "warning"
    assert col_level[0].detail["rho"] < -0.35
    # "other" was never touched - it has no blanks at all, so it cannot fire.
    assert not any(f.column == "other" for f in findings)


def test_benign_missingness_does_not_trigger():
    """Same blank count, uncorrelated with the target: no finding at all."""
    findings = check_informative_missingness(_benign_frame(), "titer")
    assert findings == []


def test_strict_mode_does_not_reject_a_sheet_with_only_this_warning():
    """This check must never be able to flip an upload from accepted to
    rejected - that is a product decision it has no business making."""
    df = _mnar_frame()
    df["batch"] = [f"B{i}" for i in range(len(df))]
    df["date"] = [f"2026-01-{(i % 28) + 1:02d}" for i in range(len(df))]
    df["operator"] = ["opA"] * len(df)
    df["condition"] = ["control"] + ["test"] * (len(df) - 1)
    report = validate_frame(
        df, target="titer", features=["feat", "other"], profile=BIOPROCESS_PROFILE, mode="strict"
    )
    assert report.status != "fail"
    assert any(f.check == "informative_missingness" for f in report.findings)
    assert all(
        f.severity != "error" for f in report.findings if f.check == "informative_missingness"
    )


# --- guard rails: never crash, never raise severity -------------------------- #


def test_no_target_produces_no_findings():
    df = pd.DataFrame({"feat": [1.0, np.nan] * 10})
    assert check_informative_missingness(df, None) == []


def test_target_not_in_frame_produces_no_findings():
    df = pd.DataFrame({"feat": [1.0, np.nan] * 10})
    assert check_informative_missingness(df, "titer") == []


def test_non_numeric_target_produces_no_findings():
    df = pd.DataFrame({"feat": [1.0, np.nan] * 10, "titer": ["x"] * 20})
    assert check_informative_missingness(df, "titer") == []


def test_constant_target_produces_no_findings():
    """rho is undefined against a target with zero variance - nothing to correlate."""
    df = pd.DataFrame({"feat": [1.0, np.nan] * 10, "titer": [5.0] * 20})
    assert check_informative_missingness(df, "titer") == []


def test_no_blanks_produces_no_findings():
    df = pd.DataFrame({"feat": list(range(20)), "titer": list(range(20))})
    assert check_informative_missingness(df, "titer") == []


def test_below_small_n_guard_produces_no_findings():
    """Fewer than 12 valid-target rows: even a perfect-looking correlation is
    within noise, so the check stays silent rather than report a confident
    number built on 8 points."""
    df = pd.DataFrame(
        {
            "feat": [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0],
            "titer": [0, 0, 0, 0, 10, 10, 10, 10],
        }
    )
    assert check_informative_missingness(df, "titer") == []
