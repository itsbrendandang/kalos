"""Tests for `check_constant_within_group` (a feature that is really group identity).

`check_constant_columns` catches a column that never varies anywhere. This
check catches the sneakier variant from the owner's real Cytena clone-funnel
data (voyager-brain-rebuild/docs/ml-review.md, finding P0.2): a column that
varies across the whole sheet but has exactly one value inside every group,
so within any single group it carries zero information and a model fit on it
is really just learning "which group," not the thing the column claims to
measure. 4 of 6 candidate features there were campaign identity in disguise.

All fixtures are hand-built (no RNG); the group column is detected the same
way `kalos.portal.analysis` infers it - the first column name matching a
`DomainProfile.group_hint` regex.
"""
from __future__ import annotations

import pandas as pd

from kalos.domains import BIOPROCESS_PROFILE
from kalos.validation.checks import check_constant_within_group
from kalos.validation.runner import validate_frame


def _campaign_frame() -> pd.DataFrame:
    """3 campaigns of 3 rows each.

    `proxy_feat` is exactly one value per campaign (1, 2, 3) though it takes
    three distinct values overall - campaign identity wearing a numeric
    costume. `real_feat` varies within every campaign, a genuine per-row
    measurement. `titer` is the outcome, unused by this check directly.
    """
    return pd.DataFrame(
        {
            "campaign": ["C1", "C1", "C1", "C2", "C2", "C2", "C3", "C3", "C3"],
            "proxy_feat": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "real_feat": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "titer": [0.1, 0.2, 0.15, 0.5, 0.6, 0.55, 0.9, 0.95, 0.85],
        }
    )


# --- the core distinction ----------------------------------------------------- #


def test_group_proxy_feature_is_flagged():
    findings = check_constant_within_group(_campaign_frame(), group_hint=BIOPROCESS_PROFILE.group_hint)
    flagged = {f.column for f in findings}
    assert "proxy_feat" in flagged
    proxy_finding = next(f for f in findings if f.column == "proxy_feat")
    assert proxy_finding.severity == "warning"
    assert proxy_finding.check == "constant_within_group"
    assert proxy_finding.detail["group_column"] == "campaign"
    assert proxy_finding.detail["n_groups_checked"] == 3


def test_genuinely_varying_within_group_feature_is_not_flagged():
    findings = check_constant_within_group(_campaign_frame(), group_hint=BIOPROCESS_PROFILE.group_hint)
    assert not any(f.column == "real_feat" for f in findings)


def test_titer_itself_is_not_flagged_because_it_varies_within_every_campaign():
    findings = check_constant_within_group(_campaign_frame(), group_hint=BIOPROCESS_PROFILE.group_hint)
    assert not any(f.column == "titer" for f in findings)


# --- guard rails --------------------------------------------------------------- #


def test_no_group_column_is_silent():
    df = _campaign_frame().drop(columns=["campaign"])
    assert check_constant_within_group(df, group_hint=BIOPROCESS_PROFILE.group_hint) == []


def test_all_singleton_groups_is_silent():
    """Every group has exactly 1 row: within-group variation is unmeasurable,
    not evidence of anything, so nothing is flagged."""
    df = _campaign_frame().copy()
    df["campaign"] = [f"C{i}" for i in range(len(df))]
    assert check_constant_within_group(df, group_hint=BIOPROCESS_PROFILE.group_hint) == []


def test_blank_group_labels_are_excluded_not_treated_as_a_group():
    df = _campaign_frame().copy()
    df.loc[0, "campaign"] = None
    findings = check_constant_within_group(df, group_hint=BIOPROCESS_PROFILE.group_hint)
    # C1 still has 2 non-blank rows (1, 2), still measurable; proxy_feat still flagged
    assert any(f.column == "proxy_feat" for f in findings)


def test_globally_constant_column_is_not_double_flagged():
    """A column with zero variance sheet-wide is `check_constant_columns`'s
    finding, not this check's - it never even gets to the group comparison."""
    df = _campaign_frame().copy()
    df["always_same"] = 7
    findings = check_constant_within_group(df, group_hint=BIOPROCESS_PROFILE.group_hint)
    assert not any(f.column == "always_same" for f in findings)


def test_column_blank_within_a_group_is_not_treated_as_constant_there():
    """A column entirely blank inside one measurable group has zero non-blank
    values there (not one) - that is a missingness problem, not evidence the
    column is a group proxy, so it must not be flagged on this account."""
    df = _campaign_frame().copy()
    # blank out proxy_feat entirely within campaign C3
    df.loc[df["campaign"] == "C3", "proxy_feat"] = None
    findings = check_constant_within_group(df, group_hint=BIOPROCESS_PROFILE.group_hint)
    assert not any(f.column == "proxy_feat" for f in findings)


# --- integration: strict mode never rejects on this warning ------------------ #


def test_strict_mode_does_not_reject_a_sheet_with_only_this_warning():
    df = _campaign_frame().copy()
    df["date"] = [f"2026-01-{i + 1:02d}" for i in range(len(df))]
    df["operator"] = ["opA"] * len(df)
    df["condition"] = ["control"] + ["test"] * (len(df) - 1)
    report = validate_frame(
        df, target="titer", features=["real_feat", "proxy_feat"],
        profile=BIOPROCESS_PROFILE, mode="strict",
    )
    assert report.status != "fail"
    assert any(f.check == "constant_within_group" for f in report.findings)
    assert all(
        f.severity != "error" for f in report.findings if f.check == "constant_within_group"
    )
