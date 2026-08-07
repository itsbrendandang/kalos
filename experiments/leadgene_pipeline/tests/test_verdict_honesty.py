"""Regression tests for the verdict/honesty layer.

The blend weights, the confidence tier, and the analysis Interpretation must all
gate on the bootstrap-CI verdict (does the 95% CI on CV Spearman exclude 0?), not
on the raw CV Spearman point estimate. Gating on the point estimate lets a
NOT VALIDATED method (documented real case: CV Spearman +0.54 with the CI crossing
0) drive the blend and lets the analysis print "validated" a few lines below its own
"NOT VALIDATED" table -- the exact "trust the verdict, not the point estimate"
contract, violated.
"""
from __future__ import annotations

import numpy as np

from pipeline.analysis import write_analysis
from pipeline.blend import Blender
from pipeline.evaluation import MethodResult
from pipeline.pipeline import Report


def _result(name, cv_spearman, ci_low, ci_high, client_mean, collapsed=False):
    verdict = ("USABLE (CI excludes 0)" if ci_low > 0 else
               "NOT VALIDATED (bootstrap CI on Spearman includes 0)")
    client_mean = np.asarray(client_mean, dtype=float)
    wells = np.array([f"w{i}" for i in range(len(client_mean))])
    return MethodResult(
        name=name, n_train=10, n_unique_groups=5, n_features=3,
        cv_spearman=cv_spearman, cv_p=0.05, ci_low=ci_low, ci_high=ci_high,
        verdict=verdict, client_well_ids=wells, client_mean=client_mean,
        client_std=np.zeros(len(client_mean)), collapsed_on_client=collapsed,
    )


def test_not_validated_method_gets_zero_weight():
    """A strong point estimate whose bootstrap CI still includes 0 is NOT
    VALIDATED and must earn zero blend weight."""
    not_valid = _result("point_gb", 0.54, -0.10, 0.82, [1, 2, 3, 4])
    assert not_valid.validated is False
    assert not_valid.weight == 0.0

    validated = _result("gp", 0.60, 0.20, 0.85, [1, 2, 3, 4])
    assert validated.validated is True
    assert validated.weight == 0.60

    collapsed = _result("br", 0.60, 0.20, 0.85, [5, 5, 5, 5], collapsed=True)
    assert collapsed.weight == 0.0


def test_confidence_tier_ignores_unvalidated_agreement():
    """confidence_tier counts cross-model agreement only among methods that earned
    weight (validated, not collapsed); an unvalidated method agreeing on the top
    wells must not inflate confidence."""
    validated = _result("gp", 0.6, 0.2, 0.9, [1, 2, 3, 9])
    not_valid = _result("point_gb", 0.54, -0.1, 0.8, [1, 2, 3, 9])
    table = Blender([validated, not_valid]).blended_table()
    top_row = table.loc[table["well_id"] == "w3"].iloc[0]
    assert top_row["n_methods_agreeing_top10"] == 1  # the validated method only, not 2


def test_predictions_carry_a_blend_validated_flag():
    """The machine-readable predictions table must expose whether the blend used
    any validated weight, not hide it in a sidecar."""
    valid_a = _result("gp", 0.6, 0.2, 0.9, [1, 2, 3, 6])
    valid_b = _result("br", 0.6, 0.2, 0.9, [1, 2, 3, 7])
    report = _report([valid_a, valid_b], reference=None)
    assert bool(report.predictions_table()["blend_validated"].iloc[0]) is True

    not_valid = _result("point_gb", 0.54, -0.1, 0.8, [1, 2, 3, 6])
    report_none = _report([not_valid], reference=None)
    assert bool(report_none.predictions_table()["blend_validated"].iloc[0]) is False


def _report(results, reference):
    blender = Blender(results)
    return Report(results=results, blend_table=blender.blended_table(),
                  weights=blender.weights(), note=blender.note(),
                  cohort=None, reference=reference)


def test_collapse_detection_is_scale_invariant():
    """The collapse check must be titer-unit invariant: near-constant predictions
    on a raw-titer scale (~1600) count as collapsed even though their absolute
    spread dwarfs the old 1e-3 epsilon, while genuine differentiation on either a
    large (SF) or small (MP) scale does not."""
    from pipeline.pipeline import _collapsed_on_cohort

    assert _collapsed_on_cohort(np.array([1600.0, 1600.1, 1600.2, 1600.05])) is True
    assert _collapsed_on_cohort(np.array([1600.0, 1500.0, 1700.0, 1650.0])) is False
    assert _collapsed_on_cohort(np.array([40.0, 55.0, 67.0, 48.0])) is False


def test_analysis_does_not_claim_validated_when_reference_ci_includes_zero(tmp_path):
    """write_analysis must not print 'validated ranking signal' when the reference
    model's bootstrap CI includes 0, even when the blend differentiates the cohort
    (two other models validate, so the run is not degenerate)."""
    valid_a = _result("gp", 0.6, 0.2, 0.9, [1, 2, 3, 4, 5, 6])
    valid_b = _result("br", 0.6, 0.2, 0.9, [1, 2, 3, 4, 5, 7])
    ref_unvalidated = _result("point_gb", 0.54, -0.1, 0.8, [2, 1, 4, 8, 5, 7])
    reference = {
        "model_name": "point_gb", "cv_spearman": 0.54, "cv_p": 0.05,
        "ci_low": -0.1, "ci_high": 0.8,
        "verdict": "NOT VALIDATED (bootstrap CI on Spearman includes 0)",
        "importance": [], "titer_mean": 4.0, "feature_cols": [],
    }
    report = _report([valid_a, valid_b, ref_unvalidated], reference)
    text = write_analysis(report, tmp_path / "analysis.md").read_text()
    assert "shows a validated ranking signal" not in text
    assert "NOT VALIDATED" in text
