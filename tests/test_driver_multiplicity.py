"""Multiplicity control on the driver panel, and the band's stated coverage.

Two defects motivated this file, both of which put a false or overstated claim in
front of a scientist.

MULTIPLICITY. The driver panel tests every continuous feature against the target
and then ships the strongest by |rho|. With an uncorrected per-feature 95% CI
that is a machine for printing false process insight: over 400 simulated reports
on 30 pure-noise features against an independent target, 78.8% contained at least
one "significant" driver. Under Benjamini-Hochberg at q=0.05 that falls to 3.5%.
Selection makes it worse than the raw rate suggests, because ranking by |rho|
preferentially surfaces exactly the flukes. Drivers are what a scientist acts on
and repeats in a meeting, so this was the single most likely path to a wrong
conclusion reaching a client.

STATED COVERAGE. The engine computes the conformal band at alpha=0.1, a 90% band.
The frontend had drifted to labelling that same band "95%" on one surface and
"90%" on another. A coverage number that is wrong on screen is an overclaim, not a
hedge, so the engine now ships the coverage it actually computed and no consumer
has to hardcode it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.core.drivers import benjamini_hochberg

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from kalos.portal.analysis import (  # noqa: E402
    CONFORMAL_ALPHA,
    DRIVER_FDR_Q,
    DRIVER_TOP_K,
    _analyze,
)


# --- the BH primitive ------------------------------------------------------- #


def test_bh_rejects_nothing_when_all_p_are_large():
    assert not benjamini_hochberg([0.9, 0.8, 0.7], q=0.05).any()


def test_bh_rejects_everything_when_all_p_are_tiny():
    assert benjamini_hochberg([1e-6, 2e-6, 3e-6], q=0.05).all()


def test_bh_is_step_up_not_per_feature():
    """The defining property. With m=3 and q=0.05 the thresholds are
    0.0167 / 0.0333 / 0.05. p=0.04 fails ITS own threshold (0.0333) but a
    step-up procedure that rejected a larger p would carry it along; here the
    largest passing rank is 1 (0.01 <= 0.0167), so 0.04 is not rejected. A
    per-feature 0.05 rule would have rejected it - that difference is the entire
    point of the correction."""
    mask = benjamini_hochberg([0.01, 0.04, 0.9], q=0.05)
    assert mask.tolist() == [True, False, False]


def test_bh_carries_along_smaller_p_values():
    """Step-up: if a larger p-value passes its threshold, every smaller one is
    rejected with it, even one that fails its own threshold."""
    # m=4, q=0.05 -> thresholds 0.0125 / 0.025 / 0.0375 / 0.05
    # p=0.04 fails its own (0.0375) but p=0.045 at rank 4 passes 0.05,
    # so ranks 1..4 are all rejected.
    mask = benjamini_hochberg([0.001, 0.02, 0.04, 0.045], q=0.05)
    assert mask.all()


def test_bh_is_stricter_than_uncorrected_at_scale():
    """The property that motivated the change, asserted directly."""
    rng = np.random.default_rng(11)
    m = 30
    p = rng.uniform(0, 1, m)
    uncorrected = (p < 0.05).sum()
    corrected = int(benjamini_hochberg(p, q=0.05).sum())
    assert corrected <= uncorrected


def test_bh_handles_empty_and_nan():
    assert benjamini_hochberg([], q=0.05).shape == (0,)
    # NaN carries no evidence and must never be rejected
    mask = benjamini_hochberg([1e-9, float("nan")], q=0.05)
    assert mask[0] and not mask[1]


def test_bh_mask_is_aligned_to_input_order():
    """Returned mask must map back to the caller's feature order, not sorted
    order - misalignment would attribute significance to the wrong column."""
    mask = benjamini_hochberg([0.9, 1e-9, 0.8], q=0.05)
    assert mask.tolist() == [False, True, False]


# --- the live driver panel -------------------------------------------------- #


def _sheet_one_real_driver_many_noise(n: int = 60, n_noise: int = 14, seed: int = 0):
    """One genuine driver plus many pure-noise columns: the exact shape that
    manufactures false drivers when multiplicity is ignored."""
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {"Methanol": rng.uniform(0, 4, n)}
    for k in range(n_noise):
        cols[f"noise_{k}"] = rng.uniform(0, 1, n)
    cols["lipase_titer"] = 1.5 * cols["Methanol"] + rng.normal(0, 0.4, n)
    return pd.DataFrame(cols)


def test_only_the_real_driver_is_significant():
    """Without FDR control this sheet reports noise columns as significant with
    confidence intervals excluding zero."""
    out = _analyze(_sheet_one_real_driver_many_noise(), target="lipase_titer")
    significant = [d["name"] for d in out["drivers"] if d["significant"]]
    assert significant == ["Methanol"]


def test_a_noise_driver_can_clear_the_ci_and_still_be_rejected():
    """The specific rescue: a feature whose bootstrap CI excludes zero purely by
    chance must not be reported as significant. If this ever finds no such
    feature the test is vacuous, so it asserts the setup too."""
    out = _analyze(_sheet_one_real_driver_many_noise(), target="lipase_titer")
    noise = [d for d in out["drivers"] if d["name"].startswith("noise_")]
    tricked = [d for d in noise if d["ci_excludes_zero"]]
    assert tricked, "expected at least one noise feature to fool the raw CI"
    for d in tricked:
        assert d["survives_fdr"] is False
        assert d["significant"] is False


def test_significant_requires_both_tests():
    """`significant` is the conjunction, never either alone."""
    out = _analyze(_sheet_one_real_driver_many_noise(), target="lipase_titer")
    for d in out["drivers"]:
        assert d["significant"] == (d["ci_excludes_zero"] and d["survives_fdr"])


def test_selection_is_disclosed_not_silent():
    """Selecting the strongest of many tested features is a statistical act; the
    client has to be able to see that it happened."""
    df = _sheet_one_real_driver_many_noise(n_noise=14)
    out = _analyze(df, target="lipase_titer")
    sel = out["driver_selection"]
    assert sel["n_tested"] == 15  # 1 real + 14 noise
    assert sel["n_reported"] == len(out["drivers"]) <= DRIVER_TOP_K
    assert sel["fdr_method"] == "benjamini_hochberg"
    assert sel["fdr_q"] == DRIVER_FDR_Q
    assert sel["ranked_by"] == "abs_rho"


def test_fdr_is_applied_over_all_tested_features_not_the_reported_subset():
    """Correcting for the 8 shipped features when 30 were tested would understate
    the multiplicity. More tested features must make it no easier for a noise
    column to be called significant."""
    few = _analyze(_sheet_one_real_driver_many_noise(n_noise=3), target="lipase_titer")
    many = _analyze(_sheet_one_real_driver_many_noise(n_noise=25), target="lipase_titer")
    assert few["driver_selection"]["n_tested"] == 4
    assert many["driver_selection"]["n_tested"] == 26
    # the real driver survives either way; noise never becomes significant
    for out in (few, many):
        assert [d["name"] for d in out["drivers"] if d["significant"]] == ["Methanol"]


def test_every_driver_carries_its_evidence():
    out = _analyze(_sheet_one_real_driver_many_noise(), target="lipase_titer")
    for d in out["drivers"]:
        assert 0.0 <= d["p"] <= 1.0
        assert isinstance(d["ci_excludes_zero"], bool)
        assert isinstance(d["survives_fdr"], bool)
        assert len(d["ci95"]) == 2


# --- the band's stated coverage --------------------------------------------- #


def test_response_states_the_coverage_it_actually_computed():
    """So no consumer has to hardcode 90 or 95 and drift from the engine."""
    out = _analyze(_sheet_one_real_driver_many_noise(), target="lipase_titer")
    assert out["conformal_coverage"] == pytest.approx(1.0 - CONFORMAL_ALPHA)
    assert out["conformal_coverage"] == pytest.approx(0.9)


def test_coverage_is_present_even_when_the_band_is_not():
    """A sheet too small for out-of-fold residuals yields conformal_q None. The
    COVERAGE is a property of the method, not of the data, so it must still be
    stated rather than becoming None alongside the band."""
    rng = np.random.default_rng(5)
    n = 8
    df = pd.DataFrame(
        {
            "Methanol": rng.uniform(0, 4, n),
            "lipase_titer": rng.uniform(1, 3, n),
        }
    )
    out = _analyze(df, target="lipase_titer")
    assert out["conformal_coverage"] == pytest.approx(1.0 - CONFORMAL_ALPHA)
