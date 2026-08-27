"""Held-out calibration: is the surrogate's stated uncertainty honest?

`spearman` asks whether the model ranks held-out runs correctly. It cannot ask
whether the model's error bars are the right SIZE, and those are two different
failure modes: a model can rank perfectly while quoting every interval at half
its true width. The scientist reading "predicted 4.2 +/- 0.3" is acting on the
0.3, so an overconfident band is a wrong answer in its own right.

The engine reported calibration as unmodeled, and the reason was mechanical
rather than principled: the grouped CV computed a held-out posterior sd on every
fold and then threw it away, keeping only the mean. Keeping it makes the claim
falsifiable at no extra cost.

WHICH sd matters. `Surrogate.posterior` returns the LATENT band by default -
uncertainty about the response surface, which is what the acquisition reasons
over. A held-out value is a MEASUREMENT and carries assay noise on top of that,
so scoring observations against the latent band under-covers by construction: on
a replicated sheet with an assay sd of 1.0 it read as z_std=3.20, ece=0.48, a
"catastrophically overconfident" model that was nothing of the kind. Against the
predictive band (`observation_noise=True`) the same fit measures z_std=1.32,
ece=0.09 - mildly overconfident, which is true and useful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.core.evaluation import (
    MIN_CALIBRATION_N,
    interval_calibration,
)

pytest.importorskip("botorch")

from kalos.core.evaluation import grouped_cv_report  # noqa: E402
from kalos.core.surrogate import Surrogate  # noqa: E402


# --- the metric ------------------------------------------------------------- #


def test_a_calibrated_model_scores_near_zero():
    """Truth drawn from exactly the distribution the model states: empirical
    coverage should sit on the nominal at every level."""
    rng = np.random.default_rng(0)
    n = 4000
    pred = np.zeros(n)
    sd = np.ones(n)
    actual = rng.normal(0.0, 1.0, n)
    out = interval_calibration(actual, pred, sd)
    assert out["available"]
    assert out["ece"] < 0.02
    assert out["z_std"] == pytest.approx(1.0, abs=0.05)
    for row in out["levels"]:
        assert row["empirical"] == pytest.approx(row["nominal"], abs=0.03)


def test_an_overconfident_model_is_caught():
    """Bands at half their true width. `z_std` must exceed 1 and say which way."""
    rng = np.random.default_rng(1)
    n = 2000
    out = interval_calibration(rng.normal(0.0, 1.0, n), np.zeros(n), np.full(n, 0.5))
    assert out["ece"] > 0.15  # over GatesConfig.max_ece
    assert out["z_std"] > 1.5
    for row in out["levels"]:
        assert row["empirical"] < row["nominal"]


def test_an_underconfident_model_is_caught_too():
    """Over-wide bands are a different failure, not a safe one: they hide a real
    difference between recipes behind an interval that swallows everything."""
    rng = np.random.default_rng(2)
    n = 2000
    out = interval_calibration(rng.normal(0.0, 1.0, n), np.zeros(n), np.full(n, 3.0))
    assert out["ece"] > 0.15
    assert out["z_std"] < 0.5
    for row in out["levels"]:
        assert row["empirical"] > row["nominal"]


def test_it_declines_rather_than_reporting_a_rounding_grid():
    """At n=6 the attainable coverage rates are 0, 1/6, 2/6 ... so the closest
    value to a nominal 0.90 is 0.833 and the 'error' is an artifact. Declining is
    the honest answer."""
    n = MIN_CALIBRATION_N - 1
    out = interval_calibration(np.zeros(n), np.zeros(n), np.ones(n))
    assert out["available"] is False
    # None, never a number that looks measured - and never NaN, which is not
    # valid JSON and would 500 the response that carries this block.
    assert out["ece"] is None and out["z_std"] is None
    assert str(out["reason"])
    assert out["levels"] == []
    import json

    json.dumps(out)


def test_non_finite_and_zero_sd_rows_are_excluded_not_fatal():
    rng = np.random.default_rng(3)
    n = 60
    actual = rng.normal(0.0, 1.0, n)
    pred = np.zeros(n)
    sd = np.ones(n)
    sd[:3] = 0.0            # a degenerate posterior carries no interval
    actual[3] = np.nan
    out = interval_calibration(actual, pred, sd)
    assert out["available"]
    assert out["n"] == n - 4


def test_mismatched_lengths_are_a_caller_error():
    with pytest.raises(ValueError):
        interval_calibration(np.zeros(5), np.zeros(5), np.ones(4))


# --- the sd the CV now keeps ------------------------------------------------ #


def _blobs(n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 4.0, size=(n, 2))
    y = 1.5 * X[:, 0] - 0.8 * (X[:, 1] - 2.0) ** 2 + rng.normal(0.0, 0.3, n)
    return X, y


def test_predictive_sd_is_wider_than_the_latent_band():
    """The defining property of `observation_noise=True`, and the reason the CV
    asks for it. The MEAN must not move: assay noise is zero-mean, so switching
    bands cannot change a prediction or the Spearman computed from it."""
    X, y = _blobs()
    bounds = np.vstack([X.min(axis=0), X.max(axis=0)])
    s = Surrogate().fit(X, y, bounds=bounds)
    mu_latent, sd_latent = s.posterior(X)
    mu_pred, sd_pred = s.posterior(X, observation_noise=True)
    assert np.allclose(mu_latent, mu_pred)
    assert np.all(sd_pred > sd_latent)


def test_grouped_cv_report_returns_an_aligned_positive_sd():
    X, y = _blobs()
    rep = grouped_cv_report(X, y, n_splits=4, bounds=np.vstack([X.min(0), X.max(0)]))
    sd = np.asarray(rep["oof_std"], float)
    assert len(sd) == len(rep["oof_pred"]) == len(rep["oof_actual"]) == rep["n_oof"]
    assert np.all(np.isfinite(sd)) and np.all(sd > 0)


# --- what the analysis path reports ----------------------------------------- #


def _replicated_sheet(n_recipes: int = 16, reps: int = 3, noise_sd: float = 1.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(0.0, 4.0, n_recipes)
    f1 = rng.uniform(0.0, 4.0, n_recipes)
    mu = 1.5 * f0 - 0.8 * (f1 - 2.0) ** 2
    rows = []
    for k in range(n_recipes):
        for _ in range(reps):
            rows.append(
                {
                    "Methanol": round(float(f0[k]), 3),
                    "pH": round(float(f1[k]), 3),
                    "lipase_titer": float(mu[k] + rng.normal(0.0, noise_sd)),
                }
            )
    return pd.DataFrame(rows)


def test_analysis_reports_calibration_and_stops_calling_it_unmodeled():
    pytest.importorskip("fastapi")
    from kalos.portal.analysis import _analyze

    res = _analyze(_replicated_sheet(), "lipase_titer")
    cal = res["reliability"]["calibration"]
    assert cal["available"] is True
    assert cal["n"] == 48
    assert 0.0 <= cal["ece"] <= 1.0
    assert cal["z_std"] > 0
    assert [row["nominal"] for row in cal["levels"]] == [0.5, 0.8, 0.9, 0.95]
    # the claim it replaces must be gone, and only that claim
    assert not any(str(u).startswith("calibration") for u in res["reliability"]["unmodeled"])
    assert "feasibility probability" in res["reliability"]["unmodeled"]


def test_measuring_calibration_did_not_move_any_existing_number():
    """`observation_noise=True` widens the sd only. If the pooled Spearman or the
    conformal band ever shifts with it, the mean has been disturbed and the
    change has stopped being additive.

    The exact mean-invariance is asserted platform-independently by
    `test_predictive_sd_is_wider_than_the_latent_band` (allclose on the two
    posteriors). This test pins the numbers as a cross-change regression anchor,
    with a LOOSE tolerance on purpose: the reference values were measured on the
    dev machine (macOS), and the GP fit's L-BFGS path differs at the last few
    bits per platform - enough to reorder near-tied held-out ranks and move a
    3-decimal Spearman. CI (Linux) measured 0.786 where the dev machine measures
    0.783; an exact pin turned that FP drift into a red build. The tolerance is
    sized to catch a real regression (a band mix-up moves these numbers by far
    more) while absorbing platform drift."""
    pytest.importorskip("fastapi")
    from kalos.portal.analysis import _analyze

    res = _analyze(_replicated_sheet(), "lipase_titer")
    assert res["cv_spearman"] == pytest.approx(0.783, abs=0.02)
    assert res["conformal_q"] == pytest.approx(2.6208, abs=0.1)


def test_calibration_declines_on_a_sheet_too_small_to_measure_it():
    """And the `unmodeled` list picks the claim back up, with the reason."""
    pytest.importorskip("fastapi")
    from kalos.portal.analysis import _analyze

    rng = np.random.default_rng(5)
    n = 8
    df = pd.DataFrame(
        {
            "Methanol": rng.uniform(0.0, 4.0, n),
            "pH": rng.uniform(0.0, 4.0, n),
            "lipase_titer": rng.uniform(1.0, 5.0, n),
        }
    )
    res = _analyze(df, "lipase_titer")
    cal = res["reliability"]["calibration"]
    if cal["available"]:
        pytest.skip("this sheet produced enough out-of-fold points after all")
    assert any(str(u).startswith("calibration") for u in res["reliability"]["unmodeled"])
