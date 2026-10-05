"""Two checks the engine was missing: is the noise floor's assumption true, and
does the model rank the recipes that actually produced?

BOTH exist because a single pooled number can look acceptable while the product
fails, and `docs/BENCHMARK.md` documents exactly that happening on the real media DoE.

1. PRODUCER-ONLY RANKING. `cv_spearman` is taken over every held-out row,
   producers and non-producers together. On a sheet with a meaningful fraction of
   non-producers a model scores well on it by separating zeros from non-zeros,
   which is a feasibility classifier and not a ranking of recipes. The client's
   question is "which of my producing recipes is best".

2. THE HOMOSCEDASTICITY ASSUMPTION. `estimate_noise_floor` states openly that it
   assumes approximately constant assay noise across recipes, and nothing checked
   it. For titer there is specific reason to doubt it: non-negative, bounded below
   by zero, and on the real data the noise sd exceeds the signal sd - the
   signature of multiplicative noise near a floor. Under multiplicative noise a
   single pooled variance is the wrong summary, and it deflates the ICC that
   `docs/BENCHMARK.md` reasons from.

Neither check transforms or gates anything. They report, because changing the
target's scale would change every number in the response and tightening a gate
changes which uploads the API accepts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.core.evaluation import producer_only_spearman
from kalos.core.replicates import (
    HETERO_RHO_FLOOR,
    MIN_REPLICATED_FOR_HETERO,
    heteroscedasticity_report,
)

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from kalos.portal.analysis import _analyze  # noqa: E402


# --- producer-only ranking --------------------------------------------------- #


def test_pooled_score_can_be_carried_entirely_by_the_zeros():
    """The failure mode, constructed so it is unambiguous.

    Producers are ranked in exactly the WRONG order, yet the pooled Spearman is
    strongly positive purely because every non-producer is predicted below every
    producer. A reliability verdict reading only the pooled number would call this
    model trustworthy.
    """
    actual = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    pred = [0.0, 0.1, 0.2, 0.3, 9.0, 8.0, 7.0, 6.0]  # producers inverted

    pooled = producer_only_spearman(actual, pred, threshold=-1.0)  # nothing excluded
    prod = producer_only_spearman(actual, pred)  # producers only

    assert pooled["spearman"] > 0.5, "the pooled view looks good"
    assert prod["spearman"] == pytest.approx(-1.0), "the ranking that matters is inverted"
    assert prod["n_producers"] == 4
    assert prod["evaluable"] is True


def test_producers_are_selected_by_the_threshold():
    actual = [0.0, 0.5, 1.0, 2.0]
    pred = [0.0, 0.4, 1.1, 1.9]
    assert producer_only_spearman(actual, pred)["n_producers"] == 3
    # An LOD above 0.5 reclassifies that reading as non-producing.
    assert producer_only_spearman(actual, pred, threshold=0.6)["n_producers"] == 2


def test_not_evaluable_rather_than_a_fabricated_number():
    """Fewer than three producers cannot support a rank correlation, and must not
    be reported as one."""
    r = producer_only_spearman([0.0, 0.0, 1.0, 2.0], [0.0, 0.1, 1.0, 2.0])
    assert r["n_producers"] == 2
    assert np.isnan(r["spearman"])
    assert r["evaluable"] is False


def test_no_variation_among_producers_is_not_evaluable():
    r = producer_only_spearman([1.0, 1.0, 1.0, 1.0], [3.0, 1.0, 2.0, 4.0])
    assert r["evaluable"] is False


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        producer_only_spearman([1.0, 2.0], [1.0])


# --- the homoscedasticity diagnostic ---------------------------------------- #


def _replicated(means: list[float], sds: list[float], reps: int = 4, seed: int = 0):
    """One recipe per (mean, sd), each replicated `reps` times."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, (m, sd) in enumerate(zip(means, sds)):
        for _ in range(reps):
            X.append([float(i)])
            y.append(max(0.0, m + rng.normal(0.0, sd)))
    return np.array(X, dtype=float), np.array(y, dtype=float)


def test_constant_noise_reads_as_homoscedastic():
    means = [0.5 + 0.9 * i for i in range(10)]
    X, y = _replicated(means, [0.15] * 10)
    r = heteroscedasticity_report(X, y)
    assert r["homoscedastic"] is True
    assert r["suggests_transform"] is False
    assert abs(r["variance_mean_rho"]) < HETERO_RHO_FLOOR


def test_multiplicative_noise_is_detected():
    means = [0.5 + 0.9 * i for i in range(10)]
    X, y = _replicated(means, [0.15 * m for m in means])
    r = heteroscedasticity_report(X, y)
    assert r["homoscedastic"] is False
    assert r["variance_mean_rho"] >= HETERO_RHO_FLOOR


def test_log1p_would_have_been_the_wrong_transform_at_titer_scale():
    """A regression test for a real mistake in this diagnostic's first version.

    Titer runs around 0.005-0.02. At that magnitude `log1p(y) ~= y`, so the `+1`
    dominates and the transform is nearly the identity: it moved the ICC by 0.002
    on a sheet with a clear variance-mean coupling, and the diagnostic wrongly
    concluded a transform would not help. The offset must be on the DATA's scale.
    """
    means = [0.0 if i % 5 == 0 else 0.004 + 0.0009 * i for i in range(20)]
    X, y = _replicated(means, [0.9 * m for m in means], seed=3)

    scaled = heteroscedasticity_report(X, y)
    assert scaled["log_offset"] is not None
    assert scaled["log_offset"] < 0.01, "offset must sit on the data's scale, not at 1.0"
    assert scaled["suggests_transform"] is True
    assert scaled["icc_gain"] > 0.05

    # Forcing the log1p offset reproduces the old, useless answer.
    as_log1p = heteroscedasticity_report(X, y, log_offset=1.0)
    assert abs(as_log1p["icc_gain"]) < 0.05
    assert as_log1p["suggests_transform"] is False


def test_too_few_replicated_recipes_declines_to_answer():
    X, y = _replicated([1.0, 2.0], [0.1, 0.1])
    r = heteroscedasticity_report(X, y)
    assert r["n_replicated"] < MIN_REPLICATED_FOR_HETERO
    assert r["homoscedastic"] is None
    assert r["suggests_transform"] is False
    assert str(MIN_REPLICATED_FOR_HETERO) in (r["reason"] or "")


def test_negative_target_skips_the_log_rather_than_clipping():
    """Clipping to fit a transform would invent data."""
    means = [-1.0 + 0.9 * i for i in range(8)]
    X, y = _replicated(means, [0.2] * 8)
    y[0] = -0.5  # ensure a genuine negative survives the max(0, .) in the helper
    r = heteroscedasticity_report(X, y)
    assert np.isnan(r["icc_log"])
    assert r["suggests_transform"] is False
    assert "negative" in (r["reason"] or "")


def test_all_zero_target_is_reported_not_crashed():
    X, y = _replicated([0.0] * 6, [0.0] * 6)
    r = heteroscedasticity_report(X, y)
    assert r["suggests_transform"] is False
    assert r["reason"] is not None


# --- both, through the live analyze path ------------------------------------ #


def _media_like_sheet(seed: int = 3) -> pd.DataFrame:
    """20 recipes x 4 replicates, 20% true non-producers, multiplicative noise -
    the shape of the real media DoE."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(20):
        mean = 0.0 if i % 5 == 0 else 0.004 + 0.0009 * i
        for _ in range(4):
            rows.append(
                {
                    "Glucose_g_L": 10.0 + 0.7 * i,
                    "pH": 6.8 + 0.02 * (i % 9),
                    "Lipase_g_L": 0.0 if mean == 0 else max(0.0, mean * (1 + rng.normal(0, 0.9))),
                }
            )
    return pd.DataFrame(rows)


def test_analyze_reports_both_rankings_separately():
    r = _analyze(_media_like_sheet(), target="Lipase_g_L")["reliability"]
    assert "producer_spearman" in r
    assert "producer_clears_floor" in r
    assert r["n_producers"] > 0
    # The two are genuinely different measurements and may disagree in either
    # direction; reporting one alone hides the other.
    assert r["producer_spearman"] != r["spearman"]


def test_analyze_reports_the_scale_diagnostic():
    scale = _analyze(_media_like_sheet(), target="Lipase_g_L")["noise"]["scale"]
    assert scale is not None
    assert scale["homoscedastic"] is False
    assert scale["suggests_transform"] is True
    # The headline consequence: the reported ICC understates real signal here.
    assert scale["icc_log"] > scale["icc_raw"]
    assert scale["reason"] is not None


def test_clears_floor_is_unchanged_by_this_work():
    """The producer number is reported, not folded into the gate. Tightening the
    gate changes which uploads the API accepts, which is a product decision."""
    out = _analyze(_media_like_sheet(), target="Lipase_g_L")
    rel = out["reliability"]
    rho = rel["spearman"]
    expected = bool(rho is not None and rho >= rel["spearman_floor"])
    assert rel["clears_floor"] == expected


def test_unmodeled_admits_when_producer_ranking_cannot_be_scored():
    """A sheet with almost no producers must say so rather than stay silent."""
    rows = []
    for i in range(12):
        rows.append(
            {
                "Glucose_g_L": 10.0 + i,
                "pH": 7.0 + 0.01 * i,
                "Lipase_g_L": 2.0 if i == 0 else 0.0,  # a single producer
            }
        )
    rel = _analyze(pd.DataFrame(rows), target="Lipase_g_L")["reliability"]
    assert rel["n_producers"] < 3
    assert any("producer ranking" in u for u in rel["unmodeled"])
