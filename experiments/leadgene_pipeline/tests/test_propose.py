"""Acquisition / next-batch proposal tests.

The proposer must (a) rank by an upper-confidence bound in exploit mode, so a
high-uncertainty candidate outranks an equally-predicted but near-certain one;
(b) flip to explore mode (rank by uncertainty alone) when nothing validated,
because the predicted titers are then untrustworthy; and (c) select exactly q,
diversified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.evaluation import MethodResult
from pipeline.pipeline import Report
from pipeline.propose import propose_batch


def _mr(name, ci_low, mean, std, wells):
    verdict = "USABLE (CI excludes 0)" if ci_low > 0 else "NOT VALIDATED (CI includes 0)"
    return MethodResult(
        name=name, n_train=20, n_unique_groups=10, n_features=3,
        cv_spearman=0.6 if ci_low > 0 else -0.1, cv_p=0.05, ci_low=ci_low, ci_high=0.9,
        verdict=verdict, client_well_ids=np.array(wells),
        client_mean=np.array(mean, float), client_std=np.array(std, float),
    )


def _report(results, cohort, reference):
    return Report(results=results, blend_table=pd.DataFrame(), weights={}, note="",
                  cohort=cohort, reference=reference)


def _cohort(wells):
    return pd.DataFrame({"well_id": wells,
                         "f0": np.arange(len(wells), dtype=float),
                         "f1": np.arange(len(wells), dtype=float) * 2})


def test_exploit_ucb_prefers_uncertainty_at_equal_mean():
    wells = ["w0", "w1", "w2", "w3"]
    # w0 and w3 have the same predicted titer; w3 is far more uncertain.
    a = _mr("gp", 0.2, [90, 50, 60, 90], [1, 1, 1, 1], wells)
    b = _mr("br", 0.2, [90, 50, 60, 90], [1, 1, 1, 20], wells)
    res = propose_batch(_report([a, b], _cohort(wells), {"feature_cols": ["f0", "f1"]}),
                        q=2, beta=2.0)
    assert res.mode == "exploit"
    r = res.table.set_index("well_id")
    assert r.loc["w3", "acq_score"] > r.loc["w0", "acq_score"]
    assert res.table["selected"].sum() == 2


def test_explore_when_nothing_validated_ranks_by_uncertainty():
    wells = ["w0", "w1", "w2"]
    a = _mr("gp", -0.1, [90, 50, 60], [1, 1, 8], wells)   # not validated
    b = _mr("br", -0.1, [88, 52, 61], [1, 1, 8], wells)
    res = propose_batch(_report([a, b], _cohort(wells), None), q=1, beta=1.5)
    assert res.mode == "explore"
    assert res.table.iloc[0]["well_id"] == "w2"           # highest uncertainty leads
    assert res.table["selected"].sum() == 1


def test_batch_is_diversified():
    wells = ["w0", "w1", "w2", "w3", "w4"]
    a = _mr("gp", 0.2, [99, 98, 50, 51, 10], [1, 1, 1, 1, 1], wells)
    b = _mr("br", 0.2, [99, 98, 50, 51, 10], [1, 1, 1, 1, 1], wells)
    # w0 and w1 are the two highest-acq but sit right next to each other in feature
    # space; a diversified q=2 batch should not pick both.
    cohort = pd.DataFrame({"well_id": wells,
                           "f0": [0.0, 0.01, 5.0, 5.01, 9.0],
                           "f1": [0.0, 0.01, 5.0, 5.01, 9.0]})
    res = propose_batch(_report([a, b], cohort, {"feature_cols": ["f0", "f1"]}),
                        q=2, beta=1.0, diversity=1.0)
    picked = set(res.table[res.table["selected"]]["well_id"])
    assert not {"w0", "w1"}.issubset(picked)
