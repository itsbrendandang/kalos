"""The production analysis path (`_analyze`) is replicate-aware: when a run
sheet has replicated recipes it proposes on the reproducible (replicate-averaged)
titer with the measured assay noise floor fed to the GP - the "SNR lever" that
BENCHMARK.md shows is what actually makes BO beat random on real, noisy data -
and reports an honest `noise` block. Non-replicated sheets are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("botorch")

from kalos.portal.analysis import _analyze  # noqa: E402


def _replicated_sheet(n_recipes: int = 10, reps: int = 3, noise_sd: float = 1.2, seed: int = 0):
    """A heavily replicated media-style sheet: a real between-recipe signal buried
    under large assay noise, each recipe measured `reps` times."""
    rng = np.random.default_rng(seed)
    recipes = rng.uniform(0.0, 4.0, (n_recipes, 2))
    rows = []
    for r in recipes:
        true = 1.5 * r[0] - 0.8 * (r[1] - 2.0) ** 2  # reproducible signal
        for _ in range(reps):
            rows.append(
                {
                    "Methanol": round(float(r[0]), 3),
                    "pH": round(float(r[1]), 3),
                    "lipase_titer": true + rng.normal(0.0, noise_sd),
                }
            )
    return pd.DataFrame(rows)


def test_replicated_sheet_uses_reproducible_objective_and_reports_noise():
    df = _replicated_sheet()
    res = _analyze(df, "lipase_titer")

    noise = res["noise"]
    assert noise["replicate_aware"] is True
    # 10 distinct recipes folded from 30 rows
    assert noise["n_recipes"] == 10
    assert noise["n_replicated"] == 10
    assert res["n"] == 30
    # the honest SNR picture is populated
    assert isinstance(noise["icc"], float) and 0.0 <= noise["icc"] <= 1.0
    assert noise["noise_sd"] is not None and noise["noise_sd"] > 0
    # the reproducible best is a group mean, so it can never exceed the best
    # single (spike) measurement - which is exactly why single-measurement
    # "bests" reward noise
    assert noise["best_reproducible"] is not None
    assert noise["best_reproducible"] <= noise["best_single"]
    assert res["best"] == noise["best_single"]

    # still a real proposed batch, all finite
    assert len(res["proposals"]) == 5
    for p in res["proposals"]:
        assert np.isfinite(p["pred"]) and np.isfinite(p["std"])


def test_non_replicated_sheet_is_unchanged_and_flags_not_replicate_aware():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "Methanol": rng.uniform(0, 4, 20).round(4),  # all distinct -> no replicates
            "pH": rng.uniform(0, 4, 20).round(4),
        }
    )
    df["lipase_titer"] = 1.5 * df["Methanol"] - 0.8 * (df["pH"] - 2) ** 2 + rng.normal(0, 0.3, 20)

    res = _analyze(df, "lipase_titer")
    noise = res["noise"]
    assert noise["replicate_aware"] is False
    assert noise["n_replicated"] == 0
    assert noise["best_reproducible"] is None
    # non-replicated: best is still the (only) measurement per recipe
    assert res["best"] == noise["best_single"]
    assert len(res["proposals"]) == 5


def test_replicate_aware_analysis_is_deterministic():
    df = _replicated_sheet(seed=3)
    a = _analyze(df, "lipase_titer")
    b = _analyze(df, "lipase_titer")
    assert a["noise"] == b["noise"]
    assert [p["vals"] for p in a["proposals"]] == [p["vals"] for p in b["proposals"]]
