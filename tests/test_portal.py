"""Portal /api/run analysis path: leakage-controlled grouped CV + honest CI."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from kalos.portal.app import _analyze


def _sheet(n=40, seed=0):
    rng = np.random.default_rng(seed)
    methanol = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    other_output = rng.uniform(0, 1, n)   # a DIFFERENT measured output (leakage bait)
    titer = 1.5 * methanol - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "medium": rng.choice(["A", "B", "C"], n),   # group column
        "Methanol": methanol.round(3),
        "pH": ph.round(2),
        "biomass_od600": other_output.round(3),      # output -> must NOT become a feature
        "lipase_titer": titer.round(3),              # the target
    })


def test_analyze_excludes_outputs_and_reports_honest_cv():
    out = _analyze(_sheet())
    assert out["target"] == "lipase_titer"
    # other measured outputs must never become input features (anti-leakage)
    assert "biomass_od600" not in out["features"]
    assert "lipase_titer" not in out["features"]
    assert "Methanol" in out["features"]
    # the honest grouped-CV report is surfaced with a confidence band + group count
    assert out["cv_spearman"] is not None
    assert out["cv_ci95"] is not None and len(out["cv_ci95"]) == 2
    assert isinstance(out["cv_n_groups"], int) and out["cv_n_groups"] >= 2
    assert len(out["oof"]) > 0
    assert len(out["proposals"]) >= 1
