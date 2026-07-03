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
    # honest uncertainty band + reliability verdict (only what this path can assess)
    assert isinstance(out["conformal_q"], float) and out["conformal_q"] >= 0
    rel = out["reliability"]
    assert rel["spearman_floor"] == 0.20
    assert rel["clears_floor"] == (out["cv_spearman"] >= 0.20)
    assert set(rel) == {"spearman", "ci95", "spearman_floor", "clears_floor", "ci_excludes_zero", "unmodeled"}
    assert "scale-up transfer" in rel["unmodeled"]


def test_latest_reflects_last_upload(tmp_path, monkeypatch):
    from kalos.portal import app as portal

    monkeypatch.setattr(portal, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal, "_LATEST_PATH", tmp_path / "latest.json")
    monkeypatch.setattr(portal, "_LATEST", None)

    # before any upload the Overview must know there is no data (not fake it)
    assert portal.latest() == {"has_data": False}

    result = portal._analyze(_sheet())
    portal._save_latest(result, "runs.csv")

    got = portal.latest()
    assert got["has_data"] is True
    assert got["dataset"] == "runs.csv"
    assert got["target"] == "lipase_titer"
    assert got["cv_spearman"] is not None
    assert len(got["proposals"]) >= 1
    assert "updated" in got

    # survives a restart (reloads from disk when the in-memory copy is gone)
    monkeypatch.setattr(portal, "_LATEST", None)
    reloaded = portal.latest()
    assert reloaded["has_data"] is True and reloaded["dataset"] == "runs.csv"
