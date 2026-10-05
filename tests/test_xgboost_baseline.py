"""The XGBoost baseline the GP is scored against (`kalos.core.evaluation.
xgboost_baseline_report`, reported by `_analyze` as `cv_xgboost_baseline`).

Covers: it runs on the SAME grouped-CV partition and rows as the GP (so the
comparison is paired), the paired verdict only claims a winner when the
difference CI excludes zero, a missing `xgboost` degrades to
`available: false` instead of raising, the block is strict-JSON, and computing
it changes nothing else in the analysis response.
"""
from __future__ import annotations

import builtins
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("xgboost")

from kalos.core.evaluation import grouped_cv_report, xgboost_baseline_report  # noqa: E402


def _data(seed: int = 0, n: int = 48):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 3))
    y = 3.0 * np.sin(3.0 * X[:, 0]) + 2.0 * X[:, 1] ** 2 + rng.normal(0, 0.2, n)
    groups = np.repeat(np.arange(n // 2), 2)  # replicate pairs never straddle a fold
    return X, y, groups


def test_baseline_is_paired_with_the_gp_partition():
    X, y, groups = _data()
    rep = grouped_cv_report(X, y, groups=groups, n_splits=5)
    out = xgboost_baseline_report(
        X, y, groups=groups, n_splits=5, gp_oof_pred=rep["oof_pred"], gp_oof_actual=rep["oof_actual"],
    )
    assert out["available"] is True
    assert out["paired"] is True
    assert out["n_oof"] == rep["n_oof"]
    assert out["n_folds"] == rep["n_folds"]
    assert -1.0 <= out["spearman"] <= 1.0
    lo, hi = out["ci95"]
    assert lo <= out["spearman"] <= hi
    vs = out["vs_gp"]
    assert vs["gp_spearman"] == pytest.approx(rep["spearman"])
    assert vs["delta"] == pytest.approx(vs["gp_spearman"] - out["spearman"])
    d_lo, d_hi = vs["delta_ci95"]
    assert d_lo == d_lo and d_hi == d_hi, "24 groups must give a finite paired CI"
    expected = "gp_better" if d_lo > 0 else "xgboost_better" if d_hi < 0 else "no_detectable_difference"
    assert vs["verdict"] == expected


def test_verdict_is_not_computable_without_bootstrap_draws():
    # A constant target makes every Spearman undefined: there is nothing to
    # compare, and the report must say so rather than "no detectable difference".
    X, _, groups = _data(seed=4)
    y = np.full(len(X), 2.5)
    rep = grouped_cv_report(X, y, groups=groups, n_splits=5)
    out = xgboost_baseline_report(
        X, y, groups=groups, gp_oof_pred=rep["oof_pred"], gp_oof_actual=rep["oof_actual"],
    )
    assert out["paired"] is True
    assert out["vs_gp"]["verdict"] == "not_computable"
    assert out["reason"]


def test_baseline_is_deterministic():
    X, y, groups = _data(seed=1)
    a = xgboost_baseline_report(X, y, groups=groups)
    b = xgboost_baseline_report(X, y, groups=groups)
    assert a == b


def test_unpaired_without_gp_predictions():
    X, y, groups = _data(seed=2)
    out = xgboost_baseline_report(X, y, groups=groups)
    assert out["paired"] is False
    assert out["vs_gp"] is None


def test_mismatched_gp_rows_are_not_paired():
    X, y, groups = _data(seed=3)
    rep = grouped_cv_report(X, y, groups=groups, n_splits=5)
    shuffled = list(reversed(rep["oof_actual"]))
    out = xgboost_baseline_report(
        X, y, groups=groups, gp_oof_pred=rep["oof_pred"], gp_oof_actual=shuffled,
    )
    assert out["paired"] is False
    assert out["vs_gp"] is None


def test_too_few_groups_reports_a_reason():
    X, y, _ = _data(n=8)
    out = xgboost_baseline_report(X, y, groups=np.zeros(8))
    assert out["available"] is True
    assert out["n_folds"] == 0
    assert out["reason"]


def test_missing_xgboost_is_reported_not_raised(monkeypatch):
    real_import = builtins.__import__

    def no_xgboost(name, *args, **kwargs):
        if name == "xgboost" or name.startswith("xgboost."):
            raise ImportError("simulated: xgboost not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_xgboost)
    X, y, groups = _data()
    out = xgboost_baseline_report(X, y, groups=groups)
    assert out == {"available": False, "reason": "xgboost is not installed (pip install 'kalos[xgboost]')"}


# --- through _analyze ------------------------------------------------------------- #


def _sheet(seed: int = 5, n_recipes: int = 18, reps: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_recipes):
        temp = rng.uniform(30, 37)
        feed = rng.uniform(1, 4)
        for _ in range(reps):
            rows.append({
                "recipe": f"R{r}",
                "Temp": round(temp, 2),
                "Feed": round(feed, 3),
                "Titer_g_L": round(2 + 0.5 * feed - 0.08 * (temp - 34) ** 2 + rng.normal(0, 0.1), 3),
            })
    return pd.DataFrame(rows)


def test_analyze_reports_the_baseline_as_strict_json():
    pytest.importorskip("fastapi")
    from kalos.portal.analysis import _analyze

    out = _analyze(_sheet(), target="Titer_g_L")
    block = out["cv_xgboost_baseline"]
    assert block["available"] is True
    assert block["model"] == "xgboost"
    assert block["vs_gp"] is not None
    assert block["vs_gp"]["verdict"] in {"gp_better", "xgboost_better", "no_detectable_difference"}
    assert block["vs_gp"]["gp_spearman"] == pytest.approx(out["cv_spearman"], abs=1e-3)
    assert block["interpretation"]
    assert "NaN" not in json.dumps(block)


def test_baseline_is_purely_additive(monkeypatch):
    """Computing the baseline must change nothing else in the response. The
    "without" arm stubs the COMPUTATION (`_analyze` imports
    `xgboost_baseline_report` at call time), so the real fit never runs there."""
    pytest.importorskip("fastapi")
    import kalos.core.evaluation as evaluation_mod
    import kalos.portal.analysis as analysis_mod

    sheet = _sheet(seed=7)
    with_block = analysis_mod._analyze(sheet, target="Titer_g_L")
    assert with_block["cv_xgboost_baseline"]["available"] is True

    calls: list[int] = []

    def stub(*args, **kwargs):
        calls.append(1)
        return {"available": False, "reason": "forced off"}

    monkeypatch.setattr(evaluation_mod, "xgboost_baseline_report", stub)
    without_block = analysis_mod._analyze(sheet, target="Titer_g_L")
    assert calls == [1], "the stub must replace the real computation"
    assert without_block["cv_xgboost_baseline"] == {"available": False, "reason": "forced off"}
    skip = {"cv_xgboost_baseline", "timestamp"}
    a = {k: v for k, v in with_block.items() if k not in skip}
    b = {k: v for k, v in without_block.items() if k not in skip}
    assert a == b, "cv_xgboost_baseline must be purely additive to the rest of the response"
