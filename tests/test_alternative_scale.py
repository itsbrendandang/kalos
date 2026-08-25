"""The `alternative_scale` response block: an EXPLICIT, LABELED second
evaluation pass on the log scale, run only when `heteroscedasticity_report`
says a log1p-style transform would materially raise the ICC.

CONTEXT. `kalos.core.replicates.heteroscedasticity_report` already diagnoses
when the homoscedastic noise-floor assumption fails and a transform would help
(`suggests_transform`, `icc_log`, `icc_gain`, `log_offset`), and `_noise_block`
reports that diagnosis in `noise.scale`. Until now `_analyze` stopped there and
went on to model the PROPOSAL surrogate on the raw scale regardless - diagnosed
the problem, then modeled on the wrong scale anyway. The standing design
decision (see both docstrings) is that the target's scale is never changed
SILENTLY, because every reported number lives on it. `alternative_scale` is the
honest alternative to silence: a second, clearly-labeled fit and grouped-CV
pass on the scale the diagnostic says fits better, reported ALONGSIDE the raw
numbers - proposals and everything else in the response stay on the raw scale.

THE OFFSET TRAP this module tests for directly. `heteroscedasticity_report`
computes `icc_log` from `np.log(y + log_offset)`, NOT `np.log1p(y + log_offset)`
- literal log1p (i.e. `log(1 + y + offset)`) was the first thing tried there and
was WRONG at titer's scale (~0.005-0.02): the `+1` swamps the tiny offset and
the transform is nearly the identity. `"log1p"` in this codebase's naming is an
informal label for "log of value-plus-small-offset", not a literal call to
`np.log1p`. `test_alternative_scale_uses_log_offset_not_log1p_offset` below
pins this down directly by capturing the array actually handed to the (faked)
CV function.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from kalos.portal.analysis import _alternative_scale_block, _analyze  # noqa: E402


# --- fixtures ----------------------------------------------------------------- #


def _hetero_sheet(seed: int = 3, n_recipes: int = 20, reps: int = 3) -> pd.DataFrame:
    """20 recipes x 3 replicates = 60 rows, 20% true non-producers, multiplicative
    noise (within-recipe sd proportional to the recipe mean) - the shape that
    triggers `suggests_transform` on the real media DoE (see
    `tests/test_noise_scale_and_producers.py`'s `_media_like_sheet`, same shape at
    3 replicates instead of 4 so this fixture lands at exactly 60 rows for the
    runtime measurement below)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_recipes):
        mean = 0.0 if i % 5 == 0 else 0.004 + 0.0009 * i
        for _ in range(reps):
            rows.append(
                {
                    "Glucose_g_L": 10.0 + 0.7 * i,
                    "pH": 6.8 + 0.02 * (i % 9),
                    "Lipase_g_L": 0.0 if mean == 0 else max(0.0, mean * (1 + rng.normal(0, 0.9))),
                }
            )
    return pd.DataFrame(rows)


def _homo_sheet(seed: int = 7, n_recipes: int = 20, reps: int = 3) -> pd.DataFrame:
    """Same shape, but constant (additive) within-recipe noise regardless of the
    recipe mean - homoscedastic by construction, so `suggests_transform` must
    stay False."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_recipes):
        mean = 0.5 + 0.9 * i
        for _ in range(reps):
            rows.append(
                {
                    "Glucose_g_L": 10.0 + 0.7 * i,
                    "pH": 6.8 + 0.02 * (i % 9),
                    "Lipase_g_L": max(0.0, mean + rng.normal(0.0, 0.15)),
                }
            )
    return pd.DataFrame(rows)


assert len(_hetero_sheet()) == 60, "runtime measurement below is tuned against 60 rows"


# --- offset semantics, tested directly (no GP, no torch needed) -------------- #


def test_alternative_scale_uses_log_offset_not_log1p_offset():
    """The trap: fitting on `np.log1p(y + offset)` (literal `log(1 + y + offset)`)
    would silently reproduce the exact historical bug
    `heteroscedasticity_report`'s docstring documents, and would desynchronize
    this block's numbers from `icc_log`. The transform actually handed to the
    (faked) CV function must be `np.log(y + offset)`, byte-for-byte what
    `heteroscedasticity_report` used to compute `icc_log`."""
    y = np.array([0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.02])
    X = np.zeros((len(y), 1))
    offset = 0.002
    het = {
        "suggests_transform": True,
        "log_offset": offset,
        "icc_log": 0.4231,
        "reason": "within-recipe variance rises with recipe mean",
    }
    captured: dict = {}

    def fake_cv(X_, y_, **kwargs):
        captured["y"] = np.asarray(y_, dtype=float)
        n = len(y_)
        return {
            "spearman": 0.5,
            "ci95": (0.1, 0.9),
            "oof_actual": list(y_),
            "oof_pred": list(y_),
            "oof_std": [1.0] * n,
        }

    def fake_calibration(actual, pred, sd):
        return {"available": True, "ece": 0.01, "z_std": 1.0}

    block = _alternative_scale_block(
        het, X, y, groups=list(range(len(y))), bounds=None, cat_dims=None,
        grouped_cv_report=fake_cv, interval_calibration=fake_calibration,
    )

    expected = np.log(y + offset)
    as_literal_log1p = np.log1p(y + offset)
    assert np.allclose(captured["y"], expected)
    assert not np.allclose(captured["y"], as_literal_log1p), (
        "the surrogate must be fit on log(y + offset), not literal log1p(y + offset)"
    )
    assert block["scale"] == "log1p"
    assert block["offset"] == pytest.approx(offset)
    # `icc` is rounded to 3 places (matching `noise.scale.icc_log`'s rounding).
    assert block["icc"] == pytest.approx(0.4231, abs=5e-4)


def test_null_block_when_not_suggested():
    het = {"suggests_transform": False, "reason": "no variance-mean coupling detected"}
    block = _alternative_scale_block(
        het, np.zeros((4, 1)), np.zeros(4), groups=[0, 1, 2, 3], bounds=None, cat_dims=None,
        grouped_cv_report=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")),
        interval_calibration=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    assert block["scale"] is None
    assert block["offset"] is None
    assert block["cv_spearman"] is None
    assert block["cv_ci95"] is None
    assert block["calibration"] is None
    assert block["icc"] is None
    assert block["interpretation"] is None
    assert block["reason"] == "no variance-mean coupling detected"


# --- through the live analyze path -------------------------------------------- #


def test_heteroscedastic_sheet_triggers_block_with_consistent_numbers():
    out = _analyze(_hetero_sheet(), target="Lipase_g_L")
    scale = out["noise"]["scale"]
    alt = out["alternative_scale"]

    assert scale is not None
    assert scale["suggests_transform"] is True, "fixture must actually trigger the diagnostic"

    assert alt["scale"] == "log1p"
    # Consistency the task exists to guarantee: the offset and ICC reported here
    # must be the SAME numbers `noise.scale` already reports, not independently
    # (and possibly differently) derived ones.
    assert alt["offset"] == pytest.approx(scale["log_offset"])
    assert alt["icc"] == pytest.approx(scale["icc_log"])
    assert alt["reason"] == scale["reason"]
    assert isinstance(alt["cv_spearman"], float) or alt["cv_spearman"] is None
    assert alt["cv_ci95"] is None or len(alt["cv_ci95"]) == 2
    assert alt["calibration"] is not None
    assert "The log scale fits materially better" in alt["interpretation"]
    assert "raw scale" in alt["interpretation"]


def test_homoscedastic_sheet_yields_null_with_reason():
    out = _analyze(_homo_sheet(), target="Lipase_g_L")
    scale = out["noise"]["scale"]
    alt = out["alternative_scale"]

    assert scale is not None
    assert scale["suggests_transform"] is False, "fixture must not trigger the diagnostic"
    assert alt["scale"] is None
    assert alt["offset"] is None
    assert alt["cv_spearman"] is None
    assert alt["cv_ci95"] is None
    assert alt["calibration"] is None
    assert alt["icc"] is None
    assert alt["interpretation"] is None
    assert isinstance(alt["reason"], str) and alt["reason"]


def test_block_is_json_safe():
    for sheet in (_hetero_sheet(), _homo_sheet()):
        out = _analyze(sheet, target="Lipase_g_L")
        # json.dumps raises on NaN by default (strict-JSON), so this fails loudly
        # if any field slipped through as a raw float("nan") instead of None.
        encoded = json.dumps(out["alternative_scale"])
        assert "NaN" not in encoded


def test_block_is_purely_additive():
    """Computing `alternative_scale` must change nothing else in the response -
    not the proposals, not the CV numbers, not the driver panel. Forces the
    block off via monkeypatch and diffs the rest of the response against a
    normal run on the identical (deterministic-seed) sheet."""
    import kalos.portal.analysis as analysis_mod

    sheet = _hetero_sheet()
    with_block = _analyze(sheet, target="Lipase_g_L")

    original = analysis_mod._alternative_scale_block
    try:
        analysis_mod._alternative_scale_block = lambda *a, **k: {
            "scale": None, "offset": None, "cv_spearman": None, "cv_ci95": None,
            "calibration": None, "icc": None, "reason": "forced off for the test",
            "interpretation": None,
        }
        without_block = _analyze(sheet, target="Lipase_g_L")
    finally:
        analysis_mod._alternative_scale_block = original

    # `timestamp` is `int(time.time())`, wall-clock and unrelated to this block;
    # excluded alongside `alternative_scale` itself.
    skip = {"alternative_scale", "timestamp"}
    a = {k: v for k, v in with_block.items() if k not in skip}
    b = {k: v for k, v in without_block.items() if k not in skip}
    assert a == b, "alternative_scale must be purely additive to the rest of the response"


def test_runtime_bound_is_loose():
    """No hard per-block budget asserted here (machine-dependent); this is the
    stated `< 10s total analyze` loose bound on the heteroscedastic sheet, which
    pays for both the raw analysis and the extra log-scale pass."""
    t0 = time.monotonic()
    _analyze(_hetero_sheet(), target="Lipase_g_L")
    elapsed = time.monotonic() - t0
    assert elapsed < 10.0, f"analyze took {elapsed:.2f}s, expected < 10s"
