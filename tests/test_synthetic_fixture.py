"""Tests for the non-proprietary synthetic bioprocess fixture (`examples/synthetic_bioprocess`).

`kalos/bench` otherwise has only abstract math surfaces (`objectives.py`); its
pool benchmark (`kalos.bench.pool.pool_from_frame`) needs a real run sheet,
and the real ones all require client data that cannot be committed. This
fixture - ported from the owner's earlier engine (voyager-brain-rebuild,
samples/synthetic_bioprocess.csv + make_synthetic.py) - is the non-proprietary
stand-in: fully synthetic, seeded, safe to commit.

These tests check the fixture itself, not the engine: the CSV has the shape
and columns the docs promise, the feasibility gate zeroes the documented
fraction of rows, and a FAST classifier (cross-validated RandomForest, no GP
fit) recovers the known feasibility signal well above chance - confirming the
signal is real and strong, the way `samples/README.md` originally documented
it (AUC ~ 0.97 there). Everything here runs in well under a second; no GP
touches this file, keeping the whole thing comfortably inside ~15s.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from kalos.bench.pool import pool_from_frame

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "examples" / "synthetic_bioprocess"
CSV_PATH = FIXTURE_DIR / "synthetic_bioprocess.csv"

_CONTINUOUS_INPUTS = [
    "pH", "temperature_C", "glucose_gL", "glutamine_mM", "feed_rate_mLh",
    "DO_pct", "agitation_rpm", "inoculum_1e6", "antifoam_mgL", "osmolality_mOsm",
]


def _load() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


# --- the fixture on disk ------------------------------------------------------- #


def test_fixture_csv_has_documented_shape_and_columns():
    df = _load()
    assert len(df) == 160
    assert set(_CONTINUOUS_INPUTS) <= set(df.columns)
    assert "run_id" in df.columns  # id, meant to be ignored downstream
    assert "medium_base" in df.columns  # categorical, meant to be ignored downstream
    assert "titer_g_L" in df.columns  # the target


def test_feasibility_gate_zero_fraction_is_in_documented_range():
    # samples/README.md documents "~44% non-producers"; the generator's own
    # docstring says "~30%". Both describe the same gate observed at different
    # points in its history, so this asserts the documented band rather than
    # pinning one figure - a change to the RNG draw order (but not the gate
    # logic or its seed) should not need this test rewritten.
    df = _load()
    zero_frac = float((df["titer_g_L"] == 0).mean())
    assert 0.25 <= zero_frac <= 0.50


def test_generator_reproduces_the_csv_byte_for_byte(tmp_path):
    """The generator is seeded; running it must reproduce the committed CSV
    exactly, not just approximately - anything else means the fixture on disk
    has drifted from the script that is supposed to produce it."""
    import runpy

    script = FIXTURE_DIR / "make_synthetic.py"
    out_dir = tmp_path / "synthetic_bioprocess"
    out_dir.mkdir()
    # Copy the script into an isolated directory so it writes its output there
    # (it writes next to itself), not back over the real fixture.
    script_copy = out_dir / "make_synthetic.py"
    script_copy.write_text(script.read_text())
    runpy.run_path(str(script_copy), run_name="__main__")

    generated = (out_dir / "synthetic_bioprocess.csv").read_text()
    committed = CSV_PATH.read_text()
    assert generated == committed


# --- the fixture's own signal (loader + a fast classifier, no GP) ------------- #


def test_pool_loader_recovers_the_ten_continuous_inputs():
    df = _load()
    X, y, feats = pool_from_frame(df, "titer_g_L")
    assert sorted(feats) == sorted(_CONTINUOUS_INPUTS)
    assert X.shape == (160, 10)
    assert y.shape == (160,)


def test_feasibility_signal_is_strong_and_fast_to_recover():
    """A cross-validated RandomForest (not a GP - this file must stay fast)
    should clearly separate producers from non-producers, confirming the
    fixture's feasibility gate is a real, learnable signal and not noise."""
    df = _load()
    X, y, _feats = pool_from_frame(df, "titer_g_L")
    labels = (y > 0).astype(int)

    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    proba = cross_val_predict(clf, X, labels, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(labels, proba)

    assert auc > 0.8
