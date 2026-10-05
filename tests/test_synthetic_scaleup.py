"""Tests for the synthetic scale-up fixture (`examples/synthetic_scaleup`),
built for the Scale-Up Readout.

These tests check the fixture itself, not the modeling code: the generator
is deterministic under its seed, the committed CSV cannot silently drift
from the script that produces it, and the dataset has the shape the readout
depends on (documented columns, at least 5 distinct scales, at least 3 runs
per scale, at least 10 rows at the third-smallest scale and above, and no
non-finite values anywhere).
"""
from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "examples" / "synthetic_scaleup"
CSV_PATH = FIXTURE_DIR / "synthetic_scaleup.csv"

_REQUIRED_COLUMNS = [
    "scale_L",
    "agitation_rpm",
    "airflow_L_per_min",
    "ph_setpoint",
    "temperature_C",
    "titer_g_per_L",
]


def _load_committed() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def _import_make_synthetic():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_synthetic_scaleup_make_synthetic", FIXTURE_DIR / "make_synthetic.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the generator itself ------------------------------------------------- #


def test_generator_is_deterministic_for_a_fixed_seed():
    make_synthetic = _import_make_synthetic()
    df1 = make_synthetic.make_synthetic_scaleup()
    df2 = make_synthetic.make_synthetic_scaleup()
    pd.testing.assert_frame_equal(df1, df2)


def test_committed_csv_matches_a_fresh_generation(tmp_path):
    """The generator is seeded; running it must reproduce the committed CSV
    byte-for-byte, not just approximately - anything else means the fixture
    on disk has drifted from the script that is supposed to produce it."""
    script = FIXTURE_DIR / "make_synthetic.py"
    out_dir = tmp_path / "synthetic_scaleup"
    out_dir.mkdir()
    # Copy the script into an isolated directory so it writes its output
    # there (it writes next to itself), not back over the real fixture.
    script_copy = out_dir / "make_synthetic.py"
    script_copy.write_text(script.read_text())
    runpy.run_path(str(script_copy), run_name="__main__")

    generated = (out_dir / "synthetic_scaleup.csv").read_text()
    committed = CSV_PATH.read_text()
    assert generated == committed


# --- shape the Scale-Up Readout depends on --------------------------------- #


def test_fixture_has_the_documented_columns():
    df = _load_committed()
    assert set(_REQUIRED_COLUMNS) <= set(df.columns)


def test_fixture_has_at_least_five_distinct_scales():
    df = _load_committed()
    assert df["scale_L"].nunique() >= 5


def test_every_scale_has_at_least_three_runs():
    df = _load_committed()
    counts = df.groupby("scale_L").size()
    assert (counts >= 3).all(), counts[counts < 3]


def test_third_smallest_scale_and_above_has_at_least_ten_rows():
    df = _load_committed()
    scales = sorted(df["scale_L"].unique())
    third_smallest = scales[2]
    assert len(df[df["scale_L"] >= third_smallest]) >= 10


def test_no_non_finite_values():
    df = _load_committed()
    numeric = df.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()
