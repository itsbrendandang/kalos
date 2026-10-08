"""A synthetic dataset harder than v0's fixture and than the real
`mab-scaleup-synthetic` dataset on purpose: more scales, more replicates
per scale, and a PLANTED recipe-by-scale interaction strong enough that the
best `ph_setpoint` genuinely flips between the smallest and largest scale
(the real dataset's per-scale correlations swing too, but a pooled F-test
finds no significant interaction there - see `test_scale_v1_real_data.py` -
so the real data cannot answer "can any of these models learn a rank
crossing at all"). This dataset exists to answer that question directly:
if v0/candidates fail to extrapolate ranking here too, that would point at
the modeling approach; if they succeed here, the real dataset's -0.7 is
about a weak/absent true signal at small n, not an architectural blind
spot. See `kalos/scale/transfer.py`'s "V1 CANDIDATES" docstring and this
repo's v1 report for how this result was used in the promotion decision.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from kalos.scale.candidates import MultiFidelitySurrogate, PhysicsMeanSurrogate
from kalos.scale.evaluation import leave_one_scale_out_report

# The generator this module tests now lives in `examples/synthetic_scaleup`
# (promoted there so the Scale-Up Readout demo dataset and this test fixture
# are the same code, not two copies that can drift apart). It is loaded by
# file path, under a unique module name, rather than a plain `import
# make_synthetic` - `examples/synthetic_bioprocess` also ships a
# `make_synthetic.py`, and a bare import would collide with it in
# `sys.modules`.
_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "synthetic_scaleup" / "make_synthetic.py"
)
_spec = importlib.util.spec_from_file_location(
    "examples_synthetic_scaleup_make_synthetic", _EXAMPLE_PATH
)
assert _spec is not None and _spec.loader is not None
_make_synthetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_make_synthetic)

fabricate_harder_synthetic = _make_synthetic.make_synthetic_scaleup
_SCALES = _make_synthetic.SCALES
_N_PER_SCALE = _make_synthetic.N_PER_SCALE
_PROCESS_COLUMNS = _make_synthetic.PROCESS_COLUMNS
_TARGET_COLUMN = _make_synthetic.TARGET_COLUMN


def test_the_planted_interaction_is_a_genuine_rank_crossing():
    """Locks in the fixture's own premise: the sign of spearman(ph, titer)
    must differ between the smallest and largest scale - otherwise this
    would not be testing what its docstring claims."""
    df = fabricate_harder_synthetic()
    smallest, largest = min(_SCALES), max(_SCALES)
    r_small = spearmanr(
        df.loc[df["scale_L"] == smallest, "ph_setpoint"],
        df.loc[df["scale_L"] == smallest, "titer_g_per_L"],
    ).statistic
    r_large = spearmanr(
        df.loc[df["scale_L"] == largest, "ph_setpoint"],
        df.loc[df["scale_L"] == largest, "titer_g_per_L"],
    ).statistic
    assert r_small < -0.3, r_small
    assert r_large > 0.3, r_large


def test_all_three_models_recover_the_strong_rank_crossing_at_extrapolate_up():
    """The harder-synthetic counterpart to
    `test_scale_v1_real_data.test_no_candidate_extrapolate_up_spearman_clears_the_noise_floor`:
    here the interaction is real, strong, and well-powered (n=10/scale), and
    the expectation flips - v0 AND both candidates should recover a strongly
    positive extrapolate-up Spearman (measured ~0.86-0.89 on the run behind
    this repo's v1 report; asserting a loose >0.5 floor here, not the exact
    figure, for the platform-FP-drift reasons `test_scale_v1_real_data.py`'s
    module docstring explains). This is what rules out "the harness or these
    architectures simply cannot learn a rank crossing" as the explanation
    for v0's real-data result.
    """
    df = fabricate_harder_synthetic()
    factories = {
        "v0": None,
        "physics_mean": lambda names: PhysicsMeanSurrogate(names),
        "multi_fidelity": lambda names: MultiFidelitySurrogate(names),
    }
    for label, factory in factories.items():
        report = leave_one_scale_out_report(
            df, _TARGET_COLUMN, _PROCESS_COLUMNS, model_factory=factory, model_label=label
        )
        bucket = report["by_direction"]["extrapolate_up"]
        assert bucket["n"] == _N_PER_SCALE, label
        assert bucket["spearman"] > 0.5, (
            f"{label}: extrapolate-up spearman={bucket['spearman']} did not recover the "
            "planted rank crossing - would contradict the v1 report's finding that all "
            "three architectures CAN learn a genuine, well-powered scale interaction"
        )


def test_fabrication_is_deterministic_for_a_fixed_seed():
    df1 = fabricate_harder_synthetic(seed=7)
    df2 = fabricate_harder_synthetic(seed=7)
    pd.testing.assert_frame_equal(df1, df2)
