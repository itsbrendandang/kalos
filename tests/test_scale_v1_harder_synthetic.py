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

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kalos.scale.candidates import MultiFidelitySurrogate, PhysicsMeanSurrogate
from kalos.scale.evaluation import leave_one_scale_out_report

_RNG_SEED = 20260825
_SCALES = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 5000.0)
_N_PER_SCALE = 10
_PROCESS_COLUMNS = ["ph_setpoint", "temperature_C"]
_TARGET_COLUMN = "titer_g_per_L"

# The optimal pH shifts linearly (in log10-scale-normalized units) from 6.6
# at the smallest scale to 7.4 at the largest - a swing wide enough, against
# a ph_setpoint sampling range of [6.4, 7.6], to flip which end of the range
# wins between the two extremes.
_OPTIMAL_PH_AT_SMALLEST_SCALE = 6.6
_OPTIMAL_PH_AT_LARGEST_SCALE = 7.4


def fabricate_harder_synthetic(
    scales: tuple[float, ...] = _SCALES, n_per_scale: int = _N_PER_SCALE, seed: int = _RNG_SEED
) -> pd.DataFrame:
    """Fabricate the dataset described in this module's docstring.
    Deterministic for a fixed `(scales, n_per_scale, seed)`."""
    rng = np.random.default_rng(seed)
    log_scales = np.log10(np.asarray(scales, dtype=float))
    lmin, lmax = log_scales.min(), log_scales.max()

    rows = []
    for s, log_s in zip(scales, log_scales):
        l_norm = (log_s - lmin) / (lmax - lmin)
        optimal_ph = _OPTIMAL_PH_AT_SMALLEST_SCALE + (
            _OPTIMAL_PH_AT_LARGEST_SCALE - _OPTIMAL_PH_AT_SMALLEST_SCALE
        ) * l_norm
        for _ in range(n_per_scale):
            ph = float(rng.uniform(6.4, 7.6))
            temp = float(rng.uniform(36.0, 38.0))  # nuisance feature, no interaction
            rpm = float(rng.uniform(80, 250))
            airflow = float(max(s * 0.05, 0.05) * rng.uniform(0.8, 1.2))
            # Genuine recipe x scale interaction: titer peaks near the
            # scale-dependent optimal pH (a quadratic penalty for being off
            # it), plus a mild overall scale trend and observation noise.
            titer = 6.0 - 4.0 * (ph - optimal_ph) ** 2 + 0.3 * l_norm + float(rng.normal(0, 0.15))
            rows.append(
                {
                    "scale_L": s,
                    "agitation_rpm": rpm,
                    "airflow_L_per_min": airflow,
                    "ph_setpoint": ph,
                    "temperature_C": temp,
                    "titer_g_per_L": titer,
                }
            )
    return pd.DataFrame(rows)


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
