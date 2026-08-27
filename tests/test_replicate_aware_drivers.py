"""The driver panel's unit of analysis is the recipe, not the row.

WHY THIS FILE EXISTS. `test_driver_multiplicity.py` locked in Benjamini-Hochberg
control of the driver panel and measured it on sheets of INDEPENDENT rows: 78.8%
of reports on 30 pure-noise features contained a false "significant" driver
uncorrected, 3.5% under BH.

Media DoE sheets are not independent rows. The same recipe is run three or four
times because the ASSAY is noisy, and the engine says so everywhere else - the CV
groups by recipe, the noise floor is estimated within recipe, the proposal is fit
on the replicate-averaged objective. The driver panel was the one place that
still counted rows. Three wells of one recipe are one recipe's worth of evidence
about the process plus three draws of assay noise, so testing rows computes every
p-value against an `n` the sheet does not have, and BH stops correcting anything.

Measured on 20 recipes x 3 replicates, 30 noise features, 300 reports:

    60 independent rows          7.3% of reports had a "significant" driver
    20 recipes x 3 replicates   87.0%
    ... after averaging first     8.0%

A cluster bootstrap alone does NOT fix it (82.5%): the row-level p-values are
what BH reads, so the aggregation has to happen before the test, not around it.

The tests below assert both halves of the contract - noise stays quiet on a
replicated sheet, and a real driver is still found rather than muted - plus the
no-op property that keeps unreplicated uploads byte-for-byte unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from kalos.portal.analysis import _analyze  # noqa: E402

N_RECIPES = 20
N_REPS = 3
ASSAY_SD = 0.5


def _replicated_sheet(
    *, n_noise: int = 12, real_driver: bool = False, seed: int = 7
) -> pd.DataFrame:
    """`N_RECIPES` recipes, each run `N_REPS` times.

    Every replicate of a recipe carries the SAME feature vector and a fresh draw
    of assay noise on the target - the structure a real replicated run sheet has.
    With `real_driver=False` the target is drawn independently of every column, so
    any driver the panel calls significant is false by construction.
    """
    rng = np.random.default_rng(seed)
    names = [f"Component_{i}_mM" for i in range(n_noise)]
    Z = rng.uniform(1.0, 100.0, size=(N_RECIPES, n_noise))
    if real_driver:
        # a strong, unambiguous effect: rho ~ 1 at the recipe level
        y_recipe = 0.08 * Z[:, 0] + rng.normal(0.0, 0.2, N_RECIPES)
    else:
        y_recipe = rng.uniform(1.0, 10.0, N_RECIPES)

    rows: list[dict[str, float]] = []
    for k in range(N_RECIPES):
        for _ in range(N_REPS):
            row = dict(zip(names, Z[k]))
            row["Titer_g_L"] = float(y_recipe[k] + rng.normal(0.0, ASSAY_SD))
            rows.append(row)
    return pd.DataFrame(rows)


def _unreplicated_sheet(*, n_noise: int = 12, seed: int = 7) -> pd.DataFrame:
    """The same shape and size, but every row a distinct recipe."""
    rng = np.random.default_rng(seed)
    n = N_RECIPES * N_REPS
    cols = {f"Component_{i}_mM": rng.uniform(1.0, 100.0, n) for i in range(n_noise)}
    cols["Titer_g_L"] = rng.uniform(1.0, 10.0, n)
    return pd.DataFrame(cols)


# --- noise stays quiet ------------------------------------------------------ #


def test_replicated_noise_sheet_reports_no_significant_driver():
    """The canonical reproduction. Before the fix this exact sheet reported three
    significant drivers - p=0.0006, p=0.0034, p=0.0036 - for a target drawn
    independently of every column."""
    out = _analyze(_replicated_sheet())
    significant = [d["name"] for d in out["drivers"] if d["significant"]]
    assert significant == []


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_replicated_noise_stays_quiet_across_seeds(seed: int):
    """One quiet seed could be luck. The false-positive rate is what changed, so
    check it does not come back on a different draw."""
    out = _analyze(_replicated_sheet(seed=seed))
    assert [d["name"] for d in out["drivers"] if d["significant"]] == []


def test_replication_no_longer_inflates_the_evidence():
    """The mechanism, asserted directly: duplicating each recipe must not shrink
    the p-values, because it adds no information about the process.

    Row-level testing had exactly the opposite behaviour - that is what let BH
    through."""
    base = _replicated_sheet()
    doubled = pd.concat([base, base], ignore_index=True)  # 6 wells per recipe now
    p_base = {d["name"]: d["p"] for d in _analyze(base)["drivers"]}
    p_doubled = {d["name"]: d["p"] for d in _analyze(doubled)["drivers"]}
    shared = set(p_base) & set(p_doubled)
    assert shared, "expected the same features to be reported on both sheets"
    for name in shared:
        assert p_doubled[name] == pytest.approx(p_base[name], abs=1e-9)


# --- real signal still gets through ----------------------------------------- #


def test_a_real_driver_on_a_replicated_sheet_is_still_significant():
    """The fix must not work by muting the panel. A genuine strong driver has to
    survive averaging - it is the one thing averaging makes CLEARER, since the
    assay noise is what gets removed."""
    out = _analyze(_replicated_sheet(real_driver=True))
    significant = [d["name"] for d in out["drivers"] if d["significant"]]
    assert significant == ["Component_0_mM"]


# --- disclosure and the no-op property -------------------------------------- #


def test_the_panel_discloses_its_unit_of_analysis():
    """A client reading `n_tested` alone cannot tell how much independent
    evidence stood behind these p-values. The unit and the count are stated."""
    out = _analyze(_replicated_sheet())
    sel = out["driver_selection"]
    assert sel["unit"] == "recipe"
    assert sel["n_rows"] == N_RECIPES * N_REPS
    assert sel["n_units"] == N_RECIPES
    assert sel["n_units"] < sel["n_rows"]


def test_an_unreplicated_sheet_is_unchanged():
    """Every group is a singleton, so aggregation is an exact no-op and the panel
    behaves as it always did. This is what makes the fix safe to ship."""
    out = _analyze(_unreplicated_sheet())
    sel = out["driver_selection"]
    assert sel["n_units"] == sel["n_rows"] == N_RECIPES * N_REPS


def test_the_unit_count_matches_the_cv_group_count_when_recipes_are_the_barrier():
    """With no declared group column the recipe IS the CV group, so the two
    counts describe the same partition and must agree. If they ever diverge, one
    of the two grouping paths has drifted."""
    out = _analyze(_replicated_sheet())
    assert out["driver_selection"]["n_units"] == out["cv_n_groups"] == N_RECIPES
