"""One definition of "same recipe", and it is not the zero-filled matrix.

`_analyze` keeps two frames: `Xc_raw` with NaN preserved, and `Xc_zf` with blanks
filled to 0.0 for the GP and the design box. The CV grouper deliberately runs on
the RAW frame, with a comment saying why - rows missing different components must
not be merged into one group by the zero-fill.

The replicate machinery did not get that treatment. `noise_report` and
`aggregate_replicates` were handed the zero-filled `X` and identified replicates
by comparing rounded feature vectors, so a component that was BLANK looked
identical to a component genuinely dosed at zero. On a sparse media sheet - which
is what a media DoE is, since most formulations omit most candidate components -
distinct recipes were merged into one replicate group. That fabricates
within-recipe variance, which inflates the estimated assay-noise floor and
deflates the reported ICC.

That matters beyond a wrong field: `BENCHMARK.md` reasons from ICC ~= 0.26 to the
conclusion that assay noise, not the optimizer, is the limiting factor. An
inflated noise floor is exactly the artifact that would manufacture that number.

The fix threads an explicit recipe key, built from the raw frame, into both
functions. Note the key is kept SEPARATE from the CV group: a declared group
column is a leakage barrier and is deliberately coarser than a recipe (every run
on one medium lot shares a fold, but those runs are not replicates of each other),
so reusing it as a recipe key would collapse different recipes and reintroduce the
same fabricated variance from the other direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.core.replicates import aggregate_replicates, noise_report

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from kalos.portal.analysis import _analyze  # noqa: E402


# --- the primitive: an explicit key beats re-deriving one from values -------- #


def test_blank_and_zero_are_one_group_without_a_key():
    """Baseline, documenting the old behaviour so the fix is legible.

    Two DIFFERENT recipes - one that omits component B, one that doses it at
    zero - are indistinguishable once the blank has been filled with 0.0.
    """
    X_zero_filled = np.array([[1.0, 0.0], [1.0, 0.0]])
    _, y_mean, _, n_reps = aggregate_replicates(X_zero_filled, np.array([10.0, 20.0]))
    assert len(n_reps) == 1, "value-based grouping cannot tell them apart"
    assert n_reps[0] == 2
    # and it invents a within-group variance from two unrelated recipes
    assert y_mean[0] == pytest.approx(15.0)


def test_an_explicit_key_keeps_distinct_recipes_apart():
    X_zero_filled = np.array([[1.0, 0.0], [1.0, 0.0]])
    keys = np.array(["recipe-omits-B", "recipe-doses-B-at-zero"])
    X_u, y_mean, y_var, n_reps = aggregate_replicates(
        X_zero_filled, np.array([10.0, 20.0]), groups=keys
    )
    assert len(n_reps) == 2
    assert n_reps.tolist() == [1, 1]
    # singletons carry no variance, so nothing is fabricated
    assert y_var.tolist() == [0.0, 0.0]
    assert sorted(y_mean.tolist()) == [10.0, 20.0]
    assert X_u.shape == (2, 2)


def test_an_explicit_key_still_folds_real_replicates():
    keys = np.array(["r1", "r1", "r2"])
    _, y_mean, _, n_reps = aggregate_replicates(
        np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]),
        np.array([10.0, 12.0, 50.0]),
        groups=keys,
    )
    assert n_reps.tolist() == [2, 1]
    assert y_mean[0] == pytest.approx(11.0)


def test_noise_report_forwards_the_key():
    """The noise floor and the ICC are what the key actually protects."""
    X = np.array([[1.0, 0.0]] * 4)
    y = np.array([10.0, 20.0, 30.0, 40.0])
    merged = noise_report(X, y)
    split = noise_report(X, y, groups=np.array(["a", "b", "c", "d"]))
    assert merged["n_recipes"] == 1
    assert split["n_recipes"] == 4
    # Merging four unrelated recipes manufactures a large within-recipe variance;
    # with the key, nothing is replicated so there is no noise estimate at all.
    assert merged["noise_var"] > 0
    assert split["n_replicated"] == 0
    assert np.isnan(split["noise_var"])


def test_groups_length_is_validated():
    with pytest.raises(ValueError, match="one entry per row"):
        aggregate_replicates(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]), groups=np.array(["a"]))


# --- the live path: a sparse media-style sheet ------------------------------- #


def _sparse_media_sheet(n: int = 12) -> pd.DataFrame:
    """A media sheet where half the formulations OMIT a candidate component.

    Every row is a genuinely distinct recipe - glucose is unique per row - and
    `Yeast_extract_g_L` is blank on even rows. So nothing here is a replicate,
    and any replicate the engine reports is manufactured by the zero-fill.
    """
    rows = []
    for i in range(n):
        rows.append(
            {
                "Glucose_g_L": 10.0 + 0.5 * i,  # unique per row
                "Yeast_extract_g_L": None if i % 2 == 0 else 2.0 + 0.25 * i,
                "Feed_rate_mL_h": 0.2 + 0.01 * i,
                "lipase_titer": 1.0 + 0.1 * i,
            }
        )
    return pd.DataFrame(rows)


def _blank_versus_dosed_zero_sheet() -> pd.DataFrame:
    """The exact collision the fix exists for, isolated.

    Two recipes share every other component. One OMITS the supplement (blank);
    the other deliberately DOSES IT AT ZERO. Chemically and in the raw sheet those
    are different recipes. After `Xc_raw.fillna(0.0)` they are byte-identical, so
    value-based grouping merges them and reports a within-recipe variance built
    from two unrelated titers.
    """
    rows = []
    # Four recipes with a REAL dose, so the supplement column varies and is not
    # dropped as constant. Without these the column is all-zeros after the fill,
    # correctly removed as zero-variance, and the collision never gets a chance to
    # happen - which is what made an earlier version of this fixture degenerate.
    for i in range(4):
        rows.append(
            {"Glucose_g_L": 30.0 + i, "Supplement_g_L": 5.0 + i, "lipase_titer": 8.0 + 0.1 * i}
        )
    # Three collision pairs: same glucose, one OMITS the supplement and one doses
    # it at zero. Identical after fillna(0.0), different in the raw sheet.
    for i in range(3):
        g = 10.0 + i
        rows.append({"Glucose_g_L": g, "Supplement_g_L": None, "lipase_titer": 1.0 + 0.1 * i})
        rows.append({"Glucose_g_L": g, "Supplement_g_L": 0.0, "lipase_titer": 5.0 + 0.1 * i})
    return pd.DataFrame(rows)


def test_sparse_sheet_does_not_fabricate_replicates():
    """The whole point: blanks must not become replicate structure."""
    out = _analyze(_sparse_media_sheet(), target="lipase_titer")
    noise = out["noise"]
    # No two rows here are the same recipe, so nothing may be reported as
    # replicated and no assay-noise floor may be claimed from this sheet.
    assert noise["n_replicated"] == 0, (
        "blanks were merged into replicate groups, which fabricates within-recipe "
        "variance and inflates the noise floor"
    )
    assert noise["icc"] is None or np.isnan(noise["icc"] or float("nan"))
    assert noise["replicate_aware"] is False


def test_a_blank_is_not_the_same_recipe_as_a_deliberate_zero():
    """The precise collision the fix exists for.

    Omitting a supplement and dosing it at zero are different recipes, and only
    the raw frame can tell them apart. Merging them fabricates a within-recipe
    variance from titers that differ by 4 g/L, which is exactly how an assay-noise
    floor gets inflated and an ICC gets deflated.
    """
    noise = _analyze(_blank_versus_dosed_zero_sheet(), target="lipase_titer")["noise"]
    # 4 dosed + 3 blank + 3 dosed-at-zero = 10 distinct recipes, none replicated.
    # Under the old value-based grouping the 3 pairs merged, giving 7 recipes and
    # 3 "replicates" whose variance came from titers 4 g/L apart.
    assert noise["n_recipes"] == 10, "blank and dosed-zero were merged into one recipe"
    assert noise["n_replicated"] == 0
    assert noise["noise_sd"] is None


def test_genuine_replicates_are_still_detected_on_a_sparse_sheet():
    """The fix must not break real replicate detection when blanks are present."""
    df = _sparse_media_sheet()
    doubled = pd.concat([df, df], ignore_index=True)  # every recipe run twice
    noise = _analyze(doubled, target="lipase_titer")["noise"]
    assert noise["n_recipes"] == 12
    assert noise["n_replicated"] == 12
    assert noise["noise_sd"] is not None


def test_declared_group_column_is_not_used_as_the_recipe_key():
    """A group column is a leakage barrier, deliberately coarser than a recipe.

    Every row here shares one medium lot, so a single CV group is correct - but
    the twelve rows are twelve different recipes, and treating the lot as the
    recipe key would report them as twelve replicates of one and fabricate a huge
    noise floor.
    """
    df = _sparse_media_sheet()
    df["medium"] = "LOT-A"  # matches BIOPROCESS_PROFILE's group hint
    noise = _analyze(df, target="lipase_titer")["noise"]
    assert noise["n_recipes"] == 12, "the group column must not collapse recipes"
    assert noise["n_replicated"] == 0
