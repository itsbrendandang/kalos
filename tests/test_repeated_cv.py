"""Repeated grouped CV: the CI must cover PARTITION variance, not just resample
GROUPS within one fixed partition.

On a continuous target `make_splits` took the unshuffled `GroupKFold` branch,
so the split was fully determined by the group labels - one fixed partition,
every time. `grouped_cv_report`'s bootstrap resampled groups WITHIN that one
partition, so its CI only ever answered "how much would this number move if I
drew different groups from the same split?" It could not answer "how much
would this number move under a different, equally valid split?" - and
`splits.py`'s own docstring recorded the gap: on the real media DoE the pooled
Spearman moved 0.44 to 0.71 across n_splits 3 to 8, and the fix it prescribed
("quote the CI alongside a sensitivity sweep") was never wired up anywhere.

`make_splits(shuffle=True)` plus `grouped_cv_report(n_repeats=...)` is that
sweep, folded into the number the client actually reads. These tests assert
the properties that motivated it: the default path is untouched, repeats are
reproducible and genuinely different, leakage is still checked on every one of
them, the CI widens on data engineered to be partition-sensitive, and a sheet
with no room for a second partition does not fabricate variance it does not
have.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.core.splits import assert_no_group_leakage, make_splits

pytest.importorskip("botorch")

from kalos.core.evaluation import grouped_cv_report  # noqa: E402


# --- fixtures ----------------------------------------------------------- #


def _many_groups(n_groups: int = 20, reps: int = 2, seed: int = 0):
    """Plenty of groups relative to n_splits, so shuffling actually has room to
    produce a genuinely different partition (not the leave-one-group-out
    degenerate case, which is covered separately below)."""
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(0.0, 4.0, n_groups)
    y_mu = 1.3 * f0
    X, y, g = [], [], []
    for k in range(n_groups):
        for _ in range(reps):
            X.append([f0[k] + rng.normal(0.0, 0.05)])
            y.append(y_mu[k] + rng.normal(0.0, 0.3))
            g.append(k)
    return np.array(X), np.array(y), np.array(g)


def _partition_sensitive(seed: int = 3, n_normal: int = 14, n_flip: int = 4):
    """Engineered so different partitions genuinely disagree, not just resample
    noise: most groups follow y = 1.2*f + noise, but a handful of groups follow
    the OPPOSITE relationship (y = -1.2*f + offset + noise) - a hidden
    categorical effect the single continuous feature GP never sees. Held-out
    Spearman on any one partition depends heavily on how many "flip" groups
    that partition's folds happen to isolate together versus spread out, which
    is exactly the partition-to-partition variance a single-split CI cannot
    see."""
    rng = np.random.default_rng(seed)
    n = n_normal + n_flip
    f0 = np.linspace(0.0, 4.0, n)
    rng.shuffle(f0)
    is_flip = np.zeros(n, dtype=bool)
    idx = np.arange(n)
    rng.shuffle(idx)
    is_flip[idx[:n_flip]] = True
    y = np.empty(n)
    y[~is_flip] = 1.2 * f0[~is_flip] + rng.normal(0.0, 0.25, int((~is_flip).sum()))
    y[is_flip] = -1.2 * f0[is_flip] + 6.0 + rng.normal(0.0, 0.25, int(is_flip.sum()))
    X = f0.reshape(-1, 1)
    g = np.arange(n)  # one group per row: groups are the unit that gets shuffled
    return X, y, g


def _degenerate_groups(n_groups: int = 4, reps: int = 3, seed: int = 7):
    """Exactly one group per fold (n_splits == n_groups): leave-one-group-out.
    There is only ONE partition here - which groups are held out together does
    not depend on fold order, so no amount of shuffling can produce a second,
    genuinely different split."""
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(0.0, 4.0, n_groups)
    y_mu = 1.3 * f0
    X, y, g = [], [], []
    for k in range(n_groups):
        for _ in range(reps):
            X.append([f0[k] + rng.normal(0.0, 0.05)])
            y.append(y_mu[k] + rng.normal(0.0, 0.2))
            g.append(k)
    return np.array(X), np.array(y), np.array(g)


def _replicated_sheet(n_recipes: int = 16, reps: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(0.0, 4.0, n_recipes)
    f1 = rng.uniform(0.0, 4.0, n_recipes)
    mu = 1.5 * f0 - 0.8 * (f1 - 2.0) ** 2
    rows = []
    for k in range(n_recipes):
        for _ in range(reps):
            rows.append(
                {
                    "Methanol": round(float(f0[k]), 3),
                    "pH": round(float(f1[k]), 3),
                    "lipase_titer": float(mu[k] + rng.normal(0.0, 1.0)),
                }
            )
    return pd.DataFrame(rows)


# --- make_splits: shuffle is opt-in, default is untouched ---------------- #


def test_default_make_splits_ignores_random_state_on_the_continuous_branch():
    """The whole point of `shuffle=False` being the default: every existing
    caller, none of which passes `shuffle`, must see exactly today's fold
    assignment no matter what `random_state` a caller happens to be carrying
    around for something else (e.g. `grouped_cv_report`'s bootstrap seed)."""
    X, y, g = _many_groups()
    a = make_splits(X, y, g, n_splits=4)
    b = make_splits(X, y, g, n_splits=4, random_state=999)
    assert [(tr.tolist(), va.tolist()) for tr, va in a] == [
        (tr.tolist(), va.tolist()) for tr, va in b
    ]


def test_shuffle_true_requires_no_new_argument_to_be_leakage_checked():
    """`assert_no_group_leakage` must hold on a shuffled partition exactly as it
    does on the unshuffled one - shuffling changes WHICH groups land together,
    never whether a group straddles train/val."""
    X, y, g = _many_groups()
    splits = make_splits(X, y, g, n_splits=4, shuffle=True, random_state=1)
    assert_no_group_leakage(splits, g)  # must not raise


def test_shuffled_partitions_differ_across_seeds():
    X, y, g = _many_groups()
    a = make_splits(X, y, g, n_splits=4, shuffle=True, random_state=1)
    b = make_splits(X, y, g, n_splits=4, shuffle=True, random_state=2)
    a_sets = [(frozenset(tr.tolist()), frozenset(va.tolist())) for tr, va in a]
    b_sets = [(frozenset(tr.tolist()), frozenset(va.tolist())) for tr, va in b]
    assert set(a_sets) != set(b_sets)


def test_shuffle_true_is_itself_deterministic():
    X, y, g = _many_groups()
    a = make_splits(X, y, g, n_splits=4, shuffle=True, random_state=1)
    b = make_splits(X, y, g, n_splits=4, shuffle=True, random_state=1)
    assert [(tr.tolist(), va.tolist()) for tr, va in a] == [
        (tr.tolist(), va.tolist()) for tr, va in b
    ]


# --- grouped_cv_report: default path is byte-identical to before --------- #


def test_default_n_repeats_matches_a_bare_unshuffled_call():
    """`n_repeats=1` (the default) must reproduce exactly what calling the CV
    machinery directly, unshuffled, already gave - the same code path
    `grouped_cv_report` used before repeats existed."""
    X, y, g = _many_groups()
    bounds = np.vstack([X.min(0), X.max(0)])
    rep = grouped_cv_report(X, y, groups=g, n_splits=4, bounds=bounds, n_repeats=1, random_state=0)
    assert rep["n_repeats"] == 1
    assert rep["spearman_per_repeat"] == [rep["spearman"]]

    # The point estimate and every OOF array must be independent of
    # `random_state` when there is only one (unshuffled) repeat - that
    # parameter drives the bootstrap and, for r >= 1, the split shuffle, but
    # repeat 0 never shuffles, so it must ignore random_state entirely.
    rep_other_seed = grouped_cv_report(
        X, y, groups=g, n_splits=4, bounds=bounds, n_repeats=1, random_state=999,
    )
    assert rep["oof_pred"] == rep_other_seed["oof_pred"]
    assert rep["oof_actual"] == rep_other_seed["oof_actual"]
    assert rep["oof_std"] == rep_other_seed["oof_std"]
    assert rep["spearman"] == rep_other_seed["spearman"]


# --- grouped_cv_report: n_repeats > 1 is reproducible and genuinely repeats  #


def test_repeated_report_is_deterministic_across_two_calls():
    X, y, g = _partition_sensitive()
    bounds = np.vstack([X.min(0), X.max(0)])
    a = grouped_cv_report(X, y, groups=g, n_splits=3, bounds=bounds, n_repeats=3, random_state=0)
    b = grouped_cv_report(X, y, groups=g, n_splits=3, bounds=bounds, n_repeats=3, random_state=0)
    assert a == b


def test_repeats_draw_genuinely_different_partitions():
    """Not just a formality: if every repeat silently drew the same partition,
    `spearman_per_repeat` would be a list of one number copied N times."""
    X, y, g = _partition_sensitive()
    bounds = np.vstack([X.min(0), X.max(0)])
    rep = grouped_cv_report(X, y, groups=g, n_splits=3, bounds=bounds, n_repeats=3, random_state=0)
    assert rep["n_repeats"] == 3
    assert len(rep["spearman_per_repeat"]) == 3
    assert len(set(round(s, 6) for s in rep["spearman_per_repeat"])) > 1


# --- the CI actually widens on partition-sensitive data ------------------ #


def test_repeated_ci_is_at_least_as_wide_as_the_single_partition_ci():
    """The property that motivated this whole change. On data engineered so
    partitions disagree (see `_partition_sensitive`), a CI that only resamples
    groups within one fixed split cannot see that disagreement; pooling the
    bootstrap over multiple partitions must not be narrower than the
    single-partition band, and on this dataset it is measurably wider."""
    X, y, g = _partition_sensitive()
    bounds = np.vstack([X.min(0), X.max(0)])
    single = grouped_cv_report(X, y, groups=g, n_splits=3, bounds=bounds, n_repeats=1, random_state=0)
    repeated = grouped_cv_report(X, y, groups=g, n_splits=3, bounds=bounds, n_repeats=2, random_state=0)

    w_single = single["ci95"][1] - single["ci95"][0]
    w_repeated = repeated["ci95"][1] - repeated["ci95"][0]
    # >= rather than >, per the design contract: a strictly wider CI is not
    # mathematically guaranteed on every dataset (pooling more draws can in
    # principle land on about the same percentile), only that it cannot be
    # narrower than what one partition alone supports. On this engineered
    # dataset it IS strictly wider, which is the useful case to also check.
    assert w_repeated >= w_single
    assert w_repeated > w_single + 0.02, (
        f"expected a measurable widening on partition-sensitive data, got "
        f"single={w_single:.4f} repeated={w_repeated:.4f}"
    )


# --- _analyze surfaces the new fields and stays JSON-safe ----------------- #


def test_analyze_surfaces_repeat_fields_and_stays_json_safe():
    pytest.importorskip("fastapi")
    from kalos.portal.analysis import CV_N_REPEATS, _analyze

    res = _analyze(_replicated_sheet(), "lipase_titer")
    assert res["cv_n_repeats"] == CV_N_REPEATS
    assert isinstance(res["cv_spearman_per_repeat"], list)
    assert len(res["cv_spearman_per_repeat"]) == res["cv_n_repeats"]
    for s in res["cv_spearman_per_repeat"]:
        assert s is None or isinstance(s, float)
    # Strict JSON, no NaN - this dict is serialized straight into an API
    # response and `json.dumps` rejects NaN outright.
    json.dumps(res)


# --- too few groups to repeat: must not fabricate variance ---------------- #


def test_leave_one_group_out_collapses_repeats_instead_of_faking_variance():
    """4 groups, n_splits=4: every group is already its own fold, so there is
    only one possible partition. Asking for 5 repeats must not silently narrow
    (or otherwise move) the CI by diluting the bootstrap with duplicate draws
    of the same partition - it must collapse to the single-repeat result."""
    X, y, g = _degenerate_groups()
    bounds = np.vstack([X.min(0), X.max(0)])
    single = grouped_cv_report(X, y, groups=g, n_splits=4, bounds=bounds, n_repeats=1, random_state=0)
    requested = grouped_cv_report(X, y, groups=g, n_splits=4, bounds=bounds, n_repeats=5, random_state=0)

    assert requested["n_repeats"] == 1  # effective, not the requested 5
    assert requested["spearman_per_repeat"] == [requested["spearman"]]
    assert requested["oof_pred"] == single["oof_pred"]
    assert requested["oof_actual"] == single["oof_actual"]
    assert requested["ci95"] == single["ci95"]
