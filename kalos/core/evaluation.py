"""Honest evaluation: grouped cross-validation of the surrogate.

All grouped CV routes through the single leakage-checked splitter in `splits.py`
(no second, weaker splitter lives here anymore). Each fold is fit under the SAME
input-normalization box the deployed model uses (pass `bounds`), out-of-fold
predictions are POOLED, and the rank correlation is reported with a group-level
bootstrap confidence band. At a few dozen rows a bare point estimate reads as far
more precise than it is, so `grouped_cv_report` is the number to quote.
"""
from __future__ import annotations

from typing import Iterator, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from .splits import make_splits, row_hash_groups
from .surrogate import Surrogate


def grouped_folds(groups: Sequence, n_splits: int = 5) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Leakage-checked grouped folds. Kept for compatibility; delegates to the
    one splitter in `splits.make_splits`, so there is a single grouping code path.
    Yields nothing when there are too few groups for an honest split (the old
    round-robin version returned a train==validation dummy)."""
    groups = list(groups)
    n = len(groups)
    yield from make_splits(np.zeros((n, 1)), np.zeros(n), groups, n_splits=n_splits)


def _oof(X, y, groups, n_splits, bounds, cat_dims=None, shuffle: bool = False, random_state: int = 42):
    """Pooled out-of-fold predictions under one fixed normalization box.
    Returns (pred, std, actual, group, n_folds).

    `shuffle`/`random_state` are threaded straight through to `make_splits`.
    Defaults (`shuffle=False`, `random_state=42`) reproduce today's single
    fixed partition exactly; `grouped_cv_report`'s repeated-CV path is the only
    caller that passes `shuffle=True`, one distinct `random_state` per repeat.

    The posterior STANDARD DEVIATION is kept alongside the mean, because it is
    the only thing that makes the surrogate's uncertainty falsifiable: with
    held-out mean, sd, and truth in hand, `interval_calibration` can ask whether
    a stated 90% band actually covers 90%. Discarding it was why calibration had
    to be reported as unmeasured.

    It is the PREDICTIVE sd (`observation_noise=True`) - latent uncertainty plus
    fitted assay noise - not the latent band the acquisition uses. The held-out
    value it will be compared against is a measurement, and a measurement carries
    assay noise; scoring one against the latent band under-covers by
    construction, which would report every noisy assay as a wildly overconfident
    model.

    `cat_dims` (optional) marks categorical columns so each fold is fit with the
    same mixed GP the deployed model uses; omitted, every fold is the continuous
    SingleTaskGP (unchanged behavior)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    if groups is None:
        groups = row_hash_groups(X)
    groups = np.asarray(groups)
    # One normalization box for every fold AND for production. Fitting each fold
    # to its own training-data envelope (the old default) measures a different
    # input transform than the shipped model, so the CV number would not describe
    # what actually ships.
    if bounds is None and len(X):
        bounds = np.vstack([X.min(axis=0), X.max(axis=0)])
    pred, sd, actual, grp, n_folds = [], [], [], [], 0
    for tr, te in make_splits(X, y, groups, n_splits=n_splits, shuffle=shuffle, random_state=random_state):
        if len(tr) < 4 or len(te) < 1:
            continue
        n_folds += 1
        s = Surrogate().fit(X[tr], y[tr], bounds=bounds, cat_dims=cat_dims)
        mean, std = s.posterior(X[te], observation_noise=True)
        pred.extend(np.asarray(mean).ravel().tolist())
        sd.extend(np.asarray(std).ravel().tolist())
        actual.extend(y[te].tolist())
        grp.extend(groups[te].tolist())
    return (
        np.asarray(pred),
        np.asarray(sd),
        np.asarray(actual),
        np.asarray(grp, dtype=object),
        n_folds,
    )


def _spearman(pred, actual) -> float:
    if len(pred) < 3 or np.std(pred) == 0 or np.std(actual) == 0:
        return float("nan")
    r = spearmanr(pred, actual).statistic
    return float(r) if r == r else float("nan")


def grouped_cv_spearman(X, y, groups=None, n_splits: int = 5, bounds=None) -> float:
    """Pooled out-of-fold Spearman of the surrogate, leakage-controlled by group.
    Pools OOF predictions instead of averaging tiny per-fold rhos (which can only
    be +/-1 at this n), so the number reflects held-out ranking on real scale."""
    pred, _sd, actual, _, _ = _oof(X, y, groups, n_splits, bounds)
    return _spearman(pred, actual)


def grouped_cv_report(
    X, y, groups=None, n_splits: int = 5, bounds=None, n_boot: int = 1000, random_state: int = 0,
    cat_dims=None, n_repeats: int = 1,
) -> dict:
    """The honest CV number: pooled OOF Spearman plus a group-level bootstrap 95%
    CI, the count of held-out points and groups, and the fold count. The CI is
    wide on purpose at small n - relative ranking is more trustworthy than the
    absolute value.

    `cat_dims` (optional) marks categorical columns so the CV evaluates the same
    mixed GP the deployed model uses; omitted, behavior is unchanged.

    REPEATED CV (`n_repeats`, default 1 = today's behavior, unchanged). A single
    grouped CV partition is one fixed assignment of groups to folds; on a
    continuous target `make_splits` takes the unshuffled `GroupKFold` branch, so
    that partition is fully determined by the group labels. The group-level
    bootstrap below resamples GROUPS within that ONE partition, so it only ever
    covers group-resampling variance - it cannot see how much the estimate would
    move under a different, equally valid partition. On the real media DoE the
    pooled Spearman moved 0.44 to 0.71 across n_splits 3 to 8: partition variance
    was the dominant source of uncertainty, and a CI computed on one partition
    understated it.

    With `n_repeats > 1`, repeat 0 uses the SAME unshuffled partition as today
    (`shuffle=False`), so the POINT ESTIMATE (`spearman`, and every OOF array
    this function returns - `oof_actual`/`oof_pred`/`oof_std`) stays anchored to
    that one partition, exactly reproducing today's number. Repeats 1..n-1 each
    draw a DIFFERENT partition via `make_splits(shuffle=True, random_state=...)`,
    one distinct seed per repeat (`random_state + r`, deterministic). The
    group-level bootstrap runs independently within EACH repeat's own pooled
    OOF, and `ci95` is the 2.5/97.5 percentile of the UNION of every repeat's
    bootstrap draws - so the reported band now covers both group-resampling
    variance (within a partition) and partition variance (across partitions),
    instead of only the first. `spearman_per_repeat` carries the per-repeat
    pooled Spearman so the spread is visible on its own, not just folded into
    the CI width, and `n_repeats` reports how many repeats actually ran.

    WHY oof_actual/oof_pred/oof_std stay repeat-0-only rather than pooling
    across repeats: those arrays are ONE prediction per held-out row, consumed
    downstream (conformal quantile, calibration, the producer-only ranking) as
    if the rows were independent draws. Pooling every repeat's OOF would put
    each row in the response multiple times, once per repeat, with correlated
    (same GP, overlapping training folds) errors - inflating the apparent
    sample size for those consumers without adding independent evidence, and
    silently changing what `oof_actual` means for any caller that assumed one
    row in, one row out. Anchoring on repeat 0 keeps that contract intact and
    keeps every OOF-derived number in this module traceable to one partition.

    TOO FEW GROUPS TO REPEAT. When every group already gets its own fold
    (`n_splits` reduced to `n_groups`, i.e. leave-one-group-out), there is only
    ONE partition: which groups are held out together does not depend on fold
    order, so shuffling the fold *labels* cannot produce a different train/val
    split. Requesting `n_repeats > 1` there would not add real evidence - it
    would just duplicate one draw and dilute the bootstrap toward whatever
    noise a single repeat happened to produce, silently narrowing the CI rather
    than widening it. So repeats collapse to 1 whenever that degeneracy is
    detected, and the returned `n_repeats` reports the EFFECTIVE count (1), not
    the requested one."""
    X_arr = np.asarray(X, float)
    if groups is None:
        groups = row_hash_groups(pd.DataFrame(X_arr))
    groups = np.asarray(groups)

    # Mirror make_splits' own "not enough groups" reduction so this degeneracy
    # check agrees with what make_splits will actually do, rather than carrying
    # a second copy of its threshold that could drift out of sync.
    g_str = np.asarray(pd.Series(groups).astype(str).values) if len(groups) else groups
    n_groups_total = len(np.unique(g_str)) if len(g_str) else 0
    effective_splits = min(int(n_splits), n_groups_total) if n_groups_total else int(n_splits)
    degenerate = n_groups_total > 0 and effective_splits >= n_groups_total
    n_repeats_eff = 1 if degenerate else max(1, int(n_repeats))

    preds: list[np.ndarray] = []
    sds: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    grps: list[np.ndarray] = []
    n_folds0 = 0
    rng = np.random.default_rng(random_state)
    boot: list[float] = []
    for r in range(n_repeats_eff):
        if r == 0:
            # Unshuffled: the exact partition every existing caller already
            # gets, so the point estimate and OOF arrays are unchanged.
            pred, sd, actual, grp, n_folds = _oof(X_arr, y, groups, n_splits, bounds, cat_dims=cat_dims)
            n_folds0 = n_folds
        else:
            # A distinct deterministic seed per repeat: simple integer offset
            # from the caller's random_state, not cryptographic separation -
            # all that matters is each repeat draws a different shuffled
            # GroupKFold partition, reproducibly.
            seed_r = (int(random_state) + r) & 0x7FFFFFFF
            pred, sd, actual, grp, _n_folds = _oof(
                X_arr, y, groups, n_splits, bounds, cat_dims=cat_dims,
                shuffle=True, random_state=seed_r,
            )
        preds.append(pred)
        sds.append(sd)
        actuals.append(actual)
        grps.append(grp)

        uniq = np.unique(grp)
        if len(uniq) >= 3 and len(pred) >= 3:
            for _ in range(int(n_boot)):
                take = rng.choice(uniq, size=len(uniq), replace=True)
                mask = np.concatenate([np.where(grp == g)[0] for g in take])
                rr = _spearman(pred[mask], actual[mask])
                if rr == rr:
                    boot.append(rr)

    point = _spearman(preds[0], actuals[0])
    spearman_per_repeat = [_spearman(p, a) for p, a in zip(preds, actuals)]
    lo = hi = float("nan")
    if boot:
        lo = float(np.percentile(boot, 2.5))
        hi = float(np.percentile(boot, 97.5))
    return {
        "spearman": point,
        "ci95": (lo, hi),
        "n_oof": int(len(preds[0])),
        "n_groups": int(len(np.unique(grps[0]))),
        "n_folds": int(n_folds0),
        "oof_actual": actuals[0].tolist(),
        "oof_pred": preds[0].tolist(),
        # Held-out posterior sd, so a caller can check whether the model's stated
        # uncertainty is honest (`interval_calibration`) instead of assuming it.
        "oof_std": sds[0].tolist(),
        "n_repeats": int(n_repeats_eff),
        # Per-repeat pooled Spearman (repeat 0 first, matching `spearman`), so
        # the spread across partitions is visible on its own, not only folded
        # into the CI width. May contain NaN, same convention as `spearman`.
        "spearman_per_repeat": spearman_per_repeat,
    }


def producer_only_spearman(
    oof_actual, oof_pred, *, threshold: float = 0.0
) -> dict:
    """Rank agreement restricted to the rows that actually PRODUCED.

    WHY THE POOLED NUMBER IS NOT ENOUGH. `grouped_cv_report`'s Spearman is taken
    over every held-out row, producers and non-producers together. On a sheet with
    a substantial fraction of non-producers, a model can score well on that number
    by doing nothing more than separating zeros from non-zeros - which is a
    feasibility classifier, not a ranking of recipes. The client's decision is
    "which of my producing recipes is best", and that is a different question.

    `docs/BENCHMARK.md` documents this concretely: on the real media DoE the pooled
    held-out Spearman was a plausible-looking 0.37-0.52 while feasibility was
    "mostly a stable property" and never the bottleneck, so most of that pooled
    agreement was the easy part of the problem.

    `threshold` is the value above which a measurement counts as production. The
    default 0.0 means "any non-zero measurement", which is a PROXY. The correct
    value is the assay's limit of detection: below LOD a reading is censored, not
    zero, and treating it as a true zero is a different modelling error. Pass the
    LOD when it is known and say so in the report.

    Returns `spearman` as `nan` when it cannot be computed - fewer than three
    producing rows, or no variation among them - rather than a number that looks
    like a measurement. `evaluable` says which case you are in.
    """
    actual = np.asarray(oof_actual, dtype=float).reshape(-1)
    pred = np.asarray(oof_pred, dtype=float).reshape(-1)
    if actual.shape != pred.shape:
        raise ValueError("oof_actual and oof_pred must be the same length")

    mask = np.isfinite(actual) & np.isfinite(pred) & (actual > threshold)
    n_prod = int(mask.sum())
    n_total = int(np.isfinite(actual).sum())
    rho = _spearman(pred[mask], actual[mask]) if n_prod >= 3 else float("nan")
    return {
        "spearman": rho,
        "n_producers": n_prod,
        "n_total": n_total,
        "threshold": float(threshold),
        # A pooled score can look fine while this is unmeasurable, so the caller
        # must be able to tell "no signal" from "not enough producers to ask".
        "evaluable": bool(n_prod >= 3 and rho == rho),
    }


CALIBRATION_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)
# The nominal central intervals coverage is checked at. Four levels, spanning
# from the middle of the distribution to its tail, so a model that is honest at
# 50% but overconfident at 95% cannot hide behind a single number.

MIN_CALIBRATION_N = 10
# Below this many held-out points, an empirical coverage rate is not a
# measurement: at n=6 the achievable rates are 0, 1/6, 2/6, ..., so the nearest
# attainable value to a nominal 0.90 is 0.833 and the "error" is an artifact of
# the grid. Declining is more honest than reporting that.


def interval_calibration(
    oof_actual, oof_pred, oof_std, *, levels: Sequence[float] = CALIBRATION_LEVELS
) -> dict:
    """Is the surrogate's stated uncertainty honest on held-out runs?

    A posterior mean that ranks well can still be badly calibrated: the ranking
    is right and every interval is half the width it should be. A scientist
    reading "predicted 4.2 +/- 0.3" acts on the 0.3, so an overconfident band is
    its own failure mode, independent of `spearman`.

    For each nominal central level (0.9 means the +/-1.645 sigma interval), this
    counts how often the held-out truth actually fell inside. Perfect
    calibration puts empirical coverage on the nominal for every level.

    Returns `ece` as the mean absolute gap between nominal and empirical
    coverage across `levels`. That is the interval analogue of a classifier's
    expected calibration error, on the same 0-to-1 scale `GatesConfig.max_ece`
    is written against, so the promotion gate can finally read a number here
    instead of blocking on an unmeasured metric.

    `z_std` is the standard deviation of the held-out z-scores
    `(actual - pred) / sd`, which says WHICH WAY a miscalibrated model is wrong
    in one number: 1.0 is calibrated, above 1.0 means the bands are too narrow
    (overconfident), below 1.0 too wide.

    ASSUMPTION, stated because it is doing real work: coverage is computed from
    Gaussian quantiles of the posterior sd, so this measures whether the GP's
    own Gaussian band is honest. It is not distribution-free - that is what
    `kalos.core.conformal` is for, and the two answer different questions. Under
    grouped CV the held-out points are not iid either, so treat this as a
    diagnostic, not a guarantee.

    Never raises on data. Returns `available=False` with a reason - and `ece` /
    `z_std` as `None`, not NaN, because this dict is serialized into an API
    response and NaN is not valid JSON - when there is not enough held-out data,
    or no usable positive sd, to measure anything.
    """
    actual = np.asarray(oof_actual, dtype=float).reshape(-1)
    pred = np.asarray(oof_pred, dtype=float).reshape(-1)
    sd = np.asarray(oof_std, dtype=float).reshape(-1)
    if not (actual.shape == pred.shape == sd.shape):
        raise ValueError("oof_actual, oof_pred and oof_std must be the same length")

    usable = np.isfinite(actual) & np.isfinite(pred) & np.isfinite(sd) & (sd > 0)
    n = int(usable.sum())
    # `None`, never NaN: this dict is serialized straight into an API response,
    # and NaN is not valid JSON - `json.dumps` rejects it and the request 500s.
    # It also reads correctly, since there is no number to report.
    unavailable: dict = {
        "available": False,
        "n": n,
        "ece": None,
        "z_std": None,
        "levels": [],
    }
    if n < MIN_CALIBRATION_N:
        return {
            **unavailable,
            "reason": (
                f"only {n} held-out points with a positive posterior sd; "
                f"{MIN_CALIBRATION_N} are needed before an empirical coverage "
                "rate is a measurement rather than a rounding grid"
            ),
        }

    z = (actual[usable] - pred[usable]) / sd[usable]
    rows: list[dict] = []
    gaps: list[float] = []
    for level in levels:
        lv = float(level)
        if not 0.0 < lv < 1.0:
            raise ValueError("calibration levels must lie strictly between 0 and 1")
        half = float(norm.ppf(0.5 + lv / 2.0))
        empirical = float(np.mean(np.abs(z) <= half))
        gaps.append(abs(empirical - lv))
        rows.append({"nominal": round(lv, 4), "empirical": round(empirical, 4)})

    return {
        "available": True,
        "reason": None,
        "n": n,
        "ece": round(float(np.mean(gaps)), 4),
        "z_std": round(float(np.std(z, ddof=1)) if n > 1 else float("nan"), 4),
        "levels": rows,
    }


def logo_report(X, y, groups, *, bounds=None, cat_dims=None) -> dict:
    """Leave-one-group-out CV: every fold holds out exactly one group.

    Delegates to `_oof` with `n_splits` set to the number of distinct groups,
    so it reuses the ONE leakage-checked splitter (`make_splits`) rather than
    adding a second grouping code path that could drift out of sync with it.
    `GroupKFold(n_splits=n_groups)`, with `n_splits` exactly equal to the
    group count, IS leave-one-group-out: `make_splits`' own "too many splits"
    branch already reduces `n_splits` down to `n_groups` and documents that
    reduced case as "every group already gets its own fold" (see
    `grouped_cv_report`'s "TOO FEW GROUPS TO REPEAT" note) - calling it with
    that value directly, rather than arriving at it via reduction, is the same
    partition. Every group is disjoint from every other, there are exactly as
    many folds as groups, and GroupKFold never splits one group across folds,
    so by construction each fold holds out precisely one group.

    WHY THIS EARNS A PLACE ALONGSIDE THE K-FOLD NUMBER. Grouped K-fold with
    several groups per fold tests "can this model generalize to a FEW new
    groups sprinkled among many familiar ones" - an easier question than the
    one a client actually needs answered, which is "can this model generalize
    to the NEXT batch it has never seen anything like". LOGO is the harder,
    honest version of that question. On the owner's real clone-selection data
    (see the owner's earlier engine, `voyager-brain-rebuild`) the gap was not
    subtle: leave-one-strain-out Spearman was -0.12 - worse than chance -
    while a shuffled 5-fold read 0.91 on the SAME model and data. The 5-fold
    number was measuring how well the model memorizes strain identity, not
    how it predicts an unseen strain; only LOGO caught that.

    Returns the pooled out-of-fold Spearman (`spearman`, `nan` if
    unmeasurable - fewer than 3 pooled points or no variation, same
    convention as `_spearman` throughout this module), `n_groups` (how many
    folds, i.e. how many distinct groups), and `n_oof` (pooled held-out row
    count, which can be less than `len(y)` when a fold's training split ends
    up too small to fit - the same `len(tr) < 4` guard `_oof` applies to
    every grouped CV in this module).

    Never raises: fewer than 2 groups makes leave-ONE-out undefined (there
    would be nothing left to train on), so that case returns `nan`/`n_oof=0`
    rather than calling `_oof` at all.

    COST, stated because a caller reads this before deciding whether to run
    it: one held-out group means one Surrogate GP fit, so this costs
    `n_groups` GP fits - the same per-fit cost as every fold in
    `grouped_cv_report`, just as many folds as there are groups instead of a
    fixed `n_splits`. See `kalos.portal.analysis`'s `LOGO_MAX_GROUPS` for the
    cap this motivates on the analysis path.
    """
    X_arr = np.asarray(X, float)
    g_arr = np.asarray(groups)
    g_str = np.asarray(pd.Series(g_arr).astype(str).values) if len(g_arr) else g_arr
    n_groups = int(len(np.unique(g_str))) if len(g_str) else 0
    if n_groups < 2:
        return {"spearman": float("nan"), "n_groups": n_groups, "n_oof": 0}
    pred, _sd, actual, _grp, _n_folds = _oof(X_arr, y, g_arr, n_groups, bounds, cat_dims=cat_dims)
    return {
        "spearman": _spearman(pred, actual),
        "n_groups": n_groups,
        "n_oof": int(len(pred)),
    }


def top_k_overlap(oof_actual, oof_pred, k: int) -> dict:
    """|true top-k INTERSECT predicted top-k| / k, over the pooled OOF.

    WHY SPEARMAN CAN OBSCURE THIS. A rank correlation is computed over every
    held-out point, weighted equally from the top of the distribution to the
    bottom. The client's actual decision is narrower: "which N recipes do I
    advance to the next round". On a zero-inflated target (most of the
    response mass sitting at or near zero, docs/BENCHMARK.md's media DoE being the
    running example in this module) a model can post a respectable pooled
    Spearman by ranking the large flat region of near-zero rows correctly
    relative to each other, while getting the handful of rows that actually
    matter - the real top producers - out of order. Top-k overlap asks the
    narrower, more decision-relevant question directly instead of hoping
    Spearman is a good proxy for it.

    TIES. Ranking is done with `np.argsort(-values, kind="stable")`, so rows
    tied on value keep their ORIGINAL relative order (the order they appear
    in `oof_actual`/`oof_pred`) when deciding which side of the top-k cut they
    land on. This is a deterministic, reproducible tie-break - re-running on
    the same arrays always draws the same boundary - but it is arbitrary with
    respect to the tied values themselves; it is NOT "every row tied for the
    boundary value gets counted as top-k", which would let more than k rows
    into either set and make the k-in-the-denominator fraction meaningless.

    Returns `evaluable=False` with a `reason` - never a number that looks
    measured - when `k <= 0` or there are fewer than `k` finite, paired rows
    to rank: a "top-5 overlap" computed from 3 rows is not the measurement it
    claims to be.
    """
    actual = np.asarray(oof_actual, dtype=float).reshape(-1)
    pred = np.asarray(oof_pred, dtype=float).reshape(-1)
    if actual.shape != pred.shape:
        raise ValueError("oof_actual and oof_pred must be the same length")

    k_int = int(k)
    usable = np.isfinite(actual) & np.isfinite(pred)
    actual = actual[usable]
    pred = pred[usable]
    n = int(len(actual))

    if k_int <= 0:
        return {"overlap": float("nan"), "k": k_int, "n": n, "evaluable": False,
                "reason": "k must be positive"}
    if n < k_int:
        return {
            "overlap": float("nan"), "k": k_int, "n": n, "evaluable": False,
            "reason": f"only {n} evaluable held-out rows; need at least k={k_int}",
        }

    true_top = set(np.argsort(-actual, kind="stable")[:k_int].tolist())
    pred_top = set(np.argsort(-pred, kind="stable")[:k_int].tolist())
    overlap = len(true_top & pred_top) / k_int
    return {"overlap": float(overlap), "k": k_int, "n": n, "evaluable": True, "reason": None}


def group_mean_baseline_spearman(y, groups) -> dict:
    """How much of the apparent signal is just group identity?

    Predicts each row by the MEAN OF THE OTHER rows in its group (leave-one-
    row-out within the group, so a recipe's own measurement never predicts
    itself), then reports the Spearman of that prediction against the actual
    `y`. A model this simple - it does not look at a single process feature,
    only which recipe a row belongs to - sets the floor a real model has to
    clear to be adding anything. On the owner's real data a campaign-mean-only
    baseline of this shape captured about 0.75 of a trained model's ~0.86
    headline Spearman: most of the apparent skill was the model learning
    "which recipe is this", which a lookup table would also learn, not
    learning the PROCESS.

    Rows in SINGLETON groups (the only row for their recipe) are unevaluable:
    there is no "other row" to average, so leave-one-row-out has nothing to
    leave. They are excluded from both the prediction and the Spearman, not
    imputed with some other value that would silently change what the number
    measures.

    Never raises. Returns `evaluable=False` with a `reason` when fewer than 3
    rows have a usable within-group leave-one-out mean (the same floor
    `_spearman` uses everywhere else in this module - a rank correlation
    below n=3 is not a measurement), or when `y` and `groups` disagree in
    length (a caller error, but the contract here is still "never raises" -
    see `_unavailable`; this is the one function in this module where a
    length mismatch is reported rather than raised, so a shape bug in the
    caller does not propagate into an already-fail-closed pipeline as an
    exception the caller might not be catching).
    """
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    g_arr = np.asarray(pd.Series(groups).astype(str).values) if len(np.asarray(groups)) else np.asarray(groups)

    def _unavailable(reason: str, n_evaluable: int = 0) -> dict:
        return {
            "spearman": float("nan"),
            "n_evaluable": n_evaluable,
            "n_groups_used": 0,
            "evaluable": False,
            "reason": reason,
        }

    if y_arr.shape[0] != g_arr.shape[0]:
        return _unavailable("y and groups must be the same length")

    finite = np.isfinite(y_arr)
    y_arr = y_arr[finite]
    g_arr = g_arr[finite]

    df = pd.DataFrame({"y": y_arr, "g": g_arr})
    counts = df.groupby("g")["y"].transform("count")
    evaluable_mask = (counts >= 2).to_numpy()
    n_evaluable = int(evaluable_mask.sum())
    if n_evaluable < 3:
        return _unavailable(
            f"only {n_evaluable} rows belong to a group with >=2 members "
            "(singleton groups have no 'other row' to average); need >=3 to "
            "compute a rank correlation",
            n_evaluable,
        )

    sums = df.groupby("g")["y"].transform("sum")
    loo_mean = (sums - df["y"]) / (counts - 1)
    pred = loo_mean.to_numpy()[evaluable_mask]
    actual = y_arr[evaluable_mask]
    rho = _spearman(pred, actual)
    n_groups_used = int(df.loc[evaluable_mask, "g"].nunique())
    if rho != rho:
        return _unavailable(
            "the leave-one-row-out group means have no variation to correlate "
            "against y (or fewer than 3 evaluable rows survived)",
            n_evaluable,
        )
    return {
        "spearman": rho,
        "n_evaluable": n_evaluable,
        "n_groups_used": n_groups_used,
        "evaluable": True,
        "reason": None,
    }


__all__ = [
    "grouped_cv_spearman",
    "grouped_cv_report",
    "grouped_folds",
    "producer_only_spearman",
    "interval_calibration",
    "logo_report",
    "top_k_overlap",
    "group_mean_baseline_spearman",
    "CALIBRATION_LEVELS",
    "MIN_CALIBRATION_N",
]
