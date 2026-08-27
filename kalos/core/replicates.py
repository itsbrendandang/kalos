"""Replicate-aware aggregation and assay noise-floor estimation.

Media DoE run sheets are frequently heavily replicated: the same recipe
(feature row) is run several times because the assay itself is noisy, not
because the process is. Replicates are identified purely by feature identity —
two rows are the "same recipe" iff their feature vectors are equal after
rounding to `decimals` places. That rounding absorbs float round-trip noise
(e.g. a value written out and re-read from a spreadsheet) without merging
recipes that are genuinely different by design.

Aggregating replicates down to one row per recipe (mean titer) gives BO a
reproducible objective to optimize, and the within-replicate spread gives an
honest estimate of the assay noise floor (sigma^2) that a GP can be told about
explicitly instead of inferring from scratch on a handful of points.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
from scipy.stats import spearmanr

# Per-row recipe keys. An ndarray is included because `kalos.core.splits.
# row_hash_groups` - the canonical grouper callers should be passing - returns
# one; iterating either type yields the per-row key.
GroupKeys = Union[Sequence[object], np.ndarray]

MIN_REPLICATED_FOR_HETERO = 4
# Replicated recipes needed before variance-versus-mean is called a measurement.
# Four is a convention, not a derived bound: a Spearman correlation over three
# points takes only a handful of values and would read as confident nonsense.

HETERO_RHO_FLOOR = 0.4
# Spearman correlation between recipe mean and within-recipe variance at or above
# which the homoscedastic assumption is treated as violated. A convention chosen
# to sit well clear of the noise floor of a correlation estimated from a handful
# of recipes, so this flags a coupling strong enough to act on rather than every
# mild positive slope.

ICC_GAIN_FLOOR = 0.05
# How much log1p must raise the ICC before a transform is worth recommending.
# Five points of ICC is the smallest change that would alter how a scientist reads
# the signal-to-noise verdict; below that, changing the scale every reported
# number lives on is not worth it.

__all__ = ["aggregate_replicates", "estimate_noise_floor", "noise_report", "heteroscedasticity_report"]


def _group_indices_from_keys(keys: GroupKeys) -> list[np.ndarray]:
    """Group row indices by an EXPLICIT recipe key, in first-occurrence order.

    Preferred over `_group_indices` whenever the caller knows which rows are the
    same recipe, because deriving it from the feature matrix cannot distinguish a
    genuine zero from an absent value that was filled with zero. See
    `aggregate_replicates`'s `groups` argument for why that distinction changed
    a headline statistic.
    """
    seen: dict[object, list[int]] = {}
    order: list[object] = []
    for i, key in enumerate(keys):
        if key not in seen:
            seen[key] = []
            order.append(key)
        seen[key].append(i)
    return [np.asarray(seen[k], dtype=int) for k in order]


def _group_indices(X: np.ndarray, decimals: int) -> list[np.ndarray]:
    """Return index arrays grouping rows of `X` by equal rounded feature vectors.

    Groups are returned in first-occurrence order (the order each distinct
    rounded row first appears in `X`), so aggregation is deterministic.
    """
    Xr = np.round(np.asarray(X, dtype=float), decimals)
    n = Xr.shape[0]
    seen: dict[tuple[float, ...], list[int]] = {}
    order: list[tuple[float, ...]] = []
    for i in range(n):
        key = tuple(Xr[i].tolist())
        if key not in seen:
            seen[key] = []
            order.append(key)
        seen[key].append(i)
    return [np.asarray(seen[key], dtype=int) for key in order]


def aggregate_replicates(
    X: np.ndarray,
    y: np.ndarray,
    *,
    decimals: int = 6,
    groups: GroupKeys | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse replicate rows (identical rounded feature vectors) to one row each.

    Replicates are identified as rows of `X` whose feature vectors are equal
    after `np.round(X, decimals)`. Returns `(X_unique, y_mean, y_var, n_reps)`
    where each output has one row per distinct recipe, in first-occurrence
    order (deterministic given the input order):
      - `X_unique`: the (unrounded) feature vector of the first occurrence of
        each recipe.
      - `y_mean`: the mean of `y` within each group.
      - `y_var`: the within-group sample variance (`ddof=1`); `0.0` for
        singleton groups (a variance needs at least 2 observations).
      - `n_reps`: the number of replicate rows folded into each group.
    """
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float).reshape(-1)
    if Xa.ndim != 2 or Xa.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D array (n x d)")
    if Xa.shape[0] != ya.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    if groups is not None:
        if len(groups) != Xa.shape[0]:
            raise ValueError("groups must have one entry per row of X")
        row_groups = _group_indices_from_keys(list(groups))
    else:
        row_groups = _group_indices(Xa, decimals)
    n_groups = len(row_groups)
    d = Xa.shape[1]
    X_unique = np.empty((n_groups, d), dtype=float)
    y_mean = np.empty(n_groups, dtype=float)
    y_var = np.empty(n_groups, dtype=float)
    n_reps = np.empty(n_groups, dtype=int)

    for g, idx in enumerate(row_groups):
        X_unique[g] = Xa[idx[0]]
        vals = ya[idx]
        y_mean[g] = vals.mean()
        y_var[g] = float(vals.var(ddof=1)) if vals.shape[0] >= 2 else 0.0
        n_reps[g] = vals.shape[0]

    return X_unique, y_mean, y_var, n_reps


def estimate_noise_floor(
    X: np.ndarray, y: np.ndarray, *, decimals: int = 6, groups: GroupKeys | None = None
) -> float:
    """Estimate the pooled within-replicate (assay noise) variance, sigma^2.

    Assumes approximately homoscedastic assay noise across recipes: the
    estimate is the mean of the per-group sample variances, taken only over
    groups with at least 2 replicates (a singleton contributes no noise
    information). Returns `float("nan")` if no group has 2+ replicates.
    """
    _, _, y_var, n_reps = aggregate_replicates(X, y, decimals=decimals, groups=groups)
    replicated = n_reps >= 2
    if not np.any(replicated):
        return float("nan")
    return float(y_var[replicated].mean())


def heteroscedasticity_report(
    X: np.ndarray,
    y: np.ndarray,
    *,
    decimals: int = 6,
    groups: GroupKeys | None = None,
    log_offset: float | None = None,
) -> dict:
    """Does the assay noise scale with the measurement? A diagnostic, not a fix.

    `estimate_noise_floor` states its assumption openly: approximately
    HOMOSCEDASTIC assay noise across recipes, estimated as the mean of per-recipe
    variances. Nothing has ever checked that assumption, and there is a specific
    reason to doubt it for titer. Titer is non-negative and bounded below by zero,
    a substantial fraction of readings can be zero, and on the real media DoE the
    estimated noise sd (~0.0079) EXCEEDS the between-recipe signal sd (~0.0047).
    That combination is the signature of MULTIPLICATIVE noise near a floor, not of
    constant additive noise.

    It matters because a single pooled variance is the wrong summary under
    multiplicative noise: high-titer recipes contribute large variances that
    inflate the floor for everyone, which deflates the ICC. `BENCHMARK.md` reasons
    from ICC ~= 0.26 to "assay noise, not the optimizer, is the limiting factor",
    so if that ICC is partly a scale artifact then the conclusion drawn from it is
    too.

    What this returns:
      - `variance_mean_rho`: Spearman correlation between each replicated recipe's
        MEAN and its within-recipe VARIANCE. Spearman rather than a regression
        slope because it needs no distributional assumption and is not dragged by
        one high-variance recipe, which is exactly the situation here.
      - `icc_raw` / `icc_log`: the ICC on the measured scale and on `log(y + c)`.
      - `log_offset`: the `c` used. Defaults to half the smallest positive
        measurement, which keeps the transform on the data's own scale. `log1p` is
        WRONG here and was the first thing tried: titer runs around 0.005-0.02, so
        `log1p(y) ~= y`, the `+1` dominates, and the transform is nearly the
        identity - it moved the ICC by 0.002 on a sheet with a clear
        variance-mean coupling. Pass the assay LOD as `log_offset` when known;
        that is the principled choice, since it is the scale below which a
        reading carries no information.
      - `icc_gain`: `icc_log - icc_raw`. Positive means the transform recovered
        signal that the raw scale was attributing to noise.
      - `homoscedastic`: False when variance rises with the mean at or above
        `HETERO_RHO_FLOOR`, True when it does not, None when unmeasurable.
      - `suggests_transform`: True only when BOTH the variance-mean coupling is
        present AND log1p materially improves the ICC. Either alone is not enough
        to recommend changing the scale a client's numbers are reported on.

    This function deliberately does NOT transform anything. Silently changing the
    target's scale would change every number the engine reports - the titer in a
    proposal, the conformal band's units, the driver signs - so the decision to
    transform belongs to a human who can also re-label the axes. `log1p` requires
    y >= 0; with any negative value the log branch is skipped and reported as
    unmeasurable rather than clipped, since clipping would invent data.

    Needs at least `MIN_REPLICATED_FOR_HETERO` replicated recipes: correlating
    variance against mean over two points is not a measurement.
    """
    ya = np.asarray(y, dtype=float).reshape(-1)
    _, y_mean, y_var, n_reps = aggregate_replicates(X, y, decimals=decimals, groups=groups)
    replicated = n_reps >= 2
    n_replicated = int(replicated.sum())

    out: dict[str, object] = {
        "n_replicated": n_replicated,
        "variance_mean_rho": float("nan"),
        "icc_raw": float("nan"),
        "icc_log": float("nan"),
        "log_offset": None,
        "icc_gain": float("nan"),
        "homoscedastic": None,
        "suggests_transform": False,
        "reason": None,
    }

    if n_replicated < MIN_REPLICATED_FOR_HETERO:
        out["reason"] = (
            f"only {n_replicated} replicated recipe(s); at least "
            f"{MIN_REPLICATED_FOR_HETERO} are needed to relate within-recipe "
            "variance to recipe mean"
        )
        return out

    means = y_mean[replicated]
    variances = y_var[replicated]
    rho, _p = spearmanr(means, variances)
    rho_f = float("nan") if rho != rho else float(rho)
    out["variance_mean_rho"] = rho_f
    if rho_f == rho_f:
        out["homoscedastic"] = bool(rho_f < HETERO_RHO_FLOOR)

    out["icc_raw"] = _icc(y_mean, y_var, n_reps)

    if np.any(ya < 0):
        out["reason"] = (
            "log transform not evaluated: the target has negative values, and "
            "clipping them to fit it would invent data"
        )
        return out

    positive = ya[ya > 0]
    if positive.size == 0:
        out["reason"] = "log transform not evaluated: every measurement is zero"
        return out
    # Half the smallest positive reading keeps the offset on the data's own scale,
    # so the transform actually bites at the magnitudes titer occupies.
    c = float(log_offset) if log_offset is not None else float(positive.min()) / 2.0
    if not np.isfinite(c) or c <= 0:
        out["reason"] = "log transform not evaluated: no usable positive offset"
        return out
    out["log_offset"] = c

    # Re-derive the decomposition on the transformed scale, grouped identically so
    # the two ICCs differ only by the transform.
    _, lm, lv, ln = aggregate_replicates(X, np.log(ya + c), decimals=decimals, groups=groups)
    out["icc_log"] = _icc(lm, lv, ln)

    raw_icc, log_icc = out["icc_raw"], out["icc_log"]
    if isinstance(raw_icc, float) and isinstance(log_icc, float) and raw_icc == raw_icc and log_icc == log_icc:
        gain = log_icc - raw_icc
        out["icc_gain"] = float(gain)
        out["suggests_transform"] = bool(
            out["homoscedastic"] is False and gain >= ICC_GAIN_FLOOR
        )
        if out["suggests_transform"]:
            out["reason"] = (
                f"within-recipe variance rises with recipe mean (rho={rho_f:.2f}) and "
                f"a log(y+{c:.3g}) transform raises the ICC by {gain:.3f}, so the "
                "measured-scale noise "
                "floor is partly a scale artifact"
            )
        elif out["homoscedastic"] is False:
            out["reason"] = (
                f"variance rises with mean (rho={rho_f:.2f}) but log(y+{c:.3g}) does "
                f"not materially improve the ICC ({gain:+.3f}), so a transform is "
                "not the fix here"
            )
        else:
            out["reason"] = "no variance-mean coupling detected; the homoscedastic estimate holds"
    return out


def _icc(y_mean: np.ndarray, y_var: np.ndarray, n_reps: np.ndarray) -> float:
    """Intraclass correlation from a replicate decomposition: the fraction of
    total variance that is real between-recipe signal.

    Mirrors `noise_report`'s definition exactly so the two can never disagree:
    pooled within-recipe variance over replicated recipes as the noise term, and
    the sample variance of per-recipe means as the signal term.
    """
    replicated = n_reps >= 2
    if not np.any(replicated) or y_mean.size < 2:
        return float("nan")
    noise_var = float(y_var[replicated].mean())
    signal_var = float(np.var(y_mean, ddof=1))
    total = signal_var + noise_var
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return float(signal_var / total)


def noise_report(
    X: np.ndarray, y: np.ndarray, *, decimals: int = 6, groups: GroupKeys | None = None
) -> dict:
    """Summarize replicate structure and signal-to-noise for a design matrix.

    Returns a dict with:
      - `n_rows`: total input rows.
      - `n_recipes`: distinct recipes (groups) found.
      - `n_replicated`: recipes with 2 or more replicate rows.
      - `noise_var`: pooled within-replicate variance (see
        `estimate_noise_floor`); `nan` if no recipe is replicated.
      - `signal_var`: sample variance (`ddof=1`) of the per-recipe means;
        `nan` if fewer than 2 recipes.
      - `icc`: intraclass correlation `signal_var / (signal_var + noise_var)`,
        the fraction of total variance attributable to real recipe-to-recipe
        differences rather than assay noise; `nan` if either input is `nan`
        or the denominator is 0.
    """
    X_unique, y_mean, y_var, n_reps = aggregate_replicates(
        X, y, decimals=decimals, groups=groups
    )
    n_recipes = X_unique.shape[0]
    replicated = n_reps >= 2
    n_replicated = int(replicated.sum())

    noise_var = float(y_var[replicated].mean()) if n_replicated > 0 else float("nan")
    signal_var = float(y_mean.var(ddof=1)) if n_recipes >= 2 else float("nan")

    if np.isnan(noise_var) or np.isnan(signal_var):
        icc = float("nan")
    else:
        denom = signal_var + noise_var
        icc = float(signal_var / denom) if denom > 0 else float("nan")

    return {
        "n_rows": int(np.asarray(X).shape[0]),
        "n_recipes": n_recipes,
        "n_replicated": n_replicated,
        "noise_var": noise_var,
        "signal_var": signal_var,
        "icc": icc,
    }
