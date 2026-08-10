"""DO NOT WIRE THIS INTO THE PORTAL. Superseded - see the note below.

STATUS (2026-08-09): this module is a documented DEAD END, kept on this branch
only so its shape classifier and out-of-fold gate can be lifted onto the GP
posterior. It must not be imported from `kalos/portal/*` or `kalos/runner/*`.

Why: xgboost and torch each carry their own OpenMP runtime, and in one process
the second one to enter a parallel region crashes. Verified on macOS with
torch 2.13 + xgboost 3.4:

    import torch      -> ok
    torch op          -> ok
    import xgboost    -> ok
    xgb.train(...)    -> SIGSEGV (exit 139), even with nthread=1

The crash is at USE, not import, so the `available=False` degradation in this
module cannot protect against it - a SIGSEGV is not catchable, so a portal that
called this would die with no traceback and no 500. That is disqualifying for a
process that already imports torch on every analyze.

The replacement keeps the science and drops the dependency: the GP already fits
ARD lengthscales, and sweeping `Surrogate.posterior` gives the same shape
classification WITH credible intervals, from the model that actually produces
the proposals rather than a second model held to a weaker bar.

Original design notes follow.

Nonlinear driver analysis: out-of-fold gradient-boosted trees, EXPLANATORY ONLY.

`kalos.core.drivers` reports a signed Spearman rho per feature, which is the
right tool for a monotonic relationship but structurally blind to two shapes
that are common and important in bioprocess data:

  - an INTERIOR OPTIMUM (titer peaks at pH 7.0 and falls off both sides): the
    rank correlation on either side of the peak cancels out, so rho lands near
    zero and the single most important variable gets reported as "no signal".
  - an INTERACTION (pH only matters at high temperature): a univariate rank
    correlation cannot represent a relationship that depends on a second
    feature.

Gradient-boosted trees see both, which is the entire justification for this
module. It is deliberately kept OUT of the acquisition path: the GP surrogate
(`kalos.core.surrogate`) keeps sole ownership of proposals, uncertainty, and
`reliability`. This module never feeds a number back into a recipe, a
conformal band, or a promotion gate - it only explains, after the fact, what
an already-fit process might look like beyond what Spearman can show. Adding
a second authority on trustworthiness is exactly the design this avoids; the
codebase already has one.

The hard constraint this module is built around: the engine runs on TINY
datasets (hard floor 6 rows, `MAX_FIT_ROWS` 2000, and the real reference
dataset is 55 rows x 38 columns). A gradient-boosted tree ensemble overfits
savagely at that size, and a single in-sample `feature_importances_` on 55
rows is noise dressed as insight - reporting it would be worse than reporting
nothing. Every number this module reports is therefore:

  1. computed OUT OF FOLD, never in-sample, through the same leakage-checked
     grouped splitter the rest of the engine uses (`kalos.core.splits`);
  2. reported as a MEAN AND SPREAD across folds, not a single point, so a
     feature whose importance swings fold to fold reads as unestablished
     rather than as a ranked "driver";
  3. GATED on out-of-fold predictive skill first: if the model cannot predict
     held-out rows, its importances are meaningless and the report refuses to
     emit them (`available=False`), the same fail-closed contract
     `kalos.core.gates.check_gates` and `reliability.clears_floor` in
     `kalos.portal.analysis` already use elsewhere in this engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, SupportsFloat

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import spearmanr

from kalos.core.splits import assert_no_group_leakage, make_splits, row_hash_groups

Shape = Literal[
    "monotonic_up", "monotonic_down", "interior_optimum", "interior_minimum", "flat"
]

# --------------------------------------------------------------------------- #
# Thresholds. Every one below is a product decision made once, here, and named
# so it can be argued with in one place rather than rediscovered from scattered
# magic numbers.
# --------------------------------------------------------------------------- #

MIN_ROWS = 24
# Refuse outright below this. The engine's absolute floor elsewhere is 6 rows
# (`kalos/portal/analysis.py`), but that floor exists for a GP with a proper
# posterior variance, not for a tree ensemble whose only honesty check is an
# out-of-fold score computed over held-out folds. With `N_SPLITS` folds, 24
# rows gives each validation fold roughly 4-5 points - already thin, but
# enough that the pooled out-of-fold score (computed over ALL rows at once,
# not averaged per tiny fold) is a real measurement rather than a coin flip.
# The real 55-row reference dataset clears this with room to spare; a sheet
# at the engine's bare floor of 6 does not get anywhere near it, which is the
# point - six points cannot support both fitting a tree AND validating it.

N_SPLITS = 5

CV_SCORE_FLOOR = 0.20
# The out-of-fold Spearman rho between XGBoost's held-out predictions and the
# actual target must clear this before ANY importance is reported. Chosen to
# exactly match `reliability.spearman_floor` in `kalos/portal/analysis.py`
# and `GatesConfig.min_spearman` in `kalos/core/gates.py`, rather than
# inventing a fresh number for XGBoost: the question this floor answers
# ("does this model generalize at all?") is the same question in all three
# places, so a client should see one consistent bar across every part of the
# engine that asks it, not three different ones that happen to look similar.

MISSED_BY_SPEARMAN_RHO_FLOOR = 0.20
# Same floor, reused again: "Spearman missed this" means |rho| is below the
# same bar the rest of the engine already treats as "no established linear
# signal" (see above). Reusing it keeps "missed by Spearman" a statement
# about the SAME standard Spearman itself is held to elsewhere, not a
# separately-tuned trigger for a flashier headline.

NEGLIGIBLE_IMPORTANCE_SHARE = 0.25
# Below this multiple of an equal share (1/n_features), the model is treated as
# having nothing to say about a feature, and its shape is reported `flat`
# regardless of how its partial-dependence profile wiggles. 0.25 means "using a
# feature less than a quarter as much as it would under equal attribution", a
# convention rather than a derived quantity, deliberately well below the
# `HIGH_IMPORTANCE_MULTIPLE` of 1.5 that marks a feature as a real driver, so
# there is a wide neutral band between "ignored" and "important" where a shape is
# still reported and a human can judge it.

PD_GRID_POINTS = 21
# Partial-dependence grid resolution per feature. Odd, so the grid has a
# literal midpoint; large enough to resolve a smooth interior optimum, small
# enough that `N_SPLITS` folds x `n_features` profiles stays cheap at the row
# counts this module is built for (n < 100).

PD_RISE_FRACTION = 0.10
# An interior peak (or trough) must clear BOTH grid endpoints by at least
# this fraction of the target's observed standard deviation before it is
# called a real interior optimum rather than fold-to-fold wobble sitting on
# top of an essentially monotonic or flat trend. 10% is a deliberately loose
# bar: it is set to catch a real-but-modest optimum, not just a dramatic one,
# while still requiring the bump to be a meaningfully sized fraction of the
# target's own spread (a bump worth a fraction of a percent of y's spread is
# not something a scientist should act on).

PD_FLAT_FRACTION = 0.05
# When the partial-dependence profile's own range (max - min across the grid)
# is below this fraction of the target's standard deviation, the model has
# learned essentially nothing about how this feature shapes the target across
# its observed range. Reported as `flat`, not forced into a monotonic label
# on what is really noise.

TOP_K_INTERACTIONS = 5
# How many pairwise SHAP interactions to surface. This is a reporting cap,
# not a modeling choice - the underlying interaction matrix is computed in
# full; only the top few pairs are worth a scientist's attention.

HIGH_IMPORTANCE_MULTIPLE = 1.5
# What "importance is high" means for `missed_by_spearman`, the headline
# finding. A model with zero real signal still spreads its gain roughly
# 1/n_features per feature on average (some baseline share is unavoidable -
# every split has to pick something), so "high" is defined relative to that
# uniform baseline rather than as a fixed absolute number: a feature must
# carry at least 1.5x the share a completely uninformative feature would get
# by chance before its importance counts as established enough to headline.

# XGBoost hyperparameters for the n < 100 regime. These are REGULARIZATION
# choices to keep the model from memorizing a few dozen rows, not a tuned
# configuration chasing accuracy - there is no held-out set large enough at
# this n to tune hyperparameters honestly, so tuning them would just be
# overfitting the overfitting-prevention.
XGB_PARAMS: dict[str, object] = {
    "max_depth": 3,  # shallow: a depth-3 tree already has 8 leaves, plenty
    # of capacity to represent an interior optimum or a two-way interaction
    # without carving the training set into single-row leaves.
    "min_child_weight": 5,  # a split must leave at least 5 rows of Hessian
    # weight in each child, so a tree cannot split off a leaf of 1-2 rows and
    # call that a "pattern".
    "subsample": 0.8,  # each tree sees 80% of rows, decorrelating trees so
    # the ensemble is not just one memorized tree copied `n_estimators` times.
    "colsample_bytree": 0.8,  # same idea over columns.
    "n_estimators": 100,  # modest; combined with a shallow depth and the
    # subsampling above, more trees mainly average down variance rather than
    # add capacity to memorize.
    "learning_rate": 0.1,
    "reg_lambda": 2.0,  # explicit positive L2 leaf-weight penalty on top of
    # xgboost's default of 1.0 - a deliberate extra pull toward small,
    # conservative leaf values at this row count.
    "n_jobs": 1,  # determinism: xgboost's histogram builder is not
    # guaranteed bit-identical across thread counts; a portal report must be
    # reproducible from the same sheet + seed, not from the same sheet +
    # seed + whatever core count happened to build it.
    "verbosity": 0,
}

_UNMODELED: tuple[str, ...] = (
    "causality - this analysis shows associations the model found predictive, "
    "not what would happen if a variable were deliberately changed",
    "anything outside the observed range - an interior optimum or minimum is "
    "located only within the range of values actually seen for that feature "
    "and says nothing about what happens beyond it",
    "scale-up transfer - this analysis says nothing about whether a "
    "lab-scale relationship holds at production scale",
    "combinations of feature values never actually observed together in the "
    "uploaded rows, even if each value individually falls inside range",
)


@dataclass(frozen=True)
class NonlinearDriver:
    """One feature's out-of-fold-validated nonlinear profile.

    `importance` and `importance_sd` are the mean and standard deviation of
    that feature's normalized gain share ACROSS CV FOLDS, never an in-sample
    number from a single fit. `shape` and `optimum_at` come from a
    partial-dependence profile averaged across the same folds. `spearman_rho`
    is the plain linear/monotonic view for direct comparison, and
    `missed_by_spearman` is the headline flag: real importance that Spearman's
    univariate rank correlation could not see.
    """

    name: str
    importance: float
    importance_sd: float
    shape: Shape
    optimum_at: float | None
    spearman_rho: float
    missed_by_spearman: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "importance": self.importance,
            "importance_sd": self.importance_sd,
            "shape": self.shape,
            "optimum_at": self.optimum_at,
            "spearman_rho": self.spearman_rho,
            "missed_by_spearman": self.missed_by_spearman,
        }


@dataclass(frozen=True)
class Interaction:
    """A pairwise SHAP tree-interaction strength, normalized across all pairs."""

    pair: tuple[str, str]
    strength: float

    def to_dict(self) -> dict[str, object]:
        return {"pair": list(self.pair), "strength": self.strength}


@dataclass(frozen=True)
class NonlinearReport:
    """The result of one `nonlinear_drivers` call.

    `available=False` means exactly what it says: the report below carries no
    drivers or interactions because either the data could not support an
    honest out-of-fold measurement (too few rows, xgboost absent) or the
    out-of-fold measurement WAS made and came back too weak to trust
    (`cv_score` is populated in that case, so the caller can see the number
    the refusal is based on). `unmodeled` is always populated, whether or not
    the analysis ran, because it describes limits of the METHOD, not of one
    particular fit.
    """

    available: bool
    reason: str | None
    n_rows: int
    n_features: int
    cv_score: float | None
    drivers: tuple[NonlinearDriver, ...]
    interactions: tuple[Interaction, ...]
    unmodeled: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "reason": self.reason,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "cv_score": self.cv_score,
            "drivers": [d.to_dict() for d in self.drivers],
            "interactions": [i.to_dict() for i in self.interactions],
            "unmodeled": list(self.unmodeled),
        }


def _safe_float(x: SupportsFloat, default: float = 0.0) -> float:
    """Coerce to a plain, finite Python float; NaN/inf become `default`.

    Takes `SupportsFloat` (covers both a plain Python float and a numpy
    scalar like the `np.float64` that indexing a numpy array or unpacking
    `scipy.stats.spearmanr` returns) so every call site can pass either
    without a separate cast, matching `kalos.core.drivers`'s convention of
    wrapping every numpy scalar in `float(...)` before it can reach a caller.

    The portal serializes with `allow_nan=False` (see
    `kalos/validation/report.py::_json_safe`), and JSON has no literal for
    NaN or Infinity, so a non-finite value reaching `json.dumps` here would be
    a 500, not a bad report. Guarding it at construction time, rather than at
    a later serialization boundary, means `NonlinearDriver`/`NonlinearReport`
    are JSON-safe BY CONSTRUCTION - a caller does not have to remember a
    separate sanitizing pass to safely serialize what this module returns.
    """
    v = float(x)
    return v if np.isfinite(v) else default


def _safe_optional_float(x: SupportsFloat | None) -> float | None:
    """Like `_safe_float`, but for a value that is legitimately absent
    (`optimum_at` on a feature with no interior optimum): `None` stays `None`
    rather than being coerced to a misleading 0.0."""
    if x is None:
        return None
    v = float(x)
    return v if np.isfinite(v) else None


def _empty(n_rows: int, n_features: int, reason: str, cv_score: float | None = None) -> NonlinearReport:
    return NonlinearReport(
        available=False,
        reason=reason,
        n_rows=n_rows,
        n_features=n_features,
        cv_score=cv_score,
        drivers=(),
        interactions=(),
        unmodeled=_UNMODELED,
    )


def _feature_grids(X: np.ndarray) -> np.ndarray:
    """A `(n_features, PD_GRID_POINTS)` evenly-spaced grid per feature's
    observed range, `nan`-safe. A constant feature gets a single repeated
    value rather than a degenerate `linspace`."""
    lo = np.nanmin(X, axis=0)
    hi = np.nanmax(X, axis=0)
    grids = np.zeros((X.shape[1], PD_GRID_POINTS))
    for j in range(X.shape[1]):
        if hi[j] > lo[j]:
            grids[j] = np.linspace(lo[j], hi[j], PD_GRID_POINTS)
        else:
            grids[j] = np.full(PD_GRID_POINTS, lo[j])
    return grids


def _classify_shape(
    grid: np.ndarray,
    profile: np.ndarray,
    y_std: float,
    *,
    importance: float,
    n_features: int,
) -> tuple[Shape, float | None]:
    """Classify one feature's averaged partial-dependence profile.

    `flat` when either the model barely uses the feature (see below) or the
    profile barely moves relative to the target's own spread
    (`PD_FLAT_FRACTION`). Otherwise, an interior grid point that both (a) is
    strictly interior - not the first or last grid cell, so an endpoint
    extremum is never mistaken for an interior one - and (b) rises or falls
    meaningfully above BOTH endpoints (`PD_RISE_FRACTION`) is reported as an
    interior optimum or minimum, with `optimum_at` in the feature's own
    original units (read straight off `grid`, which was built from the raw,
    unstandardized column). Anything else is monotonic, signed by whether the
    profile ends higher or lower than it started.

    The importance precondition exists because a shape read off a feature the
    model ignores is a description of overfitting, not of the process. A pure
    noise column was being reported as `monotonic_up`: boosted trees will always
    carve SOME structure out of noise, and its partial-dependence profile
    wandered just past the flatness threshold. The direction of that wander is
    meaningless, but "monotonic_up" reads to a scientist as a real trend. Below
    `NEGLIGIBLE_IMPORTANCE_SHARE` of the total importance the honest label is
    `flat`, which here means "this model has nothing to say about this feature"
    rather than "the response is genuinely level".
    """
    if n_features > 0 and importance < NEGLIGIBLE_IMPORTANCE_SHARE / n_features:
        return "flat", None
    lo, hi = float(profile[0]), float(profile[-1])
    prange = float(profile.max() - profile.min())
    if y_std <= 0 or prange < PD_FLAT_FRACTION * y_std:
        return "flat", None
    last = len(profile) - 1
    j_max = int(np.argmax(profile))
    j_min = int(np.argmin(profile))
    if 0 < j_max < last and (profile[j_max] - max(lo, hi)) >= PD_RISE_FRACTION * y_std:
        return "interior_optimum", float(grid[j_max])
    if 0 < j_min < last and (min(lo, hi) - profile[j_min]) >= PD_RISE_FRACTION * y_std:
        return "interior_minimum", float(grid[j_min])
    return ("monotonic_up" if hi >= lo else "monotonic_down"), None


def nonlinear_drivers(
    X: ArrayLike,
    y: ArrayLike,
    *,
    feature_names: Sequence[str],
    groups: ArrayLike | None = None,
    seed: int = 1234,
) -> NonlinearReport:
    """Out-of-fold XGBoost driver analysis: shape, importance, and interactions.

    EXPLANATORY ONLY - see the module docstring for the scope boundary. This
    function never raises for an ordinary "not enough signal" or "not enough
    data" outcome; those come back as `NonlinearReport(available=False, ...)`
    with a `reason`, so a caller can render "not available: <reason>" without
    a try/except. It DOES raise `ValueError` for a genuine caller bug (shape
    mismatch), same as `kalos.core.drivers.spearman_driver_matrix`.

    `xgboost` is imported LAZILY, inside this function, and only after the row
    count is checked - so a sheet too small to bother fitting never pays the
    import cost, and a machine with no working xgboost (the NORMAL state on a
    fresh macOS install without `libomp`; see `pyproject.toml`) gets a report
    back, not an `ImportError` traceback.
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    names = list(feature_names)
    n_rows, n_features = X_arr.shape

    if len(names) != n_features:
        raise ValueError(f"feature_names has {len(names)} entries but X has {n_features} columns")
    if y_arr.shape[0] != n_rows:
        raise ValueError(f"X has {n_rows} rows but y has {y_arr.shape[0]}")

    if n_rows < MIN_ROWS:
        return _empty(
            n_rows, n_features,
            f"only {n_rows} rows; need at least {MIN_ROWS} for an honest "
            "out-of-fold XGBoost gate (see MIN_ROWS in kalos/core/nonlinear_drivers.py)",
        )

    try:
        import xgboost as xgb
    except (ImportError, OSError) as exc:
        return _empty(
            n_rows, n_features,
            "xgboost is not installed or failed to load "
            f"({exc}). Install with `pip install 'kalos[xgb]'`; on macOS "
            "xgboost also needs an OpenMP runtime that Python packaging does "
            "not bundle - `brew install libomp` - which is why a missing or "
            "broken xgboost is the normal state on a fresh Mac.",
        )

    if groups is None:
        groups_arr = row_hash_groups(pd.DataFrame(X_arr))
    else:
        groups_arr = np.asarray(groups)

    splits = make_splits(X_arr, y_arr, groups_arr, n_splits=N_SPLITS, random_state=seed)
    if len(splits) < 2:
        return _empty(
            n_rows, n_features,
            "not enough independent replicate groups for a leakage-free "
            "out-of-fold split",
        )
    assert_no_group_leakage(splits, groups_arr)

    grid = _feature_grids(X_arr)
    medians = np.nanmedian(X_arr, axis=0)

    oof_pred = np.full(n_rows, np.nan)
    importance_folds: list[np.ndarray] = []
    pd_sum = np.zeros((n_features, PD_GRID_POINTS))
    pd_folds = 0
    interaction_sum = np.zeros((n_features, n_features))
    interaction_rows = 0
    interactions_ok = n_features >= 2

    for tr, va in splits:
        if len(tr) < 2 or len(va) < 1:
            continue
        model = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed)
        model.fit(X_arr[tr], y_arr[tr])
        oof_pred[va] = model.predict(X_arr[va])

        gains = np.nan_to_num(np.asarray(model.feature_importances_, dtype=float))
        total = float(gains.sum())
        importance_folds.append(gains / total if total > 0 else np.zeros(n_features))

        queries = np.tile(medians, (n_features, PD_GRID_POINTS, 1))
        for j in range(n_features):
            queries[j, :, j] = grid[j]
        preds = model.predict(queries.reshape(-1, n_features)).reshape(n_features, PD_GRID_POINTS)
        pd_sum += preds
        pd_folds += 1

        if interactions_ok:
            try:
                booster = model.get_booster()
                dval = xgb.DMatrix(X_arr[va])
                inter = booster.predict(dval, pred_interactions=True)
                interaction_sum += np.abs(inter[:, :n_features, :n_features]).sum(axis=0)
                interaction_rows += len(va)
            except Exception:
                # Degrade to no interactions rather than a partial/misleading
                # matrix built from only SOME folds - see module docstring.
                interactions_ok = False
                interaction_sum[:] = 0.0
                interaction_rows = 0

    valid = np.isfinite(oof_pred)
    if int(valid.sum()) >= 3:
        cv_rho, _p = spearmanr(y_arr[valid], oof_pred[valid], nan_policy="omit")
        cv_score = _safe_float(cv_rho, default=0.0)
    else:
        cv_score = 0.0

    if cv_score < CV_SCORE_FLOOR:
        return _empty(
            n_rows, n_features,
            f"out-of-fold Spearman skill ({cv_score:.3f}) does not clear the "
            f"{CV_SCORE_FLOOR} floor (CV_SCORE_FLOOR) - the model cannot "
            "predict held-out rows, so its importances would be noise "
            "dressed as insight, not a finding",
            cv_score=cv_score,
        )

    importance_mat = np.vstack(importance_folds)
    mean_importance = importance_mat.mean(axis=0)
    total_mean = float(mean_importance.sum())
    if total_mean > 0:
        mean_importance = mean_importance / total_mean
    sd_importance = (
        importance_mat.std(axis=0, ddof=1) if len(importance_folds) > 1 else np.zeros(n_features)
    )

    pd_mean = pd_sum / max(pd_folds, 1)
    y_std = float(np.std(y_arr))

    drivers = []
    for j, name in enumerate(names):
        rho_j, _p = spearmanr(X_arr[:, j], y_arr, nan_policy="omit")
        rho = _safe_float(rho_j, default=0.0)
        importance = _safe_float(mean_importance[j], default=0.0)
        matters = importance >= HIGH_IMPORTANCE_MULTIPLE / n_features
        shape, optimum_at = _classify_shape(
            grid[j], pd_mean[j], y_std, importance=importance, n_features=n_features
        )
        # `missed_by_spearman` means "a scientist reading only rho would be
        # misled about this feature". There are TWO ways that happens, and an
        # earlier version of this flag only caught the first:
        #
        #   1. Spearman saw nothing: |rho| below the floor while the model
        #      leans on the feature heavily.
        #   2. Spearman saw a TREND where the truth is a PEAK. This is the case
        #      the module exists for, and it is the one that slipped through. A
        #      simulated titer peaking at pH 7.0 produced rho = -0.211 - just
        #      over the 0.20 floor, so condition 1 was False and the headline
        #      case went unflagged. But "weak negative trend" is not a mild
        #      understatement of "optimum at 7.0", it is the wrong shape
        #      entirely, and acting on it means pushing pH the wrong way.
        #      Magnitude of rho cannot rescue this: a monotonic rank statistic
        #      has no way to express an interior extremum at ANY value.
        #
        # So an interior extremum in a feature the model actually uses is always
        # a Spearman mischaracterization, whatever rho happens to be.
        #
        # The two branches carry DIFFERENT evidence bars on purpose:
        #
        #   - Contradicting a low rho ("Spearman saw nothing, but this matters")
        #     is an argument from importance alone, so it needs the strong bar:
        #     `HIGH_IMPORTANCE_MULTIPLE` of an equal share.
        #   - An interior extremum needs only that the feature is not ignored,
        #     because the SHAPE is the evidence, not the importance ranking.
        #     `_classify_shape` already refuses to report a shape at all below
        #     `NEGLIGIBLE_IMPORTANCE_SHARE`, so reaching this branch is itself
        #     proof the model uses the feature.
        #
        # Holding the extremum branch to the strong bar is what hid the headline
        # case: pH carried 31% of importance with a clean peak at 7.0, but with
        # three features the strong bar demands 50%, so the single most
        # actionable finding in the report was silently not flagged.
        interior = shape in ("interior_optimum", "interior_minimum")
        missed = interior or (matters and abs(rho) < MISSED_BY_SPEARMAN_RHO_FLOOR)
        drivers.append(
            NonlinearDriver(
                name=str(name),
                importance=importance,
                importance_sd=_safe_float(sd_importance[j], default=0.0),
                shape=shape,
                optimum_at=_safe_optional_float(optimum_at),
                spearman_rho=rho,
                missed_by_spearman=bool(missed),
            )
        )
    drivers.sort(key=lambda d: -d.importance)

    interactions: tuple[Interaction, ...] = ()
    if interactions_ok and interaction_rows > 0 and n_features >= 2:
        mean_inter = interaction_sum / interaction_rows
        pairs = [
            (j, k, float(mean_inter[j, k]))
            for j in range(n_features) for k in range(j + 1, n_features)
        ]
        total_strength = sum(p[2] for p in pairs)
        if total_strength > 0:
            pairs.sort(key=lambda p: -p[2])
            interactions = tuple(
                Interaction(pair=(str(names[j]), str(names[k])), strength=float(s / total_strength))
                for j, k, s in pairs[:TOP_K_INTERACTIONS]
                if s > 0
            )

    return NonlinearReport(
        available=True,
        reason=None,
        n_rows=n_rows,
        n_features=n_features,
        cv_score=cv_score,
        drivers=tuple(drivers),
        interactions=interactions,
        unmodeled=_UNMODELED,
    )


__all__ = ["NonlinearDriver", "Interaction", "NonlinearReport", "nonlinear_drivers"]
