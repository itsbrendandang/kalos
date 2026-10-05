"""Response shape and feature relevance read off the GP that makes the proposals.

`kalos.core.drivers` reports a signed Spearman rho per feature. Spearman is
univariate and MONOTONIC, so two shapes are invisible to it no matter how strong
they are:

  - an INTERIOR OPTIMUM. A titer that peaks at pH 7.0 and falls off both sides
    produces a rank correlation near zero, so the panel reports "no signal" for
    the single most important variable on the sheet. Bioprocess optima are
    usually interior, so this is the common case rather than a corner case, and
    acting on the rho sign pushes the process the wrong way.
  - an INTERACTION. pH mattering only at high temperature cannot be expressed as
    a per-feature rank statistic at all.

This module closes that gap using the surrogate the engine ALREADY fits, rather
than a second model. Three properties follow from that choice, and all three
were problems with the gradient-boosted-tree approach this replaces:

  1. ONE AUTHORITY. The shapes describe the same posterior that produced the
     proposed batch. A separate model can rank features differently from the
     model actually being optimized, which leaves a client with two answers and
     no way to choose.
  2. UNCERTAINTY COMES FREE. `Surrogate.posterior` returns a mean AND a standard
     deviation, so an interior optimum is only claimed when the peak is
     separated from the profile's endpoints by more than the posterior's own
     credible width. A boosted tree gives a point prediction, so the equivalent
     claim there rests on nothing.
  3. RELEVANCE COMES FREE. The Matern kernel is fitted with ARD, one lengthscale
     per input dimension. A short lengthscale means the response varies quickly
     along that input, which is exactly per-feature relevance, already paid for
     during the fit.

WHERE THE OTHER FEATURES ARE HELD, and why it matters. Sweeping one feature
requires fixing the rest, and the usual choice - hold them at the column medians
- is a recipe that may never have been run. That puts the whole profile in a
region where the GP has no data, reverts toward its prior, and reports a shape
of the prior rather than of the process. This module sweeps around the INCUMBENT
instead: the observed row with the best measured target. That is a real recipe
someone actually ran, so the sweep stays near data, and it answers the question
a scientist is actually asking, which is how the response moves around their
best run so far.

WHAT THIS IS NOT. These are associations in an observational fit, not causal
effects, and a shape holds only along the swept axis at the incumbent - it is not
a claim about the response surface everywhere. Both limits are carried in the
report's `unmodeled` list rather than left to the reader.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike

Shape = Literal[
    "monotonic_up", "monotonic_down", "interior_optimum", "interior_minimum", "flat"
]

# --------------------------------------------------------------------------- #
# Thresholds. Product decisions, made once and named so they can be argued with
# in one place rather than rediscovered from scattered magic numbers.
# --------------------------------------------------------------------------- #

GRID_POINTS = 21
# Resolution of each 1-D sweep. Odd, so the grid has a literal midpoint. Large
# enough to resolve a smooth interior optimum, small enough that d sweeps stay
# cheap: the cost is d posterior calls of GRID_POINTS rows each, which at the row
# counts this engine targets is negligible next to the GP fit itself.

PEAK_SD_MULTIPLE = 1.0
# How far an interior peak must clear the better endpoint before it is called an
# interior optimum: at least this many COMBINED posterior standard deviations,
# sqrt(sd_peak^2 + sd_endpoint^2). This is the check the tree-based version could
# not make, and it is the difference between "the fitted mean happens to bend"
# and "the model resolves a peak here". One sigma is deliberately modest - the
# aim is to catch a real but unremarkable optimum, not only a dramatic one - and
# both `peak_gain` and `peak_separation_sd` are reported so a human can apply a
# stricter bar without re-running anything.

FLAT_FRACTION = 0.05
# A profile whose mean moves less than this fraction of the target's observed
# standard deviation is `flat`. Below that the shape is not worth a claim.

NEGLIGIBLE_RELEVANCE_SHARE = 0.25
# Below this multiple of an equal share (1/d) of ARD relevance, the GP is barely
# using the feature and its shape is reported `flat` whatever the profile does.
# A shape read off a feature the model ignores describes the fit, not the
# process; with a very long lengthscale the profile is a nearly flat line whose
# direction is arbitrary, and calling that "monotonic_up" reads to a scientist as
# a real trend.

_UNMODELED: tuple[str, ...] = (
    "causation (these are associations in an observational fit, not causal effects)",
    "the response away from the incumbent (each shape holds along one axis, with "
    "the other features fixed at the best observed run)",
    "combinations never actually observed together",
    "scale-up transfer",
)


@dataclass(frozen=True)
class FeatureShape:
    """One feature's ARD relevance and the shape of the response along its axis."""

    name: str
    relevance: float
    """ARD-derived share of relevance, normalized across features to sum to 1."""
    lengthscale: float
    """The raw fitted ARD lengthscale, reported for audit. Shorter means the
    response varies faster along this input."""
    shape: Shape
    optimum_at: float | None
    """Feature value at an interior extremum, in the feature's ORIGINAL units, so
    a scientist can act on the number. None unless the shape is interior."""
    peak_gain: float | None
    """How much better the interior peak is than the better endpoint, in the
    target's units. None unless the shape is interior."""
    peak_separation_sd: float | None
    """That gain expressed in combined posterior standard deviations. This is what
    `PEAK_SD_MULTIPLE` gates on; reported so a stricter bar can be applied without
    re-running."""
    spearman_rho: float
    """The monotonic view of the same feature, for direct comparison."""
    missed_by_spearman: bool
    """A scientist reading only rho would be misled about this feature."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GPShapeReport:
    available: bool
    reason: str | None
    swept_at: str
    """Where the non-swept features were held, named rather than implied."""
    n_features: int
    features: tuple[FeatureShape, ...] = ()
    unmodeled: tuple[str, ...] = field(default_factory=lambda: _UNMODELED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "swept_at": self.swept_at,
            "n_features": self.n_features,
            "features": [f.to_dict() for f in self.features],
            "unmodeled": list(self.unmodeled),
        }


def ard_lengthscales(surrogate: Any, d: int) -> np.ndarray | None:
    """The fitted per-dimension ARD lengthscales, or None if the kernel has none.

    Walks the fitted module tree for the first submodule exposing `lengthscale`.
    Returns None when the kernel turned out to be isotropic (a single shared
    lengthscale), because a scalar cannot rank features and inventing a ranking
    from it would be worse than saying nothing. Also returns None on a mixed
    (categorical) GP whose continuous kernel covers only a subset of dimensions,
    since the lengthscale vector would not align with the feature list.
    """
    model = getattr(surrogate, "model", None)
    if model is None:
        return None
    for _name, module in model.named_modules():
        raw = getattr(module, "lengthscale", None)
        if raw is None:
            continue
        values = np.asarray(raw.detach().cpu().numpy(), dtype=float).reshape(-1)
        if values.size != d:
            return None  # isotropic, or misaligned with the feature list
        if not np.isfinite(values).all() or (values <= 0).any():
            return None
        return values
    return None


def _relevance_from_lengthscales(lengthscales: np.ndarray) -> np.ndarray:
    """Normalize inverse lengthscales into shares summing to 1.

    Inverse because a SHORT lengthscale means the response changes quickly along
    that input, which is high relevance. This is a monotone re-expression of the
    fitted kernel, not a significance test: a large share says the GP varies
    fastest along that axis, not that the association is established.
    """
    inv = 1.0 / lengthscales
    total = float(inv.sum())
    if total <= 0 or not np.isfinite(total):
        return np.full(lengthscales.size, 1.0 / lengthscales.size)
    return inv / total


def _classify(
    grid: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    y_std: float,
    *,
    relevance: float,
    d: int,
) -> tuple[Shape, float | None, float | None, float | None]:
    """Classify one posterior sweep. Returns (shape, optimum_at, gain, gain_sd)."""
    if d > 0 and relevance < NEGLIGIBLE_RELEVANCE_SHARE / d:
        return "flat", None, None, None
    span = float(mean.max() - mean.min())
    if y_std <= 0 or span < FLAT_FRACTION * y_std:
        return "flat", None, None, None

    last = mean.size - 1
    lo_m, hi_m = float(mean[0]), float(mean[last])
    lo_s, hi_s = float(sd[0]), float(sd[last])

    j_max = int(np.argmax(mean))
    if 0 < j_max < last:
        # Compare against the BETTER endpoint: a peak that only beats the worse
        # one is a monotonic trend with a bend, not an interior optimum.
        ref_m, ref_s = (hi_m, hi_s) if hi_m >= lo_m else (lo_m, lo_s)
        gain = float(mean[j_max]) - ref_m
        combined = float(np.hypot(sd[j_max], ref_s))
        sep = gain / combined if combined > 0 else float("inf")
        if gain > 0 and sep >= PEAK_SD_MULTIPLE:
            return "interior_optimum", float(grid[j_max]), gain, sep

    j_min = int(np.argmin(mean))
    if 0 < j_min < last:
        ref_m, ref_s = (hi_m, hi_s) if hi_m <= lo_m else (lo_m, lo_s)
        drop = ref_m - float(mean[j_min])
        combined = float(np.hypot(sd[j_min], ref_s))
        sep = drop / combined if combined > 0 else float("inf")
        if drop > 0 and sep >= PEAK_SD_MULTIPLE:
            return "interior_minimum", float(grid[j_min]), drop, sep

    return ("monotonic_up" if hi_m >= lo_m else "monotonic_down"), None, None, None


def gp_shape_report(
    surrogate: Any,
    X: ArrayLike,
    y: ArrayLike,
    *,
    feature_names: Sequence[str],
    cv_spearman: float | None = None,
    rho_floor: float = 0.20,
) -> GPShapeReport:
    """Per-feature ARD relevance and response shape, read off a FITTED surrogate.

    `surrogate` must already be fit; this function never fits anything, which is
    what keeps it a description of the deployed model rather than a second
    opinion. `X`/`y` are the same arrays the surrogate was fit on, used to locate
    the incumbent, to bound each sweep by the observed range of that feature, and
    to compute the Spearman comparison.

    `rho_floor` is the bar below which Spearman is treated as having found no
    linear signal; it defaults to the same 0.20 the engine's reliability verdict
    uses, so "missed by Spearman" means missed by the same standard applied
    elsewhere.

    Never raises on an unusable input: returns `available=False` with a reason,
    because a diagnostic that crashes on the data it is meant to describe is
    worse than one that declines.
    """
    from scipy.stats import spearmanr  # local: keeps the import cost off callers

    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float).reshape(-1)
    names = [str(n) for n in feature_names]

    if Xa.ndim != 2 or Xa.shape[0] == 0:
        return GPShapeReport(False, "no rows to sweep", "incumbent", 0)
    d = Xa.shape[1]
    if d == 0 or len(names) != d:
        return GPShapeReport(False, "feature names do not match X columns", "incumbent", d)
    if getattr(surrogate, "model", None) is None:
        return GPShapeReport(False, "surrogate is not fitted", "incumbent", d)

    # SKILL GATE. Posterior separation alone is not enough to justify a shape, and
    # the negative control proved it: on a target that is pure noise the GP fits
    # small wiggles, and near the training data its posterior sd is tiny, so the
    # separation ratio (gain / combined sd) becomes LARGE for a meaningless bend.
    # A well-resolved bump in a model that cannot predict anything is still
    # nothing. So the shapes are conditioned on the engine's own out-of-fold
    # measurement rather than on a second opinion invented here: the same
    # `cv_spearman` and 0.20 floor the reliability verdict already uses. One
    # authority, applied consistently.
    if cv_spearman is not None:
        if cv_spearman != cv_spearman or cv_spearman < rho_floor:
            shown = "nan" if cv_spearman != cv_spearman else f"{cv_spearman:.3f}"
            return GPShapeReport(
                False,
                f"out-of-fold Spearman ({shown}) does not clear the {rho_floor:.2f} "
                "floor, so this model has not shown it can predict held-out runs "
                "and the shape of its fitted response is not a finding",
                "incumbent",
                d,
            )
    unmodeled = _UNMODELED
    if cv_spearman is None:
        # Never silently imply the shapes were validated when no skill measure
        # was supplied.
        unmodeled = _UNMODELED + (
            "whether this model predicts held-out runs at all (no out-of-fold "
            "score was supplied to condition these shapes on)",
        )

    lengthscales = ard_lengthscales(surrogate, d)
    if lengthscales is None:
        return GPShapeReport(
            False,
            "the fitted kernel exposes no per-dimension ARD lengthscales, so "
            "per-feature relevance cannot be read from it",
            "incumbent",
            d,
        )
    relevance = _relevance_from_lengthscales(lengthscales)

    # The incumbent: the observed row with the best measured target. A real
    # recipe, so every sweep stays near data instead of wandering into a corner
    # of the box nobody ran.
    incumbent = Xa[int(np.argmax(ya))].astype(float)
    y_std = float(np.std(ya))

    shapes: list[FeatureShape] = []
    for j, name in enumerate(names):
        col = Xa[:, j]
        lo, hi = float(col.min()), float(col.max())
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            # A constant feature has no axis to sweep.
            shapes.append(
                FeatureShape(
                    name=name, relevance=float(relevance[j]),
                    lengthscale=float(lengthscales[j]), shape="flat",
                    optimum_at=None, peak_gain=None, peak_separation_sd=None,
                    spearman_rho=0.0, missed_by_spearman=False,
                )
            )
            continue

        grid_vals = np.linspace(lo, hi, GRID_POINTS)
        sweep = np.tile(incumbent, (GRID_POINTS, 1))
        sweep[:, j] = grid_vals
        try:
            mean, sd = surrogate.posterior(sweep)
        except Exception as exc:  # a diagnostic must not take down the caller
            return GPShapeReport(
                False, f"posterior sweep failed: {type(exc).__name__}", "incumbent", d
            )
        mean = np.asarray(mean, dtype=float).reshape(-1)
        sd = np.asarray(sd, dtype=float).reshape(-1)

        shape, opt_at, gain, sep = _classify(
            grid_vals, mean, sd, y_std, relevance=float(relevance[j]), d=d
        )

        rho_raw, _p = spearmanr(col, ya, nan_policy="omit")
        rho = 0.0 if rho_raw != rho_raw else float(rho_raw)

        # Two ways a scientist reading only rho is misled, mirroring the rule the
        # engine already applies elsewhere: rho saw nothing while the model leans
        # on the feature, or rho reported a TREND where the model resolves a PEAK.
        # The second cannot be rescued by a large |rho| - a monotonic rank
        # statistic has no way to express an interior extremum at any value.
        interior = shape in ("interior_optimum", "interior_minimum")
        matters = float(relevance[j]) >= 1.0 / d
        missed = interior or (matters and abs(rho) < rho_floor)

        shapes.append(
            FeatureShape(
                name=name,
                relevance=round(float(relevance[j]), 4),
                lengthscale=round(float(lengthscales[j]), 4),
                shape=shape,
                optimum_at=None if opt_at is None else round(opt_at, 4),
                peak_gain=None if gain is None else round(gain, 4),
                peak_separation_sd=None if sep is None else round(min(sep, 1e6), 3),
                spearman_rho=round(rho, 4),
                missed_by_spearman=bool(missed),
            )
        )

    shapes.sort(key=lambda f: -f.relevance)
    return GPShapeReport(
        available=True,
        reason=None,
        swept_at="incumbent",
        n_features=d,
        features=tuple(shapes),
        unmodeled=unmodeled,
    )


__all__ = [
    "FeatureShape",
    "GPShapeReport",
    "Shape",
    "ard_lengthscales",
    "gp_shape_report",
    "GRID_POINTS",
    "PEAK_SD_MULTIPLE",
    "FLAT_FRACTION",
    "NEGLIGIBLE_RELEVANCE_SHARE",
]
