"""GP surrogate on BoTorch.

A `SingleTaskGP` with input normalization and output standardization — the
production-grade replacement for the lean hand-rolled GP. Fits by exact marginal
likelihood and returns a posterior mean + standard deviation in the target's
original units.

Device note: BoTorch GPs run in float64 for numerical stability, and Apple's MPS
backend does not support float64, so the GP runs on CPU. This is the right call
for the small-sample regime BO targets anyway (GP fits on ~100 rows are fast on
CPU). The deep-learning protein embedder runs on MPS separately.
"""
from __future__ import annotations

from typing import Sequence

import gpytorch
import numpy as np
import torch
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import AdditiveKernel, ProductKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.utils.errors import NotPSDError

DTYPE = torch.double
DEVICE = torch.device("cpu")

# Escalating Cholesky jitter for the fit-retry ladder. A near-duplicate or
# otherwise ill-conditioned design makes the kernel matrix numerically
# non-positive-definite, so the first Cholesky can fail (LinAlgError / NotPSDError
# / ModelFittingError). We retry with progressively larger jitter before giving
# up, so a numerically-hard-but-valid dataset still fits instead of 500-ing.
_FIT_JITTERS: tuple[float, ...] = (1e-4, 1e-3, 1e-2)


class FitError(RuntimeError):
    """The GP could not be fit even after escalating the Cholesky jitter.

    Raised when every retry in the jitter ladder fails on a numerical error
    (`torch.linalg.LinAlgError`, GPyTorch `NotPSDError`, or a botorch
    `ModelFittingError`). Carries an honest, caller-safe message; the portal maps
    it to a distinct 400 so an ill-conditioned file is not confused with a parse
    failure.
    """


# Numerical fit failures we retry with more jitter. A LinAlgError from the
# Cholesky, GPyTorch's NotPSDError, and botorch's ModelFittingError all mean the
# same thing here: the kernel matrix was not positive-definite at this jitter.
_FIT_NUMERICAL_ERRORS = (torch.linalg.LinAlgError, NotPSDError, ModelFittingError)


def _fit_mll_with_retry(mll) -> None:
    """Fit `mll` in place, retrying on a numerical failure with escalating jitter.

    Tries `fit_gpytorch_mll` under each jitter in `_FIT_JITTERS`; the first
    success wins. If every attempt raises a numerical error, raise `FitError` with
    an honest message. Any non-numerical exception propagates unchanged. Works for
    both the single-objective `ExactMarginalLogLikelihood` and the multi-objective
    `SumMarginalLogLikelihood`.
    """
    last: Exception | None = None
    for jitter in _FIT_JITTERS:
        try:
            with gpytorch.settings.cholesky_jitter(jitter):
                fit_gpytorch_mll(mll)
            return
        except _FIT_NUMERICAL_ERRORS as exc:
            last = exc
    raise FitError(
        "the model could not be fit on this data - likely near-duplicate or "
        "ill-conditioned rows"
    ) from last


def sanitize_bounds(bounds) -> tuple[np.ndarray, np.ndarray]:
    """Split `bounds` (2 x d) into finite `(lower, upper)` rows with `lower <= upper`.

    A degenerate or near-constant feature (lower == upper, or a NaN/inf bound)
    would let input normalization divide by a zero-width range and poison the fit
    (and let the acquisition optimizer wander to an absurd value). We order the
    rows, replace any non-finite bound with the finite partner (or 0.0 if both are
    non-finite), and widen a zero-width interval by a tiny epsilon so both the GP
    normalization and `optimize_acqf` see a valid, non-inverted box. This is the
    guard behind the historical out-of-range blow-up (e.g. a constant
    `Culture_Volume` proposing ~= 33,000,000).
    """
    arr = np.asarray(bounds, float)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError("bounds must have shape (2, d) = [lower_row, upper_row]")
    lower = arr[0].astype(float).copy()
    upper = arr[1].astype(float).copy()
    for row, partner in ((lower, upper), (upper, lower)):
        bad = ~np.isfinite(row)
        if bad.any():
            fill = np.where(np.isfinite(partner), partner, 0.0)
            row[bad] = fill[bad]
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    degenerate = upper - lower <= 0.0
    if degenerate.any():
        pad = 1e-6 * np.maximum(np.abs(lower), 1.0)
        upper = np.where(degenerate, lower + pad, upper)
    return lower, upper


class Surrogate:
    """SingleTaskGP over a continuous design space."""

    def __init__(self) -> None:
        self.model: SingleTaskGP | None = None
        self._y: np.ndarray | None = None
        self._X: torch.Tensor | None = None

    def fit(self, X, y, bounds, *, noise=None) -> "Surrogate":
        """Fit on (X, y). `bounds` (2 x d) ties input normalization to the fixed
        design box rather than the training-data envelope; it is required so a
        caller can never fall into the training-envelope normalization footgun.

        `noise` optionally fixes the observation noise variance instead of
        inferring it from the data:
          - `None` (default): infer noise as before (unchanged behavior).
          - a scalar: the same noise variance sigma^2, in the target's original
            units, applied to every training point.
          - an array of length `n`: a per-point noise variance sigma^2_i, in
            the target's original units.
        This is for a known assay noise floor (e.g. from `estimate_noise_floor`
        on replicated recipes), so the GP stops re-inferring noise it already
        has an external estimate for and stops chasing single-replicate noise
        spikes as if they were signal.
        """
        Xa = np.asarray(X, float)
        ya = np.asarray(y, float).reshape(-1)
        if Xa.ndim != 2 or Xa.shape[0] == 0:
            raise ValueError("X must be a non-empty 2-D array (n x d)")
        if Xa.shape[0] != ya.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if not np.isfinite(Xa).all() or not np.isfinite(ya).all():
            raise ValueError("X and y must be finite (no NaN/inf)")
        Xt = torch.as_tensor(Xa, dtype=DTYPE, device=DEVICE)
        yt = torch.as_tensor(ya, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        d = Xt.shape[-1]
        lower, upper = sanitize_bounds(bounds)  # widen zero-width dims so Normalize is finite
        box = torch.as_tensor(np.vstack([lower, upper]), dtype=DTYPE, device=DEVICE)
        normalize = Normalize(d=d, bounds=box)
        if noise is None:
            # No explicit likelihood is passed, so SingleTaskGP uses BoTorch's
            # default noise model: a GaussianLikelihood with a
            # LogNormalPrior(-4, 1) on the noise and a positive floor. That
            # default is the right noise prior for the small-sample regime, so
            # we deliberately do not override it here.
            self.model = SingleTaskGP(Xt, yt, input_transform=normalize, outcome_transform=Standardize(m=1))
        else:
            # Fixed observation noise: pass train_Yvar (n x 1, original y-units)
            # alongside outcome_transform=Standardize. BoTorch scales Yvar
            # through the Standardize transform internally, and SingleTaskGP
            # automatically selects a FixedNoiseGaussianLikelihood whenever
            # train_Yvar is given, so no explicit likelihood override is
            # needed here either.
            noise_arr = np.broadcast_to(np.asarray(noise, float), (Xa.shape[0],)).astype(float)
            if not np.isfinite(noise_arr).all() or (noise_arr < 0).any():
                raise ValueError("noise must be finite and non-negative")
            yvar = torch.as_tensor(noise_arr, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
            self.model = SingleTaskGP(
                Xt, yt, train_Yvar=yvar, input_transform=normalize, outcome_transform=Standardize(m=1)
            )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        _fit_mll_with_retry(mll)
        self._y = ya
        # Keep the raw (untransformed) training inputs so the noisy-EI acquisition
        # can use them as X_baseline; the model applies its input transform to them
        # internally at acquisition time.
        self._X = Xt
        return self

    def posterior(self, X):
        """Return (mean, std) in the target's original units."""
        assert self.model is not None, "fit() first"
        Xt = torch.as_tensor(np.asarray(X, float), dtype=DTYPE, device=DEVICE)
        self.model.eval()
        with torch.no_grad():
            post = self.model.posterior(Xt)
            mean = post.mean.squeeze(-1).cpu().numpy()
            std = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()
        return mean, std

    @property
    def best_f(self) -> float:
        assert self._y is not None
        return float(self._y.max())


# Composite kernels combine several sub-kernels, each potentially covering a
# different subset of input dimensions with its OWN lengthscale (this is the
# shape `MixedSingleTaskGP` builds: a sum/product of a continuous-dims kernel and
# a separate categorical-dims kernel). There is no single per-original-feature
# lengthscale on a kernel like this, so `_ard_lengthscale` refuses rather than
# arbitrarily picking one term.
_COMPOSITE_KERNEL_TYPES = (AdditiveKernel, ProductKernel)


def _ard_lengthscale(model: SingleTaskGP) -> tuple[np.ndarray | None, str | None]:
    """Pull a single per-input-dimension ARD lengthscale array off a fitted
    model's `covar_module`, or explain why one is not available.

    Returns `(lengthscale, None)` on success - `lengthscale` has shape `(d,)` and
    is in the model's INPUT-TRANSFORMED (normalized) units, see
    `ard_main_effects` for what that means and why it matters. Returns
    `(None, reason)` when there is no single, unambiguous per-feature lengthscale
    to report: the kernel is a composite (additive/product) kernel over per-term
    dimension subsets (see `_COMPOSITE_KERNEL_TYPES` above), the kernel has no
    `lengthscale` parameter at all, or the kernel is not ARD (`ard_num_dims` is
    `None`, meaning BoTorch/GPyTorch fell back to one lengthscale shared across
    every dimension, which carries no per-feature information to rank).
    """
    kernel = getattr(model, "covar_module", None)
    if kernel is None:
        return None, "model has no covar_module"
    if isinstance(kernel, _COMPOSITE_KERNEL_TYPES):
        return None, (
            "model's kernel is a composite (additive/product) kernel over per-term "
            "dimension subsets (e.g. MixedSingleTaskGP's separate continuous + "
            "categorical kernels) - there is no single per-feature lengthscale to report"
        )
    # A ScaleKernel wraps the actual distance kernel in `base_kernel`. The
    # default path `Surrogate.fit()` takes today (no explicit `covar_module`
    # passed to `SingleTaskGP`) installs a bare ARD kernel with no ScaleKernel
    # wrapper, so `covar_module.lengthscale` already works directly - fall back to
    # `kernel` itself when there is no `base_kernel`, so both shapes are handled.
    base = getattr(kernel, "base_kernel", kernel)
    if isinstance(base, _COMPOSITE_KERNEL_TYPES):
        return None, (
            "model's kernel wraps a composite (additive/product) kernel over "
            "per-term dimension subsets - there is no single per-feature "
            "lengthscale to report"
        )
    lengthscale = getattr(base, "lengthscale", None)
    if lengthscale is None:
        return None, "kernel has no fitted lengthscale parameter"
    if getattr(base, "ard_num_dims", None) is None:
        return None, (
            "kernel is not ARD - a single lengthscale is shared across every "
            "input dimension, so there is no per-feature signal to rank"
        )
    return lengthscale.detach().cpu().numpy().reshape(-1), None


def ard_main_effects(surrogate: Surrogate, feature_names: Sequence[str]) -> dict[str, object]:
    """Model-based per-feature main effects, from the fitted GP's ARD lengthscales.

    This is a RELATIVE SENSITIVITY HEURISTIC, not a variance decomposition and not
    a calibrated effect size. BoTorch's default kernel for `SingleTaskGP` (what
    `Surrogate.fit()` always builds when no `covar_module` is passed, the only
    path this codebase uses today) is ARD: one lengthscale per input dimension. A
    short lengthscale means the fitted posterior mean tends to vary faster along
    that dimension near the training data, so the model leans on it more to
    explain what it has seen. `relative_importance` below is
    `(1 / lengthscale_j) / sum_k(1 / lengthscale_k)`, normalized to sum to 1
    across the ranked features.

    What this does NOT measure: it is not a Sobol index, not a fraction of
    variance explained, and not a percentage-based effect size of any kind - do
    not present it as one. It ignores the kernel's outputscale/signal amplitude
    (a dimension can have a short lengthscale and still barely move the output if
    its amplitude there is tiny) and it says nothing about interactions between
    features. It ranks features WITHIN one fitted model only; the lengthscale
    ratios are not meaningful compared across different fits or campaigns.

    Scale dependence - read this before trusting the numbers: ARD lengthscales
    are only comparable across dimensions when every dimension was fit on a
    common scale. `Surrogate.fit()` (`kalos/core/surrogate.py`) always normalizes
    inputs to `[0, 1]` per dimension via botorch's `Normalize(d=d, bounds=box)`
    input transform, tied to the fixed design `bounds` box (not the training-data
    envelope), and the kernel operates on those normalized inputs - so the
    lengthscales this function reads are already on a common per-dimension scale
    and comparable across features FOR A MODEL FIT THROUGH `Surrogate.fit()`.
    This function trusts that contract and does not independently re-normalize
    anything; it must not be pointed at a GP fit without input normalization; on
    such a model "shorter lengthscale" for a wide-range feature versus a
    narrow-range one would just reflect the input's raw units, not any real
    difference in sensitivity, and the ranking would be meaningless.

    `feature_names` must have exactly one name per fitted input dimension, in the
    same column order used to build the training `X` passed to `Surrogate.fit()`;
    a length mismatch raises immediately rather than silently mapping a name to
    the wrong dimension.

    Returns a dict:
      `{"available": True, "reason": None, "effects": [...]}`, where `effects` is
      `[{"name": str, "lengthscale": float, "relative_importance": float}, ...]`
      sorted by `relative_importance`, descending;
    or, when no single per-feature lengthscale exists on this model (a non-ARD
    kernel, or a composite continuous/categorical kernel like `MixedSingleTaskGP`
    builds):
      `{"available": False, "reason": <str explaining why>, "effects": None}`.
    """
    assert surrogate.model is not None, "fit() first"
    assert surrogate._X is not None, "fit() first"
    d = int(surrogate._X.shape[-1])
    names = list(feature_names)
    assert len(names) == d, (
        f"feature_names has {len(names)} entries but the fitted model has {d} "
        "input dimensions - refusing to guess a name-to-dimension mapping"
    )

    lengthscale, reason = _ard_lengthscale(surrogate.model)
    if lengthscale is None:
        return {"available": False, "reason": reason, "effects": None}
    if lengthscale.shape[0] != d:
        # Defensive: would only happen if a future covar_module override reports a
        # lengthscale count that disagrees with the training data's dimension
        # count. Refuse rather than mapping names onto the wrong dimensions.
        return {
            "available": False,
            "reason": (
                f"kernel reports {lengthscale.shape[0]} lengthscale(s) but the "
                f"model has {d} input dimensions - refusing to guess a mapping"
            ),
            "effects": None,
        }

    inv = 1.0 / np.clip(lengthscale, 1e-12, None)
    total = float(inv.sum())
    relative = inv / total if total > 0 else np.zeros_like(inv)
    effects = [
        {
            "name": names[j],
            "lengthscale": round(float(lengthscale[j]), 6),
            "relative_importance": round(float(relative[j]), 6),
        }
        for j in range(d)
    ]
    effects.sort(key=lambda e: -float(e["relative_importance"]))  # type: ignore[arg-type]
    return {"available": True, "reason": None, "effects": effects}
