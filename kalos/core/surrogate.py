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

import gpytorch
import numpy as np
import torch
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
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


def _build_mixed_gp(
    Xt: torch.Tensor,
    yt: torch.Tensor,
    box: torch.Tensor,
    cat_dims: list[int],
    d: int,
    yvar: torch.Tensor | None,
) -> MixedSingleTaskGP:
    """A `MixedSingleTaskGP` with the continuous dims input-normalized to the
    design box and the categorical dims (integer codes) passed through.

    `box` is the sanitized full `(2, d)` box. Only the continuous columns are
    normalized (via `Normalize(indices=...)`); normalizing the categorical codes
    would corrupt the CategoricalKernel's equality test. When every dim is
    categorical there is nothing to normalize, so no input transform is applied.
    """
    cat_set = set(cat_dims)
    cont_dims = [i for i in range(d) if i not in cat_set]
    input_transform = (
        Normalize(d=d, indices=cont_dims, bounds=box[:, cont_dims])
        if cont_dims
        else None
    )
    kwargs: dict = {
        "cat_dims": cat_dims,
        "input_transform": input_transform,
        "outcome_transform": Standardize(m=1),
    }
    if yvar is not None:
        kwargs["train_Yvar"] = yvar
    return MixedSingleTaskGP(Xt, yt, **kwargs)


class Surrogate:
    """SingleTaskGP over a continuous design space, or a MixedSingleTaskGP when
    some dimensions are categorical (`cat_dims`)."""

    def __init__(self) -> None:
        self.model: SingleTaskGP | MixedSingleTaskGP | None = None
        self._y: np.ndarray | None = None
        self._X: torch.Tensor | None = None
        # Integer indices of categorical columns in X (None/empty = all continuous).
        # Recorded so the acquisition path knows which dims to enumerate.
        self._cat_dims: list[int] | None = None

    def fit(self, X, y, bounds, *, noise=None, cat_dims=None) -> "Surrogate":
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

        `cat_dims` optionally marks a subset of columns as categorical (their
        values are integer level codes, not measurements). When given, the GP is
        a `MixedSingleTaskGP` with a CategoricalKernel on those dims and a Matern
        kernel on the rest; the continuous dims are input-normalized to the box
        and the categorical codes pass through untouched. When `None`/empty the
        continuous `SingleTaskGP` path is unchanged.
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

        yvar: torch.Tensor | None = None
        if noise is not None:
            # Fixed observation noise: build train_Yvar (n x 1, original y-units).
            # BoTorch scales Yvar through the Standardize outcome transform and
            # selects a FixedNoiseGaussianLikelihood automatically, so no explicit
            # likelihood override is needed on either the continuous or mixed path.
            noise_arr = np.broadcast_to(np.asarray(noise, float), (Xa.shape[0],)).astype(float)
            if not np.isfinite(noise_arr).all() or (noise_arr < 0).any():
                raise ValueError("noise must be finite and non-negative")
            yvar = torch.as_tensor(noise_arr, dtype=DTYPE, device=DEVICE).reshape(-1, 1)

        cats = [int(i) for i in cat_dims] if cat_dims else []
        if cats:
            self.model = _build_mixed_gp(Xt, yt, box, cats, d, yvar)
        else:
            normalize = Normalize(d=d, bounds=box)
            if yvar is None:
                # No explicit likelihood is passed, so SingleTaskGP uses BoTorch's
                # default noise model: a GaussianLikelihood with a
                # LogNormalPrior(-4, 1) on the noise and a positive floor. That
                # default is the right noise prior for the small-sample regime, so
                # we deliberately do not override it here.
                self.model = SingleTaskGP(Xt, yt, input_transform=normalize, outcome_transform=Standardize(m=1))
            else:
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
        self._cat_dims = cats or None
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
