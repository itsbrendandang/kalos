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

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

DTYPE = torch.double
DEVICE = torch.device("cpu")


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

    def fit(self, X, y, bounds=None) -> "Surrogate":
        """Fit on (X, y). Pass `bounds` (2 x d) to tie input normalization to the
        fixed design box rather than the training data envelope."""
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
        if bounds is not None:
            lower, upper = sanitize_bounds(bounds)  # widen zero-width dims so Normalize is finite
            box = torch.as_tensor(np.vstack([lower, upper]), dtype=DTYPE, device=DEVICE)
            normalize = Normalize(d=d, bounds=box)
        else:
            normalize = Normalize(d=d)
        self.model = SingleTaskGP(Xt, yt, input_transform=normalize, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self._y = ya
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
