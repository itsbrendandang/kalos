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


class Surrogate:
    """SingleTaskGP over a continuous design space."""

    def __init__(self) -> None:
        self.model: SingleTaskGP | None = None
        self._y: np.ndarray | None = None

    def fit(self, X, y) -> "Surrogate":
        Xt = torch.as_tensor(np.asarray(X, float), dtype=DTYPE, device=DEVICE)
        yt = torch.as_tensor(np.asarray(y, float), dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        self.model = SingleTaskGP(
            Xt,
            yt,
            input_transform=Normalize(d=Xt.shape[-1]),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self._y = np.asarray(y, float).reshape(-1)
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
