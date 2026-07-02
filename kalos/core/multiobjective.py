"""Multi-objective Bayesian optimization (e.g. titer AND purity together).

A bioprocess rarely optimizes one number — you want high titer without losing
purity. This fits one GP per objective (a `ModelListGP`) and proposes the next
batch by q-Log Noisy Expected Hypervolume Improvement (qLogNEHVI), the current
BoTorch state of the art for noisy multi-objective BO. Proposals expand the
Pareto front: the set of recipes where you can't improve one objective without
giving up another.

All objectives are MAXIMIZED. Runs on CPU float64 (same as the single-objective
surrogate).
"""
from __future__ import annotations

import inspect

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import SumMarginalLogLikelihood

from .surrogate import DEVICE, DTYPE


class MultiObjectiveSurrogate:
    """One GP per objective over a shared design space; all objectives maximized."""

    def __init__(self) -> None:
        self.model: ModelListGP | None = None
        self._X: torch.Tensor | None = None
        self._Y: torch.Tensor | None = None

    def fit(self, X, Y, bounds=None) -> "MultiObjectiveSurrogate":
        Xa = np.asarray(X, float)
        Ya = np.asarray(Y, float)
        if Xa.ndim != 2 or Ya.ndim != 2 or Xa.shape[0] != Ya.shape[0]:
            raise ValueError("X (n x d) and Y (n x m) must share n rows")
        if not np.isfinite(Xa).all() or not np.isfinite(Ya).all():
            raise ValueError("X and Y must be finite (no NaN/inf)")
        Xt = torch.as_tensor(Xa, dtype=DTYPE, device=DEVICE)
        Yt = torch.as_tensor(Ya, dtype=DTYPE, device=DEVICE)
        d = Xt.shape[-1]
        norm = (
            Normalize(d=d, bounds=torch.as_tensor(np.asarray(bounds, float), dtype=DTYPE, device=DEVICE))
            if bounds is not None
            else Normalize(d=d)
        )
        models = [
            SingleTaskGP(Xt, Yt[:, i : i + 1], input_transform=norm, outcome_transform=Standardize(m=1))
            for i in range(Yt.shape[-1])
        ]
        self.model = ModelListGP(*models)
        fit_gpytorch_mll(SumMarginalLogLikelihood(self.model.likelihood, self.model))
        self._X = Xt
        self._Y = Yt
        return self

    def pareto(self):
        """Return (X, Y) of the non-dominated observed points (the current front)."""
        assert self._Y is not None
        mask = is_non_dominated(self._Y)
        return self._X[mask].cpu().numpy(), self._Y[mask].cpu().numpy()

    def default_ref_point(self, margin: float = 0.1) -> np.ndarray:
        """A reference point dominated by every observation (per-objective min,
        nudged down) — the anchor the hypervolume is measured against."""
        assert self._Y is not None
        y = self._Y
        span = (y.max(dim=0).values - y.min(dim=0).values).clamp_min(1e-9)
        return (y.min(dim=0).values - margin * span).cpu().numpy()


def _make_sampler(mc_samples: int, seed: int) -> SobolQMCNormalSampler:
    """Build the MC sampler, passing `seed` if this BoTorch supports it; otherwise
    seed the global torch RNG before constructing it so sampling is reproducible."""
    sample_shape = torch.Size([mc_samples])
    if "seed" in inspect.signature(SobolQMCNormalSampler.__init__).parameters:
        return SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)
    torch.manual_seed(seed)
    return SobolQMCNormalSampler(sample_shape=sample_shape)


def propose_multiobjective(
    surrogate: MultiObjectiveSurrogate,
    bounds,
    ref_point=None,
    q: int = 3,
    num_restarts: int = 10,
    raw_samples: int = 128,
    mc_samples: int = 128,
    seed: int = 0,
) -> np.ndarray:
    """Return q proposed points (q x d) that best expand the Pareto front.

    seed: seeds the MC sampler and the acquisition optimizer for reproducibility.
    """
    assert surrogate.model is not None and surrogate._X is not None, "fit the surrogate first"
    b = torch.as_tensor(np.asarray(bounds, float), dtype=DTYPE, device=DEVICE)
    rp = surrogate.default_ref_point() if ref_point is None else np.asarray(ref_point, float)
    rp = torch.as_tensor(rp, dtype=DTYPE, device=DEVICE)
    acq = qLogNoisyExpectedHypervolumeImprovement(
        model=surrogate.model,
        ref_point=rp,
        X_baseline=surrogate._X,
        prune_baseline=True,
        sampler=_make_sampler(mc_samples, seed),
    )
    torch.manual_seed(seed)
    candidates, _ = optimize_acqf(
        acq_function=acq, bounds=b, q=q, num_restarts=num_restarts, raw_samples=raw_samples
    )
    return candidates.detach().cpu().numpy()


__all__ = ["MultiObjectiveSurrogate", "propose_multiobjective"]
