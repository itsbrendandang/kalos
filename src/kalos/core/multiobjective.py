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

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import SumMarginalLogLikelihood

from .surrogate import DEVICE, DTYPE, _fit_mll_with_retry, sanitize_bounds


class MultiObjectiveSurrogate:
    """One GP per objective over a shared design space; all objectives maximized."""

    def __init__(self) -> None:
        self.model: ModelListGP | None = None
        self._X: torch.Tensor | None = None
        self._Y: torch.Tensor | None = None

    def fit(self, X, Y, bounds) -> "MultiObjectiveSurrogate":
        """Fit one GP per objective. `bounds` (2 x d) is required so input
        normalization is always tied to the fixed design box, never the
        training-data envelope (the same footgun the single-objective fit closes)."""
        Xa = np.asarray(X, float)
        Ya = np.asarray(Y, float)
        if Xa.ndim != 2 or Ya.ndim != 2 or Xa.shape[0] != Ya.shape[0]:
            raise ValueError("X (n x d) and Y (n x m) must share n rows")
        if not np.isfinite(Xa).all() or not np.isfinite(Ya).all():
            raise ValueError("X and Y must be finite (no NaN/inf)")
        Xt = torch.as_tensor(Xa, dtype=DTYPE, device=DEVICE)
        Yt = torch.as_tensor(Ya, dtype=DTYPE, device=DEVICE)
        d = Xt.shape[-1]
        # sanitize_bounds widens zero-width (constant-feature) dims and repairs
        # non-finite bounds so Normalize never divides by a zero-width range and
        # NaN-poisons the fit. Mirrors the single-objective Surrogate.fit guard.
        lower, upper = sanitize_bounds(bounds)
        box = torch.as_tensor(np.vstack([lower, upper]), dtype=DTYPE, device=DEVICE)
        # A fresh Normalize per model: Normalize is a stateful torch.nn.Module, and
        # sharing one instance across every SingleTaskGP in the ModelListGP means a
        # future learn_bounds=True (or any other transform mutation) on one
        # objective would silently leak into every other objective's model through
        # the shared object. Bounds are identical across objectives today, so this
        # is numerically a no-op, but each model owning its own transform removes
        # the aliasing hazard rather than relying on the bounds staying fixed.
        models = [
            SingleTaskGP(
                Xt,
                Yt[:, i : i + 1],
                input_transform=Normalize(d=d, bounds=box),
                outcome_transform=Standardize(m=1),
            )
            for i in range(Yt.shape[-1])
        ]
        self.model = ModelListGP(*models)
        _fit_mll_with_retry(SumMarginalLogLikelihood(self.model.likelihood, self.model))
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


def propose_multiobjective(
    surrogate: MultiObjectiveSurrogate,
    bounds,
    ref_point=None,
    q: int = 3,
    num_restarts: int = 10,
    raw_samples: int = 128,
    mc_samples: int = 128,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Return q proposed points (q x d) that best expand the Pareto front.

    `seed` governs the stochastic work: the qLogNEHVI sampler's Sobol draws
    (`SobolQMCNormalSampler(seed=seed)`) and the multi-start acquisition
    optimization's raw-sample initial conditions inside `optimize_acqf`. `None`
    (the default) seeds nothing here, byte-identical to the historical
    behavior - reproducibility remains the caller's responsibility via seeding
    the global torch RNG beforehand, as the portal and bench already do. When
    given, the seed is applied inside `torch.random.fork_rng()`, which snapshots
    the caller's RNG state and restores it on exit, so a seeded call is
    reproducible without clobbering the caller's global RNG state as a side
    effect.

    Every returned coordinate is clamped into `[lower, upper]` per feature so a
    proposal can never fall outside the observed design box. This mirrors the
    single-objective `core.optimize.propose` guard against the historical
    out-of-range blow-up (a constant `Culture_Volume` proposing ~= 33,000,000)
    for a degenerate or near-constant feature. See `sanitize_bounds`.
    """
    assert surrogate.model is not None and surrogate._X is not None, "fit the surrogate first"
    lower, upper = sanitize_bounds(bounds)  # widen zero-width dims so optimize_acqf sees a valid box
    b = torch.stack(
        [
            torch.as_tensor(lower, dtype=DTYPE, device=DEVICE),
            torch.as_tensor(upper, dtype=DTYPE, device=DEVICE),
        ]
    )
    rp = surrogate.default_ref_point() if ref_point is None else np.asarray(ref_point, float)
    rp = torch.as_tensor(rp, dtype=DTYPE, device=DEVICE)

    def _run_acqf_optimization() -> torch.Tensor:
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=surrogate.model,
            ref_point=rp,
            X_baseline=surrogate._X,
            prune_baseline=True,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]), seed=seed),
        )
        candidates, _ = optimize_acqf(
            acq_function=acq, bounds=b, q=q, num_restarts=num_restarts, raw_samples=raw_samples
        )
        return candidates

    if seed is None:
        candidates = _run_acqf_optimization()
    else:
        # devices=[] restricts the fork to CPU RNG state (DEVICE is always CPU
        # here), avoiding an unnecessary probe of CUDA RNG state on machines
        # without a GPU.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            candidates = _run_acqf_optimization()
    out = candidates.detach().cpu().numpy()
    # Belt-and-suspenders: clamp each coordinate back into the observed box. The
    # optimizer respects the bounds it is given, but a NaN/degenerate corner or a
    # numerical overshoot must never surface as an absurd out-of-range recipe.
    return np.clip(out, lower, upper)


__all__ = ["MultiObjectiveSurrogate", "propose_multiobjective"]
