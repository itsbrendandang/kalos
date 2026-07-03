"""Acquisition + proposal: pick the next experiments with BoTorch.

Single-objective: q-batch Log Expected Improvement (qLogEI), the current BoTorch
default for noisy EI. Proposals are optimized inside the design bounds with
multi-start L-BFGS (`optimize_acqf`). Multi-objective (qNEHVI) is a documented
next step.
"""
from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf

from .surrogate import DEVICE, DTYPE, Surrogate, sanitize_bounds


def propose(
    surrogate: Surrogate,
    bounds,
    q: int = 3,
    num_restarts: int = 10,
    raw_samples: int = 256,
) -> np.ndarray:
    """Return `q` proposed points (shape q x d) maximizing constrained-free qLogEI.

    bounds: array-like of shape (2, d) = [lower_row, upper_row].

    Every returned coordinate is clamped into `[lower, upper]` per feature so a
    proposal can never fall outside the observed design box. This guards against
    the historical out-of-range blow-up (e.g. a `Culture_Volume ~= 33,000,000`
    proposal) that the acquisition optimizer could produce for a degenerate or
    near-constant feature. See `_sanitize_bounds` for the degenerate handling.
    """
    assert surrogate.model is not None, "fit the surrogate first"
    lower, upper = sanitize_bounds(bounds)
    b = torch.stack(
        [
            torch.as_tensor(lower, dtype=DTYPE, device=DEVICE),
            torch.as_tensor(upper, dtype=DTYPE, device=DEVICE),
        ]
    )
    acq = qLogExpectedImprovement(surrogate.model, best_f=surrogate.best_f)
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=b,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    out = candidates.detach().cpu().numpy()
    # Belt-and-suspenders: clamp each coordinate back into the observed box. The
    # optimizer respects the bounds it is given, but a NaN/degenerate corner or a
    # numerical overshoot must never surface as an absurd out-of-range recipe.
    return np.clip(out, lower, upper)
