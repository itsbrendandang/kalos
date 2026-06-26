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

from .surrogate import DEVICE, DTYPE, Surrogate


def propose(
    surrogate: Surrogate,
    bounds,
    q: int = 3,
    num_restarts: int = 10,
    raw_samples: int = 256,
) -> np.ndarray:
    """Return `q` proposed points (shape q x d) maximizing constrained-free qLogEI.

    bounds: array-like of shape (2, d) = [lower_row, upper_row].
    """
    assert surrogate.model is not None, "fit the surrogate first"
    b = torch.as_tensor(np.asarray(bounds, float), dtype=DTYPE, device=DEVICE)
    acq = qLogExpectedImprovement(surrogate.model, best_f=surrogate.best_f)
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=b,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return candidates.detach().cpu().numpy()
