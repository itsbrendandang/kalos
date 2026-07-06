"""Acquisition + proposal: pick the next experiments with BoTorch.

Single-objective: q-batch Log Noisy Expected Improvement (qLogNEI). Titer / yield
are noisy measurements, so the NOISY variant is the right choice: it integrates
improvement over the posterior at the observed baseline points (`X_baseline`)
rather than trusting a single noiseless incumbent (`best_f`), which the plain
qLogEI does. This mirrors the multi-objective path, which already uses the noisy
qLogNEHVI. Proposals are optimized inside the design bounds with multi-start
L-BFGS (`optimize_acqf`).
"""
from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf

from .surrogate import DEVICE, DTYPE, Surrogate, sanitize_bounds


def propose(
    surrogate: Surrogate,
    bounds,
    q: int = 3,
    num_restarts: int = 10,
    raw_samples: int = 256,
) -> np.ndarray:
    """Return `q` proposed points (shape q x d) maximizing constrained-free qLogNEI.

    bounds: array-like of shape (2, d) = [lower_row, upper_row].

    Every returned coordinate is clamped into `[lower, upper]` per feature so a
    proposal can never fall outside the observed design box. This guards against
    the historical out-of-range blow-up (e.g. a `Culture_Volume ~= 33,000,000`
    proposal) that the acquisition optimizer could produce for a degenerate or
    near-constant feature. See `sanitize_bounds` for the degenerate handling.
    """
    assert surrogate.model is not None and surrogate._X is not None, "fit the surrogate first"
    lower, upper = sanitize_bounds(bounds)
    b = torch.stack(
        [
            torch.as_tensor(lower, dtype=DTYPE, device=DEVICE),
            torch.as_tensor(upper, dtype=DTYPE, device=DEVICE),
        ]
    )
    # Noisy EI over the observed baseline (titer/yield are noisy), pruning baseline
    # points that cannot be optimal so the acquisition stays cheap. X_baseline is
    # the raw training design; the model applies its input transform internally.
    acq = qLogNoisyExpectedImprovement(
        surrogate.model, X_baseline=surrogate._X, prune_baseline=True
    )
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
