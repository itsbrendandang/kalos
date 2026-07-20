"""Acquisition + proposal: pick the next experiments with BoTorch.

Single-objective: q-batch Log Noisy Expected Improvement (qLogNEI). Titer / yield
are noisy measurements, so the NOISY variant is the right choice: it integrates
improvement over the posterior at the observed baseline points (`X_baseline`)
rather than trusting a single noiseless incumbent (`best_f`), which the plain
qLogNEHVI. Proposals are optimized inside the design bounds with multi-start
L-BFGS (`optimize_acqf`).

When the design space has categorical dimensions (`cat_dims`), the continuous
`optimize_acqf` is replaced by `optimize_acqf_mixed`, which optimizes the
continuous coordinates for every categorical assignment and keeps the best. For
a large categorical space that enumeration is skipped in favor of
`optimize_acqf_mixed_alternating`.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import (
    optimize_acqf,
    optimize_acqf_mixed,
    optimize_acqf_mixed_alternating,
)

from .surrogate import DEVICE, DTYPE, Surrogate, sanitize_bounds

# Enumerating one continuous optimization per categorical combination is exact
# but costs a full multi-start solve per combination. Past this many combinations
# we switch to the alternating optimizer, which does not enumerate. 64 keeps the
# exact path (the better optimizer) for realistic small categorical spaces.
MAX_MIXED_COMBOS = 64


def propose(
    surrogate: Surrogate,
    bounds,
    q: int = 3,
    num_restarts: int = 10,
    raw_samples: int = 256,
    *,
    cat_dims: list[int] | None = None,
    cat_cardinalities: list[int] | None = None,
) -> np.ndarray:
    """Return `q` proposed points (shape q x d) maximizing constrained-free qLogNEI.

    bounds: array-like of shape (2, d) = [lower_row, upper_row].

    `cat_dims` marks categorical columns (integer level codes); when given,
    `cat_cardinalities` gives the number of levels for each, aligned to
    `cat_dims`. With no categoricals the continuous `optimize_acqf` path is
    unchanged.

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
    if cat_dims:
        candidates = _optimize_mixed(
            acq, b, q, num_restarts, raw_samples, cat_dims, cat_cardinalities or []
        )
    else:
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
    out = np.clip(out, lower, upper)
    # Snap categorical coordinates to their integer level codes: optimize_acqf_mixed
    # already pins them, but the clip + float round-trip must never leave a
    # fractional code that would decode to the wrong level.
    for j in cat_dims or []:
        out[:, j] = np.rint(out[:, j])
    return out


def _optimize_mixed(
    acq,
    b: torch.Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    cat_dims: list[int],
    cat_cardinalities: list[int],
) -> torch.Tensor:
    """Optimize a mixed continuous/categorical acquisition.

    Enumerates every categorical assignment through `optimize_acqf_mixed` (exact:
    one continuous solve per combination) when the number of combinations is
    small; otherwise falls back to `optimize_acqf_mixed_alternating`, which
    alternates continuous and categorical moves without enumerating.
    """
    cardinalities = [max(int(k), 1) for k in cat_cardinalities]
    n_combos = 1
    for k in cardinalities:
        n_combos *= k
    if n_combos <= MAX_MIXED_COMBOS:
        fixed_features_list = [
            {dim: float(code) for dim, code in zip(cat_dims, combo)}
            for combo in itertools.product(*(range(k) for k in cardinalities))
        ]
        candidates, _ = optimize_acqf_mixed(
            acq_function=acq,
            bounds=b,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features_list=fixed_features_list,
        )
        return candidates
    cat_map = {
        dim: [float(code) for code in range(k)]
        for dim, k in zip(cat_dims, cardinalities)
    }
    candidates, _ = optimize_acqf_mixed_alternating(
        acq_function=acq,
        bounds=b,
        cat_dims=cat_map,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return candidates
