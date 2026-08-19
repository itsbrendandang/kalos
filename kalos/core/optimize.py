"""Acquisition + proposal: pick the next experiments with BoTorch.

Single-objective: q-batch Log Noisy Expected Improvement (qLogNEI). Titer / yield
are noisy measurements, so the NOISY variant is the right choice: it integrates
improvement over the posterior at the observed baseline points (`X_baseline`)
rather than trusting a single noiseless incumbent (`best_f`), which the plain
qLogEI does. This mirrors the multi-objective path, which already uses the noisy
qLogNEHVI. Proposals are optimized inside the design bounds with multi-start
L-BFGS (`optimize_acqf`).

When the design space has categorical dimensions (`cat_dims`), the continuous
`optimize_acqf` is replaced by `optimize_acqf_mixed`, which optimizes the
continuous coordinates for every categorical assignment and keeps the best. For
a large categorical space that enumeration is skipped in favor of
`optimize_acqf_mixed_alternating`.

Recipes that are already running but not yet measured are passed as `pending`
and become the acquisition's `X_pending`. Without them a re-proposal after a
partial round is computed as if the in-flight runs did not exist, so the
optimizer happily proposes a recipe already incubating in the shaker: the
scientist spends budget twice on one point and the round returns less
information than it cost.
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
    pending: np.ndarray | None = None,
) -> np.ndarray:
    """Return `q` proposed points (shape q x d) maximizing constrained-free qLogNEI.

    bounds: array-like of shape (2, d) = [lower_row, upper_row].

    `cat_dims` marks categorical columns (integer level codes); when given,
    `cat_cardinalities` gives the number of levels for each, aligned to
    `cat_dims`. With no categoricals the continuous `optimize_acqf` path is
    unchanged.

    `pending` is an optional `(m, d)` block of recipes that have been STARTED but
    not yet measured. They are handed to the acquisition as `X_pending`, which
    integrates over their unknown outcomes, so the batch returned here explores
    away from work already in flight instead of re-proposing it. Rows that are
    not encodable in this design (wrong width, non-finite) are dropped rather
    than raising: a malformed in-flight record must not be able to block the
    optimizer from proposing anything at all. Pass `None` (the default) when
    nothing is running, which leaves the acquisition unchanged.

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
    X_pending = _sanitize_pending(pending, lower, upper, cat_dims)
    acq = qLogNoisyExpectedImprovement(
        surrogate.model,
        X_baseline=surrogate._X,
        prune_baseline=True,
        X_pending=X_pending,
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


def _sanitize_pending(
    pending,
    lower: np.ndarray,
    upper: np.ndarray,
    cat_dims: list[int] | None,
) -> torch.Tensor | None:
    """Coerce `pending` into an `(m, d)` tensor of in-flight design points, or None.

    Applies the same discipline the returned proposals get: coordinates are
    clamped into the design box and categorical coordinates are snapped to their
    integer level codes, so an in-flight recipe recorded slightly outside the box
    (a hand-typed value, a unit round-trip) still marks the right neighborhood as
    taken instead of being silently ignored or corrupting the acquisition.

    Returns None - meaning "no pending points", the unchanged acquisition - when
    `pending` is None, empty, or has no usable row. A row is unusable if it is
    not finite; rows are dropped individually. A block whose width does not match
    the design is rejected wholesale (it is a caller bug, not a bad row), again by
    returning None rather than raising, because failing to propose is a worse
    outcome than proposing without the pending penalty.
    """
    if pending is None:
        return None
    arr = np.asarray(pending, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    if arr.shape[1] != lower.shape[0]:
        return None
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] == 0:
        return None
    arr = np.clip(arr, lower, upper)
    for j in cat_dims or []:
        arr[:, j] = np.rint(arr[:, j])
    return torch.as_tensor(arr, dtype=DTYPE, device=DEVICE)


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
