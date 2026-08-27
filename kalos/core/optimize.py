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
import logging

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import (
    optimize_acqf,
    optimize_acqf_mixed,
    optimize_acqf_mixed_alternating,
)

from .surrogate import DEVICE, DTYPE, Surrogate, sanitize_bounds

logger = logging.getLogger(__name__)

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
    seed: int | None = None,
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

    `seed` governs the stochastic multi-start acquisition optimization (the
    raw-sample initial condition draws inside `optimize_acqf` /
    `optimize_acqf_mixed*`). `None` (the default) seeds nothing here, which is
    byte-identical to the historical behavior: reproducibility is entirely the
    caller's responsibility, via seeding the global torch RNG before calling
    (as the portal's `_seed_everything` and the bench harness already do). A
    library consumer that never seeds the global RNG gets silently
    non-reproducible proposals under that contract - passing an explicit `seed`
    here closes that gap. It is applied inside `torch.random.fork_rng()`, which
    snapshots the caller's RNG state, seeds only within the block, and restores
    the snapshot on exit - so a seeded call is reproducible without clobbering
    the caller's own global RNG state as a side effect (a library function that
    resets global state on the caller behind their back is a separate bug, not
    a fix for this one).

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
    X_pending = _sanitize_pending(pending, lower, upper, cat_dims)

    def _run_acqf_optimization() -> torch.Tensor:
        # Noisy EI over the observed baseline (titer/yield are noisy), pruning
        # baseline points that cannot be optimal so the acquisition stays cheap.
        # X_baseline is the raw training design; the model applies its input
        # transform internally. `prune_baseline=True` draws posterior samples to
        # decide what to prune, so building `acq` is itself stochastic - it must
        # live inside this closure (and therefore inside the `seed` fork below)
        # alongside `optimize_acqf`'s own raw-sample draws, or a seeded call
        # would still leak unseeded randomness into the caller's global RNG.
        acq = qLogNoisyExpectedImprovement(
            surrogate.model,
            X_baseline=surrogate._X,
            prune_baseline=True,
            X_pending=X_pending,
        )
        if cat_dims:
            return _optimize_mixed(
                acq, b, q, num_restarts, raw_samples, cat_dims, cat_cardinalities or []
            )
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=b,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        return candidates

    if seed is None:
        candidates = _run_acqf_optimization()
    else:
        # devices=[] restricts the fork to CPU RNG state (DEVICE is always CPU
        # here), avoiding an unnecessary and potentially warning-raising probe
        # of CUDA RNG state on machines without a GPU.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            candidates = _run_acqf_optimization()
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

    The wholesale width-mismatch rejection and any row-level drop from the
    finite-value filter are each logged once at warning level, not silent: a
    caller bug that disables the in-flight guard (e.g. passing the wrong
    design's pending block) would otherwise be indistinguishable downstream
    from "nothing is running" - both read as `n_pending_considered == 0`. The
    documented "nothing running" fast paths (`pending is None` / empty) are the
    normal, expected case and are not logged.
    """
    if pending is None:
        return None
    arr = np.asarray(pending, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    if arr.shape[1] != lower.shape[0]:
        logger.warning(
            "pending block rejected: expected width %d, got width %d (%d rows)",
            lower.shape[0],
            arr.shape[1],
            arr.shape[0],
        )
        return None
    total = arr.shape[0]
    finite_mask = np.isfinite(arr).all(axis=1)
    arr = arr[finite_mask]
    if arr.shape[0] < total:
        logger.warning(
            "pending block dropped %d of %d rows: non-finite coordinates",
            total - arr.shape[0],
            total,
        )
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
