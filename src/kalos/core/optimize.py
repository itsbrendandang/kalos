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

Two optional compositions on top of the base qLogNEI, both opt-in via
`propose`'s keyword-only arguments and both OFF by default (byte-identical to
the historical behavior when neither is passed):

- `feasibility_classifier`: gates the acquisition by a fitted
  `FeasibilityClassifier`'s P(feasible), in LOG space -
  `log(EI * p_feasible) = logEI(X) + log p_feasible(X)` - via a small
  torch-differentiable reimplementation of the classifier's fitted sklearn
  pipeline (`_FeasibilityGate`). See `_FeasibilityGatedLogNEI` below for the
  q-batch composition rule. The POLICY of *when* to gate (is the classifier
  really fit, is the sheet zero-inflated enough, is its CV AUC trustworthy)
  is decided by the caller (`kalos.portal.analysis._analyze`), not here -
  this module only knows how to compose the acquisition once told to.
- `constraint_surrogate` + `constraint_floor`: a second outcome (fit as its
  own `Surrogate`, e.g. a purity column) the proposed batch must respect,
  via BoTorch's native `qLogNoisyExpectedImprovement(..., constraints=...)`
  on a `ModelListGP([target, constraint])`. Also decided by the caller
  (whether the column exists, has enough rows, and the constraint model
  actually fits).

The two compose freely: passing both wraps the constrained base acquisition
with the feasibility gate on top.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim import (
    optimize_acqf,
    optimize_acqf_mixed,
    optimize_acqf_mixed_alternating,
)

from .feasibility import FeasibilityClassifier
from .surrogate import DEVICE, DTYPE, Surrogate, sanitize_bounds

logger = logging.getLogger(__name__)

# Enumerating one continuous optimization per categorical combination is exact
# but costs a full multi-start solve per combination. Past this many combinations
# we switch to the alternating optimizer, which does not enumerate. 64 keeps the
# exact path (the better optimizer) for realistic small categorical spaces.
MAX_MIXED_COMBOS = 64


class _FeasibilityGate(torch.nn.Module):
    """Differentiable log P(feasible), rebuilt from a fitted
    `Pipeline(StandardScaler(), LogisticRegression())`.

    `StandardScaler` + `LogisticRegression` compose to an affine map followed by
    a sigmoid: `p = sigmoid(((x - mean) / scale) @ coef + intercept)`. Rebuilding
    that identity as a torch module (float64, CPU - matching `DTYPE`/`DEVICE`,
    the surrogate's own precision and device) keeps the whole gated acquisition
    score differentiable end to end, which `optimize_acqf`'s gradient-based
    multi-start L-BFGS requires; calling back into the sklearn/numpy pipeline
    from inside `forward()` would break autograd at that boundary and leave the
    optimizer with no gradient to climb.

    CATEGORICAL DIMS, STATED EXPLICITLY: the classifier this gate reproduces was
    fit on the same integer level codes the GP sees for categorical columns
    (`kalos.portal.analysis._analyze` fits `FeasibilityClassifier` on the same
    `X` the surrogate uses). A logistic regression treats an integer code as an
    ORDINAL number, not an unordered label, so "level 2 is between level 1 and
    level 3" is an assumption the model makes that the data does not actually
    support. This is a known, deliberate approximation carried over unchanged
    from that fit - one-hot encoding the categorical dims for the classifier is
    a possible refinement, not implemented here.
    """

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        coef: np.ndarray,
        intercept: float,
    ) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.as_tensor(mean, dtype=DTYPE, device=DEVICE))
        self.register_buffer("_scale", torch.as_tensor(scale, dtype=DTYPE, device=DEVICE))
        self.register_buffer("_coef", torch.as_tensor(coef, dtype=DTYPE, device=DEVICE))
        self.register_buffer(
            "_intercept", torch.as_tensor(float(intercept), dtype=DTYPE, device=DEVICE)
        )

    def log_p_feasible(self, X: torch.Tensor) -> torch.Tensor:
        """`log sigmoid(w . (x - mu) / s + b)`, elementwise over `X`'s leading dims.

        `X` is `... x d`; the affine map contracts the trailing (feature) dim, so
        the result has `X`'s shape minus that trailing dimension - e.g. a
        `batch_shape x q x d` candidate tensor from `optimize_acqf` comes back as
        `batch_shape x q`, one log-probability per candidate POINT (not yet
        reduced over `q` - see `_FeasibilityGatedLogNEI` for that reduction).
        """
        z = ((X - self._mean) / self._scale) @ self._coef + self._intercept
        return torch.nn.functional.logsigmoid(z)


class _FeasibilityGatedLogNEI(AcquisitionFunction):
    """`log(EI * p_feasible) = logEI(X) + log p_feasible(X)`, wrapping a base
    qLogNEI (itself optionally already constrained, see `propose`) with a
    `_FeasibilityGate`.

    Q-BATCH COMPOSITION, STATED EXPLICITLY (this is a documented design choice,
    not an incidental implementation detail): the base acquisition already
    reduces `X`'s `q` dimension internally (qLogNEI's own smooth-max over the
    batch's joint improvement), returning one log-EI value per t-batch. The
    per-point `log p_feasible` terms, one per candidate in the q-batch, are
    summed across `q` before being added to that already-reduced log-EI. In
    probability space this is `P(batch feasible) = prod_i P(feasible_i)`: the q
    points' feasibility is treated as INDEPENDENT events. That is an
    approximation - EI itself is not a simple per-point sum over `q`, so "the
    q-batch's EI x its batch feasibility" does not literally decompose this way
    - but it is the natural per-point extension of the q=1 rule this module is
    asked to implement, it keeps every term differentiable and cheap (no extra
    MC sampling), and it errs toward gating a batch MORE harshly as `q` grows
    (each additional low-feasibility point pulls the whole batch's score down),
    which is the conservative direction to be wrong in when zero-inflation is
    exactly the failure mode being guarded against. A tighter joint treatment
    (e.g. a smoothed indicator on the batch's MINIMUM per-point feasibility,
    mirroring how botorch's own `constraints=` argument treats a q-batch
    constraint) is a possible refinement, not implemented here.
    """

    def __init__(self, base_acqf: AcquisitionFunction, gate: _FeasibilityGate) -> None:
        # Mirrors botorch's own acquisition-wrapping pattern (see
        # `FixedFeatureAcquisitionFunction`): call `Module.__init__` directly
        # rather than `AcquisitionFunction.__init__`, since this wrapper has no
        # single `model` of its own to hand the base class - it delegates to
        # `base_acqf.model` below instead, which is what `optimize_acqf`'s
        # initializers (`gen_batch_initial_conditions` et al.) actually read.
        torch.nn.Module.__init__(self)
        self.base_acqf = base_acqf
        self.gate = gate
        self.model = base_acqf.model

    # `optimize_acqf_mixed` (the categorical path) treats the acquisition's
    # pending points as an attribute it can read and write: it reads
    # `acq_function.X_pending` up front and calls `set_X_pending(...)` to pin
    # candidates while it enumerates fixed categorical assignments. A wrapper
    # that does not forward both breaks exactly and only the
    # categorical+gated combination - the continuous path never touches
    # either - which is why the scoped gate tests (continuous designs) stayed
    # green while the full suite's mixed-design tests caught it. Both forward
    # to the base acquisition, which owns the pending state.
    @property
    def X_pending(self) -> torch.Tensor | None:
        return self.base_acqf.X_pending

    def set_X_pending(self, X_pending: torch.Tensor | None = None) -> None:
        self.base_acqf.set_X_pending(X_pending)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        log_ei = self.base_acqf(X)
        log_pf = self.gate.log_p_feasible(X).sum(dim=-1)  # sum over q, see class docstring
        return log_ei + log_pf


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
    feasibility_classifier: FeasibilityClassifier | None = None,
    constraint_surrogate: Surrogate | None = None,
    constraint_floor: float | None = None,
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

    `feasibility_classifier`, when given, gates the acquisition by the
    classifier's P(feasible) in log space (see `_FeasibilityGatedLogNEI`).
    Deliberately defensive: the gate is applied only when the classifier is
    ALSO `.fitted` (not the cold-start fallback, which returns a uniform
    P(feasible)=1 and would multiply the acquisition by a no-op while still
    reading as "gated"); a caller that passes an unfitted classifier gets the
    unchanged, ungated acquisition rather than a misleading gate. Whether to
    pass a classifier AT ALL is a policy decision this function does not make
    (see `kalos.portal.analysis._analyze`'s GATE POLICY) - `None` (the
    default) is byte-identical to the pre-gating behavior.

    `constraint_surrogate` + `constraint_floor`, when both given, add a
    black-box constraint on a SECOND fitted outcome (e.g. `purity >= 95.0`) via
    BoTorch's native `constraints=` argument on `qLogNoisyExpectedImprovement`:
    the target and constraint models are combined into a `ModelListGP`, the
    objective selects the target's output index, and the constraint callable is
    satisfied where `constraint_floor - constraint_value <= 0`. `None` for
    either (the default) leaves the acquisition unconstrained; passing only one
    of the pair raises `ValueError`, since a floor with no model (or a model
    with no floor) cannot be turned into a constraint.

    Both compositions are independent and stack: gating and constraining can be
    combined by passing both.
    """
    assert surrogate.model is not None and surrogate._X is not None, "fit the surrogate first"
    if (constraint_surrogate is None) != (constraint_floor is None):
        raise ValueError(
            "constraint_surrogate and constraint_floor must be given together "
            "(a floor with no model, or a model with no floor, is not a constraint)"
        )
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
        if constraint_surrogate is not None:
            assert constraint_surrogate.model is not None, "fit constraint_surrogate first"
            # ModelListGP explicitly supports differently-shaped training data per
            # sub-model (see its docstring) - the target and constraint surrogates
            # are routinely fit on different row subsets (the constraint column is
            # often sparser than the target), which is exactly this case.
            model = ModelListGP(surrogate.model, constraint_surrogate.model)
            # Posterior samples from the ModelListGP come back with a trailing
            # output dim of 2: [target, constraint], in the order the two models
            # were listed above. The objective selects the target for EI; the
            # constraint callable is satisfied (per qLogNEI's `constraints`
            # contract) where its output is < 0, i.e. constraint_floor - value <= 0
            # <=> value >= constraint_floor.
            objective = GenericMCObjective(lambda Y, X=None: Y[..., 0])
            floor = float(constraint_floor)  # type: ignore[arg-type]
            constraints = [lambda Y: floor - Y[..., 1]]
        else:
            model = surrogate.model
            objective = None
            constraints = None
        acq: AcquisitionFunction = qLogNoisyExpectedImprovement(
            model,
            X_baseline=surrogate._X,
            prune_baseline=True,
            X_pending=X_pending,
            objective=objective,
            constraints=constraints,
        )
        if feasibility_classifier is not None and feasibility_classifier.fitted:
            mean, scale, coef, intercept = feasibility_classifier.torch_gate_params()
            acq = _FeasibilityGatedLogNEI(acq, _FeasibilityGate(mean, scale, coef, intercept))
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
