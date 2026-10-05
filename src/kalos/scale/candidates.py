"""Two candidate fixes for v0's extrapolate-up ranking failure (see this
package's report), plus the shared protocol `leave_one_scale_out_report`
uses to evaluate any of the three under one harness.

WHY THIS FILE EXISTS RATHER THAN EDITING `kalos.core.surrogate.Surrogate`.
`Surrogate.fit` hardcodes a `ConstantMean` (botorch's `SingleTaskGP` default
when no `mean_module` is passed) and a plain continuous-only kernel path.
Threading a custom `mean_module` or a fidelity-kernel model class through
`Surrogate` would mean editing `kalos/core/` - off-limits for this change.
So both candidates below are built directly on botorch/gpytorch primitives,
DELIBERATELY MIRRORING `Surrogate.fit`'s transform stack line for line:

  - `botorch.models.transforms.input.Normalize` on the same `(2, d)` box
    contract (`sanitize_bounds`, imported unchanged from
    `kalos.core.surrogate` rather than re-implemented, so a degenerate bound
    is handled identically to the v0 path).
  - `botorch.models.transforms.outcome.Standardize(m=1)`.
  - The same escalating-jitter fit retry (`kalos.core.surrogate`'s private
    `_fit_mll_with_retry`, imported rather than duplicated - the ladder is
    exact retry-count/jitter-value behavior that would silently drift if
    copied by hand).
  - The same `(mean, std)` in original units returned from `.posterior`,
    with the same `observation_noise` semantics as `Surrogate.posterior`.

This is why the comparison in `leave_one_scale_out_report` is apples-to-
apples: every knob that is not the one thing each candidate changes (the
mean function for A, the kernel + explicit fidelity dim for B) is identical
to what v0 (`Surrogate`) does.

CANDIDATE A - `PhysicsMeanSurrogate` (physics-informed MEAN function).
v0's `ConstantMean` prior is scale-blind: at fold-fit time the GP's prior
belief about titer at every point in the design space, including a 2000 L
row that shares no training neighbor, is the same flat constant. The scale
trend then has to be learned entirely through the kernel's covariance
structure, which is exactly the failure mode this candidate targets - a
kernel modeling a monotonic trend degenerates toward mean-reversion far from
training data (the textbook "GP extrapolates to its prior mean" behavior),
which is consistent with v0 beating naive-mean on MAE (it gets the LEVEL
roughly right) while losing on Spearman (it has nothing left to rank WITH
once the trend is subtracted, because the trend IS most of what the kernel
learned). `PhysicsMeanSurrogate` instead gives the GP a mean function that is
LINEAR in a fixed subset of the physics scale features
(`DEFAULT_PHYSICS_MEAN_FEATURES`: log-volume, the kLa proxy, the
hydrostatic-pressure/pCO2-driver proxy), leaving the Matern kernel to model
the RESIDUAL - the recipe-driven variation that should carry the ranking
signal - rather than re-deriving the scale trend from scratch inside the
kernel every fold.

CANDIDATE B - `MultiFidelitySurrogate` (scale-as-fidelity).
Treats `log_volume_ratio` (already `log10(scale_L / reference)`, already one
of the `PHYSICS_FEATURE_COLUMNS` `build_feature_matrix` computes) as a
BoTorch multi-fidelity dimension via `SingleTaskMultiFidelityGP`
(`botorch.models.gp_regression_fidelity`), rather than an ordinary
continuous input. This is the fix the research memo's arXiv:2508.10970
precedent argues for: scale is not "just another feature" in this framing,
it is a dimension along which correlated-but-imperfect information is
available (small-scale runs cheap and plentiful, the 2000 L target
expensive and rare) - precisely the structure multi-fidelity kernels are
built for. See `MultiFidelitySurrogate`'s docstring for the installed-class
verification (fidelity dims, Normalize/Standardize support) this is built
against.

CANDIDATE C - v0 unchanged. Not reimplemented here; `leave_one_scale_out_report`
runs `kalos.core.surrogate.Surrogate` directly when no `model_factory` is
passed, exactly as before this file existed.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import gpytorch
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from numpy.typing import ArrayLike, NDArray

# Imported, not reimplemented - see the module docstring's "WHY THIS FILE
# EXISTS" section for why reusing these exactly (rather than hand-copying
# the retry ladder or the bounds-sanitization rule) matters for the
# apples-to-apples comparison.
from kalos.core.surrogate import DEVICE, DTYPE, _fit_mll_with_retry, sanitize_bounds


@runtime_checkable
class ScaleCandidateModel(Protocol):
    """The contract `leave_one_scale_out_report` needs from ANY candidate -
    `kalos.core.surrogate.Surrogate`'s own `fit`/`posterior` shape, restated
    as a `Protocol` so the harness can accept `Surrogate` and either
    candidate below interchangeably without a shared base class."""

    def fit(
        self, X: ArrayLike, y: ArrayLike, *, bounds: ArrayLike, noise: ArrayLike | float | None = None
    ) -> "ScaleCandidateModel": ...

    def posterior(
        self, X: ArrayLike, *, observation_noise: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def _prep_fit_inputs(X: ArrayLike, y: ArrayLike, bounds: ArrayLike):
    """Shared input validation + tensor/box prep, mirroring
    `Surrogate.fit`'s own checks exactly (same error messages, same
    `sanitize_bounds` call) so a bad input fails the same way under any of
    the three candidates."""
    Xa = np.asarray(X, float)
    ya = np.asarray(y, float).reshape(-1)
    if Xa.ndim != 2 or Xa.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-D array (n x d)")
    if Xa.shape[0] != ya.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if not np.isfinite(Xa).all() or not np.isfinite(ya).all():
        raise ValueError("X and y must be finite (no NaN/inf)")
    Xt = torch.as_tensor(Xa, dtype=DTYPE, device=DEVICE)
    yt = torch.as_tensor(ya, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
    lower, upper = sanitize_bounds(bounds)
    box = torch.as_tensor(np.vstack([lower, upper]), dtype=DTYPE, device=DEVICE)
    return Xa, ya, Xt, yt, box


def _prep_noise(noise: ArrayLike | float | None, n: int) -> torch.Tensor | None:
    if noise is None:
        return None
    noise_arr = np.broadcast_to(np.asarray(noise, float), (n,)).astype(float)
    if not np.isfinite(noise_arr).all() or (noise_arr < 0).any():
        raise ValueError("noise must be finite and non-negative")
    return torch.as_tensor(noise_arr, dtype=DTYPE, device=DEVICE).reshape(-1, 1)


def _posterior(model, X: ArrayLike, observation_noise: bool):
    assert model is not None, "fit() first"
    Xt = torch.as_tensor(np.asarray(X, float), dtype=DTYPE, device=DEVICE)
    model.eval()
    with torch.no_grad():
        post = model.posterior(Xt, observation_noise=observation_noise)
        mean = post.mean.squeeze(-1).cpu().numpy()
        std = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()
    return mean, std


class _SubsetLinearMean(gpytorch.means.Mean):
    """`gpytorch.means.LinearMean`, restricted to a SUBSET of input dims.

    Mirrors `LinearMean.forward` exactly (`x.matmul(weights).squeeze(-1) +
    bias`) except `x` is first indexed down to `feature_indices` - the
    physics scale features - rather than using every column, since a plain
    `LinearMean` would also put process-feature slopes into the trend term,
    defeating the point (the kernel should see ALL the recipe variation,
    not just what a linear scale-mean leaves over).

    Weights and bias are ZERO-initialized (not `LinearMean`'s
    `torch.randn` default): an all-zero start is a no-op mean (equivalent to
    `ConstantMean` at 0) that `fit_gpytorch_mll`'s optimizer then moves via
    gradient ascent on the marginal likelihood, exactly like every other
    gpytorch parameter in this module's fits. This is a deliberate
    determinism choice - see `PhysicsMeanSurrogate`'s docstring "DETERMINISM"
    note.
    """

    def __init__(self, feature_indices: Sequence[int]) -> None:
        super().__init__()
        self.feature_indices = [int(i) for i in feature_indices]
        n = len(self.feature_indices)
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.zeros(n, 1, dtype=DTYPE)))
        self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.zeros(1, dtype=DTYPE)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.tensor(self.feature_indices, dtype=torch.long, device=x.device)
        x_sub = x.index_select(-1, idx)
        return x_sub.matmul(self.weights).squeeze(-1) + self.bias


# The three physics features the mean function trends on: log volume (the
# single most standard scale-up axis), the van't-Riet kLa proxy (the mixing/
# mass-transfer driver), and the hydrostatic-pressure proxy (the documented
# pCO2-accumulation driver at >500 L - see `kalos.scale.features`'s
# `hydrostatic_pressure_proxy` docstring). Deliberately NOT
# `specific_power_w_per_m3` / `superficial_gas_velocity_m_per_s` - those two
# feed INTO the kLa proxy (`kla_proxy_per_s = f(P/V, vs)`) rather than being
# independent drivers, so including them alongside kLa would let the linear
# mean fit collinear, redundant slopes.
DEFAULT_PHYSICS_MEAN_FEATURES: tuple[str, ...] = (
    "log_volume_ratio",
    "kla_proxy_per_s",
    "hydrostatic_pressure_mmHg",
)


class PhysicsMeanSurrogate:
    """Candidate A: a `SingleTaskGP` whose mean function is linear in
    `mean_feature_names` (default `DEFAULT_PHYSICS_MEAN_FEATURES`) instead of
    the constant `Surrogate` uses. See the module docstring for the full
    rationale and the transform-mirroring contract.

    `feature_names` must be the SAME ordered column-name list
    `kalos.scale.transfer.build_feature_matrix` returns (`process_columns +
    PHYSICS_FEATURE_COLUMNS`) - this is how the mean function finds which
    columns of `X` are the physics features to trend on; a caller passing a
    mismatched `X`/`feature_names` pair gets silently wrong indices, not a
    caught error, the same contract `ScaleUpTransferModel.predict`'s
    `feature_names` assertion protects on the v0 path.

    DETERMINISM. Every parameter in this model (the linear mean's weights/
    bias, the kernel's lengthscales, the likelihood's noise) starts from a
    fixed, non-random initialization - `_SubsetLinearMean`'s zero init,
    gpytorch's own deterministic kernel/likelihood defaults (see this
    package's tests, `test_scale_v1_determinism.py`, which fits twice under
    different global `torch` RNG states and asserts bit-identical output).
    There is no stochastic component to average over with multiple seeds.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        mean_feature_names: Sequence[str] = DEFAULT_PHYSICS_MEAN_FEATURES,
    ) -> None:
        self.feature_names = list(feature_names)
        self.mean_feature_names = list(mean_feature_names)
        missing = [f for f in self.mean_feature_names if f not in self.feature_names]
        if missing:
            raise KeyError(f"mean_feature_names not in feature_names: {missing}")
        self._mean_indices = [self.feature_names.index(f) for f in self.mean_feature_names]
        self.model: SingleTaskGP | None = None
        self._y: np.ndarray | None = None

    def fit(
        self, X: ArrayLike, y: ArrayLike, *, bounds: ArrayLike, noise: ArrayLike | float | None = None
    ) -> "PhysicsMeanSurrogate":
        Xa, ya, Xt, yt, box = _prep_fit_inputs(X, y, bounds)
        if Xa.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {Xa.shape[1]} columns but feature_names has {len(self.feature_names)} entries"
            )
        yvar = _prep_noise(noise, Xa.shape[0])
        normalize = Normalize(d=Xt.shape[-1], bounds=box)
        mean_module = _SubsetLinearMean(self._mean_indices)
        kwargs: dict = {
            "mean_module": mean_module,
            "input_transform": normalize,
            "outcome_transform": Standardize(m=1),
        }
        if yvar is not None:
            kwargs["train_Yvar"] = yvar
        self.model = SingleTaskGP(Xt, yt, **kwargs)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        _fit_mll_with_retry(mll)
        self._y = ya
        return self

    def posterior(self, X: ArrayLike, *, observation_noise: bool = False):
        return _posterior(self.model, X, observation_noise)


# The fidelity dimension for candidate B: `log_volume_ratio` is already
# `log10(scale_L / reference_scale_L)` - already monotone in scale, already
# one of `build_feature_matrix`'s columns. Using it (rather than adding a
# new raw `scale_L` column) means candidate B's `X` is IDENTICAL to v0's -
# same `build_feature_matrix` call, same column order - the only difference
# is which column is designated the fidelity dim and which kernel treats it
# specially.
DEFAULT_FIDELITY_FEATURE = "log_volume_ratio"


class MultiFidelitySurrogate:
    """Candidate B: `botorch.models.gp_regression_fidelity.SingleTaskMultiFidelityGP`
    with `fidelity_feature_name` (default `DEFAULT_FIDELITY_FEATURE`,
    `log_volume_ratio`) as the single data-fidelity dimension. See the
    module docstring for the arXiv:2508.10970 precedent this follows.

    INSTALLED-CLASS VERIFICATION (botorch 0.18.1, read from
    `botorch/models/gp_regression_fidelity.py` source before using this):
      - `SingleTaskMultiFidelityGP` subclasses `SingleTaskGP` directly and
        accepts the same `input_transform`/`outcome_transform` kwargs with
        the same defaults (`Standardize` applied automatically if omitted) -
        so `Normalize`+`Standardize` compose with it exactly as with
        `Surrogate`, no special-casing needed.
      - `data_fidelities: Sequence[int]` names the COLUMN INDICES (post any
        negative-index normalization the class does internally) to treat as
        fidelity dims; requires at least one of `iteration_fidelity` /
        `data_fidelities` non-empty or the constructor raises
        `UnsupportedError`.
      - Default `linear_truncated=True` selects `LinearTruncatedFidelityKernel`
        (Wu 2019), a product kernel of a Matern-based term over the
        non-fidelity dims and bias/interaction terms over the fidelity
        dim(s); the fidelity dim's kernel formula
        (`c_1 = (1-x[f])(1-x'[f])(1+x[f]x'[f])^p`) is written for a fidelity
        value in `[0, 1]` with 1 meaning "target/full fidelity" - matching
        this module's convention that the LARGEST scale in the fold's
        normalization box lands nearest 1.0 after `Normalize`, and the
        smallest nearest 0.0 (see `leave_one_scale_out_report`'s box
        contract: one min/max box spanning train AND the held-out scale
        together, so the held-out scale's fidelity value is always inside
        `[0, 1]`, never clamped or extrapolated past the kernel's designed
        domain, even though the FEATURE space beyond it is still genuine
        extrapolation).
      - `default nu=2.5` (Matern smoothness) is used unchanged - not
        reconfigured for this dataset, since the model is being evaluated
        AS INSTALLED per the task brief, not hand-tuned.

    Same `feature_names`/determinism contract as `PhysicsMeanSurrogate` -
    see its docstring.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        fidelity_feature_name: str = DEFAULT_FIDELITY_FEATURE,
    ) -> None:
        self.feature_names = list(feature_names)
        if fidelity_feature_name not in self.feature_names:
            raise KeyError(f"fidelity_feature_name {fidelity_feature_name!r} not in feature_names")
        self.fidelity_feature_name = fidelity_feature_name
        self._fidelity_index = self.feature_names.index(fidelity_feature_name)
        self.model: SingleTaskMultiFidelityGP | None = None
        self._y: np.ndarray | None = None

    def fit(
        self, X: ArrayLike, y: ArrayLike, *, bounds: ArrayLike, noise: ArrayLike | float | None = None
    ) -> "MultiFidelitySurrogate":
        Xa, ya, Xt, yt, box = _prep_fit_inputs(X, y, bounds)
        if Xa.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {Xa.shape[1]} columns but feature_names has {len(self.feature_names)} entries"
            )
        yvar = _prep_noise(noise, Xa.shape[0])
        normalize = Normalize(d=Xt.shape[-1], bounds=box)
        kwargs: dict = {
            "data_fidelities": [self._fidelity_index],
            "input_transform": normalize,
            "outcome_transform": Standardize(m=1),
        }
        if yvar is not None:
            kwargs["train_Yvar"] = yvar
        self.model = SingleTaskMultiFidelityGP(Xt, yt, **kwargs)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        _fit_mll_with_retry(mll)
        self._y = ya
        return self

    def posterior(self, X: ArrayLike, *, observation_noise: bool = False):
        return _posterior(self.model, X, observation_noise)


__all__ = [
    "ScaleCandidateModel",
    "DEFAULT_PHYSICS_MEAN_FEATURES",
    "PhysicsMeanSurrogate",
    "DEFAULT_FIDELITY_FEATURE",
    "MultiFidelitySurrogate",
]
