"""Honest benchmark: SingleTaskGP vs MAP-SAAS on media-DoE-shaped synthetic problems.

CONTEXT. `botorch.models.map_saas.AdditiveMapSaasSingleTaskGP` ships in the
installed environment (botorch 0.18.1) - a MAP (no MCMC) version of SAASBO with a
sparsity-inducing prior on the ARD lengthscales, additive over `num_taus` Matern
kernels each with its own shrinkage. Real media DoE sheets can carry 20-40
components where most turn out irrelevant, which is exactly the SAAS prior's
sweet spot on paper. kalos's exact `SingleTaskGP` (see `kalos.core.surrogate`)
fits one ARD lengthscale per dimension with no sparsity prior at all.

This module answers, with real numbers rather than the paper's claims, whether
switching kalos's surrogate to the SAAS variant would actually win on problems
shaped like the real use case: n in {30, 60} rows, d up to 40, only 3-5 of those
dimensions truly driving the response, replicate rows with known assay noise.

WHY THE SAAS PATH IS NOT BUILT THROUGH `kalos.core.surrogate.Surrogate` HERE.
This experiment's job is to decide WHETHER to wire a `kind="saas"` option into
`Surrogate.fit` - so it must be able to measure a negative result without having
already made the change under test. `_fit_saas` below builds the SAAS model with
the same input/outcome transforms and the same jitter-retry fit ladder
`Surrogate` uses (imported from `kalos.core.surrogate`, not duplicated), wrapped
in a `_SaasFit` object that exposes the same `.model` / `._X` / `.posterior()`
surface `kalos.core.optimize.propose` and the CV loop need. If the numbers below
justify it, `kind="saas"` gets added to `Surrogate.fit` for real and this module
stops needing its own copy.

WHY THE GROUPED-CV LOOP IS NOT `kalos.core.evaluation.grouped_cv_report`. That
function is hardwired to `Surrogate` (it calls `Surrogate().fit(...)` per fold)
and is out of this task's file scope (only `kalos/core/surrogate.py` and
`kalos/bench/**` were authorized for edits) - it cannot be parameterized by
`kind` without editing it. `_pooled_cv_spearman` below mirrors its methodology
exactly: the same grouped splitter (`kalos.core.splits.make_splits`, imported
unmodified), the same pooled-out-of-fold-then-Spearman reduction, the same
`observation_noise=True` (predictive, not latent) posterior for scoring against
a noisy held-out measurement, and the same small-fold guard (skip a fold with
fewer than 4 training rows or 0 validation rows).

Run: `python -m kalos.bench.saas_experiment` (full sweep, ~5-10 min) or
`python -m kalos.bench.saas_experiment --quick` (tiny config, <60s, what the
smoke test in `tests/test_saas_surrogate.py` runs).
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from botorch.models.map_saas import AdditiveMapSaasSingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc, spearmanr

from kalos.core.optimize import propose
from kalos.core.splits import make_splits
from kalos.core.surrogate import (
    DEVICE,
    DTYPE,
    FitError,
    Surrogate,
    _fit_mll_with_retry,
    sanitize_bounds,
)

KINDS = ("single_task", "saas")

# The portal's per-request analyze budget (see kalos/portal/analysis.py and
# BENCHMARK.md). A candidate surrogate that cannot fit within this, on the row
# counts kalos actually sees, cannot ship regardless of accuracy - the request
# would just time out or feel broken. This is the absolute bar.
ANALYZE_BUDGET_S = 2.0

# Relative bar: a candidate is "unacceptable" cost-wise if it is more than this
# multiple slower than the SingleTaskGP baseline it would replace, even when
# both are individually under ANALYZE_BUDGET_S - a 5x slowdown that happens to
# still clear 2s today will not clear it once the row count grows.
MAX_ACCEPTABLE_SLOWDOWN = 3.0


# --------------------------------------------------------------------------- #
# Synthetic problem: a sparse bump surface shaped like a media DoE sheet.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Problem:
    d: int
    active_dims: np.ndarray  # sorted indices of the truly active dimensions
    centers: np.ndarray  # per-active-dim optimum location, in [0, 1]
    widths: np.ndarray
    weights: np.ndarray
    bounds: np.ndarray  # (2, d), always [0, 1]^d
    scale: float  # f at its global optimum; used to scale observation noise

    def f(self, X: np.ndarray) -> np.ndarray:
        """Sum of Gaussian bumps over the active dims; inactive dims contribute
        nothing, so the surface is exactly as sparse as `active_dims` says."""
        Xa = X[:, self.active_dims]
        d2 = (Xa - self.centers) ** 2 / (2.0 * self.widths ** 2)
        return (self.weights * np.exp(-d2)).sum(axis=1)

    def optimum_point(self) -> np.ndarray:
        """A full-d point at the surface's global optimum (inactive dims at 0.5,
        since they do not affect `f` and 0.5 is the box center)."""
        x = np.full(self.d, 0.5)
        x[self.active_dims] = self.centers
        return x


def make_problem(d: int, seed: int, n_active: int = 4) -> Problem:
    rng = np.random.default_rng(seed)
    k = min(n_active, d)
    active = np.sort(rng.choice(d, size=k, replace=False))
    centers = rng.uniform(0.3, 0.7, size=k)
    widths = rng.uniform(0.15, 0.35, size=k)
    weights = rng.uniform(0.5, 1.5, size=k)
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    scale = float(weights.sum())  # all bumps at their peak simultaneously
    return Problem(d, active, centers, widths, weights, bounds, scale)


def make_dataset(
    problem: Problem,
    n: int,
    seed: int,
    *,
    noise_frac: float = 0.15,
    replicate_frac: float = 0.3,
    zero_inflate_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y_obs, y_true, groups, yvar): `n` rows total, a fraction
    `replicate_frac` of them replicate rows of an earlier recipe (own noise draw,
    shared group id) so grouped CV has something real to guard against leaking.

    `yvar` is the KNOWN per-row noise variance (constant across rows here, one
    assay noise floor), in the target's original units - this is what exercises
    the `train_Yvar` path on both models, matching how kalos already uses a
    measured noise floor from `estimate_noise_floor` on replicated recipes.

    `zero_inflate_frac`, when > 0, zeros out that fraction of `y_obs` rows
    (simulating a failed run) without touching `y_true` - the optional
    zero-inflation variant the task asks to check does not break the fit.
    """
    rng = np.random.default_rng(seed)
    n_unique = max(4, round(n * (1.0 - replicate_frac)))
    n_unique = min(n_unique, n)
    sampler = qmc.LatinHypercube(d=problem.d, seed=rng)
    X_unique = sampler.random(n_unique)
    if n_unique < n:
        extra_idx = rng.integers(0, n_unique, size=n - n_unique)
        X = np.vstack([X_unique, X_unique[extra_idx]])
        groups = np.concatenate([np.arange(n_unique), extra_idx])
    else:
        X = X_unique
        groups = np.arange(n_unique)

    y_true = problem.f(X)
    sigma = noise_frac * problem.scale
    y_obs = y_true + rng.normal(0.0, sigma, size=y_true.shape)
    yvar = np.full(n, sigma ** 2)

    if zero_inflate_frac > 0:
        n_zero = round(zero_inflate_frac * n)
        zero_idx = rng.choice(n, size=n_zero, replace=False)
        y_obs = y_obs.copy()
        y_obs[zero_idx] = 0.0

    return X, y_obs, y_true, groups, yvar


# --------------------------------------------------------------------------- #
# The SAAS fit path, built directly on botorch (see module docstring for why).
# --------------------------------------------------------------------------- #


class _SaasFit:
    """Mirrors the subset of `Surrogate`'s surface that `propose()` and the CV
    loop below actually use: `.model`, `._X`, `.posterior()`. Same input
    normalization to the fixed design box and output standardization as
    `Surrogate`; same jitter-retry fit ladder (imported, not reimplemented)."""

    def __init__(self) -> None:
        self.model: AdditiveMapSaasSingleTaskGP | None = None
        self._X: torch.Tensor | None = None
        self._y: np.ndarray | None = None

    def fit(self, X, y, bounds, *, noise=None) -> "_SaasFit":
        Xa = np.asarray(X, float)
        ya = np.asarray(y, float).reshape(-1)
        Xt = torch.as_tensor(Xa, dtype=DTYPE, device=DEVICE)
        yt = torch.as_tensor(ya, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        d = Xt.shape[-1]
        lower, upper = sanitize_bounds(bounds)
        box = torch.as_tensor(np.vstack([lower, upper]), dtype=DTYPE, device=DEVICE)
        normalize = Normalize(d=d, bounds=box)

        yvar: torch.Tensor | None = None
        if noise is not None:
            noise_arr = np.broadcast_to(np.asarray(noise, float), (Xa.shape[0],)).astype(float)
            yvar = torch.as_tensor(noise_arr, dtype=DTYPE, device=DEVICE).reshape(-1, 1)

        self.model = AdditiveMapSaasSingleTaskGP(
            Xt, yt, train_Yvar=yvar, input_transform=normalize, outcome_transform=Standardize(m=1)
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        _fit_mll_with_retry(mll)
        self._X = Xt
        self._y = ya
        return self

    def posterior(self, X, *, observation_noise: bool = False):
        assert self.model is not None, "fit() first"
        Xt = torch.as_tensor(np.asarray(X, float), dtype=DTYPE, device=DEVICE)
        self.model.eval()
        with torch.no_grad():
            post = self.model.posterior(Xt, observation_noise=observation_noise)
            mean = post.mean.squeeze(-1).cpu().numpy()
            std = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()
        return mean, std


def _fit(kind: str, X, y, bounds, noise=None):
    if kind == "single_task":
        return Surrogate().fit(X, y, bounds, noise=noise)
    if kind == "saas":
        return _SaasFit().fit(X, y, bounds, noise=noise)
    raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #


def timed_fit(kind: str, X, y, bounds, noise=None) -> tuple[Any, float]:
    t0 = time.perf_counter()
    fitted = _fit(kind, X, y, bounds, noise=noise)
    elapsed = time.perf_counter() - t0
    return fitted, elapsed


def pooled_cv_spearman(
    kind: str, X, y, groups, bounds, noise=None, n_splits: int = 5
) -> float:
    """Pooled out-of-fold Spearman, mirroring `kalos.core.evaluation`'s
    methodology exactly (see module docstring for why this is not a direct
    call into that module)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    noise = None if noise is None else np.asarray(noise, float).reshape(-1)
    splits = make_splits(X, y, groups, n_splits=n_splits)
    pred, actual = [], []
    for tr, te in splits:
        if len(tr) < 4 or len(te) < 1:
            continue
        try:
            fitted = _fit(kind, X[tr], y[tr], bounds, noise=None if noise is None else noise[tr])
        except FitError:
            continue
        mean, _std = fitted.posterior(X[te], observation_noise=True)
        pred.extend(np.asarray(mean).ravel().tolist())
        actual.extend(y[te].tolist())
    if len(pred) < 3 or np.std(pred) == 0 or np.std(actual) == 0:
        return float("nan")
    r = spearmanr(pred, actual).statistic
    return float(r) if r == r else float("nan")


def proposal_active_dim_error(
    kind: str, problem: Problem, X, y, bounds, noise=None, seed: int = 0
) -> float:
    """One BO step (q=1); how far the proposed point lands from the TRUE optimum
    on the dims that actually matter, normalized so 0 = exact and results are
    comparable across `d`.

    Concretely: Euclidean distance between the proposal's coordinates on
    `problem.active_dims` and `problem.centers`, divided by
    `sqrt(len(active_dims))` (the box diagonal on just those dims, since every
    dim lives in [0, 1]) - so the proxy is bounded in [0, 1] regardless of `d`
    or how many dims are active, and lower is better (closer to the true peak
    on the dims a working optimizer should have learned to chase).
    """
    fitted = _fit(kind, X, y, bounds, noise=noise)
    proposal = propose(fitted, bounds, q=1, seed=seed)[0]
    diff = proposal[problem.active_dims] - problem.centers
    denom = float(np.sqrt(len(problem.active_dims)))
    return float(np.linalg.norm(diff) / denom)


# --------------------------------------------------------------------------- #
# The sweep.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CellResult:
    n: int
    d: int
    seed: int
    kind: str
    cv_spearman: float
    fit_time_s: float
    proposal_error: float


def run_cell(n: int, d: int, seed: int, *, n_active: int = 4) -> list[CellResult]:
    problem = make_problem(d, seed, n_active=n_active)
    X, y_obs, _y_true, groups, yvar = make_dataset(problem, n, seed)
    out = []
    for kind in KINDS:
        _fitted, fit_time = timed_fit(kind, X, y_obs, problem.bounds, noise=yvar)
        cv = pooled_cv_spearman(kind, X, y_obs, groups, problem.bounds, noise=yvar)
        prop_err = proposal_active_dim_error(
            kind, problem, X, y_obs, problem.bounds, noise=yvar, seed=seed
        )
        out.append(CellResult(n, d, seed, kind, cv, fit_time, prop_err))
    return out


def run_sweep(
    *,
    ns: Sequence[int] = (30, 60),
    ds: Sequence[int] = (10, 25, 40),
    low_d: Sequence[int] = (4,),
    n_seeds: int = 5,
    n_active: int = 4,
) -> list[CellResult]:
    """The full grid: high-d regime (`ds`, where SAAS's sparsity prior should
    help if it helps anywhere) plus the low-d regime (`low_d`, where kalos lives
    today - measures the cost/regression risk of turning SAAS on where it is not
    needed)."""
    results: list[CellResult] = []
    for n in ns:
        for d in list(ds) + list(low_d):
            for seed in range(n_seeds):
                results.extend(run_cell(n, d, seed, n_active=n_active))
    return results


def run_zero_inflation_check(seed: int = 0) -> list[CellResult]:
    """One cell with zero-inflated rows (failed-run pattern), checked once per
    the task's 'zero-inflation optional' ask - not part of the main grid."""
    d, n, n_active = 25, 60, 4
    problem = make_problem(d, seed, n_active=n_active)
    X, y_obs, _y_true, groups, yvar = make_dataset(
        problem, n, seed, zero_inflate_frac=0.15
    )
    out = []
    for kind in KINDS:
        _fitted, fit_time = timed_fit(kind, X, y_obs, problem.bounds, noise=yvar)
        cv = pooled_cv_spearman(kind, X, y_obs, groups, problem.bounds, noise=yvar)
        out.append(CellResult(n, d, seed, kind, cv, fit_time, float("nan")))
    return out


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #


def _aggregate(results: Sequence[CellResult]) -> dict[tuple[int, int, str], dict[str, float]]:
    by_cell: dict[tuple[int, int, str], list[CellResult]] = {}
    for r in results:
        by_cell.setdefault((r.n, r.d, r.kind), []).append(r)
    out = {}
    for key, rows in by_cell.items():
        cv = np.array([r.cv_spearman for r in rows], float)
        ft = np.array([r.fit_time_s for r in rows], float)
        pe = np.array([r.proposal_error for r in rows], float)
        out[key] = {
            "cv_spearman_mean": float(np.nanmean(cv)),
            "cv_spearman_std": float(np.nanstd(cv)),
            "fit_time_mean": float(ft.mean()),
            "fit_time_std": float(ft.std()),
            "proposal_error_mean": float(np.nanmean(pe)),
            "proposal_error_std": float(np.nanstd(pe)),
            "n_seeds": len(rows),
        }
    return out


def print_report(results: Sequence[CellResult]) -> None:
    agg = _aggregate(results)
    ns = sorted({r.n for r in results})
    ds = sorted({r.d for r in results})
    print(f"kalos SAAS-vs-SingleTaskGP benchmark  (ANALYZE_BUDGET_S={ANALYZE_BUDGET_S}, "
          f"MAX_ACCEPTABLE_SLOWDOWN={MAX_ACCEPTABLE_SLOWDOWN})\n")
    header = (
        f"  {'n':>3} {'d':>3} {'kind':<12} {'cv_spearman':>18} {'fit_time_s':>16} "
        f"{'proposal_err':>16}"
    )
    print(header)
    for n in ns:
        for d in ds:
            for kind in KINDS:
                key = (n, d, kind)
                if key not in agg:
                    continue
                a = agg[key]
                print(
                    f"  {n:>3} {d:>3} {kind:<12} "
                    f"{a['cv_spearman_mean']:>7.3f} +/- {a['cv_spearman_std']:<6.3f} "
                    f"{a['fit_time_mean']:>7.3f} +/- {a['fit_time_std']:<6.3f} "
                    f"{a['proposal_error_mean']:>7.3f} +/- {a['proposal_error_std']:<6.3f}"
                )
        print()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    if quick:
        results = run_sweep(ns=(30,), ds=(25,), low_d=(4,), n_seeds=2, n_active=3)
    else:
        results = run_sweep()
    print_report(results)
    if not quick:
        zi = run_zero_inflation_check()
        print("Zero-inflation check (n=60, d=25, 15% zeroed rows, seed=0):")
        for r in zi:
            print(f"  {r.kind:<12} cv_spearman={r.cv_spearman:.3f} fit_time_s={r.fit_time_s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
