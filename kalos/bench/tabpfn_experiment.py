"""Honest benchmark: SingleTaskGP vs TabPFN v2 point-prediction accuracy on
kalos's exact small-tabular regime.

CONTEXT. The research memo flagged TabPFN v2 (Nature 2025 - "Accurate
predictions on small data with a tabular foundation model") as a serious
BASELINE candidate: a transformer prior-fitted network built specifically for
n < 1000, evaluated in the paper at exactly the row/feature counts kalos's
portal sees (30-100 rows, 4-40 features).

THE QUESTION THIS MODULE ANSWERS IS NOT "replace the GP". kalos's
`SingleTaskGP` (`kalos.core.surrogate.Surrogate`) gives a calibrated posterior
- mean AND uncertainty - that `kalos.core.optimize.propose`'s acquisition
function needs; TabPFN's public regressor API returns point predictions (a
`quantiles` output type exists but is not a GP-style closed-form posterior a
BoTorch acquisition function can consume). The real question: does TabPFN's
point-prediction ACCURACY embarrass the GP's on held-out ranking (grouped-CV
Spearman)? If it does, a future wave is justified in prototyping a
TabPFN-mean + GP-uncertainty hybrid. If it does not, TabPFN is not worth
further engineering time here. Either answer is a fully successful outcome of
this wave - nothing gets wired into `Surrogate` regardless.

WHY THIS MIRRORS `kalos/bench/saas_experiment.py`. That module ran the same
kind of "is a fancier surrogate worth it" question for MAP-SAAS and the
resulting negative result (see its module docstring and the wave-2 report) is
the template this module follows: same synthetic problem generator family -
`Problem`/`make_problem`/`make_dataset` are IMPORTED from `saas_experiment`,
not duplicated, so both benchmarks are measuring the same surface under the
same noise/replicate/zero-inflation model - same grouped-CV methodology (a
local `cv_eval` mirrors `kalos.core.evaluation`'s pooled-out-of-fold-then-
Spearman reduction exactly, for the same reason `saas_experiment.py`
re-implements it rather than calling `kalos.core.evaluation.grouped_cv_report`
directly: that function is hardwired to `Surrogate` and parameterizing it by
model kind is out of this task's authorized file scope), same seeds 0-4, same
`ANALYZE_BUDGET_S` / `MAX_ACCEPTABLE_SLOWDOWN` cost bars (imported from
`saas_experiment`, not redefined, so "is this candidate too slow to ship" is
judged against the one real production number in one place).

CELLS. `n` in {30, 60, 100} x `d` in {4, 10, 25} (the task's explicit grid,
narrower in `d` than `saas_experiment`'s high-d SAAS sweep because TabPFN's
comparison is about kalos's ACTUAL regime, not a stress test of a sparsity
prior), seeds 0-4, plus one zero-inflated cell (failed-run pattern, mirroring
`saas_experiment.run_zero_inflation_check`).

METRICS. Grouped-CV Spearman (`cv_spearman`) and mean per-fold wall time,
split into `fit_time_s` and `predict_time_s` because for TabPFN these are NOT
comparable in cost (see CONSTRAINTS): `fit()` is near-instant (no training in
the usual sense - it just caches the context rows), essentially all of
TabPFN's cost is in `predict()`, which runs a transformer forward pass over
the cached context for every query row. `SingleTaskGP`'s cost is the reverse
(the marginal-likelihood fit is where the work happens; `posterior()` is
cheap). Reporting them separately keeps a reader from being misled into
comparing "GP fit_time" against "TabPFN fit_time" as if they measured the same
thing.

CONSTRAINTS FOUND (verified in a throwaway venv - see below - not asserted
from memory).

  API. `from tabpfn import TabPFNRegressor`; `TabPFNRegressor(device=...,
  random_state=...).fit(X, y).predict(X, output_type="mean")` returns a 1-D
  `np.ndarray` of point predictions. `fit`/`predict` signatures are stable
  across the two releases checked (`tabpfn==2.0.9`, the original open Nature-
  2025-era release, and the current PyPI `tabpfn==8.5.0`, Prior Labs' rewrite)
  - this module only relies on that stable subset, so it runs unmodified
  against whichever version `pip install tabpfn` resolves to.

  n/d LIMITS. `tabpfn==8.5.0`'s default inference config caps at
  `MAX_NUMBER_OF_SAMPLES=10_000`, `MAX_NUMBER_OF_FEATURES=500` - both far
  above kalos's 30-100 x 4-40 regime, so this benchmark never gets near a
  limit and the numbers below are not an artifact of hitting one.

  GATED MODEL DOWNLOAD - the single biggest practical constraint. The
  CURRENT `pip install tabpfn` (8.5.0) fetches its checkpoint from a GATED
  HuggingFace repo (`Prior-Labs/tabpfn_3`): the first `.fit()` call raises
  `TabPFNHuggingFaceGatedRepoError` unless the calling machine has already
  run `hf auth login` (an interactive browser-based HuggingFace login and
  license click-through) or set `HF_TOKEN` for an account that has accepted
  the Prior Labs License. There is no offline/bundled checkpoint in the pip
  package. This is a hard blocker for any credential-free environment (CI,
  a fresh eval sandbox, an air-gapped or client-audited deployment) and,
  unlike a plain download, is not something an automated agent should push
  through on a user's behalf - it is an account/OAuth action. The real
  numbers in this module's report were therefore gathered against the
  earlier `tabpfn==2.0.9` release, whose checkpoint
  (`tabpfn-v2-regressor.ckpt`, ~44 MB) is still hosted on a PUBLIC,
  ungated HuggingFace repo and downloads with a plain `pip install` and a
  first `.fit()` call, no login required. Both releases expose the same
  `TabPFNRegressor` v2 architecture from the Nature 2025 paper; 8.5.0 is
  Prior Labs' commercial rewrite around the same model family. Reproducing
  against the CURRENT package requires the auth step above FIRST.

  LICENSE. Prior Labs License v1.2 ("Apache 2.0 with an additional
  attribution provision", modeled on the Llama 3 license) - not a plain
  Apache 2.0 grant. Read the actual license text before shipping anything
  built on it; this module only benchmarks, it does not redistribute the
  checkpoint.

  COLD START. First `.fit()` in a fresh process (after the checkpoint is
  already cached on disk) pays a one-time ~2s model-load cost; every
  subsequent `TabPFNRegressor(...).fit()` in the same process is ~0.1-0.4s.
  `predict()` is the dominant per-call cost at these sizes (roughly 1-4s for
  the row/feature counts in this grid, measured on an Apple M-series CPU -
  see the report for exact per-cell numbers), which is already close to or
  over `ANALYZE_BUDGET_S` on its own, before a caller does anything else.

REPRODUCE. TabPFN is NOT a kalos dependency (not in `pyproject.toml`, not
installed in `kalos/.venv`) - this module hard-imports it at module level
(mirroring how `saas_experiment.py` hard-imports `botorch`), so importing
this module directly requires it. `tests/test_tabpfn_experiment.py` guards
with `pytest.importorskip("tabpfn")` so the suite stays green wherever it is
absent (every CI box today).

  1. Create a throwaway venv (do NOT install into `kalos/.venv` - this is a
     trial dependency, not a kalos dependency):
       uv venv /path/to/scratch/tabpfn-venv --python 3.12
  2. Install kalos's own deps plus the trial package into THAT venv:
       uv pip install --python /path/to/scratch/tabpfn-venv/bin/python \\
           -e /path/to/kalos[ml,dev] tabpfn
     (add `HF_TOKEN=...` / run `hf auth login` first if you want the CURRENT
     gated package rather than pinning `tabpfn==2.0.9` - see CONSTRAINTS)
  3. Run the sweep:
       /path/to/scratch/tabpfn-venv/bin/python -m kalos.bench.tabpfn_experiment
     (full grid, ~10-15 min on an M-series CPU - almost all of it is TabPFN
     `predict()` calls) or
       ... -m kalos.bench.tabpfn_experiment --quick
     (tiny config, well under 60s - what `tests/test_tabpfn_experiment.py`'s
     smoke test runs when tabpfn is present).
  4. Delete the throwaway venv when done.

REAL DATASET. If a `KALOS_DATA` env var points at the root of an
`itsbrendandang/kalos-data` checkout (same pattern as
`tests/test_scale_integration.py`), `run_real_dataset_check` also runs both
models on the real `mab-scaleup-synthetic` dataset. Skips gracefully (returns
`[]`) when the env var is unset or the dataset is not found there - it was
not present on the machine this wave ran on, so that comparison is reported
as skipped, not faked.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tabpfn import TabPFNRegressor

from kalos.bench.saas_experiment import (
    ANALYZE_BUDGET_S,
    MAX_ACCEPTABLE_SLOWDOWN,
    make_dataset,
    make_problem,
)
from kalos.core.splits import make_splits
from kalos.core.surrogate import FitError, Surrogate

KINDS = ("single_task", "tabpfn")

__all__ = [
    "ANALYZE_BUDGET_S",
    "MAX_ACCEPTABLE_SLOWDOWN",
    "KINDS",
    "CellResult",
    "cv_eval",
    "run_cell",
    "run_sweep",
    "run_zero_inflation_check",
    "run_real_dataset_check",
    "print_report",
    "main",
]


# --------------------------------------------------------------------------- #
# Per-kind fit+predict, timed separately (see module docstring for why fit and
# predict cost are NOT comparable across kinds and must not be summed away).
# --------------------------------------------------------------------------- #


def _fit_predict(
    kind: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    bounds: np.ndarray,
    seed: int,
    noise: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Return (point predictions on X_te, fit_time_s, predict_time_s)."""
    if kind == "single_task":
        t0 = time.perf_counter()
        s = Surrogate().fit(X_tr, y_tr, bounds, noise=noise)
        t1 = time.perf_counter()
        mean, _std = s.posterior(X_te, observation_noise=True)
        t2 = time.perf_counter()
        return np.asarray(mean, dtype=float).ravel(), t1 - t0, t2 - t1
    if kind == "tabpfn":
        # TabPFN has no `bounds`/design-box normalization (its own internal
        # preprocessing is scale-invariant) and no fixed-noise input - both
        # args are accepted for a uniform call signature and ignored here.
        t0 = time.perf_counter()
        reg = TabPFNRegressor(device="cpu", random_state=seed)
        reg.fit(X_tr, y_tr)
        t1 = time.perf_counter()
        pred = reg.predict(X_te)
        t2 = time.perf_counter()
        return np.asarray(pred, dtype=float).ravel(), t1 - t0, t2 - t1
    raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")


def _spearman(pred: Sequence[float], actual: Sequence[float]) -> float:
    if len(pred) < 3 or np.std(pred) == 0 or np.std(actual) == 0:
        return float("nan")
    r = spearmanr(pred, actual).statistic
    return float(r) if r == r else float("nan")


def cv_eval(
    kind: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    bounds: np.ndarray,
    seed: int,
    noise: np.ndarray | None = None,
    n_splits: int = 5,
) -> dict:
    """Pooled out-of-fold Spearman plus mean per-fold fit/predict wall time,
    mirroring `kalos.core.evaluation`'s methodology exactly (see module
    docstring for why this is not a direct call into that module): the same
    grouped splitter (`kalos.core.splits.make_splits`, imported unmodified),
    the same pooled-out-of-fold-then-Spearman reduction, and the same
    small-fold guard (skip a fold with fewer than 4 training rows or 0
    validation rows).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    noise_arr = None if noise is None else np.asarray(noise, dtype=float).reshape(-1)
    splits = make_splits(X, y, groups, n_splits=n_splits)
    pred: list[float] = []
    actual: list[float] = []
    fit_times: list[float] = []
    predict_times: list[float] = []
    for tr, te in splits:
        if len(tr) < 4 or len(te) < 1:
            continue
        try:
            p, ft, pt = _fit_predict(
                kind, X[tr], y[tr], X[te], bounds, seed,
                noise=None if noise_arr is None else noise_arr[tr],
            )
        except FitError:
            continue
        pred.extend(p.tolist())
        actual.extend(y[te].tolist())
        fit_times.append(ft)
        predict_times.append(pt)
    return {
        "spearman": _spearman(pred, actual),
        "fit_time_s": float(np.mean(fit_times)) if fit_times else float("nan"),
        "predict_time_s": float(np.mean(predict_times)) if predict_times else float("nan"),
        "n_folds": len(fit_times),
    }


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
    predict_time_s: float
    n_folds: int


def run_cell(n: int, d: int, seed: int, *, n_active: int = 4) -> list[CellResult]:
    problem = make_problem(d, seed, n_active=n_active)
    X, y_obs, _y_true, groups, yvar = make_dataset(problem, n, seed)
    out = []
    for kind in KINDS:
        res = cv_eval(kind, X, y_obs, groups, problem.bounds, seed, noise=yvar)
        out.append(
            CellResult(n, d, seed, kind, res["spearman"], res["fit_time_s"], res["predict_time_s"], res["n_folds"])
        )
    return out


def run_sweep(
    *,
    ns: Sequence[int] = (30, 60, 100),
    ds: Sequence[int] = (4, 10, 25),
    n_seeds: int = 5,
    n_active: int = 4,
) -> list[CellResult]:
    """The task's grid: `n` in {30, 60, 100} x `d` in {4, 10, 25}, seeds
    0..n_seeds-1 - kalos's exact regime, not a stress test of a different one."""
    results: list[CellResult] = []
    for n in ns:
        for d in ds:
            for seed in range(n_seeds):
                results.extend(run_cell(n, d, seed, n_active=n_active))
    return results


def run_zero_inflation_check(seed: int = 0) -> list[CellResult]:
    """One mid-grid cell with zero-inflated rows (failed-run pattern),
    mirroring `saas_experiment.run_zero_inflation_check` - checked once per
    the task's 'zero-inflation optional' ask, not part of the main grid."""
    d, n, n_active = 10, 60, 4
    problem = make_problem(d, seed, n_active=n_active)
    X, y_obs, _y_true, groups, yvar = make_dataset(problem, n, seed, zero_inflate_frac=0.15)
    out = []
    for kind in KINDS:
        res = cv_eval(kind, X, y_obs, groups, problem.bounds, seed, noise=yvar)
        out.append(
            CellResult(n, d, seed, kind, res["spearman"], res["fit_time_s"], res["predict_time_s"], res["n_folds"])
        )
    return out


# --------------------------------------------------------------------------- #
# Real dataset (see module docstring - skips gracefully when KALOS_DATA is
# unset or the checkout is not found there).
# --------------------------------------------------------------------------- #

_REAL_DATASET_REL = Path("datasets/mab-scaleup-synthetic")
_REAL_PROCESS_COLUMNS = [
    "ph_setpoint", "dissolved_oxygen_pct", "temperature_C", "agitation_rpm", "airflow_L_per_min",
]
_REAL_TARGET_COLUMN = "titer_g_per_L"


def _real_dataset_dir() -> Path | None:
    root = os.environ.get("KALOS_DATA")
    if not root:
        return None
    candidate = Path(root) / _REAL_DATASET_REL
    return candidate if (candidate / "batches.csv").exists() else None


def _load_real_dataset(dataset_dir: Path) -> pd.DataFrame:
    """Same derived-sensor-column construction as
    `tests/test_scale_integration.py` (`batches.csv` has no `agitation_rpm` /
    `airflow_L_per_min` of its own - each is the time-average over that
    batch's sensor series)."""
    batches = pd.read_csv(dataset_dir / "batches.csv")
    agitation_means, airflow_means = [], []
    for batch_id in batches["batch_id"]:
        sensors = pd.read_csv(dataset_dir / "sensors" / f"{batch_id}_sensors.csv")
        agitation_means.append(sensors["agitation_rpm"].mean())
        airflow_means.append(sensors["airflow_L_per_min"].mean())
    batches["agitation_rpm"] = agitation_means
    batches["airflow_L_per_min"] = airflow_means
    return batches


def run_real_dataset_check(n_splits: int = 5, seed: int = 0) -> list[dict]:
    """GP vs TabPFN grouped-CV Spearman on the real mab-scaleup dataset.
    Returns `[]` (not an error) when `KALOS_DATA` is unset or the checkout is
    not found - see module docstring."""
    dataset_dir = _real_dataset_dir()
    if dataset_dir is None:
        return []
    df = _load_real_dataset(dataset_dir)
    X = df[_REAL_PROCESS_COLUMNS].to_numpy(dtype=float)
    y = df[_REAL_TARGET_COLUMN].to_numpy(dtype=float)
    groups = df["batch_id"].to_numpy()
    bounds = np.vstack([X.min(axis=0), X.max(axis=0)])
    out = []
    for kind in KINDS:
        res = cv_eval(kind, X, y, groups, bounds, seed, n_splits=n_splits)
        out.append({"kind": kind, "n_rows": int(len(df)), **res})
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
        cv = np.array([r.cv_spearman for r in rows], dtype=float)
        ft = np.array([r.fit_time_s for r in rows], dtype=float)
        pt = np.array([r.predict_time_s for r in rows], dtype=float)
        out[key] = {
            "cv_spearman_mean": float(np.nanmean(cv)),
            "cv_spearman_std": float(np.nanstd(cv)),
            "fit_time_mean": float(np.nanmean(ft)),
            "predict_time_mean": float(np.nanmean(pt)),
            "n_seeds": len(rows),
        }
    return out


def print_report(results: Sequence[CellResult]) -> None:
    agg = _aggregate(results)
    ns = sorted({r.n for r in results})
    ds = sorted({r.d for r in results})
    print(
        f"kalos TabPFN-vs-SingleTaskGP benchmark  (ANALYZE_BUDGET_S={ANALYZE_BUDGET_S}, "
        f"MAX_ACCEPTABLE_SLOWDOWN={MAX_ACCEPTABLE_SLOWDOWN})\n"
    )
    header = (
        f"  {'n':>3} {'d':>3} {'kind':<12} {'cv_spearman':>18} "
        f"{'fit_time_s':>14} {'predict_time_s':>16}"
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
                    f"{a['fit_time_mean']:>10.4f}   {a['predict_time_mean']:>12.4f}"
                )
        print()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    quick = "--quick" in argv
    if quick:
        results = run_sweep(ns=(30,), ds=(10,), n_seeds=2, n_active=3)
    else:
        results = run_sweep()
    print_report(results)

    zi = run_zero_inflation_check()
    print("Zero-inflation check (n=60, d=10, 15% zeroed rows, seed=0):")
    for r in zi:
        print(f"  {r.kind:<12} cv_spearman={r.cv_spearman:.3f} predict_time_s={r.predict_time_s:.4f}")

    real = run_real_dataset_check()
    if not real:
        print("\nReal mab-scaleup dataset: skipped (KALOS_DATA not set or checkout not found).")
    else:
        print("\nReal mab-scaleup dataset:")
        for row in real:
            print(
                f"  {row['kind']:<12} n_rows={row['n_rows']} cv_spearman={row['spearman']:.3f} "
                f"fit_time_s={row['fit_time_s']:.4f} predict_time_s={row['predict_time_s']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
