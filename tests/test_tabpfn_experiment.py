"""TabPFN v2 evaluation: the bench experiment that answered whether TabPFN's
point-prediction accuracy embarrasses kalos's `SingleTaskGP` surrogate on
kalos's own regime (see kalos/bench/tabpfn_experiment.py and the wave-3
report for the full benchmark and its verdict).

TabPFN is NOT a kalos dependency - it is not in `pyproject.toml` and not
installed in `kalos/.venv` (it lives only in a throwaway venv used to run the
real benchmark; see `kalos/bench/tabpfn_experiment.py`'s module docstring for
the reproduce steps).

WHY THIS IS `importlib.util.find_spec` + `pytest.mark.skipif` RATHER THAN A
BARE `pytest.importorskip("tabpfn")`. That is the more familiar idiom and was
tried first, but a MODULE-level `importorskip` raises `Skipped` during
collection itself, before pytest discovers any test function in this file -
so `pytest -q tests/test_tabpfn_experiment.py` (this file's own smoke-test
gate) collects ZERO test items and pytest exits 5 (`EXIT_NOTESTSCOLLECTED`),
not 0, even though the summary line reads "1 skipped". Verified directly: a
two-line repro (`pytest.importorskip` on a nonexistent package, one dummy
test) reproduces exit 5 on this exact pytest (9.1.1). The `skipif` form below
collects every test function normally and marks each one individually
skipped, which both matches this file's own gate ("must pass") and mirrors
the in-repo convention `tests/test_scale_integration.py` already uses for its
`KALOS_DATA`-gated skip."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_TABPFN_AVAILABLE = importlib.util.find_spec("tabpfn") is not None

pytestmark = pytest.mark.skipif(
    not _TABPFN_AVAILABLE,
    reason=(
        "tabpfn is not installed (expected in CI) - see "
        "kalos/bench/tabpfn_experiment.py's module docstring for how to "
        "install it into a throwaway venv and reproduce the real benchmark"
    ),
)

if _TABPFN_AVAILABLE:
    from kalos.bench.tabpfn_experiment import (
        KINDS,
        CellResult,
        cv_eval,
        run_cell,
        run_real_dataset_check,
        run_sweep,
        run_zero_inflation_check,
    )
    from kalos.bench.saas_experiment import make_dataset, make_problem


def test_run_cell_returns_one_result_per_kind():
    results = run_cell(30, 10, seed=0, n_active=3)
    assert len(results) == len(KINDS)
    kinds = {r.kind for r in results}
    assert kinds == set(KINDS)
    for r in results:
        assert isinstance(r, CellResult)
        assert r.n == 30 and r.d == 10 and r.seed == 0
        assert np.isnan(r.cv_spearman) or -1.0 <= r.cv_spearman <= 1.0
        assert r.n_folds >= 0


def test_cv_eval_runs_for_both_kinds_and_is_bounded():
    p = make_problem(10, seed=4, n_active=3)
    X, y_obs, _y_true, groups, yvar = make_dataset(p, n=30, seed=4)
    for kind in KINDS:
        res = cv_eval(kind, X, y_obs, groups, p.bounds, seed=4, noise=yvar)
        assert set(res) == {"spearman", "fit_time_s", "predict_time_s", "n_folds"}
        assert np.isnan(res["spearman"]) or -1.0 <= res["spearman"] <= 1.0
        if res["n_folds"] > 0:
            assert res["fit_time_s"] > 0.0
            assert res["predict_time_s"] > 0.0


def test_run_sweep_smoke_is_seeded_and_shaped():
    """The experiment module's own smoke test: a tiny config, fully
    reproducible, matching the task's requirement for a smoke test on the
    experiment module when TabPFN is not wired into
    kalos.core.surrogate.Surrogate."""
    r1 = run_sweep(ns=(30,), ds=(4,), n_seeds=1, n_active=3)
    r2 = run_sweep(ns=(30,), ds=(4,), n_seeds=1, n_active=3)
    assert len(r1) == 1 * 1 * 1 * len(KINDS)  # 1 n, 1 d, 1 seed
    for a, b in zip(r1, r2):
        assert a.n == b.n and a.d == b.d and a.seed == b.seed and a.kind == b.kind
        assert (np.isnan(a.cv_spearman) and np.isnan(b.cv_spearman)) or a.cv_spearman == pytest.approx(
            b.cv_spearman
        )


def test_run_zero_inflation_check_runs_for_both_kinds():
    results = run_zero_inflation_check(seed=0)
    assert len(results) == len(KINDS)
    assert {r.kind for r in results} == set(KINDS)
    assert all(r.n == 60 and r.d == 10 for r in results)


def test_run_real_dataset_check_skips_gracefully_without_kalos_data(monkeypatch):
    """Never raises when KALOS_DATA is unset or the checkout is missing -
    the real-data comparison is optional, not a hard requirement (see module
    docstring)."""
    monkeypatch.delenv("KALOS_DATA", raising=False)
    assert run_real_dataset_check() == []
