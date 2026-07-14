"""The Singleton runner: acceptance criteria 2-4 from docs/M2_INTEGRATION.md.

  2. `run_ready`/`run_one` processes READY -> PROCESSING -> DONE, `result`
     populated with the same analysis `/api/latest` returns, provenance
     stamped.
  3. A zero-variance-target experiment ends FAILED with the actionable
     message, no exception escapes.
  4. The Singleton lock prevents a second concurrent runner from
     double-processing.

Every store/lock path here is a `tmp_path`; the real `~/.kalos` is never
touched.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")

from kalos.portal.app import _analyze  # noqa: E402
from kalos.runner.adapter import LocalStoreAdapter  # noqa: E402
from kalos.runner.singleton import SingletonLock, run_one, run_ready  # noqa: E402
from kalos.store import SqliteStore, Status  # noqa: E402


def _good_sheet(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Mirrors tests/test_hardening.py::_good_sheet - a small, honest, varying
    run sheet that `_analyze` can actually fit and produce proposals for."""
    rng = np.random.default_rng(seed)
    methanol = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    titer = 1.5 * methanol - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "medium": rng.choice(["A", "B", "C"], n),
        "Methanol": methanol.round(3),
        "pH": ph.round(2),
        "lipase_titer": titer.round(3),
    })


def _payload_from_df(df: pd.DataFrame) -> dict:
    return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}


def _adapter(tmp_path) -> tuple[LocalStoreAdapter, SqliteStore]:
    store = SqliteStore(tmp_path / "experiments.db")
    return LocalStoreAdapter(store), store


def _lock_path(tmp_path):
    return tmp_path / "runner.lock"


# --- criterion 2: READY -> PROCESSING -> DONE, result + provenance --------- #

def test_run_one_processes_ready_experiment_to_done(tmp_path):
    adapter, store = _adapter(tmp_path)
    df = _good_sheet()
    exp = store.create("run 1", _payload_from_df(df), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)

    result = run_one(adapter, exp.id, lock_path=_lock_path(tmp_path))

    assert result.status == "DONE"
    assert result.error is None

    done = store.get(exp.id)
    assert done.status == Status.DONE
    assert done.error is None
    assert done.result is not None

    # Same analysis /api/latest would return: recompute directly on the same
    # frame and compare the deterministic, non-timestamp fields.
    expected = _analyze(df, "lipase_titer")
    assert done.result["target"] == expected["target"] == "lipase_titer"
    assert done.result["cv_spearman"] == expected["cv_spearman"]
    assert done.result["best"] == expected["best"]
    assert done.result["drivers"] == expected["drivers"]
    assert done.result["proposals"] == expected["proposals"]

    # provenance stamped
    assert done.provenance is not None
    assert done.provenance["seed"] == expected["seed"]
    assert done.provenance["engine_version"] == expected["engine_version"]
    assert "processed_at" in done.provenance
    # no replicate_group_by in config -> no noise_floor key
    assert "noise_floor" not in done.provenance


def test_run_ready_processes_all_ready_experiments(tmp_path):
    adapter, store = _adapter(tmp_path)
    df = _good_sheet(seed=1)
    exp1 = store.create("run 1", _payload_from_df(df), {"target": "lipase_titer"})
    exp2 = store.create("run 2", _payload_from_df(_good_sheet(seed=2)), {"target": "lipase_titer"})
    store.set_status(exp1.id, Status.READY)
    store.set_status(exp2.id, Status.READY)
    # a DRAFT experiment must NOT be picked up by run_ready
    draft = store.create("run 3 (draft)", _payload_from_df(df), {"target": "lipase_titer"})

    results = run_ready(adapter, lock_path=_lock_path(tmp_path))

    assert {r.id for r in results} == {exp1.id, exp2.id}
    assert all(r.status == "DONE" for r in results)
    assert store.get(exp1.id).status == Status.DONE
    assert store.get(exp2.id).status == Status.DONE
    assert store.get(draft.id).status == Status.DRAFT  # untouched


def test_run_ready_is_idempotent_and_skips_processing(tmp_path):
    adapter, store = _adapter(tmp_path)
    df = _good_sheet()
    exp = store.create("run 1", _payload_from_df(df), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)
    store.set_status(exp.id, Status.PROCESSING)  # simulate a stuck/in-flight run

    results = run_ready(adapter, lock_path=_lock_path(tmp_path))

    assert results == []  # list_ready() only returns READY, so nothing runs
    assert store.get(exp.id).status == Status.PROCESSING  # left untouched


# --- criterion 3: zero-variance target -> FAILED, actionable, no crash ----- #

def test_zero_variance_target_ends_failed_not_crashed(tmp_path):
    adapter, store = _adapter(tmp_path)
    n = 20
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "medium": rng.choice(["A", "B"], n),
        "Methanol": rng.uniform(0, 4, n).round(3),
        "lipase_titer": np.full(n, 5.0),  # zero variance
    })
    exp = store.create("bad run", _payload_from_df(df), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)

    result = run_one(adapter, exp.id, lock_path=_lock_path(tmp_path))

    assert result.status == "FAILED"
    assert result.error == (
        "The target column 'lipase_titer' has no variance (all values are "
        "identical); it cannot be modeled. Check you selected the "
        "measured-output column."
    )

    failed = store.get(exp.id)
    assert failed.status == Status.FAILED
    assert failed.error == result.error
    assert failed.result is None


def test_failed_experiment_can_be_retried_to_ready(tmp_path):
    adapter, store = _adapter(tmp_path)
    n = 20
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "medium": rng.choice(["A", "B"], n),
        "Methanol": rng.uniform(0, 4, n).round(3),
        "lipase_titer": np.full(n, 5.0),
    })
    exp = store.create("bad run", _payload_from_df(df), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)
    run_one(adapter, exp.id, lock_path=_lock_path(tmp_path))
    assert store.get(exp.id).status == Status.FAILED

    retried = store.set_status(exp.id, Status.READY)  # FAILED -> READY, no force
    assert retried.status == Status.READY


# --- replicate structure -> noise_floor in provenance ----------------------- #

def test_noise_floor_populated_when_replicate_group_by_configured(tmp_path):
    adapter, store = _adapter(tmp_path)
    rng = np.random.default_rng(3)
    recipes = np.array([[1.0, 5.5], [2.0, 6.0], [3.0, 6.5]])
    rows = []
    for m, p in recipes:
        for _ in range(4):  # 4 replicates per recipe
            rows.append({
                "Methanol": m, "pH": p,
                "lipase_titer": float(1.5 * m - 0.8 * (p - 6) ** 2 + rng.normal(0, 0.05)),
            })
    df = pd.DataFrame(rows)
    exp = store.create(
        "replicated run",
        _payload_from_df(df),
        {"target": "lipase_titer", "replicate_group_by": ["Methanol", "pH"]},
    )
    store.set_status(exp.id, Status.READY)

    result = run_one(adapter, exp.id, lock_path=_lock_path(tmp_path))

    assert result.status == "DONE"
    done = store.get(exp.id)
    assert done.provenance is not None
    assert "noise_floor" in done.provenance
    nf = done.provenance["noise_floor"]
    assert set(nf) == {"icc", "sigma"}
    assert nf["sigma"] is not None and nf["sigma"] >= 0


# --- force re-run of a DONE experiment -------------------------------------- #

def test_force_reruns_done_experiment_and_discards_prior_result(tmp_path):
    adapter, store = _adapter(tmp_path)
    df = _good_sheet()
    exp = store.create("run 1", _payload_from_df(df), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)
    first = run_one(adapter, exp.id, lock_path=_lock_path(tmp_path))
    assert first.status == "DONE"

    # Without force, re-running a DONE experiment is refused outright (a
    # pre-flight guard, not an analysis failure) - it must not clobber the
    # existing result, so it raises rather than transitioning to FAILED.
    with pytest.raises(ValueError, match="already DONE"):
        run_one(adapter, exp.id, lock_path=_lock_path(tmp_path))
    assert store.get(exp.id).status == Status.DONE

    with_force = run_one(adapter, exp.id, force=True, lock_path=_lock_path(tmp_path))
    assert with_force.status == "DONE"
    assert store.get(exp.id).status == Status.DONE


# --- criterion 4: the lock prevents a second concurrent runner -------------- #

def test_lock_refuses_a_second_acquire_while_held(tmp_path):
    lock_path = _lock_path(tmp_path)
    first = SingletonLock(lock_path)
    assert first.acquire() is True

    second = SingletonLock(lock_path)
    assert second.acquire() is False

    first.release()
    assert SingletonLock(lock_path).acquire() is True


def test_lock_reclaims_a_stale_lock_from_a_dead_pid(tmp_path):
    lock_path = _lock_path(tmp_path)
    # a pid that (almost certainly) does not exist
    dead_pid = 2 ** 30
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"pid": dead_pid, "started_at": time.time()}))

    lock = SingletonLock(lock_path)
    assert lock.acquire() is True
    lock.release()


def test_run_ready_no_ops_when_lock_already_held(tmp_path):
    adapter, store = _adapter(tmp_path)
    df = _good_sheet()
    exp = store.create("run 1", _payload_from_df(df), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)

    lock_path = _lock_path(tmp_path)
    # simulate a fresh, live lock held by this very test process
    held = SingletonLock(lock_path)
    assert held.acquire() is True
    try:
        results = run_ready(adapter, lock_path=lock_path)
        assert results == []
        # the experiment was never touched - still READY, no double-processing
        assert store.get(exp.id).status == Status.READY
    finally:
        held.release()


def test_run_one_raises_when_lock_already_held(tmp_path):
    adapter, store = _adapter(tmp_path)
    df = _good_sheet()
    exp = store.create("run 1", _payload_from_df(df), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)

    lock_path = _lock_path(tmp_path)
    held = SingletonLock(lock_path)
    assert held.acquire() is True
    try:
        with pytest.raises(RuntimeError, match="lock"):
            run_one(adapter, exp.id, lock_path=lock_path)
        assert store.get(exp.id).status == Status.READY
    finally:
        held.release()
