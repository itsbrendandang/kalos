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
from kalos.store import IllegalTransition, SqliteStore, Status  # noqa: E402


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


def test_run_one_skips_an_already_processing_experiment(tmp_path):
    """`run_one` (the on-demand, single-experiment path - CLI `--id`, or
    `POST /api/experiments/{id}/run`) does NOT call `reclaim_stale` - it is a
    per-experiment endpoint, not the batch orphan-recovery entry point - so it
    still treats an already-PROCESSING experiment as (possibly) genuinely
    in-flight and skips it rather than barging in."""
    adapter, store = _adapter(tmp_path)
    df = _good_sheet()
    exp = store.create("run 1", _payload_from_df(df), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)
    store.set_status(exp.id, Status.PROCESSING)

    result = run_one(adapter, exp.id, lock_path=_lock_path(tmp_path))

    assert result.status == "SKIPPED"
    assert store.get(exp.id).status == Status.PROCESSING  # left untouched


# --- FIX 4 (HIGH): orphan recovery - PROCESSING is reclaimed, not stranded -- #
#
# DEVIATION from the ORIGINAL pre-fix behavior (and from the pre-fix wording
# of docs/M2_INTEGRATION.md, "Idempotent - skips anything already
# PROCESSING"): before this fix, `run_ready` treated any PROCESSING
# experiment it saw as possibly legitimately in-flight elsewhere and left it
# untouched forever if the process that owned it had crashed - exactly the
# "hard crash mid-analysis strands an experiment in PROCESSING forever" bug
# this fix closes. That old assumption was inconsistent with the Singleton
# lock's OWN guarantee (docs/M2_INTEGRATION.md, "The Singleton lock prevents
# a second concurrent runner from double-processing"): if `run_ready` holds
# the lock, nothing else can be running, so a PROCESSING row found here can
# only be an orphan, never a live run. `run_ready` now reclaims it
# (`reclaim_stale`, at the top of the batch, while the lock is held) instead
# of skipping it. The doc has been updated to match (see the "Singleton
# runner" and "Status lifecycle" sections). The single-experiment path
# (`run_one`, see the test above) is UNCHANGED - it does not call
# `reclaim_stale` and still skips an already-PROCESSING experiment.
#
# This replaces the old `test_run_ready_is_idempotent_and_skips_processing`,
# which asserted exactly the stranding behavior this fix corrects.

def test_run_ready_reclaims_a_stale_processing_experiment_and_processes_it_to_done(tmp_path):
    adapter, store = _adapter(tmp_path)
    df = _good_sheet()
    exp = store.create("stuck run", _payload_from_df(df), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)
    store.set_status(exp.id, Status.PROCESSING)  # simulate a crash mid-analysis: no live run

    results = run_ready(adapter, lock_path=_lock_path(tmp_path))

    assert len(results) == 1
    assert results[0].id == exp.id
    assert results[0].status == "DONE"
    assert store.get(exp.id).status == Status.DONE
    assert store.get(exp.id).result is not None


def test_reclaim_stale_moves_processing_experiments_back_to_ready(tmp_path):
    from kalos.runner.singleton import reclaim_stale

    adapter, store = _adapter(tmp_path)
    df = _good_sheet()
    stuck = store.create("stuck", _payload_from_df(df), {"target": "lipase_titer"})
    store.set_status(stuck.id, Status.READY)
    store.set_status(stuck.id, Status.PROCESSING)
    untouched = store.create("draft", _payload_from_df(df), {"target": "lipase_titer"})

    reclaimed = reclaim_stale(adapter)

    assert reclaimed == [stuck.id]
    assert store.get(stuck.id).status == Status.READY
    assert store.get(untouched.id).status == Status.DRAFT  # never touched


def test_reclaim_stale_is_a_noop_for_a_backend_that_cannot_enumerate_processing(tmp_path):
    """`reclaim_stale` degrades to a no-op (never raises) for a backend
    without `list_processing` - e.g. `HttpBackendAdapter`, never wired to a
    live server in M2."""
    from kalos.runner.singleton import reclaim_stale

    class _BareAdapter:
        def list_ready(self):
            return []

        def fetch(self, exp_id):  # pragma: no cover - not exercised
            raise NotImplementedError

        def set_status(self, exp_id, status, *, force=False, error=None):  # pragma: no cover
            raise NotImplementedError

        def push_result(self, exp_id, result, provenance):  # pragma: no cover
            raise NotImplementedError

    assert reclaim_stale(_BareAdapter()) == []


def test_processing_to_ready_at_store_layer_requires_force(tmp_path):
    """The store-level legality that makes reclaim possible is itself
    force-gated (matching the "recovery only" framing in
    `kalos/store/models.py::legal_transition`) - a force-less
    `PROCESSING -> READY` is still illegal. The client-facing block on TOP of
    this (the PATCH endpoint rejects it even WITH force) is covered by
    `tests/test_m2_portal.py::test_patch_to_ready_while_processing_is_rejected_even_though_store_allows_reclaim`."""
    store = SqliteStore(tmp_path / "experiments.db")
    exp = store.create("x", _payload_from_df(_good_sheet()), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)
    store.set_status(exp.id, Status.PROCESSING)

    with pytest.raises(IllegalTransition):
        store.set_status(exp.id, Status.READY)  # no force -> still illegal

    forced = store.set_status(exp.id, Status.READY, force=True)
    assert forced.status == Status.READY


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


# --- FIX 2 (BLOCKER): the fallback FAILED write must never propagate, and
# run_ready must be per-experiment resilient ---------------------------------- #
#
# Before this fix, `adapter.set_status(exp_id, Status.FAILED, error=message)`
# (the fallback write after an analysis failure) sat OUTSIDE any try/except in
# `_run_one_locked`. If IT raised (e.g. a transport error on a real backend, or
# an `IllegalTransition` from a concurrent mutation), the exception propagated
# uncaught through `run_ready`'s list comprehension (as it was), aborting the
# rest of the batch, and through `--watch`'s loop (`kalos/runner/__main__.py`),
# killing the daemon.

class _PoisonedPushAdapter(LocalStoreAdapter):
    """Wraps a real `LocalStoreAdapter`; `push_result` raises for one chosen
    experiment id (everything else - including its OWN `set_status(FAILED)`
    fallback write - behaves normally)."""

    def __init__(self, store: SqliteStore, poison_id: str) -> None:
        super().__init__(store)
        self.poison_id = poison_id

    def push_result(self, exp_id, result, provenance):
        if exp_id == self.poison_id:
            raise RuntimeError("simulated push_result failure (e.g. a dead transport)")
        return super().push_result(exp_id, result, provenance)


class _DoublePoisonedAdapter(_PoisonedPushAdapter):
    """As `_PoisonedPushAdapter`, but the fallback `set_status(..., FAILED,
    ...)` write for the SAME poisoned id ALSO raises - this is what actually
    exercises FIX 2(a)'s try/except around that fallback write; with only
    `push_result` poisoned, the fallback write succeeds and never tests the
    wrapping at all."""

    def set_status(self, exp_id, status, *, force=False, error=None):
        if exp_id == self.poison_id and status == Status.FAILED:
            raise RuntimeError("simulated set_status(FAILED) failure")
        return super().set_status(exp_id, status, force=force, error=error)


class _PoisonedFetchAdapter(LocalStoreAdapter):
    """Wraps a real `LocalStoreAdapter`; `fetch` raises for one chosen id -
    a failure BEFORE `_run_one_locked`'s own try/except even starts, so only
    `run_ready`'s own per-item try/except (FIX 2(b)) can catch it."""

    def __init__(self, store: SqliteStore, poison_id: str) -> None:
        super().__init__(store)
        self.poison_id = poison_id

    def fetch(self, exp_id):
        if exp_id == self.poison_id:
            raise RuntimeError("simulated fetch failure (e.g. a dead transport)")
        return super().fetch(exp_id)


def test_run_one_survives_when_the_fallback_failed_write_also_raises(tmp_path):
    """FIX 2(a): the fallback `set_status(FAILED)` write is wrapped in its
    own try/except - even when BOTH the original analysis step (`push_result`
    here) and the fallback write raise, `run_one` still returns a `FAILED`
    `RunResult` carrying the ORIGINAL error, and does not itself raise."""
    store = SqliteStore(tmp_path / "experiments.db")
    df = _good_sheet()
    exp = store.create("run 1", _payload_from_df(df), {"target": "lipase_titer"})
    store.set_status(exp.id, Status.READY)
    adapter = _DoublePoisonedAdapter(store, poison_id=exp.id)

    result = run_one(adapter, exp.id, lock_path=_lock_path(tmp_path))  # must not raise

    assert result.status == "FAILED"
    assert result.error is not None and "simulated push_result failure" in result.error


def test_run_ready_survives_a_push_result_failure_and_still_processes_the_next_ready_item(tmp_path):
    """FIX 2(a)+(b) together, at the batch level: one experiment whose
    `push_result` (and, per `_DoublePoisonedAdapter`, whose OWN fallback
    `set_status(FAILED)` write) raises does not abort `run_ready` - the good
    experiment right after it in the same batch still runs to DONE, and
    `run_ready` itself does not raise. This is "the daemon path stays alive
    and processes the next READY item" from the task's regression spec."""
    store = SqliteStore(tmp_path / "experiments.db")
    bad = store.create("bad", _payload_from_df(_good_sheet(seed=9)), {"target": "lipase_titer"})
    good = store.create("good", _payload_from_df(_good_sheet(seed=10)), {"target": "lipase_titer"})
    store.set_status(bad.id, Status.READY)
    store.set_status(good.id, Status.READY)
    adapter = _DoublePoisonedAdapter(store, poison_id=bad.id)

    results = run_ready(adapter, lock_path=_lock_path(tmp_path))  # must not raise

    by_id = {r.id: r.status for r in results}
    assert by_id[good.id] == "DONE"
    assert by_id[bad.id] == "FAILED"
    assert store.get(good.id).status == Status.DONE


def test_run_ready_one_poisoned_and_one_good_experiment_both_reported_good_done_poisoned_failed(tmp_path):
    """FIX 2(b), the task's exact regression scenario: `run_ready` with one
    poisoned experiment (here, `push_result` raises but the fallback write
    succeeds normally) and one good experiment returns BOTH in the summary -
    good -> DONE, poisoned -> FAILED - and the store reflects both outcomes,
    without `run_ready` raising."""
    store = SqliteStore(tmp_path / "experiments.db")
    poisoned = store.create("poisoned", _payload_from_df(_good_sheet(seed=1)), {"target": "lipase_titer"})
    good = store.create("good", _payload_from_df(_good_sheet(seed=2)), {"target": "lipase_titer"})
    store.set_status(poisoned.id, Status.READY)
    store.set_status(good.id, Status.READY)
    adapter = _PoisonedPushAdapter(store, poison_id=poisoned.id)

    results = run_ready(adapter, lock_path=_lock_path(tmp_path))  # must not raise

    by_id = {r.id: r.status for r in results}
    assert by_id == {poisoned.id: "FAILED", good.id: "DONE"}
    assert store.get(poisoned.id).status == Status.FAILED
    assert store.get(good.id).status == Status.DONE


def test_run_ready_survives_a_fetch_failure_before_any_internal_try_except(tmp_path):
    """FIX 2(b)'s own safety net: `adapter.fetch` raises BEFORE
    `_run_one_locked` even reaches its internal try/except, so only
    `run_ready`'s per-item wrapping around the whole `_run_one_locked` call
    can catch this. The poisoned experiment is still reported (as FAILED,
    even though the store never actually recorded that), and the good
    experiment after it still processes to DONE."""
    store = SqliteStore(tmp_path / "experiments.db")
    poisoned = store.create("poisoned", _payload_from_df(_good_sheet(seed=3)), {"target": "lipase_titer"})
    good = store.create("good", _payload_from_df(_good_sheet(seed=4)), {"target": "lipase_titer"})
    store.set_status(poisoned.id, Status.READY)
    store.set_status(good.id, Status.READY)
    adapter = _PoisonedFetchAdapter(store, poison_id=poisoned.id)

    results = run_ready(adapter, lock_path=_lock_path(tmp_path))  # must not raise

    by_id = {r.id: r.status for r in results}
    assert by_id[good.id] == "DONE"
    assert by_id[poisoned.id] == "FAILED"
    assert store.get(good.id).status == Status.DONE
    # the poisoned experiment's status was never actually written (fetch
    # failed before any status transition) - it is still READY in the store,
    # but the BATCH still completed and reported it, which is the point.
    assert store.get(poisoned.id).status == Status.READY
