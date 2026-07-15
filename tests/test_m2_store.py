"""M2 experiment store: lifecycle truth table + SqliteStore CRUD (acceptance
criterion 1: create -> DRAFT, flip to READY)."""
from __future__ import annotations

import json
import math
import sqlite3

import pytest

from kalos.store import ExperimentNotFound, IllegalTransition, SqliteStore, Status, legal_transition


# --- legal_transition truth table --------------------------------------- #

@pytest.mark.parametrize(
    "old,new,force,expected",
    [
        (Status.DRAFT, Status.READY, False, True),
        (Status.DRAFT, Status.PROCESSING, False, False),
        (Status.DRAFT, Status.DONE, False, False),
        (Status.READY, Status.PROCESSING, False, True),
        (Status.READY, Status.DRAFT, False, False),
        (Status.READY, Status.DONE, False, False),
        (Status.PROCESSING, Status.DONE, False, True),
        (Status.PROCESSING, Status.FAILED, False, True),
        (Status.PROCESSING, Status.READY, False, False),
        (Status.FAILED, Status.READY, False, True),
        (Status.FAILED, Status.DONE, False, False),
        (Status.FAILED, Status.PROCESSING, False, False),
        (Status.DONE, Status.READY, False, False),   # only legal WITH force
        (Status.DONE, Status.READY, True, True),
        (Status.DONE, Status.PROCESSING, True, False),  # force only covers ->READY
        (Status.DRAFT, Status.DRAFT, False, False),   # same-status is never a transition
        (Status.DONE, Status.DONE, True, False),
        # FIX 4: PROCESSING -> READY is ALSO force-gated, orphan-recovery-only
        # (kalos.runner.singleton.reclaim_stale) - never reachable via the
        # client-facing PATCH endpoint, which rejects it regardless of force.
        (Status.PROCESSING, Status.READY, False, False),
        (Status.PROCESSING, Status.READY, True, True),
    ],
)
def test_legal_transition_truth_table(old, new, force, expected):
    assert legal_transition(old, new, force=force) is expected


# --- SqliteStore CRUD + lifecycle ---------------------------------------- #

def _payload():
    return {"columns": ["Methanol", "titer"], "rows": [{"Methanol": 1.0, "titer": 0.5}]}


def _config():
    return {"target": "titer", "anonymize": False}


def test_create_is_draft_and_flip_to_ready(tmp_path):
    store = SqliteStore(tmp_path / "experiments.db")
    exp = store.create("run 1", _payload(), _config())
    assert exp.status == Status.DRAFT
    assert exp.id.startswith("exp_")
    assert exp.result is None and exp.provenance is None and exp.error is None

    flipped = store.set_status(exp.id, Status.READY)
    assert flipped.status == Status.READY
    assert flipped.id == exp.id

    reloaded = store.get(exp.id)
    assert reloaded.status == Status.READY
    assert reloaded.payload == _payload()
    assert reloaded.config == _config()


def test_get_missing_raises(tmp_path):
    store = SqliteStore(tmp_path / "experiments.db")
    with pytest.raises(ExperimentNotFound):
        store.get("exp_does_not_exist")


def test_set_status_illegal_transition_raises_clear_error(tmp_path):
    store = SqliteStore(tmp_path / "experiments.db")
    exp = store.create("run 1", _payload(), _config())
    with pytest.raises(IllegalTransition, match="DRAFT -> PROCESSING"):
        store.set_status(exp.id, Status.PROCESSING)


def test_list_filters_by_status(tmp_path):
    store = SqliteStore(tmp_path / "experiments.db")
    a = store.create("a", _payload(), _config())
    b = store.create("b", _payload(), _config())
    store.set_status(a.id, Status.READY)

    ready = store.list(status=Status.READY)
    assert [e.id for e in ready] == [a.id]

    drafts = store.list(status=Status.DRAFT)
    assert [e.id for e in drafts] == [b.id]

    everything = store.list()
    assert {e.id for e in everything} == {a.id, b.id}


def test_save_result_marks_done_and_requires_processing(tmp_path):
    store = SqliteStore(tmp_path / "experiments.db")
    exp = store.create("run 1", _payload(), _config())
    with pytest.raises(IllegalTransition):
        store.save_result(exp.id, {"n": 1}, {"seed": 1})

    store.set_status(exp.id, Status.READY)
    store.set_status(exp.id, Status.PROCESSING)
    done = store.save_result(exp.id, {"n": 1}, {"seed": 1})
    assert done.status == Status.DONE
    assert done.result == {"n": 1}
    assert done.provenance == {"seed": 1}
    assert done.error is None


def test_force_reset_done_to_ready_discards_result(tmp_path):
    store = SqliteStore(tmp_path / "experiments.db")
    exp = store.create("run 1", _payload(), _config())
    store.set_status(exp.id, Status.READY)
    store.set_status(exp.id, Status.PROCESSING)
    store.save_result(exp.id, {"n": 1}, {"seed": 1})

    with pytest.raises(IllegalTransition):
        store.set_status(exp.id, Status.READY)  # without force: illegal

    reset = store.set_status(exp.id, Status.READY, force=True)
    assert reset.status == Status.READY
    assert reset.result is None and reset.provenance is None and reset.error is None


def test_set_status_failed_stores_error_message(tmp_path):
    store = SqliteStore(tmp_path / "experiments.db")
    exp = store.create("run 1", _payload(), _config())
    store.set_status(exp.id, Status.READY)
    store.set_status(exp.id, Status.PROCESSING)
    failed = store.set_status(exp.id, Status.FAILED, error="no varying process-input columns found")
    assert failed.status == Status.FAILED
    assert failed.error == "no varying process-input columns found"


# --- kalos#17 review nit: a retry must not carry forward a stale `error` ---- #
# A FAILED -> READY retry (and the DONE -> READY force re-run) left the OLD
# `error` field set, contradicting the contract (`error` is populated on
# FAILED only). `force_reset_done_to_ready_discards_result` above already
# covers the force path; this covers the plain FAILED -> READY retry.

def test_failed_to_ready_retry_clears_stale_error(tmp_path):
    store = SqliteStore(tmp_path / "experiments.db")
    exp = store.create("run 1", _payload(), _config())
    store.set_status(exp.id, Status.READY)
    store.set_status(exp.id, Status.PROCESSING)
    failed = store.set_status(exp.id, Status.FAILED, error="boom")
    assert failed.error == "boom"

    retried = store.set_status(exp.id, Status.READY)  # FAILED -> READY, no force needed
    assert retried.status == Status.READY
    assert retried.error is None

    reloaded = store.get(exp.id)
    assert reloaded.error is None


def test_to_dict_from_dict_round_trip(tmp_path):
    store = SqliteStore(tmp_path / "experiments.db")
    exp = store.create("run 1", _payload(), _config())
    d = exp.to_dict()
    assert set(d) == {
        "id", "name", "status", "created_at", "updated_at",
        "config", "payload", "result", "provenance", "error",
    }
    assert d["status"] == "DRAFT"
    from kalos.store import Experiment

    rebuilt = Experiment.from_dict(d)
    assert rebuilt == exp


# --- FIX 3 (HIGH): NaN in the analysis RESULT must not become invalid JSON -- #
# `_analyze`'s `cv_ci95`/`reliability.ci95` is `[nan, nan]` when Spearman is
# valid but there are fewer than 3 replicate groups. Before this fix,
# `save_result` wrote it with plain `json.dumps` (`allow_nan=True`), which
# happily writes the literal (INVALID JSON) token `NaN` into the TEXT column.

def test_save_result_with_nan_persists_as_strict_valid_json(tmp_path):
    db_path = tmp_path / "experiments.db"
    store = SqliteStore(db_path)
    exp = store.create("run 1", _payload(), _config())
    store.set_status(exp.id, Status.READY)
    store.set_status(exp.id, Status.PROCESSING)

    result = {
        "cv_spearman": 0.42,
        "cv_ci95": [float("nan"), float("nan")],
        "reliability": {"spearman": 0.42, "ci95": [float("nan"), float("nan")]},
    }
    provenance = {"seed": 1, "noise_floor": {"icc": float("nan"), "sigma": float("inf")}}
    store.save_result(exp.id, result, provenance)

    # Read the RAW stored text directly from the DB, bypassing the store's own
    # (already-lenient) `json.loads` on the way back out, and parse it with
    # the strict stdlib default (`parse_constant` rejects NaN/Infinity/-Infinity
    # tokens outright) - this is the actual bug: plain `json.dumps(allow_nan=True)`
    # writes a token that is not valid JSON at all.
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT result, provenance FROM experiments WHERE id = ?", (exp.id,)
        ).fetchone()
    finally:
        conn.close()
    raw_result, raw_provenance = row

    def _reject_non_finite(_):
        raise ValueError("non-finite token in stored JSON")

    parsed_result = json.loads(raw_result, parse_constant=_reject_non_finite)
    parsed_provenance = json.loads(raw_provenance, parse_constant=_reject_non_finite)
    assert parsed_result["cv_ci95"] == [None, None]
    assert parsed_result["reliability"]["ci95"] == [None, None]
    assert parsed_provenance["noise_floor"]["icc"] is None
    assert parsed_provenance["noise_floor"]["sigma"] is None

    # GET-shaped to_dict() round-trips cleanly - no NaN survives into the
    # Experiment either (None, not a NaN float you cannot compare/serialize).
    done = store.get(exp.id)
    d = done.to_dict()
    assert d["result"]["cv_ci95"] == [None, None]
    assert d["result"]["reliability"]["ci95"] == [None, None]
    assert d["provenance"]["noise_floor"]["icc"] is None
    assert d["provenance"]["noise_floor"]["sigma"] is None
    # to_dict() itself must be strict-JSON-serializable end to end.
    reserialized = json.loads(json.dumps(d), parse_constant=_reject_non_finite)
    assert reserialized == d


def test_create_with_nan_in_payload_or_config_persists_as_strict_valid_json(tmp_path):
    db_path = tmp_path / "experiments.db"
    store = SqliteStore(db_path)
    payload = {"columns": ["a"], "rows": [{"a": float("nan")}]}
    config = {"target": "a", "threshold": float("-inf")}
    exp = store.create("run 1", payload, config)

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT payload, config FROM experiments WHERE id = ?", (exp.id,)
        ).fetchone()
    finally:
        conn.close()
    raw_payload, raw_config = row

    def _reject_non_finite(_):
        raise ValueError("non-finite token in stored JSON")

    parsed_payload = json.loads(raw_payload, parse_constant=_reject_non_finite)
    parsed_config = json.loads(raw_config, parse_constant=_reject_non_finite)
    assert parsed_payload["rows"] == [{"a": None}]
    assert parsed_config["threshold"] is None


# --- _json_sanitize unit coverage -------------------------------------------- #

def test_json_sanitize_handles_nested_structures_and_tuples():
    from kalos.store.sqlite_store import _json_sanitize

    assert _json_sanitize(float("nan")) is None
    assert _json_sanitize(float("inf")) is None
    assert _json_sanitize(float("-inf")) is None
    assert _json_sanitize(1.5) == 1.5
    assert _json_sanitize([1.0, float("nan"), (2.0, float("inf"))]) == [1.0, None, [2.0, None]]
    assert _json_sanitize({"a": {"b": float("nan")}, "c": [float("nan")]}) == {
        "a": {"b": None}, "c": [None],
    }
    assert math.isnan(float("nan"))  # sanity: the literal we feed in is really NaN
