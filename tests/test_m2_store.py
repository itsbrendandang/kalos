"""M2 experiment store: lifecycle truth table + SqliteStore CRUD (acceptance
criterion 1: create -> DRAFT, flip to READY)."""
from __future__ import annotations

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
        (Status.DONE, Status.PROCESSING, True, False),  # force only covers DONE->READY
        (Status.DRAFT, Status.DRAFT, False, False),   # same-status is never a transition
        (Status.DONE, Status.DONE, True, False),
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
