"""BackendAdapter seam: LocalStoreAdapter wiring + the HttpBackendAdapter
contract test against a mock transport (acceptance criterion 5). No network
I/O anywhere in this file."""
from __future__ import annotations

import pytest

from kalos.runner.adapter import HttpBackendAdapter, LocalStoreAdapter, get_adapter
from kalos.store import SqliteStore, Status


def _payload():
    return {"columns": ["Methanol", "titer"], "rows": [{"Methanol": 1.0, "titer": 0.5}]}


def _config():
    return {"target": "titer"}


# --- LocalStoreAdapter ---------------------------------------------------- #

def test_local_store_adapter_round_trips_through_the_store(tmp_path):
    store = SqliteStore(tmp_path / "experiments.db")
    adapter = LocalStoreAdapter(store)

    exp = store.create("run 1", _payload(), _config())
    assert adapter.list_ready() == []

    store.set_status(exp.id, Status.READY)
    assert adapter.list_ready() == [exp.id]

    fetched = adapter.fetch(exp.id)
    assert fetched.id == exp.id and fetched.status == Status.READY

    adapter.set_status(exp.id, Status.PROCESSING)
    assert store.get(exp.id).status == Status.PROCESSING

    adapter.push_result(exp.id, {"n": 1}, {"seed": 1})
    done = store.get(exp.id)
    assert done.status == Status.DONE
    assert done.result == {"n": 1}


def test_local_store_adapter_set_status_force_and_error(tmp_path):
    store = SqliteStore(tmp_path / "experiments.db")
    adapter = LocalStoreAdapter(store)
    exp = store.create("run 1", _payload(), _config())
    store.set_status(exp.id, Status.READY)
    store.set_status(exp.id, Status.PROCESSING)

    adapter.set_status(exp.id, Status.FAILED, error="boom")
    failed = store.get(exp.id)
    assert failed.status == Status.FAILED and failed.error == "boom"

    adapter.set_status(exp.id, Status.READY)  # FAILED -> READY retry, no force needed
    assert store.get(exp.id).status == Status.READY


# --- HttpBackendAdapter contract test ------------------------------------- #

class _MockTransport:
    """Records every call; returns a scripted response per call."""

    def __init__(self, response=None):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.response = response

    def __call__(self, method, url, json_body):
        self.calls.append((method, url, json_body))
        return self.response


def test_http_adapter_list_ready_issues_get_with_status_filter():
    transport = _MockTransport(response=[{"id": "exp_1", "status": "READY"}, {"id": "exp_2", "status": "READY"}])
    adapter = HttpBackendAdapter("https://portal.example.com/", transport=transport)

    ids = adapter.list_ready()

    assert ids == ["exp_1", "exp_2"]
    assert transport.calls == [("GET", "https://portal.example.com/api/experiments?status=READY", None)]


def test_http_adapter_fetch_issues_get_and_parses_experiment():
    exp_dict = {
        "id": "exp_1", "name": "run 1", "status": "READY",
        "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00",
        "config": {"target": "titer"}, "payload": {"columns": [], "rows": []},
        "result": None, "provenance": None, "error": None,
    }
    transport = _MockTransport(response=exp_dict)
    adapter = HttpBackendAdapter("https://portal.example.com", transport=transport)

    exp = adapter.fetch("exp_1")

    assert exp.id == "exp_1" and exp.status == Status.READY
    assert transport.calls == [("GET", "https://portal.example.com/api/experiments/exp_1", None)]


def test_http_adapter_set_status_issues_patch_with_body():
    transport = _MockTransport()
    adapter = HttpBackendAdapter("https://portal.example.com", transport=transport)

    adapter.set_status("exp_1", Status.PROCESSING)

    assert transport.calls == [
        ("PATCH", "https://portal.example.com/api/experiments/exp_1",
         {"status": "PROCESSING", "force": False, "error": None}),
    ]


def test_http_adapter_set_status_force_and_error_flow_through():
    transport = _MockTransport()
    adapter = HttpBackendAdapter("https://portal.example.com", transport=transport)

    adapter.set_status("exp_1", Status.FAILED, force=True, error="boom")

    assert transport.calls == [
        ("PATCH", "https://portal.example.com/api/experiments/exp_1",
         {"status": "FAILED", "force": True, "error": "boom"}),
    ]


def test_http_adapter_push_result_issues_post_with_result_and_provenance():
    transport = _MockTransport()
    adapter = HttpBackendAdapter("https://portal.example.com", transport=transport)

    adapter.push_result("exp_1", {"n": 1}, {"seed": 1})

    assert transport.calls == [
        ("POST", "https://portal.example.com/api/experiments/exp_1/result",
         {"result": {"n": 1}, "provenance": {"seed": 1}}),
    ]


def test_http_adapter_strips_trailing_slash_from_base_url():
    transport = _MockTransport(response=[])
    adapter = HttpBackendAdapter("https://portal.example.com/", transport=transport)
    adapter.list_ready()
    method, url, _ = transport.calls[0]
    assert url == "https://portal.example.com/api/experiments?status=READY"


# --- get_adapter() factory ------------------------------------------------- #

def test_get_adapter_defaults_to_local(monkeypatch, tmp_path):
    # Never let this test touch the real ~/.kalos/experiments.db: redirect the
    # store's default path before get_adapter() constructs a bare SqliteStore().
    import kalos.store.sqlite_store as sqlite_store_module

    monkeypatch.delenv("KALOS_BACKEND", raising=False)
    monkeypatch.setattr(sqlite_store_module, "DEFAULT_DB_PATH", tmp_path / "experiments.db")
    adapter = get_adapter()
    assert isinstance(adapter, LocalStoreAdapter)


def test_get_adapter_http_requires_backend_url(monkeypatch):
    monkeypatch.setenv("KALOS_BACKEND", "http")
    monkeypatch.delenv("KALOS_BACKEND_URL", raising=False)
    with pytest.raises(ValueError, match="KALOS_BACKEND_URL"):
        get_adapter()


def test_get_adapter_http_builds_http_adapter(monkeypatch):
    monkeypatch.setenv("KALOS_BACKEND", "http")
    monkeypatch.setenv("KALOS_BACKEND_URL", "https://portal.example.com")
    adapter = get_adapter()
    assert isinstance(adapter, HttpBackendAdapter)
    assert adapter.base_url == "https://portal.example.com"


def test_get_adapter_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("KALOS_BACKEND", "carrier-pigeon")
    with pytest.raises(ValueError, match="unknown KALOS_BACKEND"):
        get_adapter()
