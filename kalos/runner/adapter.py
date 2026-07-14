"""The `BackendAdapter` seam: the Singleton never knows which backend it talks
to (`docs/M2_INTEGRATION.md`, "The seam: BackendAdapter").

`LocalStoreAdapter` wraps the SQLite store and ships in M2. `HttpBackendAdapter`
is a typed, contract-tested stub for a future portal - it is never wired to a
live server in M2 (see the doc's "Non-goals"); its four methods issue REST
calls through an injectable `transport` callable so a test can assert the
right method/path/body against a mock, with no network I/O involved.

Deviation from the doc's literal `BackendAdapter.set_status(self, exp_id,
status) -> None` signature: it gains two optional keywords, `force: bool =
False` and `error: str | None = None`. The Singleton needs a way to push a
`DONE -> READY` reset (legal only with `force`) through this same seam, and a
way to stamp the actionable failure message onto a `FAILED` transition,
without a fifth Protocol method just for that one field. The doc's lifecycle
section already requires both transitions to exist; adding optional,
default-`False`/`None` keywords is additive and does not change any existing
call site's behavior.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Protocol, runtime_checkable

from kalos.store.models import Experiment, Status
from kalos.store.sqlite_store import SqliteStore

# transport(method, url, json_body) -> parsed JSON response (dict or list).
# Injectable so `HttpBackendAdapter` can be contract-tested with a mock and
# never performs real network I/O inside this package.
Transport = Callable[[str, str, dict[str, Any] | None], Any]


@runtime_checkable
class BackendAdapter(Protocol):
    """The four operations the Singleton runner needs from a backend."""

    def list_ready(self) -> list[str]:
        """Ids of `READY` experiments."""
        ...

    def fetch(self, exp_id: str) -> Experiment:
        """Pull one experiment. Raises if it does not exist."""
        ...

    def set_status(
        self, exp_id: str, status: Status, *, force: bool = False, error: str | None = None
    ) -> None:
        """Set an experiment's status (see the module docstring for `force`/`error`)."""
        ...

    def push_result(self, exp_id: str, result: dict[str, Any], provenance: dict[str, Any]) -> None:
        """Push a processed result + provenance and mark the experiment `DONE`."""
        ...


class LocalStoreAdapter:
    """`BackendAdapter` backed directly by a `SqliteStore`. The M2 default."""

    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def list_ready(self) -> list[str]:
        return [exp.id for exp in self._store.list(status=Status.READY)]

    def list_processing(self) -> list[str]:
        """Ids of `PROCESSING` experiments - used by
        `kalos.runner.singleton.reclaim_stale` for orphan recovery. Not part
        of the `BackendAdapter` Protocol (only this default, store-backed
        adapter can enumerate by arbitrary status); `reclaim_stale` degrades
        to a no-op for a backend that lacks this method, e.g.
        `HttpBackendAdapter` (never wired to a live server in M2)."""
        return [exp.id for exp in self._store.list(status=Status.PROCESSING)]

    def fetch(self, exp_id: str) -> Experiment:
        return self._store.get(exp_id)

    def set_status(
        self, exp_id: str, status: Status, *, force: bool = False, error: str | None = None
    ) -> None:
        self._store.set_status(exp_id, status, force=force, error=error)

    def push_result(self, exp_id: str, result: dict[str, Any], provenance: dict[str, Any]) -> None:
        self._store.save_result(exp_id, result, provenance)


def _urllib_transport(method: str, url: str, json_body: dict[str, Any] | None) -> Any:
    """The real (non-test) transport: a plain `urllib` JSON request. Only
    exercised if `HttpBackendAdapter` is actually pointed at a live server,
    which M2 never does - `get_adapter()` only reaches this when a caller
    explicitly sets `KALOS_BACKEND=http` against a real `KALOS_BACKEND_URL`.
    """
    import json
    import urllib.request

    data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - operator-controlled URL, not user input
        body = resp.read()
    return json.loads(body) if body else None


class HttpBackendAdapter:
    """Typed, contract-tested stub for a future portal backend.

    Maps the four `BackendAdapter` methods onto REST calls, and every one of
    them targets a REAL portal endpoint (`kalos/portal/app.py`):
      - `list_ready`   -> `GET  {base_url}/api/experiments?status=READY`
      - `fetch`        -> `GET  {base_url}/api/experiments/{id}`
      - `set_status`   -> `PATCH {base_url}/api/experiments/{id}` body `{"status", "force", "error"}`
      - `push_result`  -> `POST {base_url}/api/experiments/{id}/result` body `{"result", "provenance"}`

    The first three mirror the portal API table in `docs/M2_INTEGRATION.md`
    verbatim (`GET /api/experiments`, `GET /api/experiments/{id}`,
    `PATCH /api/experiments/{id}`) - `GET /api/experiments` genuinely honors
    the `status` query param (filters; omitted = all, unchanged default).
    `push_result` has no listed counterpart in that table (the documented
    endpoints describe the front end driving a server-side run); since M2's
    Singleton always runs `_analyze` locally and pushes the result back, the
    portal exposes `POST /api/experiments/{id}/result` for exactly this push
    (ingests via `SqliteStore.save_result`, legal only from `PROCESSING`).
    Both endpoints are real and covered by a contract test against the actual
    FastAPI app via `TestClient`, not just the mock transport below - but this
    adapter is still not selected by default (`KALOS_BACKEND=local`) and is
    not pointed at any live server in M2 (see the doc's "Non-goals").
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._transport = transport or _urllib_transport

    def _request(self, method: str, path: str, json_body: dict[str, Any] | None = None) -> Any:
        return self._transport(method, f"{self.base_url}{path}", json_body)

    def list_ready(self) -> list[str]:
        data = self._request("GET", "/api/experiments?status=READY")
        return [item["id"] if isinstance(item, dict) else item for item in data]

    def fetch(self, exp_id: str) -> Experiment:
        data = self._request("GET", f"/api/experiments/{exp_id}")
        return Experiment.from_dict(data)

    def set_status(
        self, exp_id: str, status: Status, *, force: bool = False, error: str | None = None
    ) -> None:
        self._request(
            "PATCH",
            f"/api/experiments/{exp_id}",
            {"status": status.value, "force": force, "error": error},
        )

    def push_result(self, exp_id: str, result: dict[str, Any], provenance: dict[str, Any]) -> None:
        self._request(
            "POST", f"/api/experiments/{exp_id}/result", {"result": result, "provenance": provenance}
        )


def get_adapter() -> BackendAdapter:
    """Select the backend from env: `KALOS_BACKEND=local` (default) or `http`,
    with `KALOS_BACKEND_URL` (and optionally `KALOS_BACKEND_TOKEN`) for `http`.
    """
    backend = os.environ.get("KALOS_BACKEND", "local").strip().lower()
    if backend == "local":
        return LocalStoreAdapter(SqliteStore())
    if backend == "http":
        base_url = os.environ.get("KALOS_BACKEND_URL")
        if not base_url:
            raise ValueError("KALOS_BACKEND_URL must be set when KALOS_BACKEND=http")
        token = os.environ.get("KALOS_BACKEND_TOKEN")
        return HttpBackendAdapter(base_url, token=token)
    raise ValueError(f"unknown KALOS_BACKEND: {backend!r} (expected 'local' or 'http')")
