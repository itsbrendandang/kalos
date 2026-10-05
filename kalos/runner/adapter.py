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

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from email.message import Message
from typing import IO, Any, Callable, Protocol, runtime_checkable

from kalos.portal.config import is_loopback_bind
from kalos.store.models import Experiment, Status
from kalos.store.sqlite_store import SqliteStore

# transport(method, url, json_body, headers) -> parsed JSON response (dict or
# list). `headers` carries the `Authorization: Bearer ...` credential for the
# call (see `HttpBackendAdapter` for which one), or is `None` when none is
# configured. Injectable so `HttpBackendAdapter` can be contract-tested with a
# mock and never performs real network I/O inside this package.
Transport = Callable[[str, str, dict[str, Any] | None, dict[str, str] | None], Any]


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
    """`BackendAdapter` backed directly by a `SqliteStore`. The M2 default.

    Bound to one `tenant` (default `"default"`) so every store call it makes is
    scoped to that tenant (docs/HARDENING.md, Phase 1b/P1): the runner only ever
    sees and mutates the calling tenant's experiments."""

    def __init__(self, store: SqliteStore, *, tenant: str = "default") -> None:
        self._store = store
        self._tenant = tenant

    def list_ready(self) -> list[str]:
        return [exp.id for exp in self._store.list(status=Status.READY, tenant=self._tenant)]

    def list_processing(self) -> list[str]:
        """Ids of `PROCESSING` experiments - used by
        `kalos.runner.singleton.reclaim_stale` for orphan recovery. Not part
        of the `BackendAdapter` Protocol (only this default, store-backed
        adapter can enumerate by arbitrary status); `reclaim_stale` degrades
        to a no-op for a backend that lacks this method, e.g.
        `HttpBackendAdapter` (never wired to a live server in M2)."""
        return [exp.id for exp in self._store.list(status=Status.PROCESSING, tenant=self._tenant)]

    def fetch(self, exp_id: str) -> Experiment:
        return self._store.get(exp_id, tenant=self._tenant)

    def set_status(
        self, exp_id: str, status: Status, *, force: bool = False, error: str | None = None
    ) -> None:
        self._store.set_status(exp_id, status, force=force, error=error, tenant=self._tenant)

    def push_result(self, exp_id: str, result: dict[str, Any], provenance: dict[str, Any]) -> None:
        self._store.save_result(exp_id, result, provenance, tenant=self._tenant)


# Per-request socket timeout for `_urllib_transport`. Every call is a small JSON
# exchange with the portal (the analysis itself runs locally), so a request
# that has not completed in this long is a hung connection, not slow work.
# Without it `urlopen` blocks forever and wedges the `--watch` daemon.
_HTTP_TIMEOUT_SECONDS = 30.0
# Pause before the single retry `_urllib_transport` makes on a connection error.
_HTTP_RETRY_DELAY_SECONDS = 1.0


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect (SECURITY).

    urllib's default redirect handler re-sends every request header -
    `Authorization` included - to wherever a 3xx points: another host, or an
    `https` -> `http` downgrade. Following one could hand the runner's bearer
    token to a third party or put it on the wire in cleartext. The portal API
    never redirects, so a 3xx means a misconfigured URL or a hostile hop.
    Returning None makes urllib raise it as an `HTTPError` instead, which
    `_urllib_transport` does not retry.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


_OPENER = urllib.request.build_opener(_RefuseRedirects)


def _urllib_transport(
    method: str, url: str, json_body: dict[str, Any] | None, headers: dict[str, str] | None = None
) -> Any:
    """The real (non-test) transport: a plain `urllib` JSON request. Only
    exercised if `HttpBackendAdapter` is actually pointed at a live server,
    which M2 never does - `get_adapter()` only reaches this when a caller
    explicitly sets `KALOS_BACKEND=http` against a real `KALOS_BACKEND_URL`.

    Redirects are never followed (see `_RefuseRedirects`).

    Each attempt times out after `_HTTP_TIMEOUT_SECONDS`. A connection error
    is retried once: `urlopen` raises a plain `URLError` only when the
    connection could not be opened or the request could not be sent (refused,
    DNS failure, connect timeout), so the portal never acted on it and a retry
    is safe even for `PATCH`/`POST`. An `HTTPError` (a real response) and a
    failure after the request was sent (read timeout, reset) are NOT retried -
    the portal may already have applied a non-idempotent `PATCH`/`POST`.
    """
    data = json.dumps(json_body).encode("utf-8") if json_body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)

    def send() -> Any:
        with _OPENER.open(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            body = resp.read()
        return json.loads(body) if body else None

    try:
        return send()
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError:
        time.sleep(_HTTP_RETRY_DELAY_SECONDS)
        return send()


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

    Credentials:
      - `api_token` - the plaintext of a provisioned `KALOS_AUTH_TOKENS`
        principal with `read` and `runner` scopes on the runner's tenant
        (docs/HARDENING.md, "Provisioned-principal shape"), plus `write` only
        if `--id` is used to re-queue DRAFT/FAILED/DONE experiments. Since P1
        the routes behind `list_ready`/`fetch` require `read` and act on the
        principal's tenant; `PATCH` only accepts the runner transitions
        (`READY -> PROCESSING`, `PROCESSING -> FAILED`) from a `runner`-scoped
        principal; and `/result` accepts that principal too, writing to its
        tenant. So all four calls send it as `Authorization: Bearer
        <api_token>` when it is set. `get_adapter()` reads it from
        `KALOS_RUNNER_API_TOKEN`.
      - `token` - the legacy machine-channel secret for `/result` alone
        (`kalos/portal/experiments.py::_result_push_tenant`), which the portal
        matches against its own `KALOS_RUNNER_TOKEN` and which only reaches
        the `default` tenant. `push_result` falls back to it when no
        `api_token` is set. `get_adapter()` reads it from
        `KALOS_RUNNER_TOKEN`, falling back to `KALOS_BACKEND_TOKEN`.
    With neither set, no call sends an `Authorization` header. That only gets
    as far as reading against a portal in open mode, which never grants the
    `runner` scope, so a remote runner needs a provisioned `api_token`.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        api_token: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.api_token = api_token
        self._transport = transport or _urllib_transport

    def _api_headers(self) -> dict[str, str] | None:
        """The `Authorization` header for the scope-gated API routes, or None."""
        return {"Authorization": f"Bearer {self.api_token}"} if self.api_token else None

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return self._transport(method, f"{self.base_url}{path}", json_body, headers)

    def list_ready(self) -> list[str]:
        data = self._request("GET", "/api/experiments?status=READY", headers=self._api_headers())
        return [item["id"] if isinstance(item, dict) else item for item in data]

    def fetch(self, exp_id: str) -> Experiment:
        data = self._request("GET", f"/api/experiments/{exp_id}", headers=self._api_headers())
        return Experiment.from_dict(data)

    def set_status(
        self, exp_id: str, status: Status, *, force: bool = False, error: str | None = None
    ) -> None:
        self._request(
            "PATCH",
            f"/api/experiments/{exp_id}",
            {"status": status.value, "force": force, "error": error},
            headers=self._api_headers(),
        )

    def push_result(self, exp_id: str, result: dict[str, Any], provenance: dict[str, Any]) -> None:
        """Push a processed result. Sends `Authorization: Bearer <api_token>`
        when `self.api_token` is set (the runner principal's tenant), else
        `Bearer <token>` when `self.token` is set (the legacy channel, the
        `default` tenant only), matching the portal's gated `/result`
        endpoint - a plain, unauthenticated push would 404 (neither credential
        configured on the portal) or 401 (configured but missing/wrong here)."""
        if self.api_token:
            headers = self._api_headers()
        else:
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else None
        self._request(
            "POST",
            f"/api/experiments/{exp_id}/result",
            {"result": result, "provenance": provenance},
            headers=headers,
        )


# The deliberate opt-out for `_check_backend_url`'s cleartext refusal - a named
# variable, like the portal's `KALOS_ALLOW_OPEN_ACCESS`, so a default config can
# never carry it in by accident.
_ALLOW_INSECURE_HTTP_VAR = "KALOS_RUNNER_ALLOW_INSECURE_HTTP"


def _check_backend_url(base_url: str, *, sends_credentials: bool) -> None:
    """Refuse a `KALOS_BACKEND_URL` that would leak the runner's credentials
    (SECURITY).

    - Only `http`/`https` with a host. Anything else (`file:`, `ftp:`) is
      never a portal, and urllib would happily open it.
    - No `user:password@` in the URL: it is not how the runner authenticates,
      and a URL is the kind of string that ends up in logs.
    - No plain `http` to a non-loopback host while a token is configured:
      every bearer token would cross the network in cleartext, and nothing
      would ever say so - a silent failure, so it blocks rather than warns
      (the same rule as the portal's `assert_safe_exposure`). Loopback stays
      allowed, so a runner beside the portal or behind a local TLS-terminating
      proxy needs nothing; `KALOS_RUNNER_ALLOW_INSECURE_HTTP=1` accepts the
      risk knowingly, e.g. for a private network you trust.

    Errors name the scheme and host only, never the full URL.
    """
    parts = urllib.parse.urlsplit(base_url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(
            "KALOS_BACKEND_URL must be an http(s) URL with a host, e.g. https://kalos.example.com"
        )
    if parts.username is not None or parts.password is not None:
        raise ValueError(
            "KALOS_BACKEND_URL must not embed credentials (user:password@); "
            "the runner authenticates with KALOS_RUNNER_API_TOKEN"
        )
    insecure_ok = os.environ.get(_ALLOW_INSECURE_HTTP_VAR, "").strip().lower() in {"1", "true", "yes"}
    if (
        parts.scheme == "http"
        and sends_credentials
        and not is_loopback_bind(parts.hostname)
        and not insecure_ok
    ):
        raise ValueError(
            f"refusing to send runner credentials in cleartext to http://{parts.hostname}: "
            f"use an https:// KALOS_BACKEND_URL, or set {_ALLOW_INSECURE_HTTP_VAR}=1 "
            "to accept the risk on a network you trust"
        )


def get_adapter() -> BackendAdapter:
    """Select the backend from env: `KALOS_BACKEND=local` (default) or `http`,
    with `KALOS_BACKEND_URL` for `http`. The runner's provisioned API token
    (sent on every call, see `HttpBackendAdapter`) comes from
    `KALOS_RUNNER_API_TOKEN`. The legacy `/result` token comes from
    `KALOS_RUNNER_TOKEN` - matching the portal's `/result` gate env var
    (`kalos/portal/experiments.py::_result_push_tenant`) - falling back to
    the already-documented `KALOS_BACKEND_TOKEN` for backward compatibility.
    The URL is vetted by `_check_backend_url` before any token is attached.
    """
    backend = os.environ.get("KALOS_BACKEND", "local").strip().lower()
    if backend == "local":
        return LocalStoreAdapter(SqliteStore())
    if backend == "http":
        base_url = os.environ.get("KALOS_BACKEND_URL")
        if not base_url:
            raise ValueError("KALOS_BACKEND_URL must be set when KALOS_BACKEND=http")
        token = os.environ.get("KALOS_RUNNER_TOKEN") or os.environ.get("KALOS_BACKEND_TOKEN")
        api_token = os.environ.get("KALOS_RUNNER_API_TOKEN", "").strip() or None
        _check_backend_url(base_url, sends_credentials=bool(token or api_token))
        return HttpBackendAdapter(base_url.strip(), token=token, api_token=api_token)
    raise ValueError(f"unknown KALOS_BACKEND: {backend!r} (expected 'local' or 'http')")
