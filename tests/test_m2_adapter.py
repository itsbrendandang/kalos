"""BackendAdapter seam: LocalStoreAdapter wiring + the HttpBackendAdapter
contract test against a mock transport (acceptance criterion 5), plus the
transport's security properties. No network I/O beyond one test that talks to
its own throwaway servers on 127.0.0.1."""
from __future__ import annotations

import threading
import urllib.error
import urllib.request
from email.message import Message
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import kalos.runner.adapter as adapter_module
from kalos.runner.adapter import HttpBackendAdapter, LocalStoreAdapter, _urllib_transport, get_adapter
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
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []
        self.response = response

    def __call__(self, method, url, json_body, headers=None):
        self.calls.append((method, url, json_body, headers))
        return self.response


def test_http_adapter_list_ready_issues_get_with_status_filter():
    transport = _MockTransport(response=[{"id": "exp_1", "status": "READY"}, {"id": "exp_2", "status": "READY"}])
    adapter = HttpBackendAdapter("https://portal.example.com/", transport=transport)

    ids = adapter.list_ready()

    assert ids == ["exp_1", "exp_2"]
    assert transport.calls == [
        ("GET", "https://portal.example.com/api/experiments?status=READY", None, None),
    ]


def _exp_dict():
    return {
        "id": "exp_1", "name": "run 1", "status": "READY",
        "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00",
        "config": {"target": "titer"}, "payload": {"columns": [], "rows": []},
        "result": None, "provenance": None, "error": None,
    }


def test_http_adapter_fetch_issues_get_and_parses_experiment():
    transport = _MockTransport(response=_exp_dict())
    adapter = HttpBackendAdapter("https://portal.example.com", transport=transport)

    exp = adapter.fetch("exp_1")

    assert exp.id == "exp_1" and exp.status == Status.READY
    assert transport.calls == [
        ("GET", "https://portal.example.com/api/experiments/exp_1", None, None),
    ]


def test_http_adapter_set_status_issues_patch_with_body():
    transport = _MockTransport()
    adapter = HttpBackendAdapter("https://portal.example.com", transport=transport)

    adapter.set_status("exp_1", Status.PROCESSING)

    assert transport.calls == [
        ("PATCH", "https://portal.example.com/api/experiments/exp_1",
         {"status": "PROCESSING", "force": False, "error": None}, None),
    ]


def test_http_adapter_set_status_force_and_error_flow_through():
    transport = _MockTransport()
    adapter = HttpBackendAdapter("https://portal.example.com", transport=transport)

    adapter.set_status("exp_1", Status.FAILED, force=True, error="boom")

    assert transport.calls == [
        ("PATCH", "https://portal.example.com/api/experiments/exp_1",
         {"status": "FAILED", "force": True, "error": "boom"}, None),
    ]


def test_http_adapter_push_result_issues_post_with_result_and_provenance():
    """No token configured -> no Authorization header sent (matches the
    portal's default-disabled `/result` - see `test_m2_portal.py`)."""
    transport = _MockTransport()
    adapter = HttpBackendAdapter("https://portal.example.com", transport=transport)

    adapter.push_result("exp_1", {"n": 1}, {"seed": 1})

    assert transport.calls == [
        ("POST", "https://portal.example.com/api/experiments/exp_1/result",
         {"result": {"n": 1}, "provenance": {"seed": 1}}, None),
    ]


def test_http_adapter_push_result_sends_bearer_token_when_configured():
    """FIX A: `push_result` sends `Authorization: Bearer <token>` when the
    adapter has a token, so it can reach the portal's token-gated `/result`
    endpoint (`kalos/portal/app.py::push_experiment_result`)."""
    transport = _MockTransport()
    adapter = HttpBackendAdapter("https://portal.example.com", token="s3cr3t", transport=transport)

    adapter.push_result("exp_1", {"n": 1}, {"seed": 1})

    assert transport.calls == [
        ("POST", "https://portal.example.com/api/experiments/exp_1/result",
         {"result": {"n": 1}, "provenance": {"seed": 1}}, {"Authorization": "Bearer s3cr3t"}),
    ]


_API_AUTH = {"Authorization": "Bearer runner-api"}


def test_http_adapter_sends_api_token_on_list_ready_fetch_and_set_status():
    """Since P1 the routes behind these three require `read`/`write` scope, so
    a portal with `KALOS_AUTH_TOKENS` provisioned 401s them without the
    runner's API token - the runner's very first poll used to fail that way."""
    transport = _MockTransport()
    adapter = HttpBackendAdapter("https://portal.example.com", api_token="runner-api", transport=transport)

    transport.response = [{"id": "exp_1", "status": "READY"}]
    adapter.list_ready()
    transport.response = _exp_dict()
    adapter.fetch("exp_1")
    transport.response = None
    adapter.set_status("exp_1", Status.PROCESSING)

    assert [(method, headers) for method, _url, _body, headers in transport.calls] == [
        ("GET", _API_AUTH),
        ("GET", _API_AUTH),
        ("PATCH", _API_AUTH),
    ]


def test_http_adapter_push_result_prefers_api_token_over_legacy_token():
    """With both set, `/result` gets the API token: it resolves to the runner
    principal's tenant, where the legacy token only reaches `default`."""
    transport = _MockTransport()
    adapter = HttpBackendAdapter(
        "https://portal.example.com", token="legacy", api_token="runner-api", transport=transport
    )

    adapter.push_result("exp_1", {"n": 1}, {"seed": 1})

    assert transport.calls[0][3] == _API_AUTH


def test_http_adapter_legacy_token_is_only_sent_on_push_result():
    """The legacy machine-channel token is a `/result`-only secret; it must
    never leak onto the scope-gated routes, which would reject it anyway."""
    transport = _MockTransport(response=[])
    adapter = HttpBackendAdapter("https://portal.example.com", token="legacy", transport=transport)

    adapter.list_ready()
    adapter.set_status("exp_1", Status.READY)

    assert [headers for *_rest, headers in transport.calls] == [None, None]


def test_http_adapter_strips_trailing_slash_from_base_url():
    transport = _MockTransport(response=[])
    adapter = HttpBackendAdapter("https://portal.example.com/", transport=transport)
    adapter.list_ready()
    method, url, _json_body, _headers = transport.calls[0]
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
    monkeypatch.delenv("KALOS_RUNNER_TOKEN", raising=False)
    monkeypatch.delenv("KALOS_BACKEND_TOKEN", raising=False)
    monkeypatch.delenv("KALOS_RUNNER_API_TOKEN", raising=False)
    adapter = get_adapter()
    assert isinstance(adapter, HttpBackendAdapter)
    assert adapter.base_url == "https://portal.example.com"
    assert adapter.token is None
    assert adapter.api_token is None


def test_get_adapter_http_reads_runner_api_token(monkeypatch):
    """`KALOS_RUNNER_API_TOKEN` is its own variable, never conflated with the
    legacy `/result` token: the two authenticate against different checks."""
    monkeypatch.setenv("KALOS_BACKEND", "http")
    monkeypatch.setenv("KALOS_BACKEND_URL", "https://portal.example.com")
    monkeypatch.setenv("KALOS_RUNNER_API_TOKEN", "runner-api")
    monkeypatch.setenv("KALOS_RUNNER_TOKEN", "legacy")
    adapter = get_adapter()
    assert isinstance(adapter, HttpBackendAdapter)
    assert adapter.api_token == "runner-api"
    assert adapter.token == "legacy"


def test_get_adapter_http_token_prefers_runner_token_over_backend_token(monkeypatch):
    """FIX A: `get_adapter()` sends `KALOS_RUNNER_TOKEN` (the same env var the
    portal's `/result` token-gate reads) when set, falling back to the
    already-documented `KALOS_BACKEND_TOKEN` otherwise."""
    monkeypatch.setenv("KALOS_BACKEND", "http")
    monkeypatch.setenv("KALOS_BACKEND_URL", "https://portal.example.com")
    monkeypatch.setenv("KALOS_RUNNER_TOKEN", "runner-token")
    monkeypatch.setenv("KALOS_BACKEND_TOKEN", "backend-token")
    assert get_adapter().token == "runner-token"

    monkeypatch.delenv("KALOS_RUNNER_TOKEN", raising=False)
    assert get_adapter().token == "backend-token"


def test_get_adapter_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("KALOS_BACKEND", "carrier-pigeon")
    with pytest.raises(ValueError, match="unknown KALOS_BACKEND"):
        get_adapter()


# --- _urllib_transport: timeout + one safe retry ---------------------------- #
# The module's opener is replaced per-test, so no network I/O happens.

class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class _ScriptedOpener:
    """Stands in for `adapter._OPENER`: raises/returns each scripted outcome in
    turn and records what every attempt was called with."""

    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, str, str | None, object, bytes | None]] = []

    def open(self, req, timeout=None):
        self.calls.append(
            (req.get_method(), req.full_url, req.get_header("Authorization"), timeout, req.data)
        )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def no_retry_delay(monkeypatch):
    monkeypatch.setattr(adapter_module, "_HTTP_RETRY_DELAY_SECONDS", 0.0)


def test_urllib_transport_sends_a_timeout_headers_and_json_body(monkeypatch):
    opener = _ScriptedOpener(_FakeResponse(b'{"ok": true}'))
    monkeypatch.setattr(adapter_module, "_OPENER", opener)

    out = _urllib_transport(
        "PATCH", "https://portal.example.com/api/experiments/exp_1",
        {"status": "READY"}, {"Authorization": "Bearer runner-api"},
    )

    assert out == {"ok": True}
    assert opener.calls == [(
        "PATCH", "https://portal.example.com/api/experiments/exp_1", "Bearer runner-api",
        adapter_module._HTTP_TIMEOUT_SECONDS, b'{"status": "READY"}',
    )]
    assert adapter_module._HTTP_TIMEOUT_SECONDS > 0


def test_urllib_transport_retries_once_when_the_connection_fails(monkeypatch, no_retry_delay):
    """Refused/unreachable: the request never reached the portal, so one retry
    is safe even for a non-idempotent POST."""
    opener = _ScriptedOpener(
        urllib.error.URLError(ConnectionRefusedError("refused")), _FakeResponse(b""),
    )
    monkeypatch.setattr(adapter_module, "_OPENER", opener)

    assert _urllib_transport("POST", "https://portal.example.com/x", {"a": 1}, None) is None
    assert len(opener.calls) == 2


def test_urllib_transport_gives_up_after_the_single_retry(monkeypatch, no_retry_delay):
    refused = urllib.error.URLError(ConnectionRefusedError("refused"))
    opener = _ScriptedOpener(refused, refused)
    monkeypatch.setattr(adapter_module, "_OPENER", opener)

    with pytest.raises(urllib.error.URLError):
        _urllib_transport("GET", "https://portal.example.com/x", None, None)
    assert len(opener.calls) == 2


def test_urllib_transport_never_retries_an_http_error_response(monkeypatch, no_retry_delay):
    """A 401/409 is the portal's answer, not a transport blip - retrying it
    would only repeat the same answer (or re-apply a write)."""
    opener = _ScriptedOpener(
        urllib.error.HTTPError("https://portal.example.com/x", 401, "Unauthorized", Message(), None),
    )
    monkeypatch.setattr(adapter_module, "_OPENER", opener)

    with pytest.raises(urllib.error.HTTPError):
        _urllib_transport("GET", "https://portal.example.com/x", None, None)
    assert len(opener.calls) == 1


def test_urllib_transport_never_retries_once_the_request_was_sent(monkeypatch, no_retry_delay):
    """A read timeout surfaces from `urlopen` as a bare `TimeoutError`, not a
    `URLError`: the portal may already have applied the PATCH/POST, so a retry
    could double-apply it. It must propagate after one attempt."""
    opener = _ScriptedOpener(TimeoutError("timed out"))
    monkeypatch.setattr(adapter_module, "_OPENER", opener)

    with pytest.raises(TimeoutError):
        _urllib_transport("POST", "https://portal.example.com/x", {"a": 1}, None)
    assert len(opener.calls) == 1


# --- SECURITY: credentials never follow a redirect ---------------------------- #

def _serve(handler: type[BaseHTTPRequestHandler]) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_urllib_transport_never_forwards_credentials_through_a_redirect():
    """Real loopback sockets, no mocks. A 'portal' that answers 302 to another
    host must not get the runner's bearer token delivered there: stock urllib
    re-sends `Authorization` to the redirect target (the control below proves
    it), so the transport refuses to follow redirects at all."""
    stolen: list[str | None] = []

    class Thief(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            stolen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"[]")

        def log_message(self, format: str, *args: object) -> None:
            return None

    thief = _serve(Thief)

    class Redirector(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{thief.server_port}/steal")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return None

    portal = _serve(Redirector)
    url = f"http://127.0.0.1:{portal.server_port}/api/experiments?status=READY"
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            _urllib_transport("GET", url, None, {"Authorization": "Bearer runner-api"})
        err.value.close()
        assert err.value.code == 302
        assert stolen == []

        # Control: the stock opener really does hand the token over.
        req = urllib.request.Request(url, headers={"Authorization": "Bearer runner-api"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
        assert stolen == ["Bearer runner-api"]
    finally:
        for server in (portal, thief):
            server.shutdown()
            server.server_close()


# --- SECURITY: get_adapter() vets KALOS_BACKEND_URL ---------------------------- #

def _http_env(monkeypatch, url: str, **env: str) -> None:
    monkeypatch.setenv("KALOS_BACKEND", "http")
    monkeypatch.setenv("KALOS_BACKEND_URL", url)
    for var in ("KALOS_RUNNER_API_TOKEN", "KALOS_RUNNER_TOKEN", "KALOS_BACKEND_TOKEN",
                "KALOS_RUNNER_ALLOW_INSECURE_HTTP"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)


@pytest.mark.parametrize("url", [
    "http://portal.example.com",
    "http://10.0.0.5:8000",
    "http://kalos-engine:8000",
    "http://0.0.0.0:8000",
])
@pytest.mark.parametrize("token_var", ["KALOS_RUNNER_API_TOKEN", "KALOS_RUNNER_TOKEN"])
def test_get_adapter_refuses_cleartext_credentials_to_a_remote_host(monkeypatch, url, token_var):
    _http_env(monkeypatch, url, **{token_var: "s3cr3t-value"})

    with pytest.raises(ValueError, match="cleartext") as err:
        get_adapter()

    assert "s3cr3t-value" not in str(err.value)


@pytest.mark.parametrize("url", [
    "https://portal.example.com",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://[::1]:8000",
])
def test_get_adapter_allows_tls_or_loopback_with_credentials(monkeypatch, url):
    _http_env(monkeypatch, url, KALOS_RUNNER_API_TOKEN="runner-api")
    assert isinstance(get_adapter(), HttpBackendAdapter)


def test_get_adapter_cleartext_opt_out_is_an_explicit_variable(monkeypatch):
    _http_env(
        monkeypatch, "http://kalos-engine:8000",
        KALOS_RUNNER_API_TOKEN="runner-api", KALOS_RUNNER_ALLOW_INSECURE_HTTP="1",
    )
    assert isinstance(get_adapter(), HttpBackendAdapter)


def test_get_adapter_allows_plain_http_when_no_credential_is_sent(monkeypatch):
    """Nothing to leak: an open-mode portal on a trusted network still works."""
    _http_env(monkeypatch, "http://kalos-engine:8000")
    assert isinstance(get_adapter(), HttpBackendAdapter)


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://portal.example.com",
    "portal.example.com",
    "https://",
])
def test_get_adapter_rejects_urls_that_are_not_http(monkeypatch, url):
    _http_env(monkeypatch, url)
    with pytest.raises(ValueError, match="http\\(s\\) URL with a host"):
        get_adapter()


def test_get_adapter_rejects_credentials_in_the_url_without_echoing_them(monkeypatch):
    _http_env(monkeypatch, "https://alice:hunter2@portal.example.com")

    with pytest.raises(ValueError, match="must not embed credentials") as err:
        get_adapter()

    assert "hunter2" not in str(err.value) and "alice" not in str(err.value)
