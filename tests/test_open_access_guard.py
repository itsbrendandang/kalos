"""An unauthenticated portal must not serve callers off the box.

With no tokens provisioned every request resolves to an anonymous read+write
principal on the `default` tenant (`kalos.portal.auth`). Binding that to a
non-loopback interface publishes upload, analysis and campaign mutation to anyone
who can route to the host, and until now the only guard was a startup WARNING.

Two layers, because either alone is insufficient:

  - `assert_safe_exposure`, called from `python -m kalos.portal`, fails fast with an
    actionable message. It is bypassed entirely by `uvicorn kalos.portal.app:app
    --host 0.0.0.0`, which is what a real deployment does.
  - a peer-address guard in the app, which cannot be bypassed by choosing a
    different entrypoint, because it keys off who is actually calling.

Local development is deliberately untouched: loopback callers are always served,
so open mode still works on a bench machine and across this test suite.
"""
from __future__ import annotations

import pytest

from kalos.portal.config import (
    InsecureBindError,
    assert_safe_exposure,
    is_local_client,
    is_loopback_bind,
    open_access_explicitly_allowed,
)

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal.app import app  # noqa: E402
from kalos.portal.auth import get_authenticator  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch):
    """Every test starts from genuinely open, with no opt-out in force."""
    for var in ("KALOS_AUTH_TOKENS", "KALOS_AUTH_TOKENS_FILE", "KALOS_ALLOW_OPEN_ACCESS"):
        monkeypatch.delenv(var, raising=False)
    # A real salt by default so the auth assertions below isolate auth. The salt
    # requirement has its own tests further down.
    monkeypatch.setenv("KALOS_ANON_SALT", "test-salt-not-the-dev-one")


# --- classifying a peer address --------------------------------------------- #


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.7", "::1", "::ffff:127.0.0.1"])
def test_loopback_peers_are_local(host):
    """Parsed as addresses rather than string-matched, so the whole 127/8 range
    and IPv4-mapped IPv6 are covered, not just the two usual spellings."""
    assert is_local_client(host) is True


@pytest.mark.parametrize("host", ["10.0.0.5", "192.168.1.20", "8.8.8.8", "2001:db8::1"])
def test_routable_peers_are_not_local(host):
    assert is_local_client(host) is False


@pytest.mark.parametrize("host", [None, "", "testclient"])
def test_transports_without_an_ip_count_as_local(host):
    """Starlette's TestClient reports the peer as `testclient` and a Unix socket
    has no IP; neither is a remote caller. Not a spoofing hole - the peer address
    comes from the transport, not from a request header, so a remote TCP client
    always presents a parseable IP and cannot present this.
    """
    assert is_local_client(host) is True


# --- classifying a bind host ------------------------------------------------- #


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_binds_stay_local(host):
    assert is_loopback_bind(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", None, "10.0.0.5", "example.com"])
def test_wildcard_and_routable_binds_are_not_loopback(host):
    """A wildcard bind accepts connections from anywhere, so it is emphatically
    not loopback even though it is not itself a routable address. An unresolvable
    name is treated as not-loopback, which is the safe answer for a bind and the
    opposite of the safe answer for a peer.
    """
    assert is_loopback_bind(host) is False


# --- the bind refusal -------------------------------------------------------- #


def test_open_on_a_wildcard_bind_is_refused():
    with pytest.raises(InsecureBindError) as exc:
        assert_safe_exposure("0.0.0.0", auth_configured=False)
    msg = str(exc.value)
    # The message has to tell an operator what to actually do.
    assert "KALOS_AUTH_TOKENS" in msg
    assert "KALOS_ALLOW_OPEN_ACCESS" in msg


def test_open_on_loopback_is_allowed():
    """Local development must be untouched by this."""
    assert_safe_exposure("127.0.0.1", auth_configured=False)


def test_authenticated_on_a_wildcard_bind_is_allowed():
    assert_safe_exposure("0.0.0.0", auth_configured=True)


def test_the_opt_out_is_honoured_when_set_deliberately(monkeypatch):
    monkeypatch.setenv("KALOS_ALLOW_OPEN_ACCESS", "1")
    assert open_access_explicitly_allowed() is True
    assert_safe_exposure("0.0.0.0", auth_configured=False)


@pytest.mark.parametrize("value", ["0", "false", "no", "", "maybe"])
def test_the_opt_out_needs_an_affirmative_value(monkeypatch, value):
    """A variable that merely EXISTS must not disable the guard - an empty or
    leftover value in a config template would otherwise silently reopen it."""
    monkeypatch.setenv("KALOS_ALLOW_OPEN_ACCESS", value)
    assert open_access_explicitly_allowed() is False
    with pytest.raises(InsecureBindError):
        assert_safe_exposure("0.0.0.0", auth_configured=False)


# --- the bypass this guard originally had ----------------------------------- #


def test_an_empty_token_list_does_not_count_as_enforcing(monkeypatch):
    """The hole the first version of this guard walked straight into.

    `is_configured` tests only whether an auth config STRING is present, while
    `principal_for` falls back to the anonymous read+write principal when the
    parsed record list is empty. So `KALOS_AUTH_TOKENS=[]` - which a config
    template rendering an empty array produces easily - looked locked down and was
    wide open. Every security decision keys off `enforces` instead.
    """
    monkeypatch.setenv("KALOS_AUTH_TOKENS", "[]")
    auth = get_authenticator()
    assert auth.is_configured() is True, "a config string is present"
    assert auth.enforces() is False, "but no token can ever authenticate"
    # and the anonymous principal is what a caller would actually receive
    assert auth.principal_for(None).anonymous is True
    with pytest.raises(InsecureBindError):
        assert_safe_exposure("0.0.0.0", auth_configured=auth.enforces())


def test_a_real_token_does_count_as_enforcing(monkeypatch):
    monkeypatch.setenv(
        "KALOS_AUTH_TOKENS",
        '[{"subject": "acme", "tenant": "acme", "scopes": ["read"], '
        '"token_sha256": "' + "a" * 64 + '"}]',
    )
    assert get_authenticator().enforces() is True


# --- the peer-address guard, end to end ------------------------------------- #


def test_a_local_client_is_served_while_open():
    """The whole suite depends on this staying true."""
    with TestClient(app) as client:
        assert client.get("/api/latest").status_code == 200


def test_a_remote_client_is_refused_while_open():
    """The protection that cannot be bypassed by launching uvicorn differently."""
    with TestClient(app, client=("203.0.113.9", 51000)) as client:
        resp = client.get("/api/latest")
    assert resp.status_code == 503
    body = resp.json()
    # 503 not 401: the caller has no credential to offer and nothing they send
    # would help. It is a server posture problem, so the message names the fix.
    assert "KALOS_AUTH_TOKENS_FILE" in body["error"]


def test_a_remote_client_is_served_once_auth_enforces(monkeypatch):
    """Provisioning tokens lifts the guard with no restart, because auth config is
    re-read per request."""
    monkeypatch.setenv(
        "KALOS_AUTH_TOKENS",
        '[{"subject": "acme", "tenant": "acme", "scopes": ["read", "write"], '
        '"token_sha256": "' + "b" * 64 + '"}]',
    )
    with TestClient(app, client=("203.0.113.9", 51000)) as client:
        # 401 rather than 503: the guard is satisfied and normal auth now applies,
        # which is the correct answer for a request carrying no token.
        assert client.get("/api/latest").status_code == 401


def test_a_remote_client_is_served_when_open_access_is_allowed(monkeypatch):
    monkeypatch.setenv("KALOS_ALLOW_OPEN_ACCESS", "1")
    with TestClient(app, client=("203.0.113.9", 51000)) as client:
        assert client.get("/api/latest").status_code == 200


def test_the_guard_covers_mutating_routes_too(monkeypatch):
    """An open portal's real exposure is write access, not reads."""
    with TestClient(app, client=("198.51.100.4", 52000)) as client:
        assert client.post("/api/campaign/start", json={}).status_code == 503


# --- the salt requirement ---------------------------------------------------- #


def test_exposing_without_a_salt_is_refused(monkeypatch):
    """A silent failure, so it blocks.

    Pseudonyms are emitted successfully and look fine while being derived from a
    dev salt that is published in this repository, so anyone who reads the source
    can dictionary-attack them. The entire point of the salt is that these hashes
    leave the machine.
    """
    monkeypatch.delenv("KALOS_ANON_SALT", raising=False)
    with pytest.raises(InsecureBindError) as exc:
        assert_safe_exposure("0.0.0.0", auth_configured=True)
    assert "KALOS_ANON_SALT" in str(exc.value)


def test_missing_salt_is_fine_on_loopback(monkeypatch):
    """Nothing leaves the machine, so nothing to protect."""
    monkeypatch.delenv("KALOS_ANON_SALT", raising=False)
    assert_safe_exposure("127.0.0.1", auth_configured=False)


def test_both_problems_are_reported_together(monkeypatch):
    """An operator should learn everything they must fix in one attempt, not
    discover the second problem only after fixing the first."""
    monkeypatch.delenv("KALOS_ANON_SALT", raising=False)
    with pytest.raises(InsecureBindError) as exc:
        assert_safe_exposure("0.0.0.0", auth_configured=False)
    msg = str(exc.value)
    assert "(1)" in msg and "(2)" in msg
    assert "KALOS_AUTH_TOKENS" in msg
    assert "KALOS_ANON_SALT" in msg


def test_a_sound_posture_is_allowed(monkeypatch):
    monkeypatch.setenv("KALOS_ANON_SALT", "a-real-secret")
    assert_safe_exposure("0.0.0.0", auth_configured=True)


def test_cors_alone_does_not_block_exposure(monkeypatch):
    """CORS is deliberately NOT a blocker, and the reasoning matters.

    The default is `allow_origin_regex` limited to http://localhost and
    http://127.0.0.1 with credentials disabled, so it is not a credential-theft
    vector - it rejects evil.com, localhost.evil.com and even https://localhost.
    Its real consequence on a deployment is that the production browser client is
    rejected, which surfaces on the very first request. Loud failures get warnings,
    silent ones get refusals.
    """
    monkeypatch.delenv("KALOS_CORS_ORIGINS", raising=False)
    monkeypatch.setenv("KALOS_ANON_SALT", "a-real-secret")
    assert_safe_exposure("0.0.0.0", auth_configured=True)


def test_the_dev_cors_regex_admits_only_loopback_origins():
    """Pinning the claim the decision above rests on."""
    import re

    from kalos.portal.config import _DEV_ORIGIN_REGEX, cors_config

    pattern = re.compile(_DEV_ORIGIN_REGEX)
    assert pattern.fullmatch("http://localhost:3000")
    assert pattern.fullmatch("http://127.0.0.1:8050")
    for hostile in ("https://evil.com", "http://evil.com", "http://localhost.evil.com", "https://localhost:3000"):
        assert not pattern.fullmatch(hostile), hostile
    # credentials are not enabled, so a third-party page cannot ride a user's token
    assert "allow_credentials" not in cors_config()


def test_salt_configured_reflects_the_environment(monkeypatch):
    from kalos.data.anonymizer import salt_configured

    monkeypatch.delenv("KALOS_ANON_SALT", raising=False)
    assert salt_configured() is False
    monkeypatch.setenv("KALOS_ANON_SALT", "x")
    assert salt_configured() is True
