"""Kalos portal - deployment security configuration (docs/HARDENING.md, Phase 1c).

Centralizes the security-relevant configuration read from the environment so the
running posture is explicit, testable, and logged once at startup rather than
scattered across the app. Today that is the CORS origin policy; secrets (auth
tokens, the runner token) are read in `kalos.portal.auth` / `experiments`.

Backward compatible: with nothing configured the portal keeps its permissive
localhost CORS (dev/pilot). Setting `KALOS_CORS_ORIGINS` switches to an explicit
allowlist - the production posture - so a browser page on some other origin can
no longer make credentialed calls to the engine.
"""
from __future__ import annotations

import ipaddress
import logging
import os

log = logging.getLogger("kalos.portal")

# Dev/pilot default: any loopback origin on any port (matches the pre-1c behavior).
_DEV_ORIGIN_REGEX = r"http://(localhost|127\.0\.0\.1)(:\d+)?"


def cors_origins() -> list[str]:
    """The explicit CORS allowlist from `KALOS_CORS_ORIGINS` (comma-separated
    absolute origins, e.g. `https://app.acme.com`), or an empty list when unset."""
    raw = os.environ.get("KALOS_CORS_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


def cors_config() -> dict:
    """The CORS middleware kwargs for `app.add_middleware(CORSMiddleware, ...)`.

    With `KALOS_CORS_ORIGINS` set, restrict to that explicit allowlist (the
    production posture). Otherwise fall back to the permissive localhost regex
    (dev/pilot), preserving today's behavior. Methods/headers stay open so the
    browser client and its `Authorization` header keep working; tightening those
    is a documented follow-up.
    """
    common = {"allow_methods": ["*"], "allow_headers": ["*"]}
    origins = cors_origins()
    if origins:
        return {"allow_origins": origins, **common}
    return {"allow_origin_regex": _DEV_ORIGIN_REGEX, **common}


_ALLOW_OPEN_VAR = "KALOS_ALLOW_OPEN_ACCESS"


class InsecureBindError(RuntimeError):
    """Refused to serve the portal openly on an interface reachable off-box."""


def open_access_explicitly_allowed() -> bool:
    """Whether an operator has deliberately opted in to serving with no auth.

    The escape hatch exists because there are legitimate reasons to run open on a
    trusted network - a lab bench box behind a firewall, a demo - and a guard with
    no override gets worked around in worse ways. It has to be a DELIBERATE act,
    which is why it is a named variable rather than a flag that a default config
    could carry in by accident.
    """
    return os.environ.get(_ALLOW_OPEN_VAR, "").strip().lower() in {"1", "true", "yes"}


def is_local_client(host: str | None) -> bool:
    """Whether a request's peer address is the same machine.

    Parsed with `ipaddress` rather than string-matched, so `127.0.0.7`,
    IPv4-mapped IPv6 and `::1` are all handled instead of only the two spellings
    someone thought to list.

    An address that does not parse as an IP is treated as LOCAL. That covers
    in-process transports - Starlette's `TestClient` reports the peer as
    `testclient`, and a Unix domain socket has no IP - neither of which is a
    remote caller. This is not a spoofing hole: `request.client.host` comes from
    the transport layer, not from a request header, so a remote TCP peer always
    presents a parseable address and cannot present `testclient`. (The one way to
    influence it is running behind a proxy with forwarded-header parsing enabled,
    which is an explicit deployment choice and is called out in docs/HARDENING.md.)
    """
    if not host:
        # No peer information at all: an in-process or socket transport.
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return True
    if addr.is_loopback:
        return True
    mapped = getattr(addr, "ipv4_mapped", None)
    return bool(mapped is not None and mapped.is_loopback)


def is_loopback_bind(host: str | None) -> bool:
    """Whether a BIND host keeps the portal on this machine.

    Distinct from `is_local_client`: a wildcard bind (`0.0.0.0`, `::`, or an empty
    host) accepts connections from anywhere and is emphatically not loopback, even
    though it is not itself a routable address. An unparseable host name is
    treated as NOT loopback - the safe answer for a bind, and the opposite of the
    safe answer for a peer address.
    """
    if host is None:
        return False
    h = host.strip()
    if h == "" or h in {"0.0.0.0", "::", "*"}:
        return False
    if h in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def assert_safe_bind(host: str | None, *, auth_configured: bool) -> None:
    """Refuse to start an unauthenticated portal on an interface reachable off-box.

    With no tokens provisioned the API grants an anonymous read+write principal on
    the `default` tenant, so binding it to a non-loopback interface publishes
    upload, analysis and campaign mutation to anyone who can route to the host.
    That posture was previously guarded only by a startup WARNING, which is the
    kind of thing nobody reads until afterwards.

    Loopback binds are untouched, so `python -m kalos.portal` and local
    development keep working exactly as before. Set `KALOS_AUTH_TOKENS[_FILE]` to
    fix it properly, or `KALOS_ALLOW_OPEN_ACCESS=1` to accept the risk knowingly.
    """
    if auth_configured or is_loopback_bind(host) or open_access_explicitly_allowed():
        return
    raise InsecureBindError(
        f"refusing to serve the portal on {host!r} with authentication disabled: "
        "with no tokens configured every caller gets an anonymous read+write "
        "principal, so this would publish uploads, analyses and campaign changes "
        "to anyone who can reach this host. Set KALOS_AUTH_TOKENS_FILE (or "
        "KALOS_AUTH_TOKENS) to enforce authentication, bind 127.0.0.1 to stay "
        f"local, or set {_ALLOW_OPEN_VAR}=1 to accept this deliberately. Note that "
        "KALOS_ANON_SALT should also be set before identity hashes leave this "
        "machine, and KALOS_CORS_ORIGINS before a browser on another origin can "
        "call the engine (docs/HARDENING.md)."
    )


def log_security_posture(*, auth_enforced: bool) -> None:
    """Log the effective security posture once at startup, so an operator can see
    at a glance whether the deployment is locked down or running open."""
    origins = cors_origins()
    cors = f"allowlist({len(origins)})" if origins else "dev-localhost (open)"
    log.info(
        "kalos portal security posture: auth=%s, cors=%s",
        "enforced" if auth_enforced else "OPEN (no tokens configured)",
        cors,
    )
    if not auth_enforced or not origins:
        log.warning(
            "kalos portal is not fully locked down: set KALOS_AUTH_TOKENS[_FILE] to "
            "enforce authentication and KALOS_CORS_ORIGINS to restrict CORS before "
            "exposing the portal beyond localhost (docs/HARDENING.md)."
        )
