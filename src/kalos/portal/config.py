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

from kalos.data.anonymizer import salt_configured

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


def assert_safe_exposure(host: str | None, *, auth_configured: bool) -> None:
    """Refuse to publish the portal off-box with a silently-broken security posture.

    Two things block startup on a non-loopback bind, and they were chosen on one
    principle: BLOCK THE SILENT FAILURES, WARN ON THE LOUD ONES.

      - Authentication not enforced. Silent: the portal serves happily and nothing
        tells the operator that every caller is getting anonymous read+write.
      - `KALOS_ANON_SALT` unset. Silent: pseudonyms are emitted successfully and
        look fine, while being derived from a published dev salt and therefore
        dictionary-attackable by anyone who reads this source. The whole purpose of
        the salt is that these hashes leave the machine.

    `KALOS_CORS_ORIGINS` deliberately does NOT block. It is worth being precise
    about why, because "permissive CORS" oversells it: the default is
    `allow_origin_regex` restricted to `http://localhost` and `http://127.0.0.1`,
    with `allow_credentials` off. It rejects `evil.com`, `localhost.evil.com` and
    even `https://localhost`, and with credentials disabled a third-party page
    cannot ride a user's token. So it is not a credential-theft vector. What it
    does do on a real deployment is block the legitimate browser client at, say,
    `https://app.acme.com` - a failure that announces itself in the first request
    anyone makes, which is exactly the kind that does not need a startup refusal.
    It gets a specific warning instead.

    Loopback binds are untouched, so `python -m kalos.portal` and local
    development keep working exactly as before.

    With no tokens provisioned the API grants an anonymous read+write principal on
    the `default` tenant, so binding it to a non-loopback interface publishes
    upload, analysis and campaign mutation to anyone who can route to the host.
    That posture was previously guarded only by a startup WARNING, which is the
    kind of thing nobody reads until afterwards.

    Loopback binds are untouched, so `python -m kalos.portal` and local
    development keep working exactly as before. Set `KALOS_AUTH_TOKENS[_FILE]` to
    fix it properly, or `KALOS_ALLOW_OPEN_ACCESS=1` to accept the risk knowingly.
    """
    if is_loopback_bind(host) or open_access_explicitly_allowed():
        return

    problems: list[str] = []
    if not auth_configured:
        problems.append(
            "authentication is not enforced, so every caller would get an anonymous "
            "read+write principal on the 'default' tenant - set "
            "KALOS_AUTH_TOKENS_FILE (or KALOS_AUTH_TOKENS) to a list with at least "
            "one token"
        )
    if not salt_configured():
        problems.append(
            "KALOS_ANON_SALT is unset, so identity pseudonyms would be derived from "
            "the published dev salt and are dictionary-attackable by anyone who "
            "reads this source - set it to a secret value"
        )
    if not problems:
        # Posture is sound for an exposed deployment; CORS is a separate, loud
        # failure and is warned about in `log_security_posture`.
        return

    joined = "; ".join(f"({i + 1}) {p}" for i, p in enumerate(problems))
    raise InsecureBindError(
        f"refusing to serve the portal on {host!r}: {joined}. Bind 127.0.0.1 to "
        f"stay local, fix the above, or set {_ALLOW_OPEN_VAR}=1 to accept this "
        "deliberately (docs/HARDENING.md)."
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
    if not auth_enforced:
        log.warning(
            "kalos portal auth is OPEN: every caller gets an anonymous read+write "
            "principal. Remote callers are refused while this is true; set "
            "KALOS_AUTH_TOKENS_FILE to serve them (docs/HARDENING.md)."
        )
    if not salt_configured():
        log.warning(
            "KALOS_ANON_SALT is unset, so identity pseudonyms use the published dev "
            "salt and are dictionary-attackable. Set it before anonymized output "
            "leaves this machine."
        )
    if not origins:
        # Stated precisely, because "permissive CORS" would overstate it: the
        # default admits only http://localhost and http://127.0.0.1 origins, with
        # credentials disabled, so it is not a credential-theft vector. The real
        # consequence is that a production browser client on its own origin is
        # rejected - which surfaces on the first request rather than silently.
        log.warning(
            "KALOS_CORS_ORIGINS is unset, so only http://localhost and "
            "http://127.0.0.1 origins may call this engine from a browser. A "
            "deployed frontend on any other origin will be rejected; set it to a "
            "comma-separated allowlist of the origins that should be able to."
        )
