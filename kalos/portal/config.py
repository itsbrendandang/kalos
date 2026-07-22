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
