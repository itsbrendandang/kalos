"""python -m kalos.portal  ->  serve the portal, loopback-only by default.

Host and port come from `KALOS_HOST` / `KALOS_PORT` and default to
`127.0.0.1:8050`. They were previously hardcoded, which made this entrypoint
accidentally safe: nobody could expose the portal without editing the source.
Making them configurable is what a real deployment needs, so the bind is now
checked instead of merely being impossible.

Serving unauthenticated on a non-loopback interface fails fast with an actionable
message rather than starting and hoping the operator reads a warning. See
`kalos.portal.config.assert_safe_exposure`. The peer-address guard in
`kalos.portal.app` covers the case where uvicorn is launched directly and this
check never runs at all.
"""
from __future__ import annotations

import os
import sys

import uvicorn

from kalos.portal.auth import get_authenticator
from kalos.portal.config import InsecureBindError, assert_safe_exposure

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8050


def main() -> None:
    host = os.environ.get("KALOS_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    raw_port = os.environ.get("KALOS_PORT", "").strip()
    try:
        port = int(raw_port) if raw_port else DEFAULT_PORT
    except ValueError:
        # A typo in a port must not silently fall back to a different one than the
        # operator asked for, since they would then look for the portal in the
        # wrong place.
        print(f"kalos portal: KALOS_PORT={raw_port!r} is not an integer", file=sys.stderr)
        raise SystemExit(2) from None

    try:
        assert_safe_exposure(host, auth_configured=get_authenticator().enforces())
    except InsecureBindError as exc:
        # Exit non-zero with the reason on stderr: this is a misconfiguration an
        # operator has to fix, not a crash, so it should read like a refusal.
        print(f"kalos portal: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    uvicorn.run("kalos.portal.app:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
