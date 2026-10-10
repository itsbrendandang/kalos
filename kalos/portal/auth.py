"""Kalos portal - in-house authentication + authorization.

The portal is moving from a single-user local tool to a multi-tenant service
(see docs/HARDENING.md, Phase 1). This module is the identity foundation: a
bearer token resolves to a `Principal` (subject, tenant, scopes), and FastAPI
dependencies gate endpoints on scope.

Design (deliberately self-hosted, no external identity provider):

- Tokens are provisioned by the operator and stored as SHA-256 HASHES, never
  plaintext, never logged. At request time the presented token is hashed and
  constant-time compared (`hmac.compare_digest`) against the configured hashes,
  so neither the hash set nor a wrong guess leaks timing about the real token.
- Configuration is read AT REQUEST TIME from `KALOS_AUTH_TOKENS_FILE` (a JSON
  file, preferred) or `KALOS_AUTH_TOKENS` (inline JSON), so tokens can be
  rotated without a process restart and tests can override per-case.
- **Open mode**: when neither is configured the API stays open - every request
  gets an anonymous `Principal` on the `default` tenant with read+write scope
  (today's behavior) - and a warning is logged once at startup. `admin` is never
  granted in open mode. The moment tokens are provisioned, enforcement turns on;
  there is no half-authenticated state.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, Header, HTTPException

log = logging.getLogger("kalos.portal")

READ = "read"
WRITE = "write"
ADMIN = "admin"
_VALID_SCOPES = frozenset({READ, WRITE, ADMIN})

# The tenant/subject an unauthenticated request is attributed to in open mode.
_ANON_TENANT = "default"
_ANON_SUBJECT = "anonymous"
_OPEN_MODE_SCOPES = frozenset({READ, WRITE})  # never ADMIN without a real token


@dataclass(frozen=True)
class Principal:
    """The authenticated caller: who they are, which tenant's data they may
    touch, and what they may do. In open mode this is the anonymous principal."""

    subject: str
    tenant: str
    scopes: frozenset[str]
    anonymous: bool = False

    def has(self, scope: str) -> bool:
        return scope in self.scopes


@dataclass
class _TokenRecord:
    token_sha256: str
    subject: str
    tenant: str
    scopes: frozenset[str]


@dataclass
class Authenticator:
    """Resolves bearer tokens to `Principal`s from the current configuration.

    Config is re-read on every `principal_for` call so token rotation needs no
    restart; parsing a small JSON token list per request is negligible next to a
    GP fit, and it keeps the class stateless and test-friendly.
    """

    _warned_open: bool = field(default=False, init=False, repr=False)

    def _load_records(self) -> list[_TokenRecord]:
        raw = self._load_raw_config()
        if raw is None:
            return []
        records: list[_TokenRecord] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise AuthConfigError(f"token entry {i} is not an object")
            token_hash = entry.get("token_sha256")
            subject = entry.get("subject")
            tenant = entry.get("tenant")
            scopes_in = entry.get("scopes", [READ])
            if not isinstance(token_hash, str) or not token_hash:
                raise AuthConfigError(f"token entry {i} is missing 'token_sha256'")
            if not isinstance(subject, str) or not subject:
                raise AuthConfigError(f"token entry {i} is missing 'subject'")
            if not isinstance(tenant, str) or not tenant:
                raise AuthConfigError(f"token entry {i} is missing 'tenant'")
            if not isinstance(scopes_in, list) or not all(s in _VALID_SCOPES for s in scopes_in):
                raise AuthConfigError(
                    f"token entry {i} has invalid scopes; allowed: {sorted(_VALID_SCOPES)}"
                )
            records.append(
                _TokenRecord(
                    token_sha256=token_hash.lower(),
                    subject=subject,
                    tenant=tenant,
                    scopes=frozenset(scopes_in),
                )
            )
        return records

    @staticmethod
    def _load_raw_config() -> list[Any] | None:
        """The provisioned-principal list from file (preferred) or inline env,
        or None when auth is not configured (open mode). Read at request time."""
        path = os.environ.get("KALOS_AUTH_TOKENS_FILE", "").strip()
        if path:
            try:
                text = Path(path).expanduser().read_text()
            except OSError as exc:
                raise AuthConfigError(f"cannot read KALOS_AUTH_TOKENS_FILE: {exc}") from exc
        else:
            text = os.environ.get("KALOS_AUTH_TOKENS", "").strip()
            if not text:
                return None
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise AuthConfigError(f"auth token config is not valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise AuthConfigError("auth token config must be a JSON array")
        return data

    def is_configured(self) -> bool:
        """Whether an auth config is PRESENT. Not the same as whether it enforces.

        See `enforces` - a config that parses to an empty token list is present but
        grants everyone the anonymous principal. Prefer `enforces` for any security
        decision.
        """
        return self._load_raw_config() is not None

    def enforces(self) -> bool:
        """Whether authentication is ACTUALLY enforced: at least one usable token.

        `is_configured` is not a safe proxy for this, and the difference is a real
        misconfiguration rather than a hypothetical. With `KALOS_AUTH_TOKENS=[]` -
        which a config template rendering an empty array, or a tokens file
        containing `[]`, produces easily - `is_configured` returns True while
        `principal_for` falls back to the anonymous read+write principal, because
        it keys off whether any RECORD parsed. The deployment looks locked down and
        is wide open.

        Every security decision (the bind refusal, the remote-request guard, the
        startup posture line) keys off this method so the answer matches what
        `principal_for` will actually do.
        """
        return bool(self._load_records())

    def principal_for(self, authorization: str | None) -> Principal:
        """The `Principal` for a request's `Authorization` header.

        Open mode (no tokens configured): the anonymous principal. Configured:
        a valid `Bearer <token>` mapping to a provisioned record, else 401.
        """
        records = self._load_records()
        if not records:
            if not self._warned_open:
                log.warning(
                    "kalos auth: no tokens configured - API is OPEN (anonymous '%s' tenant, "
                    "read+write). Set KALOS_AUTH_TOKENS[_FILE] to enforce authentication.",
                    _ANON_TENANT,
                )
                self._warned_open = True
            return Principal(
                subject=_ANON_SUBJECT, tenant=_ANON_TENANT, scopes=_OPEN_MODE_SCOPES, anonymous=True
            )

        token = _bearer(authorization)
        if token is None:
            raise HTTPException(status_code=401, detail="authentication required")
        presented = hashlib.sha256(token.encode("utf-8")).hexdigest()
        for rec in records:
            # constant-time compare so a wrong token can't be recovered byte-by-byte
            if hmac.compare_digest(presented, rec.token_sha256):
                return Principal(subject=rec.subject, tenant=rec.tenant, scopes=rec.scopes)
        raise HTTPException(status_code=401, detail="invalid token")


class AuthConfigError(RuntimeError):
    """The auth token configuration is malformed. Surfaced as a 500 (an operator
    misconfiguration), never echoing token material."""


def _bearer(authorization: str | None) -> str | None:
    """Extract the token from an `Authorization: Bearer <token>` header, or None."""
    if not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    token = authorization[len(prefix):].strip()
    return token or None


# --- module singleton + FastAPI dependencies -------------------------------- #

_AUTH: Authenticator | None = None


def get_authenticator() -> Authenticator:
    """The module-level `Authenticator`. Tests override via
    `app.dependency_overrides[get_authenticator]`."""
    global _AUTH
    if _AUTH is None:
        _AUTH = Authenticator()
    return _AUTH


def require_principal(
    authorization: str | None = Header(default=None),
    auth: Authenticator = Depends(get_authenticator),
) -> Principal:
    """FastAPI dependency: the authenticated `Principal` for this request
    (anonymous in open mode, else resolved from the bearer token or 401)."""
    try:
        return auth.principal_for(authorization)
    except AuthConfigError:
        log.exception("kalos auth: token configuration is invalid")
        raise HTTPException(status_code=500, detail="authentication is misconfigured") from None


def require_scope(scope: str):
    """Build a dependency that requires the caller's principal to hold `scope`
    (403 otherwise). Use as `Depends(require_scope(WRITE))` on an endpoint."""
    if scope not in _VALID_SCOPES:
        raise ValueError(f"unknown scope {scope!r}")

    def _dep(principal: Principal = Depends(require_principal)) -> Principal:
        if not principal.has(scope):
            raise HTTPException(status_code=403, detail=f"requires '{scope}' scope")
        return principal

    return _dep
