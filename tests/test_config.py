"""Deployment security config (kalos/portal/config.py, docs/HARDENING.md 1c):
CORS is a permissive localhost default until `KALOS_CORS_ORIGINS` restricts it to
an explicit allowlist, and the security posture is logged at startup."""
from __future__ import annotations

import logging

import pytest

from kalos.portal.config import cors_config, cors_origins, log_security_posture


@pytest.mark.parametrize("raw,expected", [
    ("", []),
    ("https://app.acme.com", ["https://app.acme.com"]),
    ("https://a.com, https://b.com", ["https://a.com", "https://b.com"]),
    ("  https://a.com ,, https://b.com  ,", ["https://a.com", "https://b.com"]),  # strips + drops empties
])
def test_cors_origins_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("KALOS_CORS_ORIGINS", raw)
    assert cors_origins() == expected


def test_cors_origins_unset_is_empty(monkeypatch):
    monkeypatch.delenv("KALOS_CORS_ORIGINS", raising=False)
    assert cors_origins() == []


def test_dev_default_is_permissive_localhost_regex(monkeypatch):
    monkeypatch.delenv("KALOS_CORS_ORIGINS", raising=False)
    cfg = cors_config()
    assert "allow_origin_regex" in cfg
    assert "allow_origins" not in cfg
    assert "localhost" in cfg["allow_origin_regex"]


def test_configured_allowlist_replaces_the_regex(monkeypatch):
    monkeypatch.setenv("KALOS_CORS_ORIGINS", "https://app.acme.com,https://portal.acme.com")
    cfg = cors_config()
    assert cfg["allow_origins"] == ["https://app.acme.com", "https://portal.acme.com"]
    assert "allow_origin_regex" not in cfg  # the wide-open regex is gone in production posture


def test_log_security_posture_warns_specifically_per_problem(monkeypatch, caplog):
    """One warning per actual problem, each naming its variable.

    This replaced a single "not fully locked down" catch-all. A generic warning
    tells an operator that something is wrong without saying which of three
    variables to set or what each one costs them, so it is easy to read and
    dismiss. Each message now names the variable and the concrete consequence.
    """
    monkeypatch.delenv("KALOS_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("KALOS_ANON_SALT", raising=False)
    with caplog.at_level(logging.WARNING, logger="kalos.portal"):
        log_security_posture(auth_enforced=False)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "KALOS_AUTH_TOKENS_FILE" in text
    assert "KALOS_ANON_SALT" in text
    assert "KALOS_CORS_ORIGINS" in text
    # The auth warning must say what open mode actually grants, and that remote
    # callers are already being refused rather than silently served.
    assert "anonymous read+write" in text
    assert "refused" in text


def test_log_security_posture_quiet_when_locked_down(monkeypatch, caplog):
    monkeypatch.setenv("KALOS_CORS_ORIGINS", "https://app.acme.com")
    monkeypatch.setenv("KALOS_ANON_SALT", "a-real-secret")
    with caplog.at_level(logging.WARNING, logger="kalos.portal"):
        log_security_posture(auth_enforced=True)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_cors_warning_does_not_overstate_the_default(monkeypatch, caplog):
    """The default is origin-restricted to loopback with credentials disabled, so
    it is not a credential-theft vector. The warning must describe the real
    consequence - a deployed frontend being rejected - and not imply exposure."""
    monkeypatch.delenv("KALOS_CORS_ORIGINS", raising=False)
    monkeypatch.setenv("KALOS_ANON_SALT", "a-real-secret")
    with caplog.at_level(logging.WARNING, logger="kalos.portal"):
        log_security_posture(auth_enforced=True)
    cors = [r.getMessage() for r in caplog.records if "KALOS_CORS_ORIGINS" in r.getMessage()]
    assert len(cors) == 1
    assert "will be rejected" in cors[0]
