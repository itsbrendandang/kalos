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


def test_log_security_posture_warns_when_open(monkeypatch, caplog):
    monkeypatch.delenv("KALOS_CORS_ORIGINS", raising=False)
    with caplog.at_level(logging.WARNING, logger="kalos.portal"):
        log_security_posture(auth_enforced=False)
    assert any("not fully locked down" in r.message for r in caplog.records)


def test_log_security_posture_quiet_when_locked_down(monkeypatch, caplog):
    monkeypatch.setenv("KALOS_CORS_ORIGINS", "https://app.acme.com")
    with caplog.at_level(logging.WARNING, logger="kalos.portal"):
        log_security_posture(auth_enforced=True)
    assert not any("not fully locked down" in r.message for r in caplog.records)
