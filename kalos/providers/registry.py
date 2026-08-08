"""The provider seam's lookup surface: list them, get one, serialize status.

No global mutable state and no import-time environment reads - `all_providers`
builds a fresh tuple of provider instances on every call, and each provider
reads its own env vars lazily inside `available()`/`status()`. This keeps the
registry trivially test-hackable: `monkeypatch.setenv` before a call is always
picked up, never shadowed by a cached instance or a cached status.
"""
from __future__ import annotations

from kalos.providers.anthropic_provider import AnthropicProvider
from kalos.providers.base import Provider
from kalos.providers.benchling_provider import BenchlingProvider
from kalos.providers.bionemo_provider import BioNemoProvider


def all_providers() -> tuple[Provider, ...]:
    """Every provider slot, in a fixed, stable order."""
    return (AnthropicProvider(), BioNemoProvider(), BenchlingProvider())


def provider_status() -> list[dict]:
    """JSON-safe, ordered status for every provider - the `/api/providers`
    payload. Each entry is a `dataclasses.asdict()` of that provider's
    `ProviderStatus`, so it carries only names and sentences, never a
    credential value."""
    from dataclasses import asdict

    return [asdict(provider.status()) for provider in all_providers()]


def get(name: str) -> Provider | None:
    """The provider registered under `name`, or `None` if there isn't one."""
    for provider in all_providers():
        if provider.name == name:
            return provider
    return None


__all__ = ["all_providers", "provider_status", "get"]
