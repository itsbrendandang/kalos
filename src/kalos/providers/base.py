"""The provider contract: a name, a credential check, and an honest status.

This is a credential-presence seam, not a plugin framework. `Provider` is a
`typing.Protocol` rather than an ABC - there is no shared behavior worth
inheriting across three implementations, only a shape they each satisfy.

`available()` must read its env vars lazily, at call time, never at import
time or in `__init__`. A key set after process start (or unset in a test via
`monkeypatch`) must be picked up on the next call - caching the result would
make the seam unhackable in tests and stale in production.

`ProviderStatus` is serialized into an HTTP response (`GET /api/providers`),
so it carries env var NAMES only, never values. Nothing in this package ever
reads a credential's value - only whether it is present.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProviderStatus:
    """A provider's credential status, safe to serialize verbatim.

    `required_env` / `missing_env` are variable NAMES, never values.
    `capability` is one sentence on what this provider unlocks once its key
    is present. `fallback` is one sentence on exactly what happens today,
    without it.
    """

    name: str
    available: bool
    required_env: tuple[str, ...]
    missing_env: tuple[str, ...]
    capability: str
    fallback: str


class Provider(Protocol):
    """An external-provider slot: a name and a credential check.

    Implementations must be constructible and importable with no key present
    and no optional SDK installed - `available()` and `status()` only ever
    check `os.environ`, they never import or call the provider's own SDK.
    """

    name: str

    def available(self) -> bool:
        """Whether every credential this provider requires is present in the
        environment right now. Reads `os.environ` fresh on every call."""
        ...

    def status(self) -> ProviderStatus:
        """The current `ProviderStatus` for this provider, computed fresh."""
        ...
