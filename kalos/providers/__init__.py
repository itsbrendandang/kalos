"""External-provider / API-key seam for the kalos engine.

Every provider here is a SLOT: a name, a lazy credential-presence check, and
an honest status describing what the key unlocks and what happens without
it. No provider requires a key for any existing behavior - normalization,
protein embeddings, and validation all already have a fully offline,
keyless path, and every provider's `fallback` sentence names it.

Import-light by design: this package never imports an optional third-party
SDK (`anthropic`, an NVIDIA client) at module scope, and never reads an
environment variable at import time. Env vars are read fresh, inside
`available()`/`status()`, on every call - see `kalos.providers.base`.
"""
from __future__ import annotations

from kalos.providers.anthropic_provider import AnthropicProvider
from kalos.providers.base import Provider, ProviderStatus
from kalos.providers.bionemo_provider import BioNemoProvider
from kalos.providers.registry import all_providers, get, provider_status

__all__ = [
    "Provider",
    "ProviderStatus",
    "AnthropicProvider",
    "BioNemoProvider",
    "all_providers",
    "provider_status",
    "get",
]
