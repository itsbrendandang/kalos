"""SLOT for a future Benchling LIMS integration - not a live integration.

Benchling is per-tenant (e.g. `acme.benchling.com`), so both an API key and a
tenant identifier are required. This module implements no HTTP calls, no
request builder, no retry logic - only a name and a credential check.
"""
from __future__ import annotations

import os

from kalos.providers.base import ProviderStatus

_API_KEY_ENV_VAR = "BENCHLING_API_KEY"
_TENANT_ENV_VAR = "BENCHLING_TENANT"

_CAPABILITY = (
    "Pull run metadata, registry entities, and ELN results directly from the "
    "LIMS - the honest source of the run-id / operator / instrument / date "
    "provenance the validation gate currently has to warn about. This is a "
    "SLOT for a future integration; no HTTP calls are implemented today."
)
_FALLBACK = (
    "Provenance columns (run id, operator, instrument, date) must be present "
    "in the uploaded sheet itself, or the validation gate warns that they "
    "are missing."
)


class BenchlingProvider:
    """Credential check for `BENCHLING_API_KEY` + `BENCHLING_TENANT`. Both are
    required - Benchling is per-tenant, so a key without a tenant is useless."""

    name = "benchling"

    def available(self) -> bool:
        return bool(os.environ.get(_API_KEY_ENV_VAR)) and bool(os.environ.get(_TENANT_ENV_VAR))

    def status(self) -> ProviderStatus:
        required = (_API_KEY_ENV_VAR, _TENANT_ENV_VAR)
        missing = tuple(name for name in required if not os.environ.get(name))
        return ProviderStatus(
            name=self.name,
            available=not missing,
            required_env=required,
            missing_env=missing,
            capability=_CAPABILITY,
            fallback=_FALLBACK,
        )


__all__ = ["BenchlingProvider"]
