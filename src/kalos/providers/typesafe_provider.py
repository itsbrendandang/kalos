"""Slot for the TypeSafe API key that gates `kalos.normalize`'s TypeSafe tier.

Like `AnthropicProvider`, this reports a credential the codebase already
reads (`kalos/normalize/config.py`, `typesafe_credentials_available()`),
importing `kalos.normalize` lazily. The `typesafe-sdk` package is never
imported here - it is the optional `.[typesafe]` extra.

The key alone does not switch the tier on: `KALOS_LLM_PROVIDER=typesafe`
selects it, so a key set for another purpose never changes which provider
normalizes a run sheet.
"""
from __future__ import annotations

from kalos.providers.base import ProviderStatus

_API_KEY_ENV_VAR = "TYPESAFE_API_KEY"

_CAPABILITY = (
    "Typed per-column decisions for run-sheet normalization (role, identity, "
    "canonical name) from TypeSafe's System One model, with probabilities "
    "that gate whether each decision is acted on; selected with "
    "KALOS_LLM_PROVIDER=typesafe, and usable to decide /api/run roles (roles=auto)."
)
_FALLBACK = (
    "The deterministic offline synonym+units mapper in kalos.normalize "
    "(offline_plan) still runs, and /api/run infers roles from the "
    "bioprocess profile's header patterns."
)


class TypesafeProvider:
    """Credential check for `TYPESAFE_API_KEY`, delegating to
    `kalos.normalize.typesafe_credentials_available`."""

    name = "typesafe"

    def available(self) -> bool:
        from kalos.normalize import typesafe_credentials_available

        return typesafe_credentials_available()

    def status(self) -> ProviderStatus:
        ok = self.available()
        return ProviderStatus(
            name=self.name,
            available=ok,
            required_env=(_API_KEY_ENV_VAR,),
            missing_env=() if ok else (_API_KEY_ENV_VAR,),
            capability=_CAPABILITY,
            fallback=_FALLBACK,
        )


__all__ = ["TypesafeProvider"]
