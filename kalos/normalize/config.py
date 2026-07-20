"""Runtime configuration for the LLM tier of `kalos.normalize`.

Import-light by design (stdlib only) - this module never imports `anthropic`
or `pydantic`, so `kalos.normalize` stays usable without the `normalize`
extra installed. The only thing read from the environment is whether an
Anthropic API key is *present* - the key's value is never read into a
variable that outlives `credentials_available`, never logged, and never
persisted anywhere in this package.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_MODEL_ENV_VAR = "KALOS_NORMALIZE_MODEL"
_MAX_SAMPLE_ENV_VAR = "KALOS_NORMALIZE_MAX_SAMPLE"
_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"

_DEFAULT_MODEL = "claude-sonnet-5"
_DEFAULT_MAX_SAMPLE = 5
_MIN_MAX_SAMPLE = 1
_MAX_MAX_SAMPLE = 20


@dataclass(frozen=True)
class NormalizeConfig:
    """Resolved configuration for one `propose_plan` call.

    `enabled_live` gates the live LLM path: it is `True` only when
    `credentials_available()` is `True` at the time `load_config` ran. The
    API key itself is never stored on this object - only the boolean
    presence check.
    """

    model: str
    max_sample: int
    enabled_live: bool


def credentials_available() -> bool:
    """Return `True` iff `ANTHROPIC_API_KEY` is set to a non-empty string.

    The key's value is read once, checked for truthiness, and discarded -
    it is never logged, printed, or returned. This is the only place the
    env var is read; every other function in `kalos.normalize` should ask
    this function rather than reading the env var itself.
    """
    key = os.environ.get(_API_KEY_ENV_VAR)
    return bool(key)


def load_config() -> NormalizeConfig:
    """Build a `NormalizeConfig` from the `KALOS_NORMALIZE_*` environment.

    - `model`: `KALOS_NORMALIZE_MODEL`, default `"claude-sonnet-5"`.
    - `max_sample`: `KALOS_NORMALIZE_MAX_SAMPLE`, default `5`, clamped to
      `[1, 20]`. A non-integer value falls back to the default rather than
      raising, since a malformed env var should degrade gracefully to the
      offline-safe default, not crash plan proposal.
    - `enabled_live`: `credentials_available()` - the live path activates
      only when an API key is present in the environment. The key value
      itself is never read into this object.
    """
    model = os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL) or _DEFAULT_MODEL

    raw_max_sample = os.environ.get(_MAX_SAMPLE_ENV_VAR)
    try:
        max_sample = int(raw_max_sample) if raw_max_sample is not None else _DEFAULT_MAX_SAMPLE
    except ValueError:
        max_sample = _DEFAULT_MAX_SAMPLE
    max_sample = max(_MIN_MAX_SAMPLE, min(_MAX_MAX_SAMPLE, max_sample))

    return NormalizeConfig(model=model, max_sample=max_sample, enabled_live=credentials_available())


__all__ = ["NormalizeConfig", "load_config", "credentials_available"]
