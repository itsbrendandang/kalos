"""Runtime configuration for the LLM tier of `kalos.normalize`.

Import-light by design (stdlib only) - this module never imports `anthropic`
or `pydantic`, so `kalos.normalize` stays usable without the `normalize`
extra installed. The only thing read from the environment for the Anthropic
provider is whether an API key is *present* - the key's value is never read
into a variable that outlives `credentials_available`, never logged, and
never persisted anywhere in this package.

`KALOS_LLM_PROVIDER` selects which live provider `propose_plan` may call:
  - `"anthropic"` (default, unchanged behavior): live path gated on
    `ANTHROPIC_API_KEY` being present, exactly as before this env var existed.
  - `"ollama"`: a self-hosted, open-source model reached over HTTP (see
    `llm.py`'s `_live_plan_ollama`). No credential is required or read; the
    live path is always attempted (reachability is checked at CALL time via
    a health-check, never here), and any failure falls back to the offline
    plan exactly like an Anthropic API failure does.
  - `"typesafe"`: TypeSafe System One typed judgments (see
    `typesafe_tier.py`). Live path gated on `TYPESAFE_API_KEY` being present,
    read the same presence-only way as the Anthropic key; the value itself is
    read by the `typesafe-sdk` client from the environment, never here.
  - `"none"`: forces `enabled_live = False` unconditionally, regardless of
    `ANTHROPIC_API_KEY` - a hard "never call any LLM, deterministic only"
    switch.
An unrecognized value falls back to `"anthropic"` rather than raising, same
as a malformed `max_sample` falls back to its default.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

_MODEL_ENV_VAR = "KALOS_NORMALIZE_MODEL"
_MAX_SAMPLE_ENV_VAR = "KALOS_NORMALIZE_MAX_SAMPLE"
_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
_PROVIDER_ENV_VAR = "KALOS_LLM_PROVIDER"
_OLLAMA_URL_ENV_VAR = "KALOS_OLLAMA_URL"
_OLLAMA_MODEL_ENV_VAR = "KALOS_OLLAMA_MODEL"
_TYPESAFE_KEY_ENV_VAR = "TYPESAFE_API_KEY"
_TYPESAFE_MODEL_ENV_VAR = "KALOS_TYPESAFE_MODEL"
_TYPESAFE_MIN_CONFIDENCE_ENV_VAR = "KALOS_TYPESAFE_MIN_CONFIDENCE"

_DEFAULT_MODEL = "claude-sonnet-5"
_DEFAULT_MAX_SAMPLE = 5
_MIN_MAX_SAMPLE = 1
_MAX_MAX_SAMPLE = 20

Provider = Literal["anthropic", "ollama", "typesafe", "none"]
_PROVIDERS: tuple[Provider, ...] = ("anthropic", "ollama", "typesafe", "none")
_DEFAULT_PROVIDER: Provider = "anthropic"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"
# llama3.1 is a widely-available general-purpose Ollama model (a single
# `ollama pull llama3.1` away) reasonable at following a schema-constrained
# JSON instruction - a sensible default for self-hosters who have not pulled
# anything task-specific, not a claim that it is the best available model.
_DEFAULT_OLLAMA_MODEL = "llama3.1"
# `jev-latest` is the typesafe-sdk's own default model alias. Pinned here so a
# plan's `model` field records what was asked for, not an SDK-version default.
_DEFAULT_TYPESAFE_MODEL = "jev-latest"
# Below this Choice confidence a column's TypeSafe role is not acted on: the
# deterministic offline guess is kept instead and the note says so. A
# starting point to evaluate on real run sheets, not a calibrated constant.
_DEFAULT_TYPESAFE_MIN_CONFIDENCE = 0.6


@dataclass(frozen=True)
class NormalizeConfig:
    """Resolved configuration for one `propose_plan` call.

    `enabled_live` gates whether ANY live provider is attempted:
      - `provider == "none"`: always `False`.
      - `provider == "ollama"`: always `True` (no credential concept; a
        misconfigured/unreachable server is caught by the live call's own
        health-check + try/except, not here).
      - `provider == "anthropic"`: `credentials_available()` at load time,
        unchanged from before `provider` existed.
      - `provider == "typesafe"`: `typesafe_credentials_available()`.
    No API key is ever stored on this object - only the boolean presence
    check.
    """

    model: str
    max_sample: int
    enabled_live: bool
    provider: Provider = _DEFAULT_PROVIDER
    ollama_url: str = _DEFAULT_OLLAMA_URL
    ollama_model: str = _DEFAULT_OLLAMA_MODEL
    typesafe_model: str = _DEFAULT_TYPESAFE_MODEL
    typesafe_min_confidence: float = _DEFAULT_TYPESAFE_MIN_CONFIDENCE


def credentials_available() -> bool:
    """Return `True` iff `ANTHROPIC_API_KEY` is set to a non-empty string.

    The key's value is read once, checked for truthiness, and discarded -
    it is never logged, printed, or returned. This is the only place the
    env var is read; every other function in `kalos.normalize` should ask
    this function rather than reading the env var itself.
    """
    key = os.environ.get(_API_KEY_ENV_VAR)
    return bool(key)


def typesafe_credentials_available() -> bool:
    """Return `True` iff `TYPESAFE_API_KEY` is set to a non-empty string.

    Same contract as `credentials_available`: the value is checked for
    presence and discarded, never logged, printed, or returned.
    """
    return bool((os.environ.get(_TYPESAFE_KEY_ENV_VAR) or "").strip())


def load_config() -> NormalizeConfig:
    """Build a `NormalizeConfig` from the `KALOS_NORMALIZE_*`/`KALOS_LLM_*`/
    `KALOS_OLLAMA_*` environment.

    - `model`: `KALOS_NORMALIZE_MODEL`, default `"claude-sonnet-5"` (the
      Anthropic model id; unused by the `"ollama"`/`"none"` providers).
    - `max_sample`: `KALOS_NORMALIZE_MAX_SAMPLE`, default `5`, clamped to
      `[1, 20]`. A non-integer value falls back to the default rather than
      raising, since a malformed env var should degrade gracefully to the
      offline-safe default, not crash plan proposal.
    - `provider`: `KALOS_LLM_PROVIDER`, default `"anthropic"`. An
      unrecognized value also falls back to `"anthropic"` rather than
      raising, same as `max_sample`'s malformed-value handling.
    - `ollama_url` / `ollama_model`: `KALOS_OLLAMA_URL` (default
      `"http://localhost:11434"`) / `KALOS_OLLAMA_MODEL` (default
      `"llama3.1"`) - only consulted when `provider == "ollama"`.
    - `typesafe_model` / `typesafe_min_confidence`: `KALOS_TYPESAFE_MODEL`
      (default `"jev-latest"`) / `KALOS_TYPESAFE_MIN_CONFIDENCE` (default
      `0.6`, clamped to `[0, 1]`, malformed falls back to the default) - only
      consulted when `provider == "typesafe"`.
    - `enabled_live`: see `NormalizeConfig`'s docstring for the per-provider
      rule. The Anthropic key value itself is never read into this object.
    """
    model = os.environ.get(_MODEL_ENV_VAR, _DEFAULT_MODEL) or _DEFAULT_MODEL

    raw_max_sample = os.environ.get(_MAX_SAMPLE_ENV_VAR)
    try:
        max_sample = int(raw_max_sample) if raw_max_sample is not None else _DEFAULT_MAX_SAMPLE
    except ValueError:
        max_sample = _DEFAULT_MAX_SAMPLE
    max_sample = max(_MIN_MAX_SAMPLE, min(_MAX_MAX_SAMPLE, max_sample))

    raw_provider = (os.environ.get(_PROVIDER_ENV_VAR) or _DEFAULT_PROVIDER).strip().lower()
    provider: Provider = raw_provider if raw_provider in _PROVIDERS else _DEFAULT_PROVIDER  # type: ignore[assignment]

    ollama_url = os.environ.get(_OLLAMA_URL_ENV_VAR, _DEFAULT_OLLAMA_URL) or _DEFAULT_OLLAMA_URL
    ollama_model = os.environ.get(_OLLAMA_MODEL_ENV_VAR, _DEFAULT_OLLAMA_MODEL) or _DEFAULT_OLLAMA_MODEL

    typesafe_model = os.environ.get(_TYPESAFE_MODEL_ENV_VAR, _DEFAULT_TYPESAFE_MODEL) or _DEFAULT_TYPESAFE_MODEL
    raw_min_confidence = os.environ.get(_TYPESAFE_MIN_CONFIDENCE_ENV_VAR)
    try:
        typesafe_min_confidence = (
            float(raw_min_confidence) if raw_min_confidence else _DEFAULT_TYPESAFE_MIN_CONFIDENCE
        )
    except ValueError:
        typesafe_min_confidence = _DEFAULT_TYPESAFE_MIN_CONFIDENCE
    if math.isnan(typesafe_min_confidence):
        typesafe_min_confidence = _DEFAULT_TYPESAFE_MIN_CONFIDENCE
    typesafe_min_confidence = max(0.0, min(1.0, typesafe_min_confidence))

    if provider == "none":
        enabled_live = False
    elif provider == "ollama":
        enabled_live = True
    elif provider == "typesafe":
        enabled_live = typesafe_credentials_available()
    else:
        enabled_live = credentials_available()

    return NormalizeConfig(
        model=model,
        max_sample=max_sample,
        enabled_live=enabled_live,
        provider=provider,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        typesafe_model=typesafe_model,
        typesafe_min_confidence=typesafe_min_confidence,
    )


__all__ = [
    "NormalizeConfig",
    "Provider",
    "load_config",
    "credentials_available",
    "typesafe_credentials_available",
]
