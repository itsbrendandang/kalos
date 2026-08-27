"""Tier 3's LLM fallback: fill `BatchSummary` fields the deterministic tier
(`pdf.py`) could not recover, from the source PDF's raw text, using a
self-hosted Ollama model - last resort only, always schema-validated, always
degrading to "still unresolved" rather than a fabricated value on any
failure. See `docs/ingestion-architecture.md` (`itsbrendandang/kalos-transition`
repo) for the tier contract and `ports/pdf-extraction/PROVENANCE.md` for this
pattern's origin.

Reuses `kalos.normalize`'s existing provider seam rather than re-implementing
it:
  - `kalos.normalize.config.{NormalizeConfig,load_config}` resolve
    `KALOS_LLM_PROVIDER`/`KALOS_OLLAMA_URL`/`KALOS_OLLAMA_MODEL` exactly as
    the normalize LLM tier does - one provider configuration for the whole
    codebase, not a second copy of the same three environment variables.
  - `kalos.normalize.llm._check_ollama_server` is imported directly for the
    health-check: it is already generic (`base_url` in, `bool` out, "any
    failure means unreachable") and specific to neither module's schema, so
    re-implementing it here would only be a second copy of the exact same
    urllib probe.
  - This module's LLM tier only ever activates for `config.provider ==
    "ollama"`. `kalos.normalize`'s default provider is `"anthropic"`
    (gated on `ANTHROPIC_API_KEY`), but no Anthropic path exists here -
    `ports/pdf-extraction/llm_backend.py` (this pattern's origin) only ever
    implemented a local/Ollama call, and PDF extraction is exactly the kind
    of client-document-adjacent data this codebase prefers to keep off a
    hosted API by default. A caller who wants the LLM tier attempted must
    opt in with `KALOS_LLM_PROVIDER=ollama`; every other provider value
    (including the "anthropic" default) makes this tier a deterministic
    no-op, matching this package's fail-closed-informative design (see
    `pdf.py`'s import guard for the same posture applied to the missing
    optional dependency instead of the missing opt-in).

The one thing this module does NOT reuse verbatim is
`kalos.normalize.llm._ollama_generate`: that function hardcodes
`"format": "json"` (Ollama's generic "please emit some JSON" mode). The
upgrade this module implements - passing the actual target JSON SCHEMA in
the `format` field, which Ollama documents as constraining the model's
output to that schema rather than merely asking for JSON - needs a
`format` value `_ollama_generate` has no parameter for, so `_generate_json`
below is a sibling following the exact same pattern (health-check ->
`POST /api/generate` -> parse -> validate), not a modification of the
existing function. Verified against Ollama's own documented contract
(`docs/api.md`, "Structured Outputs" - `format` accepts a JSON Schema
object for both `/api/generate` and `/api/chat`, "The model will generate
a response that matches the schema.").
"""
from __future__ import annotations

import json
import logging
from typing import Any

from kalos.normalize.config import NormalizeConfig, load_config
from kalos.normalize.llm import _check_ollama_server

from .schema import Measurement

log = logging.getLogger("kalos.ingest.llm")

_OLLAMA_GENERATE_PATH = "/api/generate"
_OLLAMA_GENERATE_TIMEOUT = 60.0
_OLLAMA_TEMPERATURE = 0.0

_SYSTEM_PROMPT = (
    "You are extracting a fermentation batch summary from a scientific PDF "
    "report's raw text. The deterministic table parser could not recover "
    "the fields listed below; extract them from the surrounding text if "
    "present. For each field, respond with its numeric value and its unit "
    "exactly as reported in the text (do not convert units). If a field is "
    "genuinely not stated in the text, omit it from your response rather "
    "than guessing."
)


def _field_schema() -> dict[str, Any]:
    """The JSON Schema for one `BatchSummary` field's value: a number and the
    unit it was reported in. Shared by every field in `_build_schema` so the
    per-field shape cannot drift between fields."""
    return {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
            "unit": {"type": "string"},
        },
        "required": ["value", "unit"],
    }


def _build_schema(fields: list[str]) -> dict[str, Any]:
    """The JSON Schema of the extraction target, passed verbatim as Ollama's
    `format` field: an object with exactly `fields` as properties, each
    shaped by `_field_schema`, all optional (a field genuinely absent from
    the text is a valid response, not a schema violation - the model is
    instructed not to guess, and a `required` field would pressure it to)."""
    return {
        "type": "object",
        "properties": {name: _field_schema() for name in fields},
    }


def _generate_json(base_url: str, model: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    """POST `prompt` to Ollama's `/api/generate` with `schema` in the
    `format` field, returning the parsed JSON object or `None` on any
    failure (connection error, non-200, unparsable/non-dict response body).

    Mirrors `kalos.normalize.llm._ollama_generate`'s request shape (`stream:
    false`, low temperature, `urllib.request` only - no new hard
    dependency) with `format` carrying the schema object instead of the
    string `"json"` - see module docstring for why this is a sibling
    function rather than a call to `_ollama_generate` itself.
    """
    import urllib.request

    url = base_url.rstrip("/") + _OLLAMA_GENERATE_PATH
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {"temperature": _OLLAMA_TEMPERATURE},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=_OLLAMA_GENERATE_TIMEOUT) as response:  # noqa: S310 - explicit configured URL
            if response.status != 200:
                return None
            raw = response.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - any failure means "no response"; caller falls back
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    response_text = data.get("response")
    if not isinstance(response_text, str):
        return None
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _validate_fields(raw: dict[str, Any], requested: list[str]) -> dict[str, Measurement]:
    """Keep only the entries of `raw` that are both in `requested` and
    schema-shaped (`{"value": <number>, "unit": <string>}`) - the same
    "the model proposes, this validation decides what counts as well-formed"
    discipline `kalos.normalize.llm._llm_plan_model` applies to the
    normalize LLM tier, done by hand here rather than via `pydantic` so this
    package adds no dependency beyond the optional `pdf` extra. A field that
    is present but malformed (wrong types, missing key) is dropped, not
    coerced - it stays unresolved rather than accepting a guess.
    """
    out: dict[str, Measurement] = {}
    for name in requested:
        entry = raw.get(name)
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        unit = entry.get("unit")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not isinstance(unit, str) or not unit:
            continue
        out[name] = Measurement(value=float(value), unit=unit, method="llm_ollama")
    return out


def fill_unresolved_fields(
    text: str,
    unresolved_fields: list[str],
    *,
    config: NormalizeConfig | None = None,
) -> dict[str, Measurement]:
    """Attempt to fill `unresolved_fields` from `text` via the configured
    Ollama model, schema-constrained to exactly those fields.

    Returns a `dict` of only the fields it could confidently fill (possibly
    empty, possibly a subset of `unresolved_fields`) - NEVER raises. Every
    failure mode (provider not `"ollama"`, server unreachable, non-200,
    unparsable JSON, a response failing `_validate_fields`, or any other
    exception) returns `{}`, so `api.extract_run_document`'s merge step can
    treat this function's result uniformly: whatever it does not return
    stays unresolved and the deterministic tier's result stands, matching
    the fallback discipline `kalos.normalize.llm.propose_plan` established
    for the normalize LLM tier.
    """
    if not unresolved_fields:
        return {}
    config = config or load_config()
    if config.provider != "ollama":
        return {}
    try:
        if not _check_ollama_server(config.ollama_url):
            return {}
        schema = _build_schema(unresolved_fields)
        prompt = (
            f"{_SYSTEM_PROMPT}\n\nFields to extract: {', '.join(unresolved_fields)}\n\n"
            f"Text:\n{text}"
        )
        raw = _generate_json(config.ollama_url, config.ollama_model, prompt, schema)
        if raw is None:
            return {}
        return _validate_fields(raw, unresolved_fields)
    except Exception as err:  # noqa: BLE001 - must never hard-fail the caller
        log.warning(
            "ingest.llm: fill_unresolved_fields failed (%s); leaving fields unresolved",
            type(err).__name__,
        )
        return {}


__all__ = ["fill_unresolved_fields"]
