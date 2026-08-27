"""LLM tier: propose a `NormalizationPlan` from an identity-screened payload,
with a fully deterministic offline fallback and a choice of live PROVIDER.

`anthropic` and `pydantic` are imported lazily, INSIDE the live-path
functions only - this module (and `kalos.normalize` as a whole) must stay
importable, and `offline_plan`/`propose_plan` must stay fully USABLE, with
neither package installed. `urllib` (the Ollama provider's transport) is
stdlib, so it adds no new hard dependency. The live path never runs unless
`config.enabled_live` says so (`NormalizeConfig`'s docstring in `config.py`
has the exact per-provider rule), and on ANY failure it falls back to the
offline path rather than raising - this feature must never hard-fail because
an LLM is unavailable, unreachable, rate-limited, or misbehaving.

Two providers share the exact same request/validate contract:
  - `"anthropic"` (`_live_plan_anthropic`): a hosted call via the `anthropic`
    SDK, with the SDK's own structured-output support where available.
  - `"ollama"` (`_live_plan_ollama`): a self-hosted, open-source model
    reached over plain HTTP - health-check -> `POST /api/generate` ->
    regex-extract the JSON object -> validate. Ported from
    `ports/pdf-extraction/llm_backend.py`'s `check_ollama_server` /
    `extract_flask_table_with_deepseek` pattern in the kalos-transition
    repo (see that port's PROVENANCE.md); adapted here from a PDF
    table-extraction prompt to this module's column-normalization schema,
    and from `/api/health` (not a real Ollama route) to the real `/api/tags`
    endpoint for the health-check.
Both providers validate the model's response against the SAME
`LLMPlan`/`LLMColumnPlan` pydantic schema (`_llm_plan_model`) - the model
proposes, this schema (not the model, not the provider) decides what counts
as a well-formed column proposal, and a malformed or off-schema response
from EITHER provider fails exactly the same way an Anthropic API error does
today: caught by `propose_plan`, falls back to `offline_plan`.

Nothing in this module logs, prints, or persists the Anthropic API key
itself; it is read only by the `anthropic` SDK's own `Anthropic()` client
constructor from the environment, never handled as a string here. The
Ollama provider sends the same identity-screened payload the Anthropic
provider does (never raw/unscreened data) to a URL that is explicit
configuration (`KALOS_OLLAMA_URL`, default `localhost` only), never a
value observed from data.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

import pandas as pd

from kalos.data.anonymizer import Anonymizer

from .config import NormalizeConfig, load_config
from .payload import build_payload
from .plan import ColumnPlan, NormalizationPlan
from .synonyms import SYNONYMS, guess_role, snake_canonical
from .units import canonical_suffix, parse_value

log = logging.getLogger("kalos.normalize.llm")


def _canonical_base_name(raw_name: str) -> str:
    """`snake_canonical(raw_name)`, upgraded to the `SYNONYMS` canonical key
    when the snake-cased header matches a known alias (e.g. "Temp" ->
    "temp" -> "temperature", via the `"temperature"` entry in `SYNONYMS`)."""
    snake = snake_canonical(raw_name)
    for canonical, aliases in SYNONYMS.items():
        if snake in aliases:
            return canonical
    return snake


def _canonical_name_with_unit(raw_name: str, unit_token: str | None, to_base: bool) -> str:
    """`_canonical_base_name` plus the unit's canonical suffix when the
    column is being converted to its base unit (e.g. "temperature" + "_c"
    -> "temperature_c"). No suffix is added for unit-less/unconverted
    columns, and a base name that already ends with the suffix is not
    doubled up."""
    base_name = _canonical_base_name(raw_name)
    if not to_base:
        return base_name
    suffix = canonical_suffix(unit_token)
    if not suffix or base_name.endswith(suffix):
        return base_name
    return f"{base_name}{suffix}"

_SYSTEM_PROMPT = """You are a bioprocess data engineer normalizing a client run sheet.

You will be given ONLY column headers plus screened statistics and sample values \
(free-text columns are redacted). Raw identity columns have already been removed \
before you see this payload.

For each column in the payload, propose:
  - raw_name: the header exactly as given.
  - canonical_name: a snake_case name for the column, or null if the column \
should be dropped (identity or free-text).
  - role: one of "target", "feature", "group", "identity", "freetext", "metadata".
  - unit_token: the unit the values carry (e.g. "C", "g/L", "mL/h"), or null if \
the column is unit-less or you are not confident of the unit.
  - is_identity: true if this column, despite surviving the deterministic \
identity pre-screen, still looks like client-identifying information (a name, \
an operator, a sample label, contact info). Flag conservatively - only when \
you are confident.
  - note: a short (<20 word) rationale.

Return one entry per column in the payload, in the same order."""

# Appended to `_SYSTEM_PROMPT` for the Ollama provider only: local models do
# not reliably support a hosted API's structured-output/JSON mode, so the
# exact response shape has to be spelled out in the prompt text itself (the
# Anthropic path gets this for free from `output_format`/`output_config`).
_OLLAMA_JSON_INSTRUCTION = """Return ONLY a single JSON object of the exact shape:
{"columns": [{"raw_name": "...", "canonical_name": "..." or null, "role": "...", \
"unit_token": "..." or null, "is_identity": true or false, "note": "..."}, ...]}
One entry per input column, in the same order as the payload. No markdown \
code fences, no prose before or after the JSON object."""


def _llm_plan_model() -> Any:
    """Lazily import `pydantic` and build the `LLMPlan` response schema.

    Defined ONCE here (not duplicated per provider) so every live provider
    validates the model's proposal against the exact same schema - a
    provider-specific schema drift would mean the Anthropic and Ollama paths
    could silently disagree about what counts as a well-formed column
    proposal. Returns `Any` (not a precise pydantic type) because the class
    is built dynamically inside this function, matching this module's
    existing "not imported at module scope" contract for `pydantic`.
    """
    from pydantic import BaseModel

    class LLMColumnPlan(BaseModel):
        raw_name: str
        canonical_name: str | None
        role: str
        unit_token: str | None = None
        is_identity: bool = False
        note: str = ""

    class LLMPlan(BaseModel):
        columns: list[LLMColumnPlan]

    return LLMPlan


def _columns_from_llm_items(items: Iterable[Any]) -> list[ColumnPlan]:
    """Turn validated `LLMColumnPlan` items (from EITHER provider) into
    `ColumnPlan`s, applying the same is_identity-override rule to both: a
    column the model flags `is_identity=True` is always forced to
    `role="identity"`, `canonical_name=None`, regardless of what the model
    separately said for `role`/`canonical_name` - the model proposes, this
    rule (not the model, not the provider) has final say on identity.
    """
    columns: list[ColumnPlan] = []
    for item in items:
        is_identity = bool(item.is_identity)
        canonical_name = None if is_identity else item.canonical_name
        role = "identity" if is_identity else item.role
        columns.append(
            ColumnPlan(
                raw_name=item.raw_name,
                canonical_name=canonical_name,
                role=role,  # type: ignore[arg-type]
                unit_token=item.unit_token,
                to_base=item.unit_token is not None and not is_identity,
                is_identity=is_identity,
                redacted=(role == "freetext"),
                note=item.note,
            )
        )
    return columns


def offline_plan(
    df: pd.DataFrame,
    *,
    anonymizer: Anonymizer | None = None,
    max_sample: int = 5,
) -> NormalizationPlan:
    """Build a `NormalizationPlan` with zero network access.

    Reuses Phase 1 exactly:
      - identity pre-screen via `payload.build_payload` (same rules as the
        live path uses to screen its payload, so offline and live agree on
        which columns are identity before either one runs).
      - `synonyms.snake_canonical` (upgraded to a `SYNONYMS` canonical key
        when the header matches a known alias) for canonical names.
      - `synonyms.guess_role` for roles - this is the single source of truth
        for identity/group/target/feature/freetext/metadata, so a column
        that survives the header pre-screen but that `guess_role` still
        flags as identity or freetext (by content) is dropped the same way
        a pre-screened column is.
      - `units.parse_value` to detect a per-column unit token for columns
        `payload.build_payload` classified `"numeric+unit"`, both to set
        `unit_token`/`to_base` and to give `guess_role` a unit-aware numeric
        signal (a raw `pd.to_numeric` parse rate would call a column like
        "34.6 C" non-numeric and misclassify it as free-text).

    Dropped identity columns (pre-screened, or flagged by `guess_role`)
    become `ColumnPlan`s with `canonical_name=None`, `role="identity"`,
    `is_identity=True`. Columns `guess_role` calls `"freetext"` become
    `ColumnPlan`s with `canonical_name=None`, `role="freetext"`,
    `redacted=True`. Every other column gets a canonical name, a guessed
    role, and - if it looks unit-bearing - a `unit_token`/`to_base=True`.

    `created_by="offline"`, `model=None`. Always passes `plan.validate()`.
    """
    payload, dropped_identity = build_payload(df, anonymizer=anonymizer, max_sample=max_sample)
    dropped_identity_set = set(dropped_identity)
    dtype_by_header = {entry["header"]: entry["dtype"] for entry in payload["columns"]}

    columns: list[ColumnPlan] = []
    for raw_name in df.columns:
        raw_name = str(raw_name)

        if raw_name in dropped_identity_set:
            columns.append(
                ColumnPlan(
                    raw_name=raw_name,
                    canonical_name=None,
                    role="identity",
                    is_identity=True,
                    note="dropped by deterministic identity pre-screen",
                )
            )
            continue

        dtype = dtype_by_header.get(raw_name)
        series = df[raw_name]
        non_null = series.dropna()
        unit_token: str | None = None
        to_base = False

        if dtype == "numeric+unit":
            # Unit-aware parse rate: these cells fail a plain `pd.to_numeric`
            # (the unit suffix makes them non-numeric strings) but DO parse
            # via `units.parse_value` once the unit is stripped off - use
            # that as the numeric signal `guess_role` sees, or a unit-bearing
            # column like "Temp" would be misclassified as free-text.
            n_parsed = 0
            for raw in non_null:
                value, token = parse_value(raw)
                if value is not None:
                    n_parsed += 1
                    if unit_token is None:
                        unit_token = token
            parse_rate = n_parsed / len(non_null) if len(non_null) else 0.0
            to_base = unit_token is not None
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            parse_rate = float(numeric.notna().sum()) / len(non_null) if len(non_null) else 0.0

        is_numeric = parse_rate >= 0.8
        role = guess_role(raw_name, is_numeric=is_numeric, parse_rate=parse_rate)

        if role in ("identity", "freetext"):
            # guess_role can independently flag identity/freetext even for a
            # column that survived the header pre-screen (e.g. by content
            # heuristics) - keep the contract that those roles never carry a
            # canonical_name (ColumnPlan/NormalizationPlan.validate rule).
            columns.append(
                ColumnPlan(
                    raw_name=raw_name,
                    canonical_name=None,
                    role=role,
                    is_identity=(role == "identity"),
                    redacted=(role == "freetext"),
                    note="flagged by offline role guess",
                )
            )
            continue

        columns.append(
            ColumnPlan(
                raw_name=raw_name,
                canonical_name=_canonical_name_with_unit(raw_name, unit_token, to_base),
                role=role,
                unit_token=unit_token,
                to_base=to_base,
                note="offline heuristic plan",
            )
        )

    plan = NormalizationPlan(columns=columns, created_by="offline", model=None)
    plan.validate()
    return plan


def _merge_dropped_identity(
    llm_columns: list[ColumnPlan], dropped_identity: list[str], all_raw_names: list[str]
) -> list[ColumnPlan]:
    """Combine the model's proposed columns with the deterministically
    pre-screened identity columns (which never reached the model), producing
    one `ColumnPlan` per raw column in `all_raw_names`'s order."""
    by_raw_name = {c.raw_name: c for c in llm_columns}
    merged: list[ColumnPlan] = []
    for raw_name in all_raw_names:
        if raw_name in dropped_identity:
            merged.append(
                ColumnPlan(
                    raw_name=raw_name,
                    canonical_name=None,
                    role="identity",
                    is_identity=True,
                    note="dropped by deterministic identity pre-screen",
                )
            )
            continue
        col = by_raw_name.get(raw_name)
        if col is not None:
            merged.append(col)
    return merged


def _live_plan_anthropic(
    df: pd.DataFrame,
    config: NormalizeConfig,
    payload: dict,
    dropped_identity: list[str],
    anonymizer: Anonymizer | None,
) -> NormalizationPlan:
    """The live LLM call, via the hosted Anthropic API. Lazily imports
    `anthropic` and `pydantic` - never imported at module scope so the
    package stays import-light and this function is the only place either
    dependency is required for this provider."""
    import anthropic

    LLMPlan = _llm_plan_model()

    client = anthropic.Anthropic()
    payload_json = json.dumps(payload, sort_keys=True)

    # Annotated `dict[str, Any]`: mypy cannot match a `**` splat of a
    # heterogeneously-valued dict against the SDK's overloads, so an inferred
    # `dict[str, object]` fails every `create`/`parse` variant at the call site.
    kwargs: dict[str, Any] = dict(
        model=config.model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": payload_json}],
        thinking={"type": "disabled"},
    )

    parsed: Any
    if hasattr(client.messages, "parse"):
        response = client.messages.parse(output_format=LLMPlan, **kwargs)
        # `parsed_output` is None when the model returned no parseable object.
        # Raise rather than propagate a None into the plan walk below: the
        # caller treats any exception here as "LLM tier unavailable" and falls
        # back to the deterministic offline plan, which is the honest outcome.
        if response.parsed_output is None:
            raise ValueError("LLM returned no parseable normalization plan")
        parsed = response.parsed_output
    else:
        schema = LLMPlan.model_json_schema()
        response = client.messages.create(
            output_config={"format": {"type": "json_schema", "schema": schema}},
            **kwargs,
        )
        text = next(block.text for block in response.content if block.type == "text")
        parsed = LLMPlan.model_validate_json(text)

    llm_columns = _columns_from_llm_items(parsed.columns)
    all_raw_names = [str(c) for c in df.columns]
    merged = _merge_dropped_identity(llm_columns, dropped_identity, all_raw_names)
    plan = NormalizationPlan(columns=merged, created_by="llm", model=config.model)
    plan.validate()
    return plan


# --- Ollama provider: self-hosted, open-source, plain-HTTP -------------------- #
#
# Ported call shape from ports/pdf-extraction/llm_backend.py (PROVENANCE.md:
# source repo itsbrendandang/Templatinizer @
# f22e527891febf140b563c412bee7cbe94326c58): health-check the server first,
# POST /api/generate with a low temperature, regex-extract the JSON out of the
# raw text response. `urllib` (stdlib) is used instead of the port's
# `requests` - this module must add no new hard dependency.

_OLLAMA_GENERATE_PATH = "/api/generate"
# The port's health-check hit `/api/health`, which is not a route real Ollama
# servers expose. `/api/tags` (list locally-pulled models) is a genuine,
# documented Ollama endpoint that serves the same "is the server up" purpose
# and is used here instead - a deliberate correction, not a literal port.
_OLLAMA_HEALTH_PATH = "/api/tags"
_OLLAMA_HEALTH_TIMEOUT = 5.0
_OLLAMA_GENERATE_TIMEOUT = 60.0
# Near-deterministic decoding, matching the port's low-temperature setting.
_OLLAMA_TEMPERATURE = 0.0
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _check_ollama_server(base_url: str, timeout: float = _OLLAMA_HEALTH_TIMEOUT) -> bool:
    """Health-check a local/self-hosted Ollama server before sending any
    payload. Returns `True` only on a 200 response, `False` on ANY failure
    (connection refused, DNS, timeout, non-200) - callers must treat `False`
    as "fall back to the deterministic plan", never as a hard error."""
    import urllib.request

    url = base_url.rstrip("/") + _OLLAMA_HEALTH_PATH
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - explicit, non-data URL
            return bool(response.status == 200)
    except Exception:  # noqa: BLE001 - any failure means "unreachable"
        return False


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Regex-extract the first `{...}` JSON object from a raw local-model
    text response. Mirrors `llm_backend.py`'s `_extract_json_array`, adapted
    for an object shape (`{"columns": [...]}`) instead of a bare array -
    local models wrap JSON in prose/markdown fences more often than a hosted
    API's structured-output mode, so the widest brace-delimited match is
    parsed rather than trusting the whole response to be pure JSON."""
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _ollama_generate(base_url: str, model: str, prompt: str, timeout: float = _OLLAMA_GENERATE_TIMEOUT) -> str | None:
    """POST `prompt` to Ollama's `/api/generate`, returning the model's raw
    text response or `None` on any failure (connection error, non-200,
    unparsable response body). `format: "json"` asks the server to
    constrain the model to JSON output where the model/server supports it;
    the response is still regex-extracted and schema-validated downstream
    regardless, since that support is not universal across local models."""
    import urllib.request

    url = base_url.rstrip("/") + _OLLAMA_GENERATE_PATH
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": _OLLAMA_TEMPERATURE},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit configured URL
            if response.status != 200:
                return None
            raw = response.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - any failure means "no response"
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    text = data.get("response")
    return text if isinstance(text, str) else None


def _live_plan_ollama(
    df: pd.DataFrame,
    config: NormalizeConfig,
    payload: dict,
    dropped_identity: list[str],
) -> NormalizationPlan:
    """The live LLM call, via a self-hosted Ollama server.

    Health-check -> `POST /api/generate` -> regex-extract JSON -> validate
    against the EXACT SAME `LLMPlan` schema `_live_plan_anthropic` uses (see
    `_llm_plan_model`), so a malformed or off-schema local-model response
    fails exactly like a malformed Anthropic response would.

    Raises on any failure (unreachable server, bad status, unparsable or
    schema-invalid JSON) rather than returning `None` - `propose_plan` is
    the one place that catches and falls back to `offline_plan`, matching
    the Anthropic provider's contract exactly: this function's job is only
    to produce a valid plan or raise, never to decide the fallback itself.
    """
    if not _check_ollama_server(config.ollama_url):
        raise ConnectionError(f"ollama server unreachable at {config.ollama_url!r}")

    payload_json = json.dumps(payload, sort_keys=True)
    prompt = f"{_SYSTEM_PROMPT}\n\n{_OLLAMA_JSON_INSTRUCTION}\n\nInput payload:\n{payload_json}"

    text = _ollama_generate(config.ollama_url, config.ollama_model, prompt)
    if text is None:
        raise ConnectionError("ollama /api/generate request failed")

    raw_obj = _extract_json_object(text)
    if raw_obj is None:
        raise ValueError("could not extract a JSON object from the ollama response")

    LLMPlan = _llm_plan_model()
    parsed = LLMPlan.model_validate(raw_obj)  # raises pydantic.ValidationError on schema mismatch

    llm_columns = _columns_from_llm_items(parsed.columns)
    all_raw_names = [str(c) for c in df.columns]
    merged = _merge_dropped_identity(llm_columns, dropped_identity, all_raw_names)
    plan = NormalizationPlan(columns=merged, created_by="llm", model=config.ollama_model)
    plan.validate()
    return plan


def propose_plan(
    df: pd.DataFrame,
    *,
    config: NormalizeConfig | None = None,
    anonymizer: Anonymizer | None = None,
) -> NormalizationPlan:
    """Propose a `NormalizationPlan` for `df`, using the configured LLM
    provider when it is enabled and falling back to the fully offline,
    deterministic path otherwise (or on any live-call failure).

    Steps:
      1. Resolve `config` (`config.load_config()` if not given) -
         `config.provider` selects `"anthropic"` (default), `"ollama"`, or
         `"none"`; see `NormalizeConfig`'s docstring for the per-provider
         `enabled_live` rule.
      2. Build the identity-screened payload via `payload.build_payload`,
         using `config.max_sample`. This happens regardless of provider -
         it is a local, network-free screening step.
      3. If `not config.enabled_live` (`provider == "none"`, or
         `provider == "anthropic"` with no API key): return
         `offline_plan(df, ...)`.
      4. Otherwise call the configured provider (`_live_plan_anthropic` or
         `_live_plan_ollama`; lazy `anthropic`/`pydantic` import in both).
         On success, merge the model's per-column proposals with the
         deterministically pre-screened identity columns and return a plan
         with `created_by="llm"`. On ANY failure (an Anthropic SDK error, an
         unreachable Ollama server, a malformed/off-schema JSON response, or
         any other exception), log a warning containing no raw data and no
         key/URL payload, and fall back to `offline_plan(df, ...)`.
    """
    config = config or load_config()
    payload, dropped_identity = build_payload(df, anonymizer=anonymizer, max_sample=config.max_sample)

    if not config.enabled_live:
        return offline_plan(df, anonymizer=anonymizer, max_sample=config.max_sample)

    try:
        if config.provider == "ollama":
            return _live_plan_ollama(df, config, payload, dropped_identity)
        return _live_plan_anthropic(df, config, payload, dropped_identity, anonymizer)
    except Exception as err:  # noqa: BLE001 - must never hard-fail the caller
        log.warning(
            "normalize: live plan failed (%s, provider=%s); falling back to offline",
            type(err).__name__,
            config.provider,
        )
        return offline_plan(df, anonymizer=anonymizer, max_sample=config.max_sample)


__all__ = ["offline_plan", "propose_plan"]
