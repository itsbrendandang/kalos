"""LLM tier: propose a `NormalizationPlan` from an identity-screened payload,
with a fully deterministic offline fallback.

`anthropic` and `pydantic` are imported lazily, INSIDE the live-path function
only - this module (and `kalos.normalize` as a whole) must stay importable,
and `offline_plan`/`propose_plan` must stay fully USABLE, with neither
package installed. The live path never runs unless an API key is present
(`config.credentials_available`), and on any failure it falls back to the
offline path rather than raising - this feature must never hard-fail because
the LLM is unavailable, rate-limited, or misbehaving.

Nothing in this module logs, prints, or persists the API key itself; the key
is read only by the `anthropic` SDK's own `Anthropic()` client constructor
from the environment, never handled as a string here.
"""
from __future__ import annotations

import logging

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
(free-text columns and grouping/campaign-style columns are redacted). Raw identity \
columns have already been removed before you see this payload.

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


def _live_plan(
    df: pd.DataFrame,
    config: NormalizeConfig,
    payload: dict,
    dropped_identity: list[str],
    anonymizer: Anonymizer | None,
) -> NormalizationPlan:
    """The live LLM call. Lazily imports `anthropic` and `pydantic` - never
    imported at module scope so the package stays import-light and this
    function is the only place either dependency is required."""
    import json

    import anthropic
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

    client = anthropic.Anthropic()
    payload_json = json.dumps(payload, sort_keys=True)

    kwargs = dict(
        model=config.model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": payload_json}],
        thinking={"type": "disabled"},
    )

    if hasattr(client.messages, "parse"):
        response = client.messages.parse(output_format=LLMPlan, **kwargs)
        parsed: LLMPlan = response.parsed_output
    else:
        schema = LLMPlan.model_json_schema()
        response = client.messages.create(
            output_config={"format": {"type": "json_schema", "schema": schema}},
            **kwargs,
        )
        text = next(block.text for block in response.content if block.type == "text")
        parsed = LLMPlan.model_validate_json(text)

    llm_columns: list[ColumnPlan] = []
    for item in parsed.columns:
        is_identity = bool(item.is_identity)
        canonical_name = None if is_identity else item.canonical_name
        role = "identity" if is_identity else item.role
        llm_columns.append(
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

    all_raw_names = [str(c) for c in df.columns]
    merged = _merge_dropped_identity(llm_columns, dropped_identity, all_raw_names)
    plan = NormalizationPlan(columns=merged, created_by="llm", model=config.model)
    plan.validate()
    return plan


def propose_plan(
    df: pd.DataFrame,
    *,
    config: NormalizeConfig | None = None,
    anonymizer: Anonymizer | None = None,
) -> NormalizationPlan:
    """Propose a `NormalizationPlan` for `df`, using the LLM tier when
    credentials are available and falling back to the fully offline,
    deterministic path otherwise (or on any live-call failure).

    Steps:
      1. Resolve `config` (`config.load_config()` if not given).
      2. Build the identity-screened payload via `payload.build_payload`,
         using `config.max_sample`.
      3. If `not config.enabled_live`: return `offline_plan(df, ...)`.
      4. Otherwise call the model (lazy `anthropic`/`pydantic` import). On
         success, merge the model's per-column proposals with the
         deterministically pre-screened identity columns and return a plan
         with `created_by="llm"`. On ANY failure (`anthropic.APIError`,
         `anthropic.APIConnectionError`, or any other exception), log a
         warning containing no raw data and no key, and fall back to
         `offline_plan(df, ...)`.
    """
    config = config or load_config()
    payload, dropped_identity = build_payload(df, anonymizer=anonymizer, max_sample=config.max_sample)

    if not config.enabled_live:
        return offline_plan(df, anonymizer=anonymizer, max_sample=config.max_sample)

    try:
        return _live_plan(df, config, payload, dropped_identity, anonymizer)
    except Exception as err:  # noqa: BLE001 - must never hard-fail the caller
        log.warning(
            "normalize: live plan failed (%s); falling back to offline",
            type(err).__name__,
        )
        return offline_plan(df, anonymizer=anonymizer, max_sample=config.max_sample)


__all__ = ["offline_plan", "propose_plan"]
