"""Canonical unit registry + messy-cell parsing for bioprocess run sheets.

Client run sheets encode units in the cell text ("34.6 C", "88%", "0.45 mL/h")
rather than in a separate column, so a single "titer" column can silently mix
g/L and mg/mL across rows. This module is the one place that knows how to pull
a `(value, unit_token)` pair out of a messy cell and how to convert that value
into a fixed BASE unit per dimension, so every canonical column downstream is
in one unit, always.

Base units and conversion formulas (the single source of truth - do not
duplicate these elsewhere):
  temperature   -> base Celsius (C)
    F -> (v - 32) * 5 / 9
    K -> v - 273.15
    C -> v
  percent       -> base is the bare percent number (unit-less, 0-100 scale)
    % / pct -> v unchanged
  concentration -> base g/L
    g/L, g/l  -> v
    mg/mL, mg/ml -> v            (1 mg/mL == 1 g/L, since mg/mL = g/L numerically)
  flow rate     -> base mL/h
    mL/h, ml/h -> v
    uL/h, µL/h -> v / 1000       (ascii "u" and the micro sign both accepted)
    L/h        -> v * 1000
    mL/min     -> v * 60         (per-minute gas/liquid flow, e.g. sparge air)
    L/min      -> v * 60000
  time          -> base hours (h)
    h, hr -> v
    min   -> v / 60
  agitation     -> base rpm
    rpm -> v
  pressure      -> base bar
    bar -> v
  mass          -> base grams (g)
    g  -> v
    kg -> v * 1000
  dimensionless -> no unit at all (pH, OD): value passes through unchanged,
    canonical suffix is the empty string.

NOTE on what is deliberately NOT a recognized unit: bare "L"/"mL" (volume) is
NOT in the registry, on purpose. Bioprocess run sheets commonly use "L" as a
VESSEL-SIZE IDENTIFIER, not a measurement - "5L", "500L" (bioreactor scale
labels), not "5 liters of something". Recognizing "L" as a unit would
silently convert that identifier into a numeric measurement (stripping the
label and keeping the bare number) - `is_known_unit`'s docstring below and
`tests/test_validation_gate.py::test_unrecognized_unit_is_not_silently_converted`
both encode this as a deliberate, tested exclusion. Do not add it back
without addressing that collision first (e.g. requiring a compound token
like "L/well" that a vessel-size label would never produce).

Every unit token maps to a `(dimension, to_base_fn, canonical_suffix)` triple
in `_UNIT_TABLE`. This keeps the registry a small, readable, data-driven table
instead of a pile of if/elif chains: adding a unit means adding one row.
"""
from __future__ import annotations

import re
from typing import Callable, NamedTuple

# --- unit registry -------------------------------------------------------- #


class _UnitDef(NamedTuple):
    dimension: str
    to_base: Callable[[float], float]
    suffix: str  # canonical column-name suffix for this unit's dimension


# Unit tokens are matched case-sensitively where case is chemically meaningful
# (mL vs ml is fine either way, but "M" for molar vs "m" for milli would not
# be), and the lookup below tries an exact match first, then a case-insensitive
# fallback, so both "C" and "c" resolve to Celsius.
_UNIT_TABLE: dict[str, _UnitDef] = {
    # temperature -> base Celsius
    "c": _UnitDef("temperature", lambda v: v, "_c"),
    "celsius": _UnitDef("temperature", lambda v: v, "_c"),
    "f": _UnitDef("temperature", lambda v: (v - 32.0) * 5.0 / 9.0, "_c"),
    "fahrenheit": _UnitDef("temperature", lambda v: (v - 32.0) * 5.0 / 9.0, "_c"),
    "k": _UnitDef("temperature", lambda v: v - 273.15, "_c"),
    "kelvin": _UnitDef("temperature", lambda v: v - 273.15, "_c"),
    # percent -> base is the bare percent number, unchanged
    "%": _UnitDef("percent", lambda v: v, "_pct"),
    "pct": _UnitDef("percent", lambda v: v, "_pct"),
    # concentration -> base g/L
    "g/l": _UnitDef("concentration", lambda v: v, "_g_l"),
    "mg/ml": _UnitDef("concentration", lambda v: v, "_g_l"),  # 1 mg/mL == 1 g/L
    # flow rate -> base mL/h
    "ml/h": _UnitDef("flow_rate", lambda v: v, "_ml_h"),
    "ul/h": _UnitDef("flow_rate", lambda v: v / 1000.0, "_ml_h"),
    "l/h": _UnitDef("flow_rate", lambda v: v * 1000.0, "_ml_h"),
    # gas/liquid flow rate given per-minute rather than per-hour (common on
    # bioreactor gas-flow controllers, e.g. "2.5 L/min" sparge air) -> same
    # base (mL/h) and suffix as the per-hour tokens above, mirroring that
    # existing ml/h <-> l/h pairing.
    "ml/min": _UnitDef("flow_rate", lambda v: v * 60.0, "_ml_h"),
    "l/min": _UnitDef("flow_rate", lambda v: v * 60_000.0, "_ml_h"),
    # time -> base hours
    "h": _UnitDef("time", lambda v: v, "_h"),
    "hr": _UnitDef("time", lambda v: v, "_h"),
    "min": _UnitDef("time", lambda v: v / 60.0, "_h"),
    # agitation -> base rpm (a single unit; no conversion needed)
    "rpm": _UnitDef("agitation", lambda v: v, "_rpm"),
    # pressure -> base bar (a single unit; no conversion needed)
    "bar": _UnitDef("pressure", lambda v: v, "_bar"),
    # volume ("L"/"mL") is deliberately NOT registered here - see the module
    # docstring's "NOTE on what is deliberately NOT a recognized unit" above.
    # mass -> base grams
    "g": _UnitDef("mass", lambda v: v, "_g"),
    "kg": _UnitDef("mass", lambda v: v * 1000.0, "_g"),
    # dimensionless -> value unchanged, no suffix
    "ph": _UnitDef("dimensionless", lambda v: v, ""),
    "od": _UnitDef("dimensionless", lambda v: v, ""),
}

# Case is preserved for lookup of these ambiguous-if-lowercased tokens (mL vs
# ml both mean milliliters so lowercasing is safe there, but "uL"/"µL" use
# the micro prefix and are handled by normalizing the micro sign before lookup).
_MICRO_PATTERN = re.compile(r"[µμ]")  # MICRO SIGN, GREEK SMALL LETTER MU

# A unit token is letters/percent/slash, optionally attached to a number with
# a space or directly (e.g. "34.6 C", "88%", "0.45mL/h").
_VALUE_UNIT_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*([A-Za-z%/µμ]*)\s*$"
)


def _normalize_unit_token(token: str) -> str:
    """Fold a raw unit token to its lookup key: normalize micro signs to 'u',
    then lowercase. ("uL/h", "µL/h", "μL/h" all fold to "ul/h".)"""
    return _MICRO_PATTERN.sub("u", token).strip().lower()


def parse_value(raw: str | float | int) -> tuple[float | None, str | None]:
    """Extract `(numeric_value, unit_token)` from a messy run-sheet cell.

    `unit_token` is the raw token as it appeared (case preserved) so
    `canonical_suffix`/`convert` can still look up its dimension; the value is
    left in its ORIGINAL unit, not yet converted to base. Examples:
      "34.6 C"    -> (34.6, "C")
      "88%"       -> (88.0, "%")
      "0.45 mL/h" -> (0.45, "mL/h")
      "72"        -> (72.0, None)          (bare number, no unit)
      "n/a"       -> (None, None)          (non-parseable)

    Bare `float`/`int` input -> `(float(raw), None)`. Non-parseable text
    (does not match a leading number, or a trailing unit token) -> `(None,
    None)`, so callers can count coercions instead of raising.
    """
    if isinstance(raw, (float, int)) and not isinstance(raw, bool):
        return float(raw), None
    text = str(raw)
    m = _VALUE_UNIT_RE.match(text)
    if not m:
        return None, None
    try:
        value = float(m.group(1))
    except ValueError:
        return None, None
    unit_token = m.group(2).strip()
    return value, (unit_token if unit_token else None)


# Human-readable label for each canonical suffix. The suffixes above ("_c",
# "_g_l") are internal column-name fragments, not something to show a scientist.
# Without this table every consumer invents its own mapping - the web UI had to
# hand-build a Celsius/Fahrenheit lookup and guess at the rest - which means the
# label a client reads depends on which surface they read it from. One table
# here, so "converted to Celsius" reads the same everywhere.
_BASE_UNIT_LABELS: dict[str, str] = {
    "_c": "Celsius",
    "_pct": "percent",
    "_g_l": "g/L",
    "_ml_h": "mL/h",
    "_h": "hours",
    "_rpm": "rpm",
    "_bar": "bar",
    "_g": "grams",
    "": "dimensionless",
}


def base_unit_label(suffix: str) -> str:
    """Human-readable name for a canonical suffix returned by `canonical_suffix`.

    `base_unit_label("_c") == "Celsius"`. An unrecognized suffix returns itself
    unchanged rather than raising, so a new suffix added to `_UNIT_TABLE` without
    a label here degrades to showing the raw fragment instead of breaking a
    response.
    """
    return _BASE_UNIT_LABELS.get(suffix, suffix)


def is_known_unit(unit_token: str | None) -> bool:
    """Whether `unit_token` is a unit this registry can actually convert.

    This exists because `canonical_suffix` cannot answer the question: it
    returns `""` both for an UNKNOWN token and for a genuinely dimensionless
    known one (pH, OD), so a caller inspecting the suffix alone cannot tell
    "this is unitless" from "I have no idea what this is".

    That distinction decides whether it is safe to rewrite a column. Converting
    a column of `"37 C"` to Celsius is a correct normalization; "converting" a
    column of vessel labels like `"5L"` (where `L` is not in the registry)
    would strip the label and leave a bare number, silently turning an
    identifier into a measurement. Callers must only auto-convert tokens this
    function accepts. `None` (a bare number, no unit) is not a known unit -
    there is nothing to convert.
    """
    if unit_token is None:
        return False
    return _normalize_unit_token(unit_token) in _UNIT_TABLE


def canonical_suffix(unit_token: str | None) -> str:
    """Return the canonical column-name suffix for a unit token.

    Unknown or `None` tokens return `""` (no suffix - passthrough), matching
    `convert`'s passthrough behavior for the same input.
    """
    if unit_token is None:
        return ""
    key = _normalize_unit_token(unit_token)
    unit_def = _UNIT_TABLE.get(key)
    return unit_def.suffix if unit_def is not None else ""


def convert(value: float, unit_token: str | None) -> tuple[float, str]:
    """Convert `value` (in `unit_token`'s original unit) to its dimension's
    base unit, returning `(value_in_base_unit, canonical_suffix)`.

    An unknown or `None` unit token is a passthrough: `(value, "")`. This is
    a deliberate design choice - an unrecognized unit is not an error here,
    just not something we know how to convert, so the raw value is kept.
    """
    if unit_token is None:
        return value, ""
    key = _normalize_unit_token(unit_token)
    unit_def = _UNIT_TABLE.get(key)
    if unit_def is None:
        return value, ""
    return unit_def.to_base(value), unit_def.suffix


__all__ = ["parse_value", "convert", "canonical_suffix", "is_known_unit", "base_unit_label"]
