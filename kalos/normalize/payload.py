"""Build the identity-stripped payload sent to the LLM tier.

This is the privacy boundary for the whole `kalos.normalize` feature: the
ONLY function in this package that is allowed to look at raw cell values with
the intent of shipping them somewhere. Two layers of defense, in order:

  1. A deterministic pre-screen using the SAME identity rules as the rest of
     the repo (`kalos.data.anonymizer.DROP_EXACT`/`DROP_SUBSTR`) drops every
     identity column before anything else runs. Those columns never enter
     the payload at all.
  2. For every surviving column, free-text columns (the ones most likely to
     carry incidental PII in prose, e.g. lab notes) are redacted to the
     literal string `"<redacted>"` - no raw cell value from a free-text
     column is ever included in the payload.

A defense-in-depth assertion re-checks every payload header against the
anonymizer's DROP rules right before returning, and raises `ValueError` if
one ever slipped through - this should be structurally impossible given the
pre-screen above, but the guarantee is enforced in code, not just by
convention.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from kalos.data.anonymizer import DROP_EXACT, DROP_SUBSTR, Anonymizer

REDACTED = "<redacted>"

# Thresholds for dtype classification. Kept as small module-level constants
# (not a config object) since they are structural to what "numeric" /
# "categorical" / "free-text" MEAN for this payload, not a runtime knob.
_NUMERIC_PARSE_THRESHOLD = 0.8
_UNIT_PARSE_THRESHOLD = 0.6
_CATEGORICAL_MAX_UNIQUE = 20
_CATEGORICAL_MAX_UNIQUE_RATIO = 0.5


def _is_identity_header(header: str) -> bool:
    """True iff `header` matches the anonymizer's identity DROP rules."""
    key = header.strip().lower()
    return key in DROP_EXACT or any(token in key for token in DROP_SUBSTR)


def _numeric_parse_rate(series: pd.Series) -> tuple[float, int]:
    """Fraction of non-null cells that parse via `pd.to_numeric`, and n."""
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return 0.0, 0
    parsed = pd.to_numeric(non_null, errors="coerce")
    return float(parsed.notna().sum()) / n, n


def _unit_parse_rate(series: pd.Series) -> float:
    """Fraction of non-null cells for which `units.parse_value` finds a unit
    token (not just a bare number)."""
    from . import units  # local import: keeps this module's import surface small

    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return 0.0
    with_unit = 0
    for raw in non_null:
        _, unit_token = units.parse_value(raw)
        if unit_token is not None:
            with_unit += 1
    return with_unit / n


def _classify_dtype(series: pd.Series) -> tuple[str, float | None]:
    """Classify one surviving column into a payload `dtype` bucket.

    Returns `(dtype, parse_rate)` where `parse_rate` is the numeric parse
    rate for `"numeric"`/`"numeric+unit"` columns and `None` otherwise.

    Order of decisions (first match wins):
      1. `>= 0.8` of non-null cells parse via `pd.to_numeric` -> `"numeric"`.
      2. Otherwise, if `>= 0.6` of non-null cells yield a unit token from
         `units.parse_value` (e.g. "34.6 C") -> `"numeric+unit"`.
      3. Otherwise, few unique values relative to the column's size ->
         `"categorical"`.
      4. Otherwise -> `"free-text"`.
    """
    numeric_rate, n = _numeric_parse_rate(series)
    if n == 0:
        return "categorical", None
    if numeric_rate >= _NUMERIC_PARSE_THRESHOLD:
        return "numeric", numeric_rate

    unit_rate = _unit_parse_rate(series)
    if unit_rate >= _UNIT_PARSE_THRESHOLD:
        return "numeric+unit", numeric_rate

    non_null = series.dropna()
    n_unique = non_null.astype(str).nunique()
    unique_ratio = n_unique / n
    if n_unique <= _CATEGORICAL_MAX_UNIQUE and unique_ratio <= _CATEGORICAL_MAX_UNIQUE_RATIO:
        return "categorical", numeric_rate
    return "free-text", numeric_rate


def _to_json_safe(value: Any) -> Any:
    """Coerce one cell value to a JSON-primitive: numpy scalars via `.item()`,
    anything else that isn't already a `str`/`int`/`float`/`bool` via `str()`
    (e.g. a `pandas.Timestamp`). Keeps `build_payload`'s JSON-round-trip
    guarantee true regardless of the dataframe's column dtypes."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sample_values(series: pd.Series, max_sample: int) -> list[Any]:
    """Up to `max_sample` distinct, non-null sample values, in first-seen
    order (deterministic given the same dataframe - no random sampling)."""
    non_null = series.dropna()
    seen: list[Any] = []
    for raw in non_null:
        value = _to_json_safe(raw)
        if value in seen:
            continue
        seen.append(value)
        if len(seen) >= max_sample:
            break
    return seen


def build_payload(
    df: pd.DataFrame,
    *,
    anonymizer: Anonymizer | None = None,
    max_sample: int = 5,
) -> tuple[dict[str, Any], list[str]]:
    """Build the screened, JSON-serializable payload sent to the LLM tier.

    Returns `(payload, dropped_identity)`:
      - `payload["columns"]` is a list of per-column dicts, one per SURVIVING
        (non-identity) column: `{"header", "dtype", "non_null": "k/n"}` plus
        `"parse_rate"` for numeric/numeric+unit columns and `"sample"` for
        every column (the literal string `"<redacted>"` for `"free-text"`
        columns, real sample values otherwise).
      - `dropped_identity` is the list of raw column names dropped by the
        deterministic identity pre-screen before classification ever runs.

    `anonymizer` is accepted for interface symmetry with `apply_plan` and
    `offline_plan`, but this function only reads the anonymizer's static
    DROP rule sets (via the module-level `DROP_EXACT`/`DROP_SUBSTR` import),
    not any instance state - anonymizer identity rules are process-wide,
    not configurable per-instance.
    """
    _ = anonymizer  # accepted for interface symmetry; rules are module-level
    n_rows = len(df)

    dropped_identity: list[str] = []
    columns_payload: list[dict[str, Any]] = []

    for header in df.columns:
        if _is_identity_header(str(header)):
            dropped_identity.append(str(header))
            continue

        series = df[header]
        non_null_count = int(series.notna().sum())
        dtype, parse_rate = _classify_dtype(series)

        entry: dict[str, Any] = {
            "header": str(header),
            "dtype": dtype,
            "non_null": f"{non_null_count}/{n_rows}",
        }
        if dtype in ("numeric", "numeric+unit") and parse_rate is not None:
            entry["parse_rate"] = round(parse_rate, 4)

        if dtype == "free-text":
            entry["sample"] = REDACTED
        else:
            entry["sample"] = _sample_values(series, max_sample)

        columns_payload.append(entry)

    payload: dict[str, Any] = {"columns": columns_payload}

    # Defense in depth: no payload column header may match the anonymizer's
    # DROP rules. Given the pre-screen above this should be structurally
    # impossible, but the guarantee is enforced here, not just by convention.
    for entry in columns_payload:
        if _is_identity_header(entry["header"]):
            raise ValueError(
                f"internal error: identity column {entry['header']!r} leaked into the "
                "normalize payload despite the pre-screen"
            )

    # The payload must be JSON-serializable and deterministic given the same
    # input dataframe (json.dumps with a fixed dict/list construction above
    # is enough - no sets, no wall-clock, no randomness were used building it).
    json.dumps(payload)

    return payload, dropped_identity


__all__ = ["build_payload", "REDACTED"]
