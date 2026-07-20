"""Execute a `NormalizationPlan` against a dataframe, deterministically.

`apply_plan` is the one place that turns a plan into an actual cleaned
dataframe: identity/freetext columns dropped, group columns hashed to a
stable pseudonym, unit-bearing columns converted to their base unit, and
everything else coerced to numeric where the role calls for it. Given the
same `(df, plan)` pair it always returns byte-identical output - no
randomness, no wall-clock, no network - which is the reproducibility
contract this whole package exists to satisfy.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from kalos.data.anonymizer import Anonymizer, _hash, default_salt

from . import units
from .plan import ColumnProvenance, NormalizationPlan

# Short, stable prefix on hashed group values so a hashed cell is visually
# distinguishable from a raw value in any exported/inspected frame.
_GROUP_HASH_PREFIX = "cmp_"


@dataclass
class NormalizedResult:
    """The output of `apply_plan`: the cleaned frame, its column-level
    provenance, the list of raw column names that were dropped, and the
    plan that produced it (for audit)."""

    frame: pd.DataFrame
    provenance: list[ColumnProvenance]
    dropped: list[str]
    plan: NormalizationPlan


def _convert_column(series: pd.Series, unit_token: str | None) -> tuple[pd.Series, int, int]:
    """Parse + convert every cell in `series` to its base unit.

    Returns `(converted_series, n_converted, n_coerced_nan)` where
    `n_converted` counts cells that parsed to a number (converted using
    `unit_token` if given, else the per-cell unit `parse_value` found) and
    `n_coerced_nan` counts non-null raw cells that did not parse and became
    NaN.
    """
    values: list[float | None] = []
    n_converted = 0
    n_coerced_nan = 0
    for raw in series:
        if pd.isna(raw):
            values.append(None)
            continue
        value, parsed_unit = units.parse_value(raw)
        if value is None:
            values.append(None)
            n_coerced_nan += 1
            continue
        base_value, _ = units.convert(value, unit_token if unit_token is not None else parsed_unit)
        values.append(base_value)
        n_converted += 1
    return pd.Series(values, index=series.index, dtype="float64"), n_converted, n_coerced_nan


def apply_plan(
    df: pd.DataFrame, plan: NormalizationPlan, *, anonymizer: Anonymizer | None = None
) -> NormalizedResult:
    """Apply `plan` to `df`, returning the cleaned frame plus full provenance.

    Steps, per column in the plan (in the plan's column order):
      1. Validate the plan first (`plan.validate()`); an invalid plan raises
         before touching `df`.
      2. `role in ("identity", "freetext")` -> column dropped entirely,
         recorded in `dropped` and as `dropped_identity`/`dropped_freetext`
         provenance.
      3. Otherwise renamed raw -> `canonical_name` (falls back to `raw_name`
         if `canonical_name` is None).
      4. If `to_base` and a `unit_token` (or per-cell parsed unit) is
         present, each cell is parsed and converted to its base unit via
         `units.parse_value`/`units.convert`; conversion and coercion counts
         are recorded (action `converted`).
      5. If `role == "group"`, each value is hashed via the anonymizer's
         `_hash` with a stable `cmp_`-prefixed short form (action `hashed`).
      6. Else if `role in ("target", "feature")`, the column is coerced with
         `pd.to_numeric(errors="coerce")`, counting NaNs produced (action
         `coerced`).
      7. Anything else (e.g. `role == "metadata"`) is renamed only (action
         `renamed`), values untouched.

    Row order and index are preserved throughout - no sort, no reset. Given
    the same `df` and `plan`, this always returns a byte-identical frame
    (deterministic; no randomness or wall-clock is read anywhere in this
    function).
    """
    plan.validate()
    anon = anonymizer if anonymizer is not None else Anonymizer()

    dropped: list[str] = []
    provenance: list[ColumnProvenance] = []
    out_columns: dict[str, pd.Series] = {}

    for col in plan.columns:
        if col.raw_name not in df.columns:
            continue  # plan may reference columns not present in this frame
        series = df[col.raw_name]

        if col.role in ("identity", "freetext"):
            dropped.append(col.raw_name)
            action = "dropped_identity" if col.role == "identity" else "dropped_freetext"
            provenance.append(
                ColumnProvenance(
                    raw_name=col.raw_name,
                    canonical_name=None,
                    role=col.role,
                    action=action,
                )
            )
            continue

        out_name = col.canonical_name if col.canonical_name is not None else col.raw_name

        if col.to_base:
            converted, n_converted, n_coerced_nan = _convert_column(series, col.unit_token)
            out_columns[out_name] = converted
            provenance.append(
                ColumnProvenance(
                    raw_name=col.raw_name,
                    canonical_name=out_name,
                    role=col.role,
                    action="converted",
                    n_converted=n_converted,
                    n_coerced_nan=n_coerced_nan,
                )
            )
            continue

        if col.role == "group":
            # `Anonymizer.salt` is the public field; fall back to the shared
            # default salt exactly like `Anonymizer._salt()` does internally,
            # without reaching into that private method from outside the class.
            salt = anon.salt if anon.salt is not None else default_salt()
            hashed = series.map(
                lambda v, _salt=salt: v if pd.isna(v) else f"{_GROUP_HASH_PREFIX}{_hash(v, _salt)}"
            )
            out_columns[out_name] = hashed
            provenance.append(
                ColumnProvenance(
                    raw_name=col.raw_name,
                    canonical_name=out_name,
                    role=col.role,
                    action="hashed",
                )
            )
            continue

        if col.role in ("target", "feature"):
            numeric = pd.to_numeric(series, errors="coerce")
            n_coerced_nan = int((numeric.isna() & series.notna()).sum())
            out_columns[out_name] = numeric
            provenance.append(
                ColumnProvenance(
                    raw_name=col.raw_name,
                    canonical_name=out_name,
                    role=col.role,
                    action="coerced",
                    n_coerced_nan=n_coerced_nan,
                )
            )
            continue

        # metadata (or any other kept role): renamed only, values untouched.
        out_columns[out_name] = series
        provenance.append(
            ColumnProvenance(
                raw_name=col.raw_name,
                canonical_name=out_name,
                role=col.role,
                action="renamed",
            )
        )

    frame = pd.DataFrame(out_columns, index=df.index)
    return NormalizedResult(frame=frame, provenance=provenance, dropped=dropped, plan=plan)


__all__ = ["NormalizedResult", "apply_plan"]
