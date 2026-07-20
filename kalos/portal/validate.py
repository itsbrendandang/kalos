"""Ingestion preflight + per-column provenance for the /api/run analyze path.

A client uploads an arbitrary run sheet and the engine silently keeps some
columns as features, treats one as the target, and drops the rest (ids, other
measured outputs, constants, sparse columns). That silent drop is a
product-readiness hole: the client cannot see what the model actually used.

`column_provenance` reproduces the exact feature-selection decisions made in
`kalos.portal.app._analyze` and returns a typed, per-column report so the UI can
show, for every column, whether it was kept, why it was dropped, and how many of
its cells had to be coerced from non-numeric text (units like "34.6 C", "88 %").

This module is pure and side-effect free: it inspects a dataframe and the same
regexes/thresholds `_analyze` uses, and never logs raw column names or values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Pattern

import pandas as pd

ColumnStatus = Literal[
    "kept_feature",
    "target",
    "dropped_id",
    "dropped_output",
    "dropped_constant",
    "dropped_constant_on_fitted_rows",
    "dropped_sparse",
    "dropped_non_numeric",
    "dropped_all_blank",
]


RoleSource = Literal["declared", "inferred"]


@dataclass(frozen=True)
class ColumnProvenance:
    """Why one uploaded column was kept or dropped, and its coercion count.

    `source` records whether the column's role came from an explicit declaration
    (`ColumnRoles`) or was inferred from the domain profile's hint patterns, so
    the audit trail is honest about which decisions the caller made.
    """

    name: str
    status: ColumnStatus
    coerced_cells: int  # non-blank cells that did not parse as a plain number
    non_null: int  # count of non-blank cells
    source: RoleSource = "inferred"

    def to_dict(self) -> dict:
        return asdict(self)


def _coerced_count(series: pd.Series) -> tuple[int, int]:
    """Return (non_blank_count, coerced_count) for one column.

    A cell is "coerced" when it is non-blank but does not parse as a plain number
    on its own (e.g. "34.6 C", "88 %", "n/a"). We count these so the client can
    see that a units-in-cells column was silently cleaned.
    """
    s = series
    nonblank_mask = s.notna() & (s.astype(str).str.strip() != "")
    nonblank = int(nonblank_mask.sum())
    if nonblank == 0:
        return 0, 0
    parsed = pd.to_numeric(s[nonblank_mask], errors="coerce")
    coerced = int(parsed.isna().sum())
    return nonblank, coerced


def column_provenance(
    df: pd.DataFrame,
    *,
    target: str,
    features: list[str],
    numeric_cols: list[str],
    id_hint: Pattern[str],
    outcome_hint: Pattern[str],
    constant_on_fitted_rows: list[str] | None = None,
    declared: set[str] | None = None,
    declared_ids: set[str] | None = None,
    declared_features: set[str] | None = None,
) -> list[ColumnProvenance]:
    """Build a per-column provenance report mirroring `_analyze`'s selection.

    Parameters map to the exact objects `_analyze` computes so this report can
    never disagree with what the model actually used:
      - `target`: the chosen target column name.
      - `features`: the columns kept as model inputs.
      - `numeric_cols`: columns that passed the >=80% numeric-parse test.
      - `id_hint` / `outcome_hint`: the compiled regexes `_analyze` uses to drop
        identifier-like and output-like columns.
      - `constant_on_fitted_rows`: columns that vary over the full sheet but are
        constant on the target-present rows the GP actually fits, so their design
        box would collapse to zero width. `_analyze` drops these; they are flagged
        here (not silently pinned) so the client is not misled about that column.
      - `declared_ids`: columns the caller explicitly declared as ids to ignore,
        so they are reported as `dropped_id` (source `declared`) rather than
        mislabeled `dropped_sparse` by the generic non-numeric branch.
      - `declared_features`: columns the caller declared as features; if such a
        column is non-numeric (and not a categorical the engine used) it is
        reported as `dropped_non_numeric` - an honest "you likely meant to mark
        this categorical" - rather than the misleading `dropped_sparse`.

    Status precedence per column:
      target -> kept_feature -> dropped_id(declared) -> dropped_constant_on_fitted_rows
      -> dropped_all_blank -> dropped_non_numeric(declared) -> dropped_id -> dropped_output
      -> dropped_constant -> dropped_sparse.
    """
    feat_set = set(features)
    num_set = set(numeric_cols)
    constant_fitted = set(constant_on_fitted_rows or [])
    declared_set = declared or set()
    declared_id_set = declared_ids or set()
    declared_feat_set = declared_features or set()
    report: list[ColumnProvenance] = []
    for col in df.columns:
        name = str(col)
        non_null, coerced = _coerced_count(df[col])
        status: ColumnStatus
        if name == str(target):
            status = "target"
        elif col in feat_set:
            status = "kept_feature"
        elif name in declared_id_set:
            # the caller explicitly excluded this column; its drop is an
            # instruction, not a data-quality inference.
            status = "dropped_id"
        elif name in constant_fitted:
            status = "dropped_constant_on_fitted_rows"
        elif non_null == 0:
            status = "dropped_all_blank"
        elif name in declared_feat_set and col not in num_set:
            # declared as a feature but not numeric and not used as a categorical:
            # honestly say it could not be modeled as a number, not "sparse".
            status = "dropped_non_numeric"
        elif id_hint.match(name.strip()):
            status = "dropped_id"
        elif outcome_hint.search(name):
            status = "dropped_output"
        elif col not in num_set:
            # non-numeric and not an id/output: too sparse or non-parseable to model
            status = "dropped_sparse"
        else:
            # numeric, not target/feature/id/output -> excluded as constant
            # (zero variance is the only remaining reason `_analyze` drops it)
            col_num = pd.to_numeric(df[col], errors="coerce")
            std = col_num.std(skipna=True)
            status = "dropped_constant" if (std != std or std <= 1e-9) else "dropped_sparse"
        source: RoleSource = "declared" if name in declared_set else "inferred"
        report.append(
            ColumnProvenance(
                name=name, status=status, coerced_cells=coerced, non_null=non_null,
                source=source,
            )
        )
    return report


def provenance_dicts(report: list[ColumnProvenance]) -> list[dict]:
    """Serialize a provenance report to plain dicts for the JSON response."""
    return [row.to_dict() for row in report]


__all__ = [
    "ColumnProvenance",
    "ColumnStatus",
    "RoleSource",
    "column_provenance",
    "provenance_dicts",
]
