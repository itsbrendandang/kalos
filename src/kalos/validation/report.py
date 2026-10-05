"""Typed result shapes for the data-validation gate.

A validation run over a client run sheet can turn up anywhere from zero to
thousands of findings (a 100k-row sheet with one bad column can trigger one
finding PER offending row if nothing caps it). This module is the one place
that owns the shape of that result so:

  - `status` is always derived from the findings, never handed in by a check
    or the runner - a check cannot accidentally report "pass" while also
    emitting an error finding, because there is no constructor path that lets
    the two disagree (see `build_report`).
  - `rows` on a `Finding` is capped at the first 20 offending row indices, with
    the true count recorded in `detail["n_rows_affected"]`. A report is meant
    to be read by a human or rendered in a UI; it must stay small even when
    the underlying data problem is large.
  - every row index is a plain Python `int`, never a numpy scalar, so
    `report_dict(...)` is always `json.dumps`-safe without a custom encoder.

Nothing in here inspects a dataframe - that is `checks.py`'s job. This module
only defines the vocabulary the checks and the runner speak.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kalos.normalize import units

Severity = Literal["error", "warning", "info"]

ROW_CAP = 20


def cap_rows(rows: list[int] | tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Cap an offending-row list at `ROW_CAP` entries, plain-int, order preserved.

    Returns `(capped_rows, n_rows_affected)` so a caller can put the capped
    tuple on `Finding.rows` and the true total in `Finding.detail`. Every
    element is coerced with `int(...)` because boolean-mask-derived indices
    from pandas/numpy are frequently `numpy.int64`, which `json.dumps` cannot
    serialize.
    """
    plain = [int(r) for r in rows]
    return tuple(plain[:ROW_CAP]), len(plain)


@dataclass(frozen=True)
class Finding:
    """One thing a check noticed, at one severity, optionally about one column.

    `rows` holds at most `ROW_CAP` offending row indices (see `cap_rows`); the
    true count of affected rows belongs in `detail["n_rows_affected"]` for any
    finding that concerns specific rows, so the report never balloons on a
    large sheet.
    """

    check: str
    severity: Severity
    message: str
    column: str | None = None
    rows: tuple[int, ...] = ()
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "column": self.column,
            "rows": list(self.rows),
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class UnitConversion:
    """A column that used exactly one unit token, and the base unit it maps to.

    Emitted by `check_units_consistency` for columns clean enough to convert
    automatically (see that function's docstring for why ambiguous columns do
    not get one of these). `apply_unit_conversions` consumes a list of these
    to actually rewrite a dataframe's cells.
    """

    column: str
    from_units: tuple[str, ...]
    to_unit: str
    cells_converted: int

    def to_dict(self) -> dict:
        # `to_unit` is an internal suffix ("_c"); `to_unit_label` is the name a
        # scientist should actually read ("Celsius"). Both are serialized: the
        # suffix stays for programmatic consumers, and the label means no UI has
        # to maintain its own private mapping and drift from this one.
        return {
            "column": self.column,
            "from_units": list(self.from_units),
            "to_unit": self.to_unit,
            "to_unit_label": units.base_unit_label(self.to_unit),
            "cells_converted": self.cells_converted,
        }


@dataclass(frozen=True)
class ValidationReport:
    """The full result of one `validate_frame` run.

    Construct this via `build_report`, not the bare constructor - `status` and
    `counts` are derived from `findings` and there is exactly one place
    (`build_report`) that is allowed to compute them, so a caller can never
    hand-assemble a report where `status` disagrees with what `findings`
    actually contains.
    """

    status: Literal["pass", "pass_with_warnings", "fail"]
    mode: str
    n_rows: int
    n_columns: int
    checks_run: tuple[str, ...]
    counts: dict
    findings: tuple[Finding, ...]
    conversions: tuple[UnitConversion, ...]
    schema_version: int = 1

    def to_dict(self) -> dict:
        return report_dict(self)


def _derive_status(findings: list[Finding] | tuple[Finding, ...]) -> Literal[
    "pass", "pass_with_warnings", "fail"
]:
    has_error = any(f.severity == "error" for f in findings)
    if has_error:
        return "fail"
    has_warning = any(f.severity == "warning" for f in findings)
    return "pass_with_warnings" if has_warning else "pass"


def _derive_counts(findings: list[Finding] | tuple[Finding, ...]) -> dict:
    counts = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f.severity] += 1
    return counts


def build_report(
    *,
    mode: str,
    n_rows: int,
    n_columns: int,
    checks_run: list[str] | tuple[str, ...],
    findings: list[Finding] | tuple[Finding, ...],
    conversions: list[UnitConversion] | tuple[UnitConversion, ...],
) -> ValidationReport:
    """Assemble a `ValidationReport`, deriving `status` and `counts` from `findings`.

    This is the only supported way to build a report; `status` is never an
    input because it must always agree with the findings that justify it.
    """
    findings_t = tuple(findings)
    return ValidationReport(
        status=_derive_status(findings_t),
        mode=mode,
        n_rows=n_rows,
        n_columns=n_columns,
        checks_run=tuple(checks_run),
        counts=_derive_counts(findings_t),
        findings=findings_t,
        conversions=tuple(conversions),
    )


def _json_safe(value: object) -> object:
    """Recursively replace non-finite floats with None so the report can be served.

    Several dimensions are genuinely unbounded above (`concentration`,
    `flow_rate`, `time_h` all have `hard_hi = inf`), and that `inf` travels into
    a finding's `detail["hard_range"]`. Python's `json.dumps` emits it as the
    literal `Infinity`, which is not valid JSON, and Starlette's `JSONResponse`
    correctly refuses with "Out of range float values are not JSON compliant".
    That turned a perfectly good analysis into a 400 for any sheet with a
    negative concentration - the gate's own finding broke the response carrying
    it.

    `None` is the right replacement rather than a sentinel number or the string
    "inf": in JSON, a null upper bound reads as "no upper bound", which is
    exactly what `inf` meant here. NaN maps to None for the same reason - there
    is no number to report.

    This lives at the serialization boundary on purpose. Sanitizing inside each
    check would mean every future check has to remember to do it, and the one
    that forgets takes down the response.
    """
    if isinstance(value, float):
        return None if (value != value or value in (_POS_INF, _NEG_INF)) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


_POS_INF = float("inf")
_NEG_INF = float("-inf")


def report_dict(report: ValidationReport) -> dict:
    """Serialize a report to a plain, `json.dumps`-safe dict for the JSON response.

    Safe means safe under `allow_nan=False`, which is what Starlette uses - see
    `_json_safe` for why that distinction cost a 400.
    """
    return {
        "status": report.status,
        "mode": report.mode,
        "n_rows": report.n_rows,
        "n_columns": report.n_columns,
        "checks_run": list(report.checks_run),
        "counts": dict(report.counts),
        "findings": [_json_safe(f.to_dict()) for f in report.findings],
        "conversions": [_json_safe(c.to_dict()) for c in report.conversions],
        "schema_version": report.schema_version,
    }


__all__ = [
    "Severity",
    "Finding",
    "UnitConversion",
    "ValidationReport",
    "ROW_CAP",
    "cap_rows",
    "build_report",
    "report_dict",
]
