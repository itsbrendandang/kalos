"""Bioprocess data-validation gate: a pre-flight quality check on a run sheet.

A client upload can be wrong in ways that are perfectly numeric and will
still fit a model without complaint: a titer column silently mixing g/L and
mg/mL, a pH of 40, a five-row sheet, an OD column that is 95% blank. None of
that trips a `pd.to_numeric` coercion or a shape check - the model just
learns from bad data and produces a confident, wrong recommendation. This
package is the layer that looks for exactly those failure modes before a
run sheet reaches the engine.

`validate_frame` is the entry point: run all nine checks, aggregate their
findings into one `ValidationReport`, and never raise (a validation gate
that crashes on the data it exists to catch is worse than no gate at all).
`report.py` defines the report's shape, `bounds.py` the physically-possible
ranges checks compare against, `checks.py` the nine checks themselves.

Torch-free by contract (numpy/pandas/scipy only), matching the rest of the
domain-neutral layer this package sits alongside.
"""
from __future__ import annotations

from .bounds import DIMENSION_BOUNDS, DimensionBounds, infer_dimension
from .checks import (
    check_constant_columns,
    check_controls_present,
    check_duplicate_rows,
    check_missingness,
    check_outliers,
    check_physical_bounds,
    check_provenance_metadata,
    check_replicate_adequacy,
    check_units_consistency,
)
from .report import (
    Finding,
    Severity,
    UnitConversion,
    ValidationReport,
    build_report,
    cap_rows,
    report_dict,
)
from .runner import apply_unit_conversions, validate_frame

__all__ = [
    "validate_frame",
    "apply_unit_conversions",
    "ValidationReport",
    "Finding",
    "Severity",
    "UnitConversion",
    "build_report",
    "cap_rows",
    "report_dict",
    "DimensionBounds",
    "DIMENSION_BOUNDS",
    "infer_dimension",
    "check_units_consistency",
    "check_physical_bounds",
    "check_duplicate_rows",
    "check_missingness",
    "check_outliers",
    "check_provenance_metadata",
    "check_replicate_adequacy",
    "check_controls_present",
    "check_constant_columns",
]
