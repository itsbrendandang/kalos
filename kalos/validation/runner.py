"""The validation gate's entry point: run all nine checks, never raise.

`validate_frame` is meant to sit in front of the analyze/optimize path and
look at exactly the kind of data that path is most likely to choke on -
mixed units, impossible values, duplicate rows, a five-row upload. A
validation gate that itself crashes on that input is worse than useless: it
turns a data-quality problem the client could have fixed into an opaque
500. So every check runs inside its own try/except here; one bad column or
an unexpected shape in one check can only ever cost that check's findings,
never take down the report.

`mode` ("warn" vs., say, a future "block") is carried through to the report
untouched. The runner does not read it or change behavior based on it - that
decision belongs to the caller (the portal route), which can choose to
reject a `status == "fail"` upload when in a stricter mode. Keeping that
choice out of this module keeps it a pure "what did we find," not a policy
engine.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from kalos.domains import BIOPROCESS_PROFILE, DomainProfile
from kalos.normalize import units

from .checks import (
    check_constant_columns,
    check_constant_within_group,
    check_controls_present,
    check_duplicate_rows,
    check_informative_missingness,
    check_missingness,
    check_outliers,
    check_physical_bounds,
    check_provenance_metadata,
    check_replicate_adequacy,
    check_units_consistency,
)
from .report import Finding, UnitConversion, ValidationReport, build_report

__all__ = ["validate_frame", "apply_unit_conversions"]


def validate_frame(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    features: list[str] | None = None,
    profile: DomainProfile = BIOPROCESS_PROFILE,
    mode: str = "warn",
) -> ValidationReport:
    """Run all eleven validation checks over `df` and return an aggregated report.

    `target`/`features` feed `check_replicate_adequacy` (both optional -
    without them that check still runs its row-count checks, just skips the
    noise-floor estimate); `target` alone also feeds `check_informative_missingness`
    (`None` -> that check reports nothing, see its own docstring). `profile`
    supplies the `id_hint`/`outcome_hint`/`group_hint` regexes for
    `check_provenance_metadata`, `check_duplicate_rows`, and
    `check_constant_within_group` respectively. `mode` is stored on the
    returned report for the caller to act on; this function does not enforce
    it.

    Never raises: each check runs inside its own try/except, and an
    unexpected exception becomes a `warning`-severity finding
    (`"check failed to run: <ExceptionType>"`) instead of propagating, so one
    bad column cannot take the whole report down.
    """
    feats = list(features) if features else []
    findings: list[Finding] = []
    conversions: list[UnitConversion] = []
    checks_run: list[str] = []

    def run(name: str, fn: Callable[[], list[Finding]]) -> None:
        checks_run.append(name)
        try:
            result = fn()
        except Exception as exc:  # deliberate: a check must never crash the gate
            findings.append(
                Finding(
                    check=name,
                    severity="warning",
                    message=f"check failed to run: {type(exc).__name__}",
                )
            )
            return
        findings.extend(result)

    def run_units() -> list[Finding]:
        unit_findings, unit_conversions = check_units_consistency(df)
        conversions.extend(unit_conversions)
        return unit_findings

    run("units_consistency", run_units)
    run("physical_bounds", lambda: check_physical_bounds(df))
    run("duplicate_rows", lambda: check_duplicate_rows(df, outcome_hint=profile.outcome_hint))
    run("missingness", lambda: check_missingness(df))
    run("outliers", lambda: check_outliers(df))
    run("provenance_metadata", lambda: check_provenance_metadata(df, profile.id_hint))
    run("replicate_adequacy", lambda: check_replicate_adequacy(df, target, feats))
    run("controls_present", lambda: check_controls_present(df))
    run("constant_columns", lambda: check_constant_columns(df))
    run("informative_missingness", lambda: check_informative_missingness(df, target))
    run(
        "constant_within_group",
        lambda: check_constant_within_group(df, group_hint=profile.group_hint),
    )

    return build_report(
        mode=mode,
        n_rows=len(df),
        n_columns=df.shape[1],
        checks_run=checks_run,
        findings=findings,
        conversions=conversions,
    )


def apply_unit_conversions(
    df: pd.DataFrame, conversions: list[UnitConversion] | tuple[UnitConversion, ...]
) -> pd.DataFrame:
    """Return a COPY of `df` with each converted column's cells in its base unit.

    For every `UnitConversion` record, every cell in that column is
    re-parsed with `units.parse_value` and converted with `units.convert`
    individually (rather than trusting the single token recorded on the
    `UnitConversion`), so a stray bare number sitting alongside a labeled
    unit still passes through correctly (`convert(value, None)` is a
    passthrough). A cell that fails to parse is left exactly as it was - this
    function converts units, it does not repair unparseable data. The
    caller's frame is never mutated.
    """
    out = df.copy(deep=True)
    for conv in conversions:
        if conv.column not in out.columns:
            continue

        def _convert_cell(raw: object) -> object:
            if not isinstance(raw, (str, float, int)):
                return raw  # e.g. NaN, None: not parseable, leave untouched
            value, token = units.parse_value(raw)
            if value is None:
                return raw
            base_value, _ = units.convert(value, token)
            return base_value

        out[conv.column] = out[conv.column].map(_convert_cell)
    return out
