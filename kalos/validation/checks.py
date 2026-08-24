"""The eleven data-quality checks the validation gate runs over a run sheet.

Every function here is pure: `df: pd.DataFrame -> list[Finding]`, never
mutating the input frame, never logging a raw cell value or column name to
anything global (findings carry column names/values because that is the
report's whole job, but nothing here writes them anywhere else). A check
that needs domain vocabulary (which columns look like ids, which look like
outcomes) takes that as an explicit parameter instead of hardcoding a
regex, so the caller's `DomainProfile` stays the single source of truth for
that vocabulary.

Two families of numeric check live here, on purpose using different
machinery:

  - `check_units_consistency` and `check_physical_bounds` are UNIT-AWARE:
    they run every cell through `kalos.normalize.units.parse_value` so
    "34.6 C" and "94.3 F" are compared as what they actually are, not as the
    strings "34.6" and "94.3". This is the only unit parser used anywhere in
    this package - see `units.py`'s own docstring for why a second one must
    never be written.
  - `check_outliers` and `check_constant_columns` work on `pd.to_numeric`
    coercion instead. They care about the numeric *shape* of a column
    (spread, variance), not what unit it is in, and a cell with an embedded
    unit token ("34.6 C") is not a plain number to `pd.to_numeric` - it
    coerces to NaN and is correctly excluded from a statistic that would
    otherwise be nonsense on mixed unit-tagged text.

The single most important check in this file is `check_units_consistency`:
a column that silently mixes g/L and mg/mL (or C and F) is the bug class
that quietly ruins a client dataset, because every downstream statistic on
that column is computed as if the numbers were all in the same unit.
"""
from __future__ import annotations

import re
from typing import Pattern

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kalos.core.replicates import noise_report
from kalos.normalize import units

from .bounds import DIMENSION_BOUNDS, infer_dimension
from .report import Finding, UnitConversion, cap_rows

__all__ = [
    "check_units_consistency",
    "check_physical_bounds",
    "check_duplicate_rows",
    "check_missingness",
    "check_outliers",
    "check_provenance_metadata",
    "check_replicate_adequacy",
    "check_controls_present",
    "check_constant_columns",
    "check_informative_missingness",
    "check_constant_within_group",
]


def _nonblank_mask(series: pd.Series) -> pd.Series:
    """True where a cell is neither NaN nor an all-whitespace string.

    Matches `kalos.portal.validate`'s own blank test so "blank" means the
    same thing everywhere in the codebase.
    """
    return series.notna() & (series.astype(str).str.strip() != "")


def _row_index(idx: object) -> int:
    """Coerce a pandas row label to a plain `int`.

    `Series.items()` types its index as `Hashable` (pandas-stubs cannot know
    it is actually an integer RangeIndex label), so a bare `int(idx)` does
    not satisfy mypy even though every row label in this package's frames is
    integer-valued. Centralizing the cast here keeps the justification in
    one place instead of a `# type: ignore` scattered across every check.
    """
    return int(idx)  # type: ignore[call-overload]


# --- 1. unit consistency ---------------------------------------------------- #


def check_units_consistency(
    df: pd.DataFrame,
) -> tuple[list[Finding], list[UnitConversion]]:
    """Flag columns whose cells carry more than one distinct unit token.

    Per column, every non-blank cell is run through `units.parse_value`
    to get `(value, unit_token)`. What happens next depends on the set of
    distinct `unit_token`s seen (ignoring cells that fail to parse at all -
    that is a different failure mode, not this check's concern):

      - Zero or only-`None` tokens (every parsed cell was a bare number):
        nothing to report. Most numeric columns look like this.
      - Exactly one distinct non-`None` token: the column consistently uses
        one unit. INFO finding plus a `UnitConversion` record so the caller
        can auto-convert the column to its base unit via
        `apply_unit_conversions`.
      - Two or more distinct non-`None` tokens: ERROR. If every token maps
        to the same dimension via `units.canonical_suffix` (e.g. "C" and
        "F", both temperature), the message says the column mixes units
        within one dimension - the classic g/L-vs-mg/mL bug. If the tokens
        map to different dimensions, the message says the column is
        ambiguous (it is not clear what quantity the column even measures).

    A bare number mixed into a column that otherwise uses one explicit unit
    token is deliberately NOT flagged here: `canonical_suffix(None)` is `""`
    same as a dimensionless unit, but bare numbers are excluded from the
    token set entirely (not conflated with a `""`-dimension unit) because a
    spreadsheet routinely omits the unit on repeat rows once it has been
    stated once. Only genuinely differing explicit unit tokens are an error.
    """
    findings: list[Finding] = []
    conversions: list[UnitConversion] = []

    for col in df.columns:
        name = str(col)
        series = df[col]
        mask = _nonblank_mask(series)
        if not mask.any():
            continue

        token_rows: dict[str | None, list[int]] = {}
        for idx, raw in series[mask].items():
            value, token = units.parse_value(raw)
            if value is None:
                continue
            token_rows.setdefault(token, []).append(_row_index(idx))

        non_none_tokens = [t for t in token_rows if t is not None]
        if len(non_none_tokens) == 0:
            continue

        # Split on whether the registry can actually convert the token. This
        # matters because `canonical_suffix` returns "" both for an unknown token
        # and for a known dimensionless one, so suffix alone cannot tell them
        # apart. Recording a conversion for an unknown token is actively harmful:
        # a vessel column holding "5L"/"500L" ("L" is not in the registry) would
        # be "converted" to bare 5/500, silently turning a label into a
        # measurement. Unknown tokens therefore never produce a conversion.
        known_tokens = [t for t in non_none_tokens if units.is_known_unit(t)]
        unknown_tokens = [t for t in non_none_tokens if not units.is_known_unit(t)]

        if unknown_tokens:
            unknown_rows = sorted(r for t in unknown_tokens for r in token_rows[t])
            capped_unknown, total_unknown = cap_rows(unknown_rows)
            findings.append(
                Finding(
                    check="units_consistency",
                    severity="warning",
                    message=(
                        f"column '{name}' carries unit token(s) "
                        f"{sorted(unknown_tokens)} that this engine cannot "
                        "convert; the column is left exactly as uploaded"
                    ),
                    column=name,
                    rows=capped_unknown,
                    detail={
                        "unrecognized_tokens": sorted(unknown_tokens),
                        "n_rows_affected": total_unknown,
                    },
                )
            )

        if len(known_tokens) == 1 and not unknown_tokens:
            token = known_tokens[0]
            suffix = units.canonical_suffix(token)
            n_cells = len(token_rows[token])
            findings.append(
                Finding(
                    check="units_consistency",
                    severity="info",
                    message=(
                        f"column '{name}' consistently uses unit '{token}'; "
                        "eligible for automatic conversion to its base unit"
                    ),
                    column=name,
                    detail={"unit": token, "cells_converted": n_cells},
                )
            )
            conversions.append(
                UnitConversion(
                    column=name, from_units=(token,), to_unit=suffix, cells_converted=n_cells
                )
            )
            continue

        # Two or more CONVERTIBLE units in one column is the real bug class this
        # check exists for (the g/L-vs-mg/mL scale break). Judge the dimension
        # question on known tokens only - unknown ones were reported above and
        # would otherwise all collapse to the "" suffix and fake agreement.
        if len(known_tokens) < 2:
            continue

        dims = {t: units.canonical_suffix(t) for t in known_tokens}
        distinct_dims = set(dims.values())
        offending_rows = sorted(r for t in known_tokens for r in token_rows[t])
        capped, total = cap_rows(offending_rows)
        # Known tokens only: an unrecognized token already has its own warning,
        # and naming it here too would imply this error was about a unit the
        # engine had actually resolved.
        tokens_sorted = sorted(known_tokens)
        if len(distinct_dims) == 1:
            message = (
                f"column '{name}' mixes units {tokens_sorted} that all resolve to the "
                "same dimension - values are not comparable without conversion"
            )
        else:
            message = (
                f"column '{name}' mixes units {tokens_sorted} from different "
                "dimensions - the column is ambiguous about what it measures"
            )
        findings.append(
            Finding(
                check="units_consistency",
                severity="error",
                message=message,
                column=name,
                rows=capped,
                detail={"tokens": tokens_sorted, "n_rows_affected": total},
            )
        )

    return findings, conversions


# --- 2. physical bounds ------------------------------------------------------ #


def check_physical_bounds(df: pd.DataFrame) -> list[Finding]:
    """Flag values outside a column's inferred dimension's HARD or TYPICAL range.

    Dimension is inferred from the header (`bounds.infer_dimension`); a
    column with no recognizable dimension is silently skipped - no bounds
    check without a confident guess at what the column measures. Every cell
    is converted to its dimension's base unit via `units.parse_value` +
    `units.convert` before comparison, so a Fahrenheit cell in a Celsius
    column is judged correctly instead of compared as a raw number.

    Outside HARD range -> ERROR (physically impossible). Inside HARD but
    outside TYPICAL range -> WARNING (possible but operationally suspect).
    Each finding's `detail` carries the offending rows' min/max seen value
    (in base units) alongside the capped row list.
    """
    findings: list[Finding] = []

    for col in df.columns:
        name = str(col)
        dim = infer_dimension(name)
        if dim is None:
            continue
        bounds = DIMENSION_BOUNDS[dim]
        series = df[col]
        mask = _nonblank_mask(series)
        if not mask.any():
            continue

        error_rows: list[int] = []
        error_vals: list[float] = []
        warning_rows: list[int] = []
        warning_vals: list[float] = []
        for idx, raw in series[mask].items():
            value, token = units.parse_value(raw)
            if value is None:
                continue
            base_value, _ = units.convert(value, token)
            if base_value < bounds.hard_lo or base_value > bounds.hard_hi:
                error_rows.append(_row_index(idx))
                error_vals.append(base_value)
            elif base_value < bounds.typical_lo or base_value > bounds.typical_hi:
                warning_rows.append(_row_index(idx))
                warning_vals.append(base_value)

        if error_rows:
            capped, total = cap_rows(error_rows)
            findings.append(
                Finding(
                    check="physical_bounds",
                    severity="error",
                    message=(
                        f"column '{name}' (inferred dimension '{dim}') has values outside "
                        f"the physically possible range [{bounds.hard_lo}, {bounds.hard_hi}]"
                    ),
                    column=name,
                    rows=capped,
                    detail={
                        "dimension": dim,
                        "hard_range": [bounds.hard_lo, bounds.hard_hi],
                        "min_seen": min(error_vals),
                        "max_seen": max(error_vals),
                        "n_rows_affected": total,
                    },
                )
            )
        if warning_rows:
            capped, total = cap_rows(warning_rows)
            findings.append(
                Finding(
                    check="physical_bounds",
                    severity="warning",
                    message=(
                        f"column '{name}' (inferred dimension '{dim}') has values outside "
                        f"the typical range [{bounds.typical_lo}, {bounds.typical_hi}] "
                        "though within physical limits"
                    ),
                    column=name,
                    rows=capped,
                    detail={
                        "dimension": dim,
                        "typical_range": [bounds.typical_lo, bounds.typical_hi],
                        "min_seen": min(warning_vals),
                        "max_seen": max(warning_vals),
                        "n_rows_affected": total,
                    },
                )
            )

    return findings


# --- 3. duplicate rows -------------------------------------------------------- #


def check_duplicate_rows(df: pd.DataFrame, *, outcome_hint: Pattern[str]) -> list[Finding]:
    """Flag exact duplicate rows (WARNING) and outcome-only-differing rows (INFO).

    An exact duplicate (every column equal) is very likely a copy-paste or
    export bug - two "different" measurements cannot be bit-for-bit
    identical across every column including any id/date column, so this is
    a WARNING, not an INFO. Rows equal on every column EXCEPT the ones
    `outcome_hint` matches are a different, benign thing: a genuine
    replicate of the same recipe with a different measured result, which is
    INFO, not a defect. `outcome_hint` is passed in (rather than
    hardcoded) so the caller's `DomainProfile` stays the single source of
    truth for what counts as an outcome column.
    """
    findings: list[Finding] = []
    if df.empty:
        return findings

    full_dup_mask = df.duplicated(keep=False)
    if full_dup_mask.any():
        rows = sorted(int(i) for i in df.index[full_dup_mask])
        capped, total = cap_rows(rows)
        findings.append(
            Finding(
                check="duplicate_rows",
                severity="warning",
                message="exact duplicate rows found (identical across every column)",
                rows=capped,
                detail={"n_rows_affected": total},
            )
        )

    outcome_cols = [c for c in df.columns if outcome_hint.search(str(c))]
    compare_cols = [c for c in df.columns if c not in outcome_cols]
    if compare_cols:
        subset_dup_mask = df.duplicated(subset=compare_cols, keep=False)
        replicate_only_mask = subset_dup_mask & ~full_dup_mask
        if replicate_only_mask.any():
            rows = sorted(int(i) for i in df.index[replicate_only_mask])
            capped, total = cap_rows(rows)
            findings.append(
                Finding(
                    check="duplicate_rows",
                    severity="info",
                    message=(
                        "rows identical on every column except the outcome column(s) - "
                        "these look like replicates of the same condition"
                    ),
                    rows=capped,
                    detail={
                        "outcome_columns": [str(c) for c in outcome_cols],
                        "n_rows_affected": total,
                    },
                )
            )

    return findings


# --- 4. missingness ----------------------------------------------------------- #


def check_missingness(df: pd.DataFrame) -> list[Finding]:
    """Flag sparse columns and sparse rows.

    Per column: >90% blank -> WARNING, the column is effectively unusable;
    50-90% blank -> WARNING, the column is sparse but may still carry
    signal. Per row: rows more than 50% blank across their columns ->
    WARNING (one finding aggregating all such rows, not one per row).
    """
    findings: list[Finding] = []
    n_rows = len(df)
    n_cols = df.shape[1]
    if n_rows == 0 or n_cols == 0:
        return findings

    for col in df.columns:
        name = str(col)
        mask = _nonblank_mask(df[col])
        frac_blank = 1.0 - (int(mask.sum()) / n_rows)
        if frac_blank > 0.90:
            findings.append(
                Finding(
                    check="missingness",
                    severity="warning",
                    message=f"column '{name}' is {frac_blank:.0%} blank - effectively unusable",
                    column=name,
                    detail={"frac_blank": frac_blank},
                )
            )
        elif frac_blank >= 0.50:
            findings.append(
                Finding(
                    check="missingness",
                    severity="warning",
                    message=f"column '{name}' is {frac_blank:.0%} blank - sparse",
                    column=name,
                    detail={"frac_blank": frac_blank},
                )
            )

    nonblank_frame = df.apply(_nonblank_mask)
    row_nonblank_counts = nonblank_frame.sum(axis=1)
    row_frac_blank = 1.0 - (row_nonblank_counts / n_cols)
    bad_rows_mask = row_frac_blank > 0.50
    if bad_rows_mask.any():
        rows = sorted(int(i) for i in df.index[bad_rows_mask])
        capped, total = cap_rows(rows)
        findings.append(
            Finding(
                check="missingness",
                severity="warning",
                message="rows more than 50% blank across their columns",
                rows=capped,
                detail={"n_rows_affected": total},
            )
        )

    return findings


# --- 5. outliers --------------------------------------------------------------- #

# Orders of magnitude a strictly-positive column must span before outliers are
# judged on a log10 scale rather than a linear one. 2.0 (a 100x spread) is a
# convention, not a derived constant, chosen to sit clearly above the spread of
# ordinary process measurements - a titer range, a viability percentage, or an
# endotoxin level rarely covers 100x - and clearly below a designed scale ladder
# or any quantity proportional to it, which covers thousands-fold. See
# `check_outliers` for what goes wrong without this.
_LOG_SCALE_DECADES = 2.0


def check_outliers(df: pd.DataFrame) -> list[Finding]:
    """Flag numeric outliers via a robust MAD-based z-score, |z| > 5.

    `z = 0.6745 * (x - median) / MAD`, deliberately not mean/std: a single
    real outlier inflates the standard deviation enough to mask itself in a
    mean/std z-score, while the median and MAD barely move. Columns whose
    MAD is 0 (constant, or fewer than half the values differ from the
    median) are skipped outright - the guard exists because MAD == 0 makes
    z undefined (division by zero), not because a MAD of 0 is itself a
    problem (that is `check_constant_columns`'s job).

    Operates on `pd.to_numeric` coercion of each column, so a cell holding
    unit-tagged text ("34.6 C") coerces to NaN and is excluded rather than
    compared as a raw number; this check does not need unit-awareness
    because it only cares about a column's numeric spread.

    SCALE ASSUMPTION, and why it is not optional: a MAD z-score presumes the
    column is roughly unimodal on a LINEAR scale. A scale-up dataset breaks that
    assumption hard. Given a designed bioreactor ladder
    (0.01, 0.05, 0.25, 1, 5, 10, 50, 200, 500, 1000, 2000 L) the median is 10
    and the MAD is about 10, so every run at 200 L or above scores |z| > 5 and
    20 of 55 rows get flagged as outliers. They are not outliers, they are the
    experimental design, and the same happens to every column proportional to
    scale (harvest volume, harvest mass). A check that cries wolf on a third of
    a legitimate dataset gets switched off, taking the genuine findings with it.

    So a strictly-positive column spanning at least `_LOG_SCALE_DECADES` orders
    of magnitude is assessed on log10 instead, which is how a multiplicative
    quantity is actually distributed. This is a statement about the data's
    geometry, not a fudge factor: on the ladder above, log10 spacing is roughly
    even, so nothing is flagged, while narrow-range columns (endotoxin spanning
    10x, aggregate percent spanning 5x) stay on the linear scale and keep
    reporting their real outliers. `detail["scale"]` records which scale was
    used, so the client is never left guessing how a value was judged.
    """
    findings: list[Finding] = []

    for col in df.columns:
        name = str(col)
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid = numeric.dropna()
        if len(valid) < 3:
            continue

        # Decide the scale before computing anything, and judge on that scale
        # throughout so median/MAD/z are mutually consistent.
        scale = "linear"
        assessed = valid
        vmin = float(valid.min())
        if vmin > 0.0:
            spread_decades = np.log10(float(valid.max()) / vmin)
            if spread_decades >= _LOG_SCALE_DECADES:
                scale = "log10"
                assessed = np.log10(valid)

        median = float(assessed.median())
        mad = float((assessed - median).abs().median())
        if mad == 0.0:
            continue
        z = 0.6745 * (assessed - median) / mad
        outlier_mask = z.abs() > 5
        if outlier_mask.any():
            rows = sorted(int(i) for i in valid.index[outlier_mask])
            capped, total = cap_rows(rows)
            findings.append(
                Finding(
                    check="outliers",
                    severity="warning",
                    message=(
                        f"column '{name}' has values with MAD-based |z| > 5 "
                        f"(assessed on a {scale} scale)"
                    ),
                    column=name,
                    rows=capped,
                    # median/mad are in the assessed scale's units: for a log10
                    # assessment they are log10 values, not raw ones. Reported
                    # alongside `scale` so the two can never be misread.
                    detail={
                        "median": median,
                        "mad": mad,
                        "scale": scale,
                        "n_rows_affected": total,
                    },
                )
            )

    return findings


# --- 6. provenance metadata ---------------------------------------------------- #

_DATE_RE = re.compile(r"date|timestamp", re.I)
_OPERATOR_RE = re.compile(r"operator|instrument|analyst|technician|equipment", re.I)

# Run/batch identifier detection for provenance. This does NOT reuse the
# caller's `DomainProfile.id_hint`, even though that pattern lists the same
# vocabulary, because the two ask different questions and id_hint is anchored
# (`^(...|run|batch|lot)$`). Anchoring is right for id_hint's own job in
# `_analyze` - deciding whether a column is ENTIRELY an identifier and should be
# dropped from modeling, where matching `batch_titer` would silently discard a
# real measurement. It is wrong here, where the question is merely "does any
# traceability column exist", because every real header is compound: `batch_id`,
# `run_number`, `run_id`, `lot_number`. Reusing id_hint made the gate report "no
# run/batch identifier column found" on a sheet whose first two columns were
# `batch_id` and `run_number` - telling a client their data is untraceable when
# it is fully traceable, which is exactly the kind of false alarm that makes a
# validation report get ignored.
#
# The optional suffix group keeps specificity: `batch`/`batch_id`/`run_number`
# match, while `batch_titer` and `run_duration_days` (measurements that merely
# start with an identifier word) do not.
_RUN_ID_RE = re.compile(
    r"^(run|batch|lot|campaign|experiment|expt|sample|culture|vessel|reactor|bioreactor)"
    r"([ _\-.]?(id|ids|no|num|number|code|label|name|key|ref))?$"
    r"|^(id|uuid|index|well|barcode)$",
    re.I,
)


def check_provenance_metadata(df: pd.DataFrame, id_hint: Pattern[str]) -> list[Finding]:
    """Flag missing run-traceability columns: a run/batch id, a date, an operator/instrument.

    Each missing category gets its own WARNING naming exactly what is absent,
    because a client cannot fix what they cannot see named.

    `id_hint` is accepted for interface compatibility with the caller's
    `DomainProfile` but is deliberately NOT used to answer the identifier
    question - see `_RUN_ID_RE` above for why an anchored role-classification
    pattern gives the wrong answer here.
    """
    findings: list[Finding] = []
    names = [str(c) for c in df.columns]

    if not any(_RUN_ID_RE.match(n.strip()) for n in names):
        findings.append(
            Finding(
                check="provenance_metadata",
                severity="warning",
                message=(
                    "no run/batch identifier column found - results cannot be traced "
                    "back to a physical run"
                ),
                detail={"missing": "run_or_batch_id"},
            )
        )
    if not any(_DATE_RE.search(n) for n in names):
        findings.append(
            Finding(
                check="provenance_metadata",
                severity="warning",
                message="no date column found - results cannot be traced to when the run happened",
                detail={"missing": "date"},
            )
        )
    if not any(_OPERATOR_RE.search(n) for n in names):
        findings.append(
            Finding(
                check="provenance_metadata",
                severity="warning",
                message=(
                    "no operator/instrument column found - results cannot be traced "
                    "to who or what equipment ran the assay"
                ),
                detail={"missing": "operator_or_instrument"},
            )
        )

    return findings


# --- 7. replicate adequacy ------------------------------------------------------ #


def check_replicate_adequacy(
    df: pd.DataFrame, target: str | None, features: list[str] | None
) -> list[Finding]:
    """Flag row-count and replicate-structure problems for fitting a GP.

    `n_rows < 6` is the engine's own hard floor (ERROR - the engine will
    refuse to fit at all). `n_rows < 10 * n_features` is a WARNING: fewer
    than ten rows per feature is under-determined for a Gaussian process in
    practice, even though the engine will still attempt a fit. When `target`
    and `features` are both given, `kalos.core.replicates.noise_report` is
    reused to report the distinct-condition count, how many are replicated,
    and the estimated noise floor / ICC (INFO); zero replicated conditions
    is a further WARNING, because with no replication the assay noise
    cannot be separated from a real effect.
    """
    findings: list[Finding] = []
    n_rows = len(df)
    feats = list(features) if features else []
    n_features = len(feats)

    if n_rows < 6:
        findings.append(
            Finding(
                check="replicate_adequacy",
                severity="error",
                message=f"only {n_rows} row(s); the engine requires at least 6 rows to fit a model",
                detail={"n_rows": n_rows},
            )
        )
    if n_features > 0 and n_rows < 10 * n_features:
        findings.append(
            Finding(
                check="replicate_adequacy",
                severity="warning",
                message=(
                    f"{n_rows} rows for {n_features} feature(s) is under-determined for a "
                    f"GP (want at least {10 * n_features})"
                ),
                detail={"n_rows": n_rows, "n_features": n_features},
            )
        )

    if target is not None and feats and n_rows > 0:
        X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_valid, y_valid = X[valid], y[valid]
        if X_valid.shape[0] >= 1:
            summary = noise_report(X_valid, y_valid)
            findings.append(
                Finding(
                    check="replicate_adequacy",
                    severity="info",
                    message=(
                        f"{summary['n_recipes']} distinct condition(s) found, "
                        f"{summary['n_replicated']} replicated"
                    ),
                    detail=summary,
                )
            )
            if summary["n_recipes"] > 0 and summary["n_replicated"] == 0:
                findings.append(
                    Finding(
                        check="replicate_adequacy",
                        severity="warning",
                        message=(
                            "no condition is replicated - assay noise cannot be separated "
                            "from a real effect"
                        ),
                        detail={"n_recipes": summary["n_recipes"]},
                    )
                )

    return findings


# --- 8. controls present --------------------------------------------------------- #

_CONTROL_RE = re.compile(r"control|blank|reference|baseline", re.I)


def check_controls_present(df: pd.DataFrame) -> list[Finding]:
    """Flag a run sheet with no control/blank/reference/baseline label anywhere.

    Looks for the label both in column names (a column literally called
    "control") and in the values of any object-dtype (categorical/text)
    column, since a run sheet more commonly marks a control row with a
    label value ("control", "blank") in an existing condition column than
    with a dedicated column. Absent either way -> WARNING: without an
    internal reference, batch-to-batch or calibration drift is invisible.
    """
    for col in df.columns:
        if _CONTROL_RE.search(str(col)):
            return []

    for col in df.columns:
        series = df[col]
        # Skip genuinely numeric columns (a control label cannot live in
        # them); everything else - legacy `object` string columns and
        # pandas' newer dedicated string dtypes alike - is scanned by value.
        if pd.api.types.is_numeric_dtype(series):
            continue
        for val in series.dropna():
            if _CONTROL_RE.search(str(val)):
                return []

    return [
        Finding(
            check="controls_present",
            severity="warning",
            message=(
                "no control/blank/reference/baseline label found - no internal "
                "reference to detect batch or calibration drift"
            ),
        )
    ]


# --- 9. constant columns ----------------------------------------------------------- #


def check_constant_columns(df: pd.DataFrame) -> list[Finding]:
    """Flag zero-variance numeric columns (INFO): they carry no signal.

    Operates on `pd.to_numeric` coercion, same rationale as `check_outliers`.
    Requires at least 2 parseable values before calling a column "constant" -
    a column with a single non-blank value and the rest genuinely blank is a
    missingness problem (`check_missingness`'s job), not a constant-value one.
    """
    findings: list[Finding] = []
    for col in df.columns:
        name = str(col)
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid = numeric.dropna()
        if len(valid) < 2:
            continue
        if valid.nunique() == 1:
            findings.append(
                Finding(
                    check="constant_columns",
                    severity="info",
                    message=(
                        f"column '{name}' has zero variance - carries no signal and "
                        "will be dropped downstream"
                    ),
                    column=name,
                    detail={"value": float(valid.iloc[0])},
                )
            )
    return findings


# --- 10. informative missingness (MNAR) ---------------------------------------- #

# A Spearman rho's standard error scales roughly as 1/sqrt(n-3), so below
# about a dozen paired observations even a genuine correlation reads as
# confident nonsense - the number would look decisive while being mostly
# sampling noise. Both the row-level and column-level tests below refuse to
# report anything under this floor rather than manufacture a rho nobody
# should trust.
_MIN_N_FOR_RHO = 12

# How far |rho| has to clear before missingness is called informative rather
# than noise. At ordinary sheet sizes (n ~ 30-100) a null (label-shuffled)
# Spearman rho already wanders past +-0.3 with non-trivial probability by
# chance alone, so 0.35 sits with real margin above that noise floor - while
# staying far below the -0.835 the owner's real Cytena clone-funnel data
# actually showed (voyager-brain-rebuild/docs/ml-review.md, finding P0.1).
# This check exists to catch that class of signal, not to flag every faint
# wobble in a small sheet.
_MISSINGNESS_RHO_WARN = 0.35

# A column must be at least this blank before its missingness indicator is
# tested against the target at all. Below 5% blank, "missing" is a handful of
# rows out of the whole sheet, and any correlation computed from that few
# positive cases is dominated by which specific rows happen to be missing,
# not a real pattern - the same small-n concern as `_MIN_N_FOR_RHO`, applied
# to the minority class of a binary indicator instead of to the whole sample.
_COL_MISSINGNESS_MIN_BLANK_FRAC = 0.05


def check_informative_missingness(df: pd.DataFrame, target: str | None) -> list[Finding]:
    """Flag missingness that CORRELATES WITH THE TARGET (MNAR), not just blank cells.

    `check_missingness` reports how much of a column or row is blank; this asks
    a different question - whether being blank is itself informative, i.e.
    correlated with the outcome, rather than happening at random. On the
    owner's real Cytena clone-funnel data, per-row missing-fraction vs titer
    had Spearman rho = -0.835, and non-producers were 66.7% missing vs 37.0%
    for producers: "no measurement" was the human culling decision on low
    performers, not benign absence. `kalos`'s GP path zero-fills blanks before
    fitting, which is safe under a missing-COMPLETELY-at-random assumption and
    quietly encodes the outcome into the features the moment that assumption
    is false.

    Two rank-based tests (Spearman - neither assumes the target or the
    missing-fraction is normally distributed, and Spearman between a 0/1
    indicator and a continuous variable reduces exactly to the rank-biserial
    correlation, so no separate point-biserial formula needs to be
    maintained):

      - ROW-level: each row's fraction of blank cells across every column
        EXCEPT `target` itself, vs. that row's numeric target value. A
        confidently negative rho means rows with more blanks skew toward
        lower outcomes - "not measured" looking like a proxy for "did not
        perform."
      - COLUMN-level: for every column at least `_COL_MISSINGNESS_MIN_BLANK_FRAC`
        blank, its own 0/1 missingness indicator (1 = blank in that row) vs.
        the target.

    Both require at least `_MIN_N_FOR_RHO` rows with a valid (numeric,
    non-blank) target, and both fire only once |rho| clears
    `_MISSINGNESS_RHO_WARN`. Severity is ALWAYS "warning", never "error": this
    check changes nothing about whether a sheet is usable, only whether
    zero-filling it is safe to do silently - flipping a strict-mode upload
    from accepted to rejected on this finding would be a product decision this
    check has no business making on its own.

    Never crashes and never raises severity beyond warning: no `target` given,
    a `target` that is entirely non-numeric, a constant target (rho is
    undefined - nothing to correlate against), or a frame/target pair with no
    blanks at all in the feature columns each produce an empty finding list,
    not an exception.
    """
    findings: list[Finding] = []
    if target is None or target not in df.columns:
        return findings

    y_all = pd.to_numeric(df[target], errors="coerce")
    valid_target = y_all.notna()
    if int(valid_target.sum()) < _MIN_N_FOR_RHO:
        return findings
    if y_all.loc[valid_target].nunique() < 2:
        return findings  # constant target: rho is undefined, nothing to correlate against

    feature_cols = [c for c in df.columns if c != target]
    if not feature_cols:
        return findings

    nonblank = df[feature_cols].apply(_nonblank_mask)
    if bool(nonblank.to_numpy().all()):
        return findings  # no blanks anywhere in the feature columns: nothing to say

    # --- row-level: fraction of blank feature cells vs. the target -------------- #
    row_frac_blank = 1.0 - (nonblank.sum(axis=1) / len(feature_cols))
    frac_valid = row_frac_blank.loc[valid_target]
    y_valid = y_all.loc[valid_target]
    if frac_valid.nunique() >= 2:
        rho, _p = spearmanr(frac_valid, y_valid, nan_policy="omit")
        if not np.isnan(rho) and abs(rho) >= _MISSINGNESS_RHO_WARN:
            findings.append(
                Finding(
                    check="informative_missingness",
                    severity="warning",
                    message=(
                        f"per-row missing-fraction correlates with target '{target}' "
                        f"(Spearman rho={rho:.2f}, n={int(frac_valid.shape[0])}) - "
                        "missingness looks informative, not random; zero-filling "
                        "before the fit may be encoding the outcome into the features"
                    ),
                    detail={
                        "target": target,
                        "rho": float(rho),
                        "n": int(frac_valid.shape[0]),
                    },
                )
            )

    # --- column-level: each sparse column's own missingness vs. the target ------ #
    for col in feature_cols:
        name = str(col)
        col_nonblank = nonblank[col]
        frac_blank_col = 1.0 - (int(col_nonblank.sum()) / len(df))
        if frac_blank_col < _COL_MISSINGNESS_MIN_BLANK_FRAC or frac_blank_col >= 1.0:
            continue
        indicator = (~col_nonblank.loc[valid_target]).astype(int)
        if len(indicator) < _MIN_N_FOR_RHO or indicator.nunique() < 2:
            continue
        y_here = y_all.loc[valid_target]
        rho, _p = spearmanr(indicator, y_here, nan_policy="omit")
        if not np.isnan(rho) and abs(rho) >= _MISSINGNESS_RHO_WARN:
            findings.append(
                Finding(
                    check="informative_missingness",
                    severity="warning",
                    message=(
                        f"column '{name}' missingness correlates with target '{target}' "
                        f"(Spearman rho={rho:.2f}, {frac_blank_col:.0%} blank, "
                        f"n={len(indicator)}) - whether this cell was measured looks "
                        "informative, not random; zero-filling it may be encoding the "
                        "outcome into the features"
                    ),
                    column=name,
                    detail={
                        "target": target,
                        "rho": float(rho),
                        "frac_blank": frac_blank_col,
                        "n": int(len(indicator)),
                    },
                )
            )

    return findings


# --- 11. constant-within-group columns ------------------------------------------ #


def check_constant_within_group(df: pd.DataFrame, *, group_hint: Pattern[str]) -> list[Finding]:
    """Flag a column that varies globally but is a group-identity proxy in disguise.

    `check_constant_columns` catches a column that is constant EVERYWHERE - no
    signal at all. This catches the sneakier variant: a column that varies
    across the whole sheet (so `check_constant_columns` says nothing) but has
    exactly one value inside every group - so within any single group it
    carries zero information, and whatever it appears to predict is actually
    just "which group this row is in." On the owner's real Cytena
    clone-funnel data, 4 of 6 candidate features had exactly one unique value
    within every campaign; they were campaign identity in disguise, and they
    inflated a within-population score to ~0.86 when the campaign-mean anchor
    alone - which uses no per-clone information whatsoever - already scored
    ~0.75 (voyager-brain-rebuild/docs/ml-review.md, finding P0.2). A model fit
    on such a column mostly learns "which group," not the thing the group
    label is standing in for.

    The group column is detected the same way `kalos.portal.analysis` infers
    it for the live analyze path: the first column whose name matches
    `group_hint` (a `DomainProfile`'s `group_hint` regex, e.g.
    `medium|strain|recipe|batch|campaign|group|lot` for the bioprocess
    profile). No group column present -> nothing to check, empty list.

    A group must have at least 2 non-blank-labeled rows to say anything about
    "varies within it" at all (a singleton group trivially has exactly one
    value in every column, which is not evidence of anything); if every group
    is a singleton, the check has nothing measurable and returns no findings.
    Blank cells never count as "a value" - a column that is blank throughout
    some group is a missingness problem (`check_missingness`'s job), not
    evidence the column is constant there, so that group correctly fails the
    "exactly one unique value" test and the column is not flagged on its
    account.

    WARNING severity: a proxy feature does not make a sheet unusable, but any
    model fit on it needs the caller to know that measured skill may be group
    memorization, not real signal.
    """
    findings: list[Finding] = []
    gcol = next((c for c in df.columns if group_hint.search(str(c))), None)
    if gcol is None:
        return findings

    nonblank = df.apply(_nonblank_mask)
    group_present = nonblank[gcol]
    if not bool(group_present.any()):
        return findings

    group_labels = df.loc[group_present, gcol]
    group_sizes = group_labels.value_counts()
    measurable_groups = set(group_sizes[group_sizes >= 2].index)
    if not measurable_groups:
        return findings  # every group is a singleton: within-group variation is unmeasurable

    in_measurable = group_present & df[gcol].isin(measurable_groups)
    groups_for_rows = df.loc[in_measurable, gcol]
    blanked = df.where(nonblank)  # blank cells -> NaN, matches `_nonblank_mask` everywhere else

    gcol_name = str(gcol)
    for col in df.columns:
        if col == gcol:
            continue
        name = str(col)
        global_values = blanked[col].dropna()
        if global_values.nunique() < 2:
            continue  # already globally constant - check_constant_columns' job, not this one

        within_group_nunique = blanked.loc[in_measurable, col].groupby(groups_for_rows).nunique(
            dropna=True
        )
        # Every measurable group must show EXACTLY one non-blank value (not "at
        # most one": a group where this column is entirely blank has nunique
        # 0, which correctly fails this test rather than looking constant).
        if len(within_group_nunique) == len(measurable_groups) and bool(
            (within_group_nunique == 1).all()
        ):
            findings.append(
                Finding(
                    check="constant_within_group",
                    severity="warning",
                    message=(
                        f"column '{name}' varies globally but has exactly one value "
                        f"within every group of '{gcol_name}' ({len(measurable_groups)} "
                        "group(s) with 2+ rows checked) - looks like a proxy for group "
                        "identity, not a real feature; any predictive skill attributed "
                        "to it may be group memorization"
                    ),
                    column=name,
                    detail={
                        "group_column": gcol_name,
                        "n_groups_checked": len(measurable_groups),
                    },
                )
            )

    return findings
