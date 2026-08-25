"""Tier-1 orientation pre-pass: detect whether an uploaded run sheet is in
standard orientation (rows = runs/experiments, columns = parameters) or
transposed (rows = parameters, columns = runs/experiments), and normalize a
confidently-transposed sheet back to standard orientation.

This has to run BEFORE unit normalization or the validation gate can mean
anything - a rule checking "is this column numeric" is meaningless if the
column is actually a row of parameter labels because the sheet came in
transposed (see `docs/ingestion-architecture.md` in the kalos-transition
repo, "Tier 1: deterministic pre-passes"). Fully deterministic - no LLM
call is ever appropriate here, and this module never raises on data: a
confused heuristic degrades to "ambiguous, leave it alone", never to a
crash or a silent guess.

Provenance: adapted from `ports/csv-orientation/csv_adapter.py`
(`CSVAdapter.detect_format` / `_process_transposed_format`) in the
itsbrendandang/kalos-transition repo, itself ported near-verbatim from
`itsbrendandang/voyPlot-main` @ `722e5520944535c20b5ccfc4e75729359cedabf6`
(see that port's own PROVENANCE.md for the client-identifier scrub log on
`BIOPROCESS_KEYWORDS`, kept unchanged below - already generic). Adapted
here into kalos house style: a typed `OrientationReport` dataclass, a
three-way standard/transposed/AMBIGUOUS decision (the source used a bare
`score > 3` boolean), and defensive `try/except` wrapping so this function
can never raise on malformed input - the source's `detect_format` had no
such guard. The source's objective/target-row detection was NOT ported:
target/feature role assignment is `kalos.normalize.synonyms.guess_role`'s
job, not this pre-pass's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Orientation = Literal["standard", "transposed", "ambiguous"]

# Generic bioprocess/fermentation vocabulary used ONLY to score whether the
# first column reads like a list of parameter names (a transposed-sheet
# signal) - unrelated to, and not a substitute for, the canonical
# name/synonym tables in `synonyms.py`. Ported unchanged from
# `csv_adapter.py`'s `BIOPROCESS_KEYWORDS` (already scrubbed of
# client-specific terms per that port's PROVENANCE.md).
_BIOPROCESS_KEYWORDS: tuple[str, ...] = (
    "scale", "ph", "temp", "temperature", "time", "feed", "final", "activity",
    "batch", "induction", "glycerol", "methanol", "antifoam", "foam",
    "broth", "withdrawals", "od600", "titer", "method", "page",
    "seed", "fermentor", "shift", "duration", "setpoint",
    "yield", "output", "concentration", "production", "biomass",
)
_TYPE_COLUMN_INDICATORS: tuple[str, ...] = ("feature", "objective", "ignore", "target")
_STANDARD_HEADER_HINTS: tuple[str, ...] = ("param", "feature", "variable", "factor")

# Decision thresholds on the summed heuristic score (see `_score_signals`).
# Scores strictly above `_TRANSPOSED_THRESHOLD` are confident "transposed";
# scores strictly below `_STANDARD_THRESHOLD` are confident "standard";
# everything in between is "ambiguous" and is deliberately never guessed -
# a sheet a pre-pass this heuristic cannot confidently classify must be left
# untouched, not silently transposed (or not) on a coin flip. The source
# adapter used a single threshold (`score > 3` => transposed, else
# standard); this splits that into a symmetric band around zero so a
# standard-leaning-but-uncertain sheet does not get force-classified either
# way.
_TRANSPOSED_THRESHOLD = 3.0
_STANDARD_THRESHOLD = -3.0
# Normalizes the summed score into a 0-1 confidence; matches the source
# adapter's own `min(abs(score) / 10, 1.0)` scale.
_CONFIDENCE_SCALE = 10.0
_ROW_SAMPLE = 15  # first-column rows sampled for keyword matches
_SHAPE_SAMPLE = 20  # rows/columns sampled for the numeric-density signals
_NUMERIC_DENSITY_THRESHOLD = 0.7  # fraction of cells that must parse numeric
# If the first column is itself mostly numeric DATA, it cannot be a row of
# parameter LABELS (a transposed sheet's defining feature), so a first
# column at or above this parse rate short-circuits straight to "standard"
# - this is a harder, more reliable rule than any of the scored signals
# below, and it is what keeps a perfectly ordinary tall-and-narrow standard
# sheet (few numeric feature columns, many run rows - an extremely common
# real shape) from ever being scored at all: the shape/numeric-density
# heuristics below are noisy on exactly that shape (see
# `_score_signals`'s docstring).
_FIRST_COLUMN_NUMERIC_STANDARD_RATE = 0.7


@dataclass(frozen=True)
class OrientationReport:
    """The pre-pass's verdict for one uploaded frame.

    `signals` is the per-heuristic contribution to the summed score (keyed
    by heuristic name, e.g. `"first_column_keywords"`), kept for provenance -
    "why did the pre-pass decide this" should be answerable from the report
    itself, not by re-running the heuristic. `normalized_frame` is populated
    only when `orientation == "transposed"` AND the transpose produced at
    least one numeric-enough row to keep; it is `None` for "standard" and
    "ambiguous" (callers must never invent a normalized frame for either).
    """

    orientation: Orientation
    confidence: float
    signals: dict[str, float] = field(default_factory=dict)
    normalized_frame: pd.DataFrame | None = None


def _score_signals(df: pd.DataFrame) -> dict[str, float]:
    """Compute each heuristic's individual contribution to the transposed-
    ness score. Positive values push toward "transposed", negative toward
    "standard". Every heuristic here is defensive against short/odd-shaped
    frames - `detect_orientation` also wraps the whole call in a
    `try/except`, but each signal degrades to "no contribution" on its own
    rather than relying solely on the outer guard.
    """
    signals: dict[str, float] = {}
    if df.shape[0] == 0 or df.shape[1] == 0:
        return signals

    first_col_values = df.iloc[:, 0].astype(str).str.lower()

    # 1. bioprocess-keyword matches in the first column (first N rows) - a
    #    first column full of parameter names is a transposed-sheet signal.
    sample = first_col_values.iloc[:_ROW_SAMPLE]
    bioprocess_matches = sum(
        any(keyword in val for keyword in _BIOPROCESS_KEYWORDS) for val in sample
    )
    if bioprocess_matches:
        signals["first_column_keywords"] = float(min(bioprocess_matches * 3, 15))

    # 2. "type"-like indicator words anywhere in the first column.
    joined = " ".join(first_col_values.tolist())
    if any(indicator in joined for indicator in _TYPE_COLUMN_INDICATORS):
        signals["type_column_indicator"] = 5.0

    # 3. an actual 'type' HEADER column is a strong standard-format signal
    #    (a per-row type/category column belongs in standard orientation).
    if "type" in [str(c).lower() for c in df.columns]:
        signals["type_header_present"] = -10.0

    # 4. column names (other than the first) that read like standard
    #    per-feature headers push toward "standard".
    first_col_name = df.columns[0]
    standard_indicators = sum(
        1
        for col in df.columns
        if col != first_col_name and any(h in str(col).lower() for h in _STANDARD_HEADER_HINTS)
    )
    if standard_indicators:
        signals["standard_header_hints"] = -float(standard_indicators * 2)

    # 5. numeric density by row vs. by column (excluding the first
    #    column/row respectively, which may be labels/headers).
    numeric_rows = 0
    if df.shape[1] > 1:
        for idx in range(min(len(df), _SHAPE_SAMPLE)):
            row_data = df.iloc[idx, 1:]
            if pd.to_numeric(row_data, errors="coerce").notna().sum() > len(row_data) * _NUMERIC_DENSITY_THRESHOLD:
                numeric_rows += 1
    numeric_cols = 0
    if len(df) > 1:
        for col_idx in range(1, min(df.shape[1], _SHAPE_SAMPLE)):
            col_data = df.iloc[1:, col_idx]
            if pd.to_numeric(col_data, errors="coerce").notna().sum() > len(col_data) * _NUMERIC_DENSITY_THRESHOLD:
                numeric_cols += 1
    if numeric_rows > numeric_cols and df.shape[1] < len(df):
        signals["numeric_density"] = 2.0
    elif numeric_cols > numeric_rows:
        signals["numeric_density"] = -2.0

    # 6. wide-vs-tall shape.
    ratio = df.shape[1] / max(len(df), 1)
    if ratio > 2.0:
        signals["shape_ratio"] = -4.0
    elif ratio > 1.5:
        signals["shape_ratio"] = -2.0
    elif len(df) > df.shape[1] * 1.5:
        signals["shape_ratio"] = 2.0

    return signals


def _normalize_transposed(df: pd.DataFrame) -> pd.DataFrame | None:
    """Transpose `df` (rows = parameters, columns = runs) to standard
    orientation (rows = runs, columns = parameters).

    Mirrors `csv_adapter.py`'s `_process_transposed_format`: rows that are
    not >=70% numeric across the run columns are dropped (label/annotation
    rows, not data), and duplicate parameter names are disambiguated with a
    `__2`, `__3`, ... suffix so the transpose never silently collides two
    parameters into one column. Returns `None` (never an empty/garbage
    frame) if no row survives the numeric filter - `detect_orientation`
    treats that as "ambiguous" rather than emitting a frame with zero
    parameters.
    """
    feature_names = df.iloc[:, 0].astype(str).tolist()
    experiment_data = df.iloc[:, 1:]

    keep_mask = [
        bool(len(row) and pd.to_numeric(row, errors="coerce").notna().sum() >= len(row) * _NUMERIC_DENSITY_THRESHOLD)
        for _, row in experiment_data.iterrows()
    ]
    if not any(keep_mask):
        return None

    kept_names = [name for name, keep in zip(feature_names, keep_mask) if keep]
    kept_rows = experiment_data.loc[[i for i, keep in zip(experiment_data.index, keep_mask) if keep]]

    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for name in kept_names:
        if name in seen:
            seen[name] += 1
            unique_names.append(f"{name}__{seen[name]}")
        else:
            seen[name] = 0
            unique_names.append(name)

    transposed = kept_rows.T
    transposed.columns = pd.Index(unique_names)
    # Built column-by-column (rather than a single `.apply`) so the result's
    # static type is unambiguously `DataFrame` - `DataFrame.apply` returning
    # a per-column-transformed frame is, per pandas-stubs, ambiguous with
    # the Series-returning overload.
    numeric_columns = {name: pd.to_numeric(transposed[name], errors="coerce") for name in unique_names}
    return pd.DataFrame(numeric_columns).reset_index(drop=True)


def _first_column_numeric_rate(df: pd.DataFrame) -> float:
    """Fraction of the first column's non-null values that parse as plain
    numbers. `0.0` for an all-null or empty first column (never divides by
    zero, never treats "no data" as "numeric")."""
    first_col = df.iloc[:, 0]
    non_null = first_col.dropna()
    if len(non_null) == 0:
        return 0.0
    numeric = pd.to_numeric(non_null, errors="coerce")
    return float(numeric.notna().sum()) / len(non_null)


def detect_orientation(df: pd.DataFrame) -> OrientationReport:
    """Detect `df`'s orientation and, if confidently transposed, normalize it.

    Never raises: any failure (unexpected dtypes, a malformed frame the
    heuristics choke on) degrades to an "ambiguous", zero-confidence report
    rather than propagating an exception into the upload path. A frame with
    fewer than 2 rows or 2 columns is reported "standard" outright - there
    is not enough structure for the transposed heuristics to say anything,
    and the safe default is to leave it alone.

    Before scoring anything, a first column that is itself mostly numeric
    DATA (>= `_FIRST_COLUMN_NUMERIC_STANDARD_RATE`) short-circuits to
    "standard": a transposed sheet's first column holds parameter LABELS
    (text), so a numeric first column cannot be one, regardless of what the
    scored signals below would otherwise say. This is the guard that keeps
    an ordinary tall-and-narrow standard sheet (few numeric feature columns,
    many run rows) from ever being misread as transposed by the
    shape/numeric-density signals, which are noisy on exactly that common
    shape (more sampled rows "look numeric" than columns simply because
    there are few columns, not because the sheet is transposed).
    """
    try:
        if df is None or len(df) < 2 or df.shape[1] < 2:
            return OrientationReport(orientation="standard", confidence=0.0, signals={})

        first_col_numeric_rate = _first_column_numeric_rate(df)
        if first_col_numeric_rate >= _FIRST_COLUMN_NUMERIC_STANDARD_RATE:
            return OrientationReport(
                orientation="standard",
                confidence=first_col_numeric_rate,
                signals={"first_column_numeric_rate": first_col_numeric_rate},
            )

        signals = _score_signals(df)
        score = sum(signals.values())
        confidence = min(abs(score) / _CONFIDENCE_SCALE, 1.0)

        # A transposed verdict additionally REQUIRES a positive keyword/type
        # signal - i.e. the first column actually reads like parameter
        # names, not just "shaped like" a transposed sheet. Shape and
        # numeric-density alone (e.g. a narrow standard sheet with a
        # non-numeric id/label first column) must never be sufficient on
        # their own; they only amplify a keyword signal that is already
        # there.
        has_label_signal = signals.get("first_column_keywords", 0.0) > 0 or signals.get(
            "type_column_indicator", 0.0
        ) > 0
        if score > _TRANSPOSED_THRESHOLD and has_label_signal:
            normalized = _normalize_transposed(df)
            if normalized is None:
                return OrientationReport(orientation="ambiguous", confidence=confidence, signals=signals)
            return OrientationReport(
                orientation="transposed",
                confidence=confidence,
                signals=signals,
                normalized_frame=normalized,
            )
        if score < _STANDARD_THRESHOLD:
            return OrientationReport(orientation="standard", confidence=confidence, signals=signals)
        return OrientationReport(orientation="ambiguous", confidence=confidence, signals=signals)
    except Exception:  # noqa: BLE001 - must never raise on data, only degrade
        return OrientationReport(orientation="ambiguous", confidence=0.0, signals={})


__all__ = ["Orientation", "OrientationReport", "detect_orientation"]
