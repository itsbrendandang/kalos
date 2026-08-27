"""Hierarchical extraction-target schema for fermentation PDF reports, plus
the bridge from a filled schema instance to a tidy `pandas.DataFrame` that
`kalos.validation.validate_frame` can consume.

Provenance
----------
Adapted from `ports/pdf-extraction/fermentation_schema.py` in the
`itsbrendandang/kalos-transition` repo (ultimately sourced from
`itsbrendandang/Templatinizer` @ `f22e527891febf140b563c412bee7cbe94326c58`;
see that port's `PROVENANCE.md` for the full chain and the client-identity
scrub log). The dataclass hierarchy below is a deliberate adaptation, not a
verbatim port:

  - `Measurement`/`Parameter`/`BatchSummary`/`FlaskStage`/`BioreactorRun`/
    `DataPoint`/`TimeSeries`/`AnalyticalResult`/`DocumentMetadata` are kept -
    this is the shape both the deterministic table parser (`pdf.py`) and the
    LLM tier (`llm.py`) fill.
  - `SDSPageLane`/`SDSPageResult` (gel-electrophoresis-lane-specific) and the
    `DataType`/`UnitType` enums are DROPPED: no code in this package produces
    or consumes them, and `UnitType` duplicates the canonical unit registry
    kalos already owns in `kalos.normalize.units` - keeping a second, unused
    unit vocabulary would be a maintenance trap, not fidelity.
  - The port's `FermentationDataSchema` manager class (JSON
    file read/write, an `ml_metadata` bookkeeping dict) is replaced by
    `FermentationRecord` below - a single dataclass per extracted document,
    matching how this package actually produces one record per PDF, plus
    `records_to_frame`, the frame bridge the port's PROVENANCE.md calls for
    ("graduation criteria" #2) but does not itself implement.
  - `BatchSummary`'s six measurement fields are ALL optional here
    (`Measurement | None`, default `None`), where the port's version required
    all six. The port's own `extraction_pipeline.py` filled every one of them
    with a hardcoded `Measurement(value=0.0, ...)` placeholder regardless of
    what the table actually said - exactly the bug its `PROVENANCE.md` names
    as needing a real implementation before graduation ("placeholder
    extraction methods... need real implementations"). A field kalos could
    not actually recover must be `None`, not a fabricated zero - this
    package's whole extraction contract is "never invented" (see
    `kalos/ingest/api.py`'s module docstring), and `Measurement(0.0, ...)`
    for a titer nobody measured is exactly the kind of invented data that
    contract exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from kalos.normalize import units

# --- measurement primitives -------------------------------------------------- #


@dataclass
class Measurement:
    """A single measured or reported value: what it is, in what unit, and how
    confident/traceable it is. `method` records how the value was obtained
    (e.g. "pdfplumber_table", "llm_ollama") - the per-field half of this
    package's provenance discipline; the per-document half lives on
    `DocumentMetadata` and `ExtractionResult.provenance` (`api.py`)."""

    value: float
    unit: str
    uncertainty: float | None = None
    detection_limit: float | None = None
    method: str | None = None
    timestamp: str | None = None


@dataclass
class Parameter:
    """A process parameter (a run condition, not a measured outcome) - e.g.
    temperature setpoint, agitation. `target`/`min_value`/`max_value` describe
    an operating range when the source document states one; all three are
    `None` when the document only reports a single observed value."""

    name: str
    value: float
    unit: str
    target: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    description: str | None = None


@dataclass
class DataPoint:
    """One point of a time series: elapsed time plus the measured value."""

    time: float
    value: float
    unit: str
    uncertainty: float | None = None


# --- batch / run shapes ------------------------------------------------------ #


@dataclass
class BatchSummary:
    """Summary metrics for one batch/run, as commonly tabulated in a
    fermentation summary report. Every field is optional (see module
    docstring) - a field the extraction pipeline could not recover is `None`,
    never a placeholder value."""

    yield_value: Measurement | None = None
    duration: Measurement | None = None
    final_od: Measurement | None = None
    final_biomass: Measurement | None = None
    final_titer: Measurement | None = None
    productivity: Measurement | None = None
    specific_productivity: Measurement | None = None


# The six fields `pdf.py`'s confidence score is computed over. `specific_productivity`
# is excluded, matching the port's own treatment of it as the one genuinely-optional
# BatchSummary field (present in some reports, absent from most).
BATCH_SUMMARY_CORE_FIELDS: tuple[str, ...] = (
    "yield_value",
    "duration",
    "final_od",
    "final_biomass",
    "final_titer",
    "productivity",
)


@dataclass
class FlaskStage:
    """Data for one shake-flask cultivation stage (e.g. "veg1", "veg2")."""

    stage_name: str
    conditions: list[Parameter]
    results: list[Measurement]
    duration: Measurement
    notes: str | None = None


@dataclass
class BioreactorRun:
    """Data for one bioreactor run: scale, conditions, and its `BatchSummary`."""

    run_id: str
    scale: Measurement
    vessel_type: str
    conditions: list[Parameter]
    inoculation: dict[str, Any]
    duration: Measurement
    results: BatchSummary
    notes: str | None = None


@dataclass
class TimeSeries:
    """A named time series (e.g. dissolved oxygen over the run), sourced from
    one table or figure."""

    source: str
    run_id: str
    parameter: str
    unit: str
    data_points: list[DataPoint]
    sampling_interval: Measurement | None = None
    notes: str | None = None


@dataclass
class AnalyticalResult:
    """Results from one analytical method run (HPLC, GC, ...) on one sample."""

    method: str
    sample_id: str
    timepoint: Measurement
    parameters: list[Parameter]
    results: list[Measurement]
    chromatogram_path: str | None = None
    notes: str | None = None


@dataclass
class FigureData:
    """Metadata (and, where digitized, extracted series) for one figure."""

    figure_number: str
    caption: str
    figure_type: str
    page_number: int | None = None
    extracted_series: list[TimeSeries] = field(default_factory=list)
    image_analysis: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentMetadata:
    """Metadata about the source PDF itself, not its content.

    `version`/`approval_status` default to `None` rather than the port's
    hardcoded `"v001"`/`"Approved"` - those are workflow facts a filename
    heuristic cannot actually know for an arbitrary document; a caller who
    does know them passes them explicitly (`api.extract_run_document`'s
    kwargs), everyone else gets an honest `None` rather than an invented
    default.
    """

    title: str
    document_id: str
    extraction_date: str
    source_file: str
    md5_hash: str
    total_pages: int | None = None
    version: str | None = None
    approval_status: str | None = None


@dataclass
class FermentationRecord:
    """One extracted document: its metadata, its `BatchSummary`, and whatever
    process `Parameter`s (e.g. temperature setpoint) were recovered alongside
    it. `records_to_frame` is the bridge from a list of these to the tidy
    frame `kalos.validation.validate_frame` expects - see that function's
    docstring for the exact column-naming/unit-conversion rules.
    """

    document_metadata: DocumentMetadata
    batch_summary: BatchSummary
    batch_id: str
    date: str | None = None
    operator: str | None = None
    condition: str = "test"
    conditions: list[Parameter] = field(default_factory=list)


# --- schema -> tidy frame bridge --------------------------------------------- #

# BatchSummary field -> the DataFrame column's BASE name (before the
# unit-derived canonical suffix from `kalos.normalize.units.canonical_suffix`
# is appended). Chosen, where possible, to also match
# `kalos.validation.bounds.infer_dimension`'s header regexes, so a value
# `validate_frame` can meaningfully bounds-check gets bounds-checked (e.g.
# "titer_g_l" matches the `concentration` dimension's "titer" keyword;
# "od600" matches `\bod\d*\b` - it must NOT carry a leading `_` or the token
# boundary the regex requires disappears, see `infer_dimension`'s docstring).
# A field whose base name matches no known dimension (e.g. "biomass_g_l",
# "productivity") simply gets no bounds check - `infer_dimension`'s own
# contract for an unrecognized header is silence, not a guess, and this
# bridge relies on that rather than working around it.
_FIELD_BASE_NAME: dict[str, str] = {
    "yield_value": "yield",
    "duration": "duration",
    "final_od": "od600",
    "final_biomass": "biomass",
    "final_titer": "titer",
    "productivity": "productivity",
    "specific_productivity": "specific_productivity",
}


def _measurement_column(field_name: str, measurement: Measurement | None) -> tuple[str, float] | None:
    """`(column_name, base_unit_value)` for one `BatchSummary` field, or
    `None` if the field is unresolved (`measurement is None`) - an
    unresolved field contributes no column at all, rather than a `NaN` cell,
    so a record where every field was recovered produces no missingness
    finding purely as an artifact of this bridge.
    """
    if measurement is None:
        return None
    base_value, suffix = units.convert(float(measurement.value), measurement.unit)
    return f"{_FIELD_BASE_NAME[field_name]}{suffix}", base_value


def _parameter_column(param: Parameter) -> tuple[str, float]:
    """`(column_name, base_unit_value)` for one process `Parameter` (e.g. a
    temperature setpoint), analogous to `_measurement_column` but for the
    `FermentationRecord.conditions` list rather than a fixed `BatchSummary`
    field. The base name is the parameter's own `name`, snake-cased.
    """
    base_name = param.name.strip().lower().replace(" ", "_").replace("-", "_")
    base_value, suffix = units.convert(float(param.value), param.unit)
    return f"{base_name}{suffix}", base_value


def records_to_frame(records: Sequence[FermentationRecord]) -> pd.DataFrame:
    """Bridge a list of `FermentationRecord`s into one tidy `pandas.DataFrame`,
    one row per record, in the unit conventions `kalos.normalize.units` owns.

    Every `BatchSummary` measurement and every process `Parameter` is
    converted to its dimension's BASE unit via `units.convert` (never left in
    whatever unit the source document happened to use) and named with that
    unit's canonical suffix, exactly like every other numeric column
    `kalos.normalize` produces - a frame built here and a frame built from a
    normalized CSV upload are indistinguishable to `validate_frame`, which is
    the point: the same validation gate, the same unit conventions, for every
    ingestion path (see `docs/ingestion-architecture.md` in the
    kalos-transition repo). `batch_id`/`date`/`operator`/`condition` are
    carried through as plain columns for `validate_frame`'s provenance and
    controls-present checks. An unresolved `BatchSummary` field is simply
    absent from a record's row (see `_measurement_column`); with a ragged set
    of resolved fields across records, `pandas` fills the gaps with `NaN`,
    which is an honest representation of "this record didn't have it" -
    `validate_frame`'s `check_missingness` is the right place to react to
    that, not this bridge.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "batch_id": record.batch_id,
            "date": record.date,
            "operator": record.operator,
            "condition": record.condition,
        }
        for field_name in (*BATCH_SUMMARY_CORE_FIELDS, "specific_productivity"):
            measurement = getattr(record.batch_summary, field_name)
            entry = _measurement_column(field_name, measurement)
            if entry is not None:
                column_name, value = entry
                row[column_name] = value
        for param in record.conditions:
            column_name, value = _parameter_column(param)
            row[column_name] = value
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "Measurement",
    "Parameter",
    "DataPoint",
    "BatchSummary",
    "BATCH_SUMMARY_CORE_FIELDS",
    "FlaskStage",
    "BioreactorRun",
    "TimeSeries",
    "AnalyticalResult",
    "FigureData",
    "DocumentMetadata",
    "FermentationRecord",
    "records_to_frame",
]
