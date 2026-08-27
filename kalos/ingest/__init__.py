"""PDF report ingestion: tier 3 of kalos's ingestion architecture.

`docs/ingestion-architecture.md` (`itsbrendandang/kalos-transition` repo)
defines the three-tier contract every ingestion path in kalos follows -
deterministic pre-pass, structured extractors for known formats, LLM only as
a last resort for unstructured input, everything validated by
`kalos.validation.validate_frame`. This package is tier 3's PDF path:
`pdf.py` is the deterministic table-extraction step, `llm.py` is the
last-resort LLM fallback (self-hosted Ollama only, schema-constrained,
never the primary path), `schema.py` is the hierarchical extraction-target
schema plus the bridge to a validate-able frame, and `api.py`'s
`extract_run_document` ties all three together.

Import-light by design: importing `kalos.ingest` (or any module in it) never
requires `pdfplumber` - that is the OPTIONAL `pdf` extra
(`pip install kalos[pdf]`), needed only to actually CALL the extraction
functions (`pdf.extract_tables`, `pdf.extract_batch_summary`,
`api.extract_run_document`, ...), which raise a clear `ImportError` naming
the extra when it is missing rather than failing to import at all. The LLM
tier adds no dependency beyond kalos's existing `kalos.normalize` provider
seam (stdlib `urllib` only).
"""
from __future__ import annotations

from .api import ExtractionResult, extract_run_document
from .schema import (
    AnalyticalResult,
    BatchSummary,
    BioreactorRun,
    DataPoint,
    DocumentMetadata,
    FermentationRecord,
    FigureData,
    FlaskStage,
    Measurement,
    Parameter,
    TimeSeries,
    records_to_frame,
)

__all__ = [
    "ExtractionResult",
    "extract_run_document",
    "Measurement",
    "Parameter",
    "DataPoint",
    "BatchSummary",
    "FlaskStage",
    "BioreactorRun",
    "TimeSeries",
    "AnalyticalResult",
    "FigureData",
    "DocumentMetadata",
    "FermentationRecord",
    "records_to_frame",
]
