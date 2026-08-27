"""Tier 3's deterministic pre-pass: table extraction from fermentation-report
PDFs, before any LLM is involved (see `docs/ingestion-architecture.md` in the
`itsbrendandang/kalos-transition` repo for the tier contract this package
implements: "deterministic first, LLM last, everything validated").

Provenance
----------
Cleaned up from `ports/pdf-extraction/extraction_pipeline.py` in the
`itsbrendandang/kalos-transition` repo (ultimately `itsbrendandang/Templatinizer`
@ `f22e527891febf140b563c412bee7cbe94326c58`; see that port's `PROVENANCE.md`).
Deliberate departures from the port, each one because the port's own
`PROVENANCE.md` names it as unfinished or because kalos's conventions differ:

  - camelot + OpenCV/pytesseract (OCR) are DROPPED. The port's
    `extraction_pipeline.py` uses camelot for table extraction and
    pytesseract/OpenCV only for a `_detect_and_extract_figures` method that
    is a documented no-op stub in the port itself ("placeholder
    implementation" - never actually extracts anything). camelot also needs
    a system Ghostscript binary (`gs`) outside anything `pip`/`uv` can
    install, and `camelot-py[cv]` pulls in `opencv-python` on top of that -
    a materially heavier and more fragile dependency for a capability
    (figure OCR) this package does not implement. `pdfplumber` alone (already
    a port dependency, used there only for text) has its own
    `page.extract_tables()`, needs no system binary, and is what this module
    uses for both text and tables. See `pyproject.toml`'s `pdf` extra for the
    installed-size comparison that also factored into this call.
  - The port's `_extract_batch_summary`/`_is_batch_summary_table`/etc. are
    STUB implementations - `_extract_batch_summary` returns a `BatchSummary`
    hardcoded to `Measurement(value=0.0, ...)` for all six fields regardless
    of the table's actual content (documented inline in the port as a
    "simplified extraction" / "Placeholder"). `_parse_batch_summary_rows`
    below is a genuine implementation: it reads a two- or three-column
    "Parameter / Value[/ Unit]" table (the shape a fermentation summary
    report's batch-summary table actually has) and maps each row to a
    `BatchSummary` field by keyword.
  - `Optional[dep]` import guards are kept (pdfplumber is optional, matching
    the port's own `PDFPLUMBER_AVAILABLE` pattern) but raise a clear,
    actionable `ImportError` on first use instead of the port's
    print-a-warning-and-silently-return-empty behavior - a caller of
    `kalos.ingest` should never get an empty result and have to guess why;
    `import kalos.ingest.pdf` itself still succeeds either way (see
    `_require_pdfplumber`).

Never raises on DATA (a malformed table row is skipped, not fatal) - the same
discipline `kalos.validation.validate_frame` holds itself to. It DOES raise on
a missing optional dependency (`_require_pdfplumber`) and on a genuinely
unreadable file (bad path, corrupt PDF) - those are caller/environment
problems, not data-quality ones, and hiding them would be worse than
`validate_frame`'s original "crashes on the data it exists to catch" failure
mode it was built to avoid.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .schema import BATCH_SUMMARY_CORE_FIELDS, BatchSummary, DocumentMetadata, Measurement, Parameter
from kalos.normalize.units import parse_value

try:
    import pdfplumber
except ImportError:  # pragma: no cover - exercised via the mocked-absent tests
    pdfplumber = None  # type: ignore[assignment]

_EXTRA_INSTALL_HINT = "pip install kalos[pdf]"


def _require_pdfplumber() -> None:
    """Raise a clear, actionable error when `pdfplumber` is not installed.

    Every public function in this module that needs `pdfplumber` calls this
    first, so `import kalos.ingest.pdf` (and `import kalos.ingest`) always
    succeeds - only USING the deterministic tier requires the `pdf` extra.
    """
    if pdfplumber is None:
        raise ImportError(
            "kalos.ingest.pdf requires the optional 'pdfplumber' dependency "
            f"for PDF table extraction. Install it with: {_EXTRA_INSTALL_HINT}"
        )


# --- raw table extraction ----------------------------------------------------- #


@dataclass
class ExtractedTable:
    """One table as pdfplumber found it: which page, and its raw string cells
    (a cell pdfplumber could not read is `None`, never fabricated as "")."""

    page: int
    rows: list[list[str | None]]


def extract_tables(path: Path) -> list[ExtractedTable]:
    """Extract every table on every page of `path` via `pdfplumber`.

    Requires the `pdf` extra (`_require_pdfplumber`). A page with no
    detectable table contributes nothing (not an error) - a report entirely
    made of prose, or a page pdfplumber's line-based table finder cannot
    parse, is a normal input, not a failure.
    """
    _require_pdfplumber()
    tables: list[ExtractedTable] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            for raw_table in page.extract_tables():
                if raw_table:
                    tables.append(ExtractedTable(page=page.page_number, rows=raw_table))
    return tables


def extract_full_text(path: Path) -> str:
    """Extract all page text, concatenated with blank-line separators.

    Requires the `pdf` extra. Used to build the LLM tier's prompt context
    (`api.py`) when a field cannot be recovered from any table - never used
    by the deterministic tier itself, which only reads table cells.
    """
    _require_pdfplumber()
    parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def page_count(path: Path) -> int:
    """Number of pages in `path`. Requires the `pdf` extra."""
    _require_pdfplumber()
    with pdfplumber.open(str(path)) as pdf:
        return len(pdf.pages)


# --- batch-summary table parsing ---------------------------------------------- #

# Row-label keyword -> BatchSummary field name. Checked in order, first match
# wins (a small, data-driven table rather than an if/elif chain, matching
# `kalos.validation.bounds._DIMENSION_HINTS`'s own style). "specific
# productivity" is checked before the bare "productivity" fallback so it is
# not swallowed by the broader pattern.
_ROW_LABEL_FIELDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("specific_productivity", re.compile(r"specific.?productivity", re.I)),
    ("productivity", re.compile(r"productivity", re.I)),
    ("final_titer", re.compile(r"titer|titre", re.I)),
    ("final_od", re.compile(r"\bod\d*\b|optical.?density", re.I)),
    ("final_biomass", re.compile(r"biomass", re.I)),
    ("duration", re.compile(r"duration", re.I)),
    ("yield_value", re.compile(r"\byield\b", re.I)),
)


def _match_field(label: str) -> str | None:
    """Map one table row's label cell to a `BatchSummary` field name, or
    `None` if it names something this parser does not recognize (e.g. a
    row-header repeat, a units-legend row) - an unrecognized row is skipped,
    not an error."""
    for field_name, pattern in _ROW_LABEL_FIELDS:
        if pattern.search(label):
            return field_name
    return None


def _parse_row_value(row: list[str | None]) -> tuple[float, str] | None:
    """`(value, unit)` for one table row, or `None` if the row's value cell
    is not a parseable number.

    Handles both table shapes this parser accepts:
      - three-plus columns: `[label, value, unit, ...]`, value and unit in
        separate cells (e.g. `["Final Titer", "5.8", "g/L"]`).
      - two columns: `[label, value]`, with the unit embedded in the value
        cell (e.g. `["Final Titer", "5.8 g/L"]`) - reuses
        `kalos.normalize.units.parse_value`, the same messy-cell parser
        `kalos.normalize` itself uses, rather than duplicating that logic.
    """
    if len(row) >= 3 and row[1] is not None and row[2] is not None:
        try:
            return float(str(row[1]).strip()), str(row[2]).strip()
        except ValueError:
            pass  # fall through to the embedded-unit parse below
    if len(row) >= 2 and row[1] is not None:
        value, unit_token = parse_value(str(row[1]))
        if value is not None:
            return value, (unit_token or "")
    return None


@dataclass
class DeterministicResult:
    """What the deterministic tier recovered from one PDF's batch-summary
    table, plus the provenance `api.extract_run_document` reports.

    `resolved_fields`/`unresolved_fields` partition
    `schema.BATCH_SUMMARY_CORE_FIELDS` exactly - together they always list
    all six, so a caller never has to guess whether a field missing from
    both lists means "not attempted" (it can't; every core field is
    classified one way or the other).
    """

    batch_summary: BatchSummary
    conditions: list[Parameter] = field(default_factory=list)
    resolved_fields: list[str] = field(default_factory=list)
    unresolved_fields: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    extractor: str = "pdfplumber_table"
    confidence: float = 0.0


def _parse_batch_summary_rows(rows: list[list[str | None]]) -> tuple[BatchSummary, list[str]]:
    """Parse one table's rows into a `BatchSummary`, returning which of
    `BATCH_SUMMARY_CORE_FIELDS` were resolved. A row whose label matches no
    known field, or whose value cell does not parse, is silently skipped -
    never raises on data (see module docstring)."""
    resolved: dict[str, Measurement] = {}
    for row in rows:
        if not row or row[0] is None:
            continue
        field_name = _match_field(str(row[0]))
        if field_name is None or field_name in resolved:
            continue
        parsed = _parse_row_value(row)
        if parsed is None:
            continue
        value, unit = parsed
        resolved[field_name] = Measurement(value=value, unit=unit, method="pdfplumber_table")
    summary = BatchSummary(**resolved)
    resolved_core = [f for f in BATCH_SUMMARY_CORE_FIELDS if f in resolved]
    return summary, resolved_core


def extract_batch_summary(path: Path) -> DeterministicResult:
    """Find and parse the batch-summary table in `path`, deterministically.

    Every table `extract_tables` finds is parsed independently; the one
    resolving the most `BATCH_SUMMARY_CORE_FIELDS` wins (first table wins a
    tie), matching how a real report may carry several small tables (a
    conditions table, an analytical-results table) alongside the one that
    actually summarizes the batch. No table found, or no table resolving any
    field, returns an all-`None` `BatchSummary` with every core field
    unresolved and `confidence=0.0` - never an exception, and never a
    fabricated value (see `schema.BatchSummary`'s docstring).
    """
    tables = extract_tables(path)
    best_summary = BatchSummary()
    best_resolved: list[str] = []
    best_page: int | None = None
    for table in tables:
        summary, resolved = _parse_batch_summary_rows(table.rows)
        if len(resolved) > len(best_resolved):
            best_summary, best_resolved, best_page = summary, resolved, table.page

    unresolved = [f for f in BATCH_SUMMARY_CORE_FIELDS if f not in best_resolved]
    confidence = len(best_resolved) / len(BATCH_SUMMARY_CORE_FIELDS)
    return DeterministicResult(
        batch_summary=best_summary,
        resolved_fields=best_resolved,
        unresolved_fields=unresolved,
        pages=[best_page] if best_page is not None else [],
        confidence=confidence,
    )


def document_metadata(path: Path, *, version: str | None = None, approval_status: str | None = None) -> DocumentMetadata:
    """Build `DocumentMetadata` for `path`: filename-derived title/id, an MD5
    content hash (provenance - identifies the exact bytes extracted, matching
    the port's own use of an MD5 hash for this purpose), and the page count.

    `document_id` mirrors the port's own filename-derived rule
    (`_extract_document_metadata`): the text before the first comma in the
    filename stem, or the whole stem when there is no comma - client run
    sheets/reports in the source project were routinely named
    "<id>, <description>.pdf". Requires the `pdf` extra (for the page count).
    """
    content = Path(path).read_bytes()
    md5_hash = hashlib.md5(content).hexdigest()
    stem = Path(path).stem
    doc_id = stem.split(",")[0].strip() if "," in stem else stem
    return DocumentMetadata(
        title=stem,
        document_id=doc_id,
        extraction_date=datetime.now(timezone.utc).isoformat(),
        source_file=str(path),
        md5_hash=md5_hash,
        total_pages=page_count(path),
        version=version,
        approval_status=approval_status,
    )


__all__ = [
    "ExtractedTable",
    "DeterministicResult",
    "extract_tables",
    "extract_full_text",
    "page_count",
    "extract_batch_summary",
    "document_metadata",
]
