"""Tier 3's public entry point: `extract_run_document`, one function tying
together the deterministic table tier (`pdf.py`), the LLM fallback tier
(`llm.py`), the schema bridge (`schema.py`), and `kalos.validation`'s
existing gate.

This package's whole contract, end to end, is: deterministic first, LLM
last, everything validated, nothing invented. Concretely:
  - the deterministic tier runs unconditionally; the LLM tier is only
    consulted for fields the deterministic tier left `None`, never to
    override or "double check" a value the deterministic tier already
    resolved.
  - the LLM tier's output is schema-validated (`llm._validate_fields`)
    before it is ever merged into the result - a schema-shaped response is
    not the same as an accepted one, matching `docs/ingestion-architecture.md`
    (`itsbrendandang/kalos-transition` repo)'s framing exactly.
  - a field neither tier could resolve is listed in
    `ExtractionResult.unresolved_fields`, never silently dropped or filled
    with a placeholder.
  - the resulting frame is run through `kalos.validation.validate_frame` -
    the SAME gate every other ingestion path in kalos uses, not a
    PDF-specific one - and the report is returned on `ExtractionResult`, not
    swallowed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from kalos.validation import ValidationReport, validate_frame

from . import pdf
from .llm import fill_unresolved_fields
from .schema import BatchSummary, DocumentMetadata, FermentationRecord, Measurement, records_to_frame

LlmFillFn = Callable[[str, list[str]], dict[str, Measurement]]


@dataclass
class ExtractionResult:
    """Everything `extract_run_document` produced for one PDF.

    `tier_contributions` maps every `schema.BATCH_SUMMARY_CORE_FIELDS` name
    to which tier produced it - `"deterministic"`, `"llm"`, or
    `"unresolved"` - so a caller (or an auditor) can answer "where did this
    number come from" for any field without re-running anything.
    `unresolved_fields` is the same information filtered to just the fields
    neither tier could fill; it is always a subset of `tier_contributions`'s
    `"unresolved"` values, kept as its own list because that is the
    question a caller actually asks most often ("what am I still missing").
    """

    frame: pd.DataFrame
    document_metadata: DocumentMetadata
    validation: ValidationReport
    tier_contributions: dict[str, str] = field(default_factory=dict)
    unresolved_fields: list[str] = field(default_factory=list)
    provenance: dict[str, object] = field(default_factory=dict)


def extract_run_document(
    path: str | Path,
    *,
    schema: type = BatchSummary,
    llm: LlmFillFn | None = None,
    batch_id: str | None = None,
    condition: str = "test",
    date: str | None = None,
    operator: str | None = None,
    version: str | None = None,
    approval_status: str | None = None,
) -> ExtractionResult:
    """Extract a fermentation batch summary from the PDF at `path`.

    `schema` names the extraction target - only `schema.BatchSummary` (the
    default) is implemented today; the parameter exists so a future
    additional target (`schema.FlaskStage`, `schema.BioreactorRun`, ...) can
    be added without changing this function's signature. `llm` overrides the
    LLM fallback tier (default `llm.fill_unresolved_fields`) - mainly for
    tests, which inject a fake rather than reaching a real Ollama server.

    Raises `ImportError` (naming `pip install kalos[pdf]`) if the `pdf`
    extra is not installed - this function always attempts real table
    extraction, so it cannot succeed without it; see `pdf._require_pdfplumber`.

    `batch_id` defaults to the file's stem when not given. `condition`
    defaults to `"test"` (not `"control"`) since a run document being
    ingested is, in the overwhelming case, a regular run rather than a
    control - a caller extracting a control run's report should pass
    `condition="control"` explicitly.
    """
    if schema is not BatchSummary:
        raise NotImplementedError(f"extract_run_document only supports schema=BatchSummary today, got {schema!r}")

    path = Path(path)
    llm_fill = llm if llm is not None else fill_unresolved_fields

    deterministic = pdf.extract_batch_summary(path)
    tier_contributions: dict[str, str] = {f: "deterministic" for f in deterministic.resolved_fields}

    llm_fields: dict[str, Measurement] = {}
    if deterministic.unresolved_fields:
        text = pdf.extract_full_text(path)
        llm_fields = llm_fill(text, deterministic.unresolved_fields)
        for name in llm_fields:
            tier_contributions[name] = "llm"

    still_unresolved = [f for f in deterministic.unresolved_fields if f not in llm_fields]
    for name in still_unresolved:
        tier_contributions[name] = "unresolved"

    # `tier_contributions` keys are exactly the core fields the deterministic
    # tier resolved plus the ones it left unresolved (`BATCH_SUMMARY_CORE_FIELDS`
    # in full) - read each back off the deterministic result, then let
    # `llm_fields` override whichever of the unresolved ones the LLM tier
    # filled. Anything in neither dict keeps its `None` from `getattr`.
    merged_fields = {name: getattr(deterministic.batch_summary, name) for name in tier_contributions}
    merged_fields.update(llm_fields)
    batch_summary = BatchSummary(**merged_fields)

    doc_meta = pdf.document_metadata(path, version=version, approval_status=approval_status)

    record = FermentationRecord(
        document_metadata=doc_meta,
        batch_summary=batch_summary,
        batch_id=batch_id or path.stem,
        date=date,
        operator=operator,
        condition=condition,
        conditions=deterministic.conditions,
    )
    frame = records_to_frame([record])
    report = validate_frame(frame)

    provenance = {
        "pages": deterministic.pages,
        "extractor": deterministic.extractor,
        "confidence": deterministic.confidence,
        "content_hash": doc_meta.md5_hash,
        "extracted_at": doc_meta.extraction_date,
    }
    return ExtractionResult(
        frame=frame,
        document_metadata=doc_meta,
        validation=report,
        tier_contributions=tier_contributions,
        unresolved_fields=still_unresolved,
        provenance=provenance,
    )


__all__ = ["ExtractionResult", "extract_run_document"]
