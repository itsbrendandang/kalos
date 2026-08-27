"""Tests for `kalos.ingest.pdf` (the deterministic table-extraction tier) and
`kalos.ingest.api.extract_run_document` (the tier-0..3 orchestrator).

The extras-absent path (`test_missing_pdfplumber_raises_informative_error`)
needs no optional dependency and always runs. Everything else needs a real
PDF to extract from, so it is generated IN-TEST with `reportlab` (see
`_write_batch_summary_pdf` below for why reportlab, not camelot's own
fixture tooling or `fpdf2`) and parsed with `pdfplumber` - both are the
optional `pdf` extra's dependency plus a test-only fixture generator, and
kalos's own `.venv` intentionally does not carry either (see
`pyproject.toml`'s `pdf` extra comment for the installed-size reasoning), so
every test needing them is marked with `pytest.importorskip`, which skips
cleanly rather than erroring on a fresh `kalos[dev]` checkout.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kalos.ingest import pdf as ingest_pdf
from kalos.ingest.api import extract_run_document
from kalos.ingest.schema import BATCH_SUMMARY_CORE_FIELDS, Measurement


def _write_batch_summary_pdf(path: Path, rows: list[list[str]]) -> None:
    """Write a minimal one-table PDF (header row + `rows`) via `reportlab`.

    Chosen over `fpdf2` or hand-built minimal-PDF bytes: `reportlab` was
    already present in the throwaway venv this port was verified in (see
    the final report), its `platypus.Table` produces a real gridded table
    pdfplumber's line-based table finder reliably detects (a hand-built
    minimal PDF would need to draw the grid lines itself, error-prone and
    unnecessary when a maintained library does it correctly), and unlike
    `fpdf2` it needed no additional install to verify this port end-to-end.
    Not used by `kalos.ingest` itself - test-only, gated behind
    `pytest.importorskip("reportlab")` in every caller.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

    data = [["Parameter", "Value", "Unit"], *rows]
    table = Table(data)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    SimpleDocTemplate(str(path), pagesize=letter).build([table])


_FULL_ROWS = [
    ["Final Titer", "5.8", "g/L"],
    ["Final OD600", "42.0", "OD"],
    ["Duration", "96", "hours"],
    ["Yield", "62.0", "g/L"],
    ["Productivity", "0.06", "g/L/h"],
    ["Final Biomass", "18.4", "g/L"],
]


def _full_pdf(tmp_path: Path) -> Path:
    pytest.importorskip("reportlab")
    pytest.importorskip("pdfplumber")
    path = tmp_path / "batch_summary.pdf"
    _write_batch_summary_pdf(path, _FULL_ROWS)
    return path


# --- extras-absent path -------------------------------------------------------- #


def test_missing_pdfplumber_raises_informative_error(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest_pdf, "pdfplumber", None)
    fake_pdf = tmp_path / "whatever.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 not a real pdf")
    with pytest.raises(ImportError, match=r"pip install kalos\[pdf\]"):
        ingest_pdf.extract_tables(fake_pdf)


def test_ingest_pdf_module_importable_with_pdfplumber_genuinely_absent():
    # A stronger version of the guard above: force `import pdfplumber` itself
    # to fail (not just monkeypatch the already-imported name), the same way
    # it genuinely fails in kalos's own `.venv` (no `pdf` extra installed),
    # and confirm re-importing `kalos.ingest.pdf` still succeeds rather than
    # propagating the ImportError - the whole point of the try/except guard
    # at the top of `pdf.py`.
    #
    # Uses a scoped `pytest.MonkeyPatch.context()` (not the `monkeypatch`
    # fixture) so `sys.modules["pdfplumber"]` is restored by the time the
    # `with` block exits, BEFORE the restoring `importlib.reload` below runs
    # - with the fixture instead, its teardown (which un-sets
    # `sys.modules["pdfplumber"]`) would not run until after this function
    # returns, one step too late for a reload inside the function body to
    # see the real module again.
    import importlib
    import sys

    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "pdfplumber", None)
        reloaded = importlib.reload(ingest_pdf)
        assert reloaded.pdfplumber is None

    importlib.reload(ingest_pdf)  # back to the real module for later tests


# --- deterministic tier --------------------------------------------------------- #


def test_extract_tables_finds_the_one_table(tmp_path):
    path = _full_pdf(tmp_path)
    tables = ingest_pdf.extract_tables(path)
    assert len(tables) == 1
    assert tables[0].page == 1
    assert tables[0].rows[0] == ["Parameter", "Value", "Unit"]


def test_extract_batch_summary_recovers_all_core_fields(tmp_path):
    path = _full_pdf(tmp_path)
    result = ingest_pdf.extract_batch_summary(path)
    assert sorted(result.resolved_fields) == sorted(BATCH_SUMMARY_CORE_FIELDS)
    assert result.unresolved_fields == []
    assert result.confidence == 1.0
    assert result.pages == [1]
    assert result.extractor == "pdfplumber_table"

    summary = result.batch_summary
    assert summary.final_titer == Measurement(value=5.8, unit="g/L", method="pdfplumber_table")
    assert summary.final_od == Measurement(value=42.0, unit="OD", method="pdfplumber_table")
    assert summary.duration.value == 96.0
    assert summary.yield_value.value == 62.0
    assert summary.productivity.value == 0.06
    assert summary.final_biomass.value == 18.4


def test_extract_batch_summary_partial_table_lists_unresolved(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("pdfplumber")
    path = tmp_path / "partial.pdf"
    _write_batch_summary_pdf(path, [["Final Titer", "5.8", "g/L"], ["Duration", "96", "hours"]])
    result = ingest_pdf.extract_batch_summary(path)
    assert set(result.resolved_fields) == {"final_titer", "duration"}
    assert set(result.unresolved_fields) == {"yield_value", "final_od", "final_biomass", "productivity"}
    assert 0.0 < result.confidence < 1.0


def test_extract_batch_summary_no_table_is_all_unresolved_never_raises(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("pdfplumber")
    from reportlab.pdfgen import canvas

    path = tmp_path / "prose_only.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(72, 720, "This report has no tables, only prose text.")
    c.save()

    result = ingest_pdf.extract_batch_summary(path)
    assert result.resolved_fields == []
    assert set(result.unresolved_fields) == set(BATCH_SUMMARY_CORE_FIELDS)
    assert result.confidence == 0.0
    assert result.pages == []


def test_document_metadata_hash_and_page_count(tmp_path):
    path = _full_pdf(tmp_path)
    meta = ingest_pdf.document_metadata(path)
    assert meta.total_pages == 1
    assert meta.document_id == "batch_summary"
    assert len(meta.md5_hash) == 32  # md5 hex digest
    # version/approval_status are never invented from the filename (see
    # schema.DocumentMetadata's docstring) - default None unless supplied.
    assert meta.version is None
    assert meta.approval_status is None


# --- orchestration: extract_run_document ---------------------------------------- #


def test_extract_run_document_full_table_end_to_end(tmp_path):
    path = _full_pdf(tmp_path)
    result = extract_run_document(path, llm=lambda text, fields: {})

    assert result.unresolved_fields == []
    assert all(tier == "deterministic" for tier in result.tier_contributions.values())
    assert set(result.tier_contributions) == set(BATCH_SUMMARY_CORE_FIELDS)

    assert result.provenance["extractor"] == "pdfplumber_table"
    assert result.provenance["confidence"] == 1.0
    assert result.provenance["pages"] == [1]
    assert result.provenance["content_hash"] == result.document_metadata.md5_hash

    # A single-document extraction is expected to fail validate_frame's row-count
    # floor (n_rows=1 < 6) - that is a real, honest finding, not a bug in the
    # bridge. What matters here is that NOTHING ELSE is wrong: no units/bounds/
    # provenance-identifier error caused by this package's own column naming.
    report = result.validation
    assert report.status == "fail"
    error_checks = {f.check for f in report.findings if f.severity == "error"}
    assert error_checks == {"replicate_adequacy"}
    assert any(
        f.check == "replicate_adequacy" and "at least 6 rows" in f.message for f in report.findings
    )


def test_extract_run_document_routes_unresolved_field_to_llm_tier(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("pdfplumber")
    path = tmp_path / "missing_od.pdf"
    _write_batch_summary_pdf(
        path,
        [
            ["Final Titer", "5.8", "g/L"],
            ["Duration", "96", "hours"],
            ["Yield", "62.0", "g/L"],
            ["Productivity", "0.06", "g/L/h"],
            ["Final Biomass", "18.4", "g/L"],
            # "Final OD600" deliberately omitted - left for the LLM tier.
        ],
    )

    def fake_llm(text: str, fields: list[str]) -> dict[str, Measurement]:
        assert fields == ["final_od"]
        assert "5.8" in text or "Final Titer" in text  # got real page text, not a stub
        return {"final_od": Measurement(value=42.0, unit="OD", method="llm_ollama")}

    result = extract_run_document(path, llm=fake_llm)

    assert result.unresolved_fields == []
    assert result.tier_contributions["final_od"] == "llm"
    assert result.tier_contributions["final_titer"] == "deterministic"
    assert result.frame.loc[0, "od600"] == 42.0


def test_extract_run_document_leaves_field_unresolved_when_llm_declines(tmp_path):
    pytest.importorskip("reportlab")
    pytest.importorskip("pdfplumber")
    path = tmp_path / "missing_od_2.pdf"
    _write_batch_summary_pdf(
        path,
        [
            ["Final Titer", "5.8", "g/L"],
            ["Duration", "96", "hours"],
            ["Yield", "62.0", "g/L"],
            ["Productivity", "0.06", "g/L/h"],
            ["Final Biomass", "18.4", "g/L"],
        ],
    )
    result = extract_run_document(path, llm=lambda text, fields: {})
    assert result.unresolved_fields == ["final_od"]
    assert result.tier_contributions["final_od"] == "unresolved"
    # Never invented: no od600 column stands in for the value nobody supplied.
    assert "od600" not in result.frame.columns


def test_extract_run_document_only_supports_batch_summary_schema(tmp_path):
    path = _full_pdf(tmp_path)
    with pytest.raises(NotImplementedError):
        extract_run_document(path, schema=dict)
