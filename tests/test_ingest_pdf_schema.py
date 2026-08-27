"""Tests for `kalos.ingest.schema`: the `FermentationRecord` -> tidy-frame
bridge (`records_to_frame`), checked directly against
`kalos.validation.validate_frame` - the requirement `docs/ingestion-architecture.md`
(`itsbrendandang/kalos-transition` repo) and that port's `PROVENANCE.md`
both name as the bridge's actual job: a frame `validate_frame` can consume,
in kalos's own unit conventions. No optional dependency needed - this
exercises the schema/bridge only, never `kalos.ingest.pdf`.
"""
from __future__ import annotations

from kalos.domains import BIOPROCESS_PROFILE
from kalos.ingest.schema import BatchSummary, DocumentMetadata, FermentationRecord, Measurement, Parameter, records_to_frame
from kalos.validation import validate_frame


def _doc_meta(i: int) -> DocumentMetadata:
    return DocumentMetadata(
        title=f"ENZ-000{i} v001, Enzyme Product Fermentation Summary Report",
        document_id=f"ENZ-000{i}-v001",
        extraction_date="2026-01-01T00:00:00+00:00",
        source_file=f"ENZ-000{i}.pdf",
        md5_hash="deadbeef",
    )


def _clean_records() -> list[FermentationRecord]:
    """Ten records built to pass `validate_frame` clean (INFO allowed),
    mirroring `tests/test_validation.py::_clean_frame`'s own sizing (10 rows,
    one feature, one replicated condition) so the same "pass" bar applies to
    a frame built through this bridge as to a frame built by hand."""
    temps_c = [20.0, 20.0, 25.0, 25.0, 30.0, 30.0, 35.0, 35.0, 40.0, 40.0]
    titers_g_l = [5.0, 5.2, 6.0, 6.1, 7.0, 7.3, 8.0, 8.2, 9.0, 9.4]
    records = []
    for i, (temp_c, titer) in enumerate(zip(temps_c, titers_g_l)):
        summary = BatchSummary(final_titer=Measurement(value=titer, unit="g/L", method="test"))
        condition = "control" if i == 0 else "test"
        records.append(
            FermentationRecord(
                document_metadata=_doc_meta(i),
                batch_summary=summary,
                batch_id=f"B{i}",
                date=f"2026-01-{i + 1:02d}",
                operator="opA",
                condition=condition,
                conditions=[Parameter(name="Temperature", value=temp_c, unit="C")],
            )
        )
    return records


def test_records_to_frame_clean_multirow_passes_validate_frame():
    frame = records_to_frame(_clean_records())
    report = validate_frame(
        frame, target="titer_g_l", features=["temperature_c"], profile=BIOPROCESS_PROFILE
    )
    assert report.status == "pass"
    assert report.counts["error"] == 0
    assert report.counts["warning"] == 0


def test_records_to_frame_column_names_use_kalos_unit_conventions():
    frame = records_to_frame(_clean_records())
    # "final_titer" (unit "g/L") -> base name "titer" + canonical_suffix("g/L") == "_g_l".
    assert "titer_g_l" in frame.columns
    # The Parameter "Temperature" (unit "C") -> snake-cased name + canonical
    # suffix for Celsius, matching kalos.normalize.units' own convention.
    assert "temperature_c" in frame.columns
    assert frame.loc[0, "titer_g_l"] == 5.0


def test_records_to_frame_converts_to_base_unit_not_raw_value():
    # A Fahrenheit reading must come out already converted to Celsius - the
    # bridge uses kalos.normalize.units.convert, never the raw source value.
    summary = BatchSummary(final_titer=Measurement(value=1.0, unit="g/L"))
    record = FermentationRecord(
        document_metadata=_doc_meta(0),
        batch_summary=summary,
        batch_id="B0",
        conditions=[Parameter(name="Temperature", value=98.6, unit="F")],
    )
    frame = records_to_frame([record])
    assert abs(frame.loc[0, "temperature_c"] - 37.0) < 0.1


def test_records_to_frame_unresolved_field_produces_no_column_not_nan():
    # A BatchSummary field left None (never recovered) must not appear as a
    # column at all for a single-record frame - never a fabricated NaN cell
    # standing in for a value nobody measured.
    summary = BatchSummary(final_titer=Measurement(value=5.0, unit="g/L"))
    record = FermentationRecord(document_metadata=_doc_meta(0), batch_summary=summary, batch_id="B0")
    frame = records_to_frame([record])
    assert "od600" not in frame.columns
    assert "duration_h" not in frame.columns
    assert "titer_g_l" in frame.columns


def test_records_to_frame_provenance_columns_present():
    frame = records_to_frame(_clean_records())
    for col in ("batch_id", "date", "operator", "condition"):
        assert col in frame.columns
