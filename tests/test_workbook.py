"""Tests for `kalos.normalize.workbook`: the tier-2 multi-sheet workbook
classifier (`analyze_workbook`) and merge (`merge_workbook`), plus their
wiring into `kalos.portal.uploads._parse_upload` via `_apply_workbook_prepass`.

No real experimental data is used anywhere in this file - every workbook is
hand-built/fabricated, mirroring the style of `tests/test_orientation.py`.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from kalos.normalize.workbook import (
    WorkbookMergeError,
    analyze_workbook,
    merge_workbook,
)
from kalos.portal import uploads
from kalos.portal.uploads import MAX_COLUMNS, MAX_XLSX_CELLS, _parse_upload

# --- fixtures ----------------------------------------------------------------- #


def _summary_sheet(n: int = 6) -> pd.DataFrame:
    """Run-level: one row per batch, widest per-run coverage."""
    return pd.DataFrame(
        {
            "Batch ID": [f"Batch-{i + 1}" for i in range(n)],
            "operator": ["alice", "bob"] * (n // 2),
            "scale_L": [5.0, 5.0, 50.0, 50.0, 200.0, 200.0][:n],
            "titer_g_per_L": [1.2, 1.4, 2.1, 2.3, 3.0, 3.2][:n],
        }
    )


def _media_sheet(n: int = 6) -> pd.DataFrame:
    """Per-topic, mergeable: one row per batch, shares the Batch ID column."""
    return pd.DataFrame(
        {
            "Batch ID": [f"Batch-{i + 1}" for i in range(n)],
            "glucose_g_per_L": [10.0, 10.0, 12.0, 12.0, 15.0, 15.0][:n],
            "yeast_extract_g_per_L": [5.0, 5.0, 6.0, 6.0, 7.0, 7.0][:n],
        }
    )


def _assays_sheet(n: int = 6) -> pd.DataFrame:
    """Per-topic, mergeable: one row per batch, shares the Batch ID column."""
    return pd.DataFrame(
        {
            "Batch ID": [f"Batch-{i + 1}" for i in range(n)],
            "purity_pct": [98.0, 97.5, 99.1, 98.8, 97.0, 96.5][:n],
        }
    )


def _timeseries_sheet(n_batches: int = 6, n_points: int = 4) -> pd.DataFrame:
    """A repeated-id sheet: multiple rows per batch (a real time series)."""
    rows = []
    for i in range(n_batches):
        for t in range(n_points):
            rows.append({"Batch ID": f"Batch-{i + 1}", "time_h": t * 6, "ph": 6.8 + 0.01 * t})
    return pd.DataFrame(rows)


def _notes_sheet(n: int = 3) -> pd.DataFrame:
    """A free-text notes sheet: no id column, long prose cells."""
    return pd.DataFrame(
        {
            "note": [
                "Operator observed slightly elevated foaming during the induction "
                "phase; antifoam was added manually at hour six per the SOP.",
                "Batch proceeded without incident; all setpoints held within the "
                "expected control band for the full run duration.",
                "Feed pump alarm triggered briefly around hour twelve, cleared "
                "after a restart with no apparent impact on downstream titer.",
            ][:n]
        }
    )


def _three_sheet_workbook(n: int = 6) -> dict[str, pd.DataFrame]:
    return {
        "Summary": _summary_sheet(n),
        "Media": _media_sheet(n),
        "Assays": _assays_sheet(n),
    }


def _xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


# --- analyze_workbook: classification ------------------------------------------ #


def test_three_sheet_workbook_classifies_run_level_and_mergeable():
    report = analyze_workbook(_three_sheet_workbook())
    assert report.id_column == "Batch ID"
    assert report.run_level_sheet == "Summary"
    assert report.classifications["Summary"].role == "run_level"
    assert report.classifications["Media"].role == "mergeable"
    assert report.classifications["Assays"].role == "mergeable"
    for name in ("Summary", "Media", "Assays"):
        assert report.classifications[name].reason  # always a stated reason


def test_time_series_sheet_excluded_with_reason():
    sheets = _three_sheet_workbook()
    sheets["Sensors"] = _timeseries_sheet()
    report = analyze_workbook(sheets)
    cls = report.classifications["Sensors"]
    assert cls.role == "unmergeable"
    assert "time-series" in cls.reason or "repeated values" in cls.reason
    assert report.run_level_sheet == "Summary"  # unaffected by the extra sheet


def test_free_text_notes_sheet_excluded_with_reason():
    sheets = _three_sheet_workbook()
    sheets["Notes"] = _notes_sheet()
    report = analyze_workbook(sheets)
    cls = report.classifications["Notes"]
    assert cls.role == "unmergeable"
    assert "free-text" in cls.reason
    assert report.run_level_sheet == "Summary"


def test_empty_sheet_excluded_with_reason():
    sheets = _three_sheet_workbook()
    sheets["Blank"] = pd.DataFrame()
    report = analyze_workbook(sheets)
    assert report.classifications["Blank"].role == "unmergeable"
    assert "empty" in report.classifications["Blank"].reason


def test_duplicate_id_mergeable_sheet_demotes_to_unmergeable():
    # A sheet that LOOKS like a per-topic sheet (same id column) but has a
    # duplicated id must be classified unmergeable, not mergeable - merging
    # it would duplicate rows.
    sheets = _three_sheet_workbook()
    dup = _media_sheet().copy()
    dup.loc[len(dup)] = dup.iloc[0]  # duplicate the first Batch ID
    sheets["Media"] = dup
    report = analyze_workbook(sheets)
    cls = report.classifications["Media"]
    assert cls.role == "unmergeable"
    assert "repeated values" in cls.reason
    assert report.run_level_sheet == "Summary"


def test_sheet_missing_shared_id_column_is_unmergeable():
    sheets = _three_sheet_workbook()
    sheets["Misc"] = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    report = analyze_workbook(sheets)
    cls = report.classifications["Misc"]
    assert cls.role == "unmergeable"
    assert "Batch ID" in cls.reason


def test_no_shared_id_column_is_ambiguous_workbook():
    sheets = {
        "A": pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}),
        "B": pd.DataFrame({"p": [7.0, 8.0], "q": [9.0, 10.0]}),
    }
    report = analyze_workbook(sheets)
    assert report.id_column is None
    assert report.run_level_sheet is None
    for cls in report.classifications.values():
        assert cls.role == "ambiguous"


def test_tied_run_level_candidates_are_ambiguous_not_guessed():
    # Two sheets with the SAME unique-id coverage and the SAME column count -
    # a genuine tie the classifier must not break by guessing.
    a = pd.DataFrame({"Batch ID": ["Batch-1", "Batch-2"], "x": [1.0, 2.0]})
    b = pd.DataFrame({"Batch ID": ["Batch-1", "Batch-2"], "y": [3.0, 4.0]})
    report = analyze_workbook({"A": a, "B": b})
    assert report.run_level_sheet is None
    assert report.classifications["A"].role == "ambiguous"
    assert report.classifications["B"].role == "ambiguous"
    assert "tied" in report.classifications["A"].reason


# --- merge_workbook: the merge itself ------------------------------------------ #


def test_merge_produces_full_provenance_and_correct_shape():
    report = analyze_workbook(_three_sheet_workbook())
    merged, provenance = merge_workbook(report)

    assert len(merged) == 6
    for col in ("glucose_g_per_L", "yeast_extract_g_per_L", "purity_pct"):
        assert col in merged.columns
    assert merged["purity_pct"].tolist() == [98.0, 97.5, 99.1, 98.8, 97.0, 96.5]

    assert provenance["id_column"] == "Batch ID"
    assert provenance["run_level_sheet"] == "Summary"
    assert provenance["sheets"]["Summary"]["role"] == "run_level"
    assert provenance["sheets"]["Media"]["merged"] is True
    assert provenance["sheets"]["Media"]["rows_contributed"] == 6
    assert provenance["sheets"]["Assays"]["merged"] is True


def test_merge_reports_each_collision_rename():
    sheets = _three_sheet_workbook()
    # Both Media and Assays independently carry a "purity_pct" column - the
    # second one merged in (workbook order: Media, then Assays) must be
    # prefixed, and the rename reported. An extra dummy column keeps Summary
    # strictly the widest sheet (run-level candidate), so adding a column to
    # Media does not accidentally tie it with Summary for run-level.
    sheets["Summary"] = sheets["Summary"].assign(vessel_type=["glass"] * 6)
    sheets["Media"] = sheets["Media"].assign(purity_pct=[1.0] * 6)
    report = analyze_workbook(sheets)
    merged, provenance = merge_workbook(report)

    # Media is merged first (workbook order): its purity_pct lands unprefixed,
    # Assays' colliding purity_pct gets renamed.
    media_prov = provenance["sheets"]["Media"]
    assays_prov = provenance["sheets"]["Assays"]
    assert media_prov["renamed_columns"] == {}
    assert "purity_pct" in assays_prov["renamed_columns"]
    renamed_to = assays_prov["renamed_columns"]["purity_pct"]
    assert renamed_to in merged.columns
    assert renamed_to.startswith("assays_")


def test_merge_at_time_demotes_a_sheet_that_became_non_unique():
    # merge_workbook must revalidate uniqueness itself (defense in depth),
    # not just trust an earlier classification - simulate a report whose
    # `frames` entry was mutated to have a duplicate id after analyze ran.
    sheets = _three_sheet_workbook()
    report = analyze_workbook(sheets)
    mutated_media = report.frames["Media"].copy()
    mutated_media.loc[len(mutated_media)] = mutated_media.iloc[0]
    report.frames["Media"] = mutated_media

    merged, provenance = merge_workbook(report)
    assert provenance["sheets"]["Media"]["role"] == "unmergeable"
    assert provenance["sheets"]["Media"]["merged"] is False
    # Assays (untouched) still merged normally.
    assert provenance["sheets"]["Assays"]["merged"] is True
    assert len(merged) == 6  # row count did not explode


def test_merge_excludes_sheet_that_would_exceed_column_cap():
    sheets = _three_sheet_workbook()
    report = analyze_workbook(sheets)
    # Summary(4) + Media(2 new) = 6; cap at 5 must exclude Media.
    merged, provenance = merge_workbook(report, max_columns=5)
    assert provenance["sheets"]["Media"]["role"] == "excluded"
    assert "column cap" in provenance["sheets"]["Media"]["reason"]
    assert "glucose_g_per_L" not in merged.columns


def test_merge_workbook_raises_on_ambiguous_report():
    sheets = {
        "A": pd.DataFrame({"x": [1.0, 2.0]}),
        "B": pd.DataFrame({"y": [3.0, 4.0]}),
    }
    report = analyze_workbook(sheets)
    with pytest.raises(WorkbookMergeError):
        merge_workbook(report)


# --- upload-path integration through _parse_upload ----------------------------- #


def test_parse_upload_merges_multi_sheet_xlsx():
    raw = _xlsx_bytes(_three_sheet_workbook())
    df = _parse_upload(raw)
    assert len(df) == 6
    for col in ("glucose_g_per_L", "yeast_extract_g_per_L", "purity_pct"):
        assert col in df.columns
    workbook_attrs = df.attrs.get("kalos_workbook")
    assert workbook_attrs is not None
    assert workbook_attrs["merged"] is True
    assert workbook_attrs["run_level_sheet"] == "Summary"
    assert workbook_attrs["sheets"]["Media"]["merged"] is True
    # The orientation pre-pass still ran (per sheet, before the merge).
    assert "kalos_orientation" in df.attrs


def test_parse_upload_ambiguous_workbook_falls_back_to_first_sheet():
    sheets = {
        "A": pd.DataFrame(
            {"x": np.arange(10, dtype=float), "y": np.arange(10, dtype=float) * 2}
        ),
        "B": pd.DataFrame(
            {"p": np.arange(10, dtype=float) * 3, "q": np.arange(10, dtype=float) * 4}
        ),
    }
    raw = _xlsx_bytes(sheets)
    df = _parse_upload(raw)
    # Degrades to the first sheet only, same shape as sheet A alone.
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 10
    workbook_attrs = df.attrs.get("kalos_workbook")
    assert workbook_attrs is not None
    assert workbook_attrs["merged"] is False


def test_parse_upload_single_sheet_xlsx_byte_identical_to_before():
    df = pd.DataFrame(
        {
            "medium": ["A", "B", "C", "A", "B", "C"],
            "Methanol": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            "pH": [6.0, 6.5, 7.0, 6.1, 6.6, 7.1],
            "titer": [1.0, 2.0, 3.0, 1.1, 2.2, 3.3],
        }
    )
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    raw = buf.getvalue()

    got = _parse_upload(raw)
    # Same values as a direct single-sheet parse (sheet_name=0, the pre-wiring
    # default) through the same orientation pre-pass.
    direct = pd.read_excel(io.BytesIO(raw))
    from kalos.portal.uploads import _apply_orientation_prepass

    expected = _apply_orientation_prepass(direct)
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), expected.reset_index(drop=True)
    )
    assert "kalos_workbook" not in got.attrs  # workbook tier never ran


def test_parse_upload_csv_unaffected_by_workbook_wiring():
    text = "Methanol,pH,titer\n1.0,6.0,1.0\n2.0,6.5,2.0\n3.0,7.0,3.0\n"
    df = _parse_upload(text.encode("utf-8"))
    assert list(df.columns) == ["Methanol", "pH", "titer"]
    assert "kalos_workbook" not in df.attrs


def test_summed_cell_cap_rejects_many_sheet_bomb(monkeypatch):
    # Each sheet is small individually (well under a realistic MAX_XLSX_CELLS),
    # but five of them together exceed a monkeypatched, much smaller cap - this
    # must be rejected on the SUM across sheets, not slip through because no
    # single sheet trips a per-sheet check.
    monkeypatch.setattr(uploads, "MAX_XLSX_CELLS", 50)
    sheets = {
        f"Sheet{i}": pd.DataFrame(
            {f"c{j}": [1, 2, 3, 4, 5, 6] for j in range(6)}  # 6 rows x 6 cols = 36 cells
        )
        for i in range(5)
    }
    raw = _xlsx_bytes(sheets)  # 5 * 36 = 180 cells > 50, but no single sheet > 50
    with pytest.raises(uploads.UploadRejected) as excinfo:
        uploads._parse_upload(raw)
    assert str(excinfo.value) == "The uploaded file is too large."


def test_summed_cell_cap_still_accepts_small_multi_sheet_workbook():
    # Sanity check the guard above is not simply over-firing: an honest small
    # multi-sheet workbook, well under the REAL caps, is still accepted.
    raw = _xlsx_bytes(_three_sheet_workbook())
    assert MAX_XLSX_CELLS > 0 and MAX_COLUMNS > 0  # constants read, not re-declared
    df = _parse_upload(raw)
    assert len(df) == 6
