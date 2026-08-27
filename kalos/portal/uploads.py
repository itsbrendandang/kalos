"""Kalos portal — the untrusted-input boundary.

This engine ingests untrusted run sheets from external clients, so the raw
upload and its parsed shape are capped to bound memory and blunt zip-bomb
expansion. Sizes are configurable via env; the CSV/xlsx caps are constants.
"""
from __future__ import annotations

import io
import logging
import os
import re
import zipfile

import pandas as pd

from kalos.normalize.orientation import detect_orientation
from kalos.normalize.workbook import analyze_workbook, merge_workbook

log = logging.getLogger("kalos.portal")

# --- upload safety limits ---------------------------------------------------- #
_MAX_UPLOAD_MB = float(os.environ.get("KALOS_MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = int(_MAX_UPLOAD_MB * 1024 * 1024)
MAX_CSV_ROWS = 100_000       # rows read from a CSV/TSV upload
MAX_COLUMNS = 512            # columns allowed in any upload (CSV or xlsx)
MAX_XLSX_CELLS = 2_000_000   # rows * cols ceiling for a parsed xlsx (zip-bomb guard)
_ZIP_MAGIC = b"PK\x03\x04"   # xlsx/xls-as-zip start-of-file marker

# Separate, tighter cap on the rows the SURROGATE is actually fit on. A
# SingleTaskGP is O(n^2) in memory and O(n^3) in time, so a raw upload that is
# within MAX_CSV_ROWS can still be far too large for an exact GP. The optimizer
# targets the small-sample bioprocess regime (N <= a couple thousand), so we
# reject an over-cap fit set rather than silently subsampling (which would be
# invisible, non-deterministic data loss). Override with KALOS_MAX_FIT_ROWS.
MAX_FIT_ROWS = int(os.environ.get("KALOS_MAX_FIT_ROWS", "2000"))

# Generic, non-leaking messages. We never echo the parser error, a column name,
# or a cell value back to an unauthenticated caller.
_ERR_PARSE = "Could not parse the uploaded file. Check it is a CSV or Excel run-sheet."
_ERR_TOO_LARGE = "The uploaded file is too large."
_ERR_TOO_MANY_COLUMNS = "The uploaded file has too many columns."
_ERR_TOO_MANY_FIT_ROWS = (
    "The dataset is too large for the surrogate; the optimizer targets the "
    f"small-sample regime, N<={MAX_FIT_ROWS}."
)
# A numerically-hard-but-valid file (near-duplicate or ill-conditioned rows) that
# defeats even the jittered fit retry gets its OWN message, so it is not confused
# with the generic parse failure.
_ERR_FIT = (
    "The model could not be fit on this data - likely near-duplicate or "
    "ill-conditioned rows."
)


class UploadRejected(ValueError):
    """A client upload failed a safety guard. Carries a generic, safe message."""


_SHEET_XML_RE = re.compile(r"^xl/worksheets/sheet\d+\.xml$")
_DIMENSION_RE = re.compile(rb'<dimension\s+ref="([^"]+)"\s*/?>')
_CELL_REF_RE = re.compile(r"^([A-Za-z]+)(\d+)$")
# The <dimension> tag is schema-guaranteed to precede the (potentially huge)
# <sheetData> section, so a bounded prefix read is enough to find it - this
# keeps the guard cheap even against a sheet whose real (not just declared)
# content is itself a compressed-XML bomb.
_DIMENSION_SCAN_BYTES = 65_536


def _col_letters_to_index(letters: str) -> int:
    """Convert a 1-based spreadsheet column label (A, Z, AA, SF, ...) to an index."""
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx


def _parse_dimension_ref(ref: str) -> tuple[int, int] | None:
    """Parse a `<dimension ref="...">` value (e.g. `A1:SF1048576`, or a single
    cell like `A1`) into its 1-based (rows, cols) extent. Returns None if the
    bottom-right cell reference does not match the expected `<letters><digits>`
    shape (defensive - callers fall back to openpyxl in that case).
    """
    end = ref.split(":")[-1].strip()
    m = _CELL_REF_RE.match(end)
    if not m:
        return None
    col_letters, row_str = m.groups()
    return int(row_str), _col_letters_to_index(col_letters)


def _reject_oversized_xlsx_via_openpyxl(raw: bytes) -> None:
    """Fallback guard for the rare sheet with no `<dimension>` tag: open read-only
    with openpyxl and check its computed `max_row` / `max_column` instead.

    The cell-count cap is enforced on the SUM of `rows * cols` across every
    worksheet, not any one sheet alone: kalos now reads every sheet of a
    multi-sheet workbook (see `_apply_workbook_prepass`), so a workbook whose
    individual sheets each sit just under the cap but whose total does not
    must still be rejected before that total is ever materialized.
    """
    import openpyxl

    try:
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True)
    except Exception as exc:  # noqa: BLE001 - corrupt/non-xlsx zip -> generic 400
        log.warning("rejected an unreadable xlsx upload: %s", type(exc).__name__)
        raise UploadRejected(_ERR_PARSE) from exc
    try:
        total_cells = 0
        for ws in wb.worksheets:
            cols = ws.max_column or 0
            rows = ws.max_row or 0
            if cols > MAX_COLUMNS:
                raise UploadRejected(_ERR_TOO_MANY_COLUMNS)
            total_cells += rows * cols
            if total_cells > MAX_XLSX_CELLS:
                raise UploadRejected(_ERR_TOO_LARGE)
    finally:
        wb.close()


def _reject_oversized_xlsx(raw: bytes) -> None:
    """Reject an xlsx whose declared dimensions exceed the cell / column caps,
    BEFORE `pd.read_excel` materializes the frame (the zip-bomb guard).

    Reads each sheet's `<dimension ref="...">` tag straight out of the raw zip
    entry (`xl/worksheets/sheet*.xml`) via a lightweight regex - this is the
    DECLARED shape a crafted workbook can inflate independently of its real
    content, so it must be checked without ever asking openpyxl or pandas to
    walk (and thus materialize) the sheet. If a sheet has no `<dimension>` tag,
    falls back to opening it read-only with openpyxl and using its computed
    `max_row` / `max_column`. A corrupt or unreadable zip raises
    `UploadRejected` (client error).

    The cell-count cap (`MAX_XLSX_CELLS`) is enforced on the SUM of
    `rows * cols` across every sheet in the workbook, not any one sheet
    alone - kalos now reads every sheet (see `_apply_workbook_prepass`), so
    N sheets each declaring cells just under the cap could otherwise combine
    into a materialized frame far over it. The column cap (`MAX_COLUMNS`)
    stays a PER-SHEET check: it bounds one sheet's own width, independent of
    how many sheets the workbook has.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        log.warning("rejected an unreadable xlsx upload: %s", type(exc).__name__)
        raise UploadRejected(_ERR_PARSE) from exc

    with zf:
        sheet_names = [n for n in zf.namelist() if _SHEET_XML_RE.match(n)]
        if not sheet_names:
            # Not a recognizable xlsx layout (or a non-xlsx zip) - let openpyxl's
            # own validation produce the generic parse-rejection.
            _reject_oversized_xlsx_via_openpyxl(raw)
            return

        needs_fallback = False
        total_cells = 0
        for name in sheet_names:
            with zf.open(name) as sheet_file:
                head = sheet_file.read(_DIMENSION_SCAN_BYTES)
            match = _DIMENSION_RE.search(head)
            if not match:
                needs_fallback = True
                continue
            parsed = _parse_dimension_ref(match.group(1).decode("ascii", errors="replace"))
            if parsed is None:
                needs_fallback = True
                continue
            rows, cols = parsed
            if cols > MAX_COLUMNS:
                raise UploadRejected(_ERR_TOO_MANY_COLUMNS)
            total_cells += rows * cols
            if total_cells > MAX_XLSX_CELLS:
                raise UploadRejected(_ERR_TOO_LARGE)

    if needs_fallback:
        _reject_oversized_xlsx_via_openpyxl(raw)


def _apply_orientation_prepass(df: pd.DataFrame) -> pd.DataFrame:
    """Tier-1 deterministic pre-pass (`kalos.normalize.orientation`): detect
    whether the just-parsed sheet is standard or transposed orientation, and
    normalize a confidently-transposed sheet BEFORE anything downstream
    (unit normalization, the validation gate, analysis) ever sees it - see
    `kalos/normalize/orientation.py`'s module docstring for why this has to
    happen first.

    Never silently swaps the frame without a trace: the `OrientationReport`
    is always recorded on the returned frame's `df.attrs["kalos_orientation"]`
    - including the "standard"/"ambiguous" no-op cases - so a downstream
    provenance report can see what this pre-pass decided and why, without
    changing `_parse_upload`'s return type or every caller's signature.
    An ambiguous sheet is returned completely untouched (never guessed);
    a standard sheet is also returned untouched, so this pre-pass leaves
    kalos's existing behavior byte-identical for every sheet that was
    already standard orientation.
    """
    report = detect_orientation(df)
    out = report.normalized_frame if report.orientation == "transposed" and report.normalized_frame is not None else df
    out.attrs["kalos_orientation"] = {
        "orientation": report.orientation,
        "confidence": report.confidence,
        "signals": dict(report.signals),
    }
    return out


def _apply_workbook_prepass(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tier-2 deterministic pre-pass (`kalos.normalize.workbook`) for a
    multi-sheet xlsx upload: classify every sheet (run-level / mergeable /
    unmergeable) and left-join the mergeable sheets onto the run-level sheet
    by their shared run/batch id column, BEFORE anything downstream (the
    validation gate, analysis) ever sees more than one frame.

    Ordering: the tier-1 orientation pre-pass (`_apply_orientation_prepass`)
    runs on EACH sheet individually FIRST, then the workbook classifier/merge
    runs on the (now orientation-normalized) sheets - never the other way
    around, and never once on the already-merged frame. Workbook
    classification needs each sheet already in standard orientation (rows =
    runs, one row per run, a run id running DOWN a column) to find a shared
    id column at all: a still-transposed per-topic sheet's "id" values run
    ACROSS its column headers, not down a column, so classifying it before
    fixing orientation would misclassify it (typically as unmergeable, for
    the wrong reason) rather than recognizing it as mergeable once flipped.
    Running orientation once on the final MERGED frame instead would be too
    late for the same reason - the per-sheet transpose has to happen before
    the per-sheet id column can be found and joined on - and would also be
    ambiguous about which of the several original sheets a single verdict
    was even describing.

    Never guesses: when `analyze_workbook` cannot confidently pick a
    run-level sheet (`WorkbookReport.run_level_sheet is None` - no shared id
    column at all, or a tie between candidates), this degrades to exactly
    what an upload of just the workbook's first sheet meant before this
    tier existed - `kalos`'s pre-wiring behavior - rather than rejecting the
    upload or guessing a merge. The ambiguity itself is still recorded, on
    `df.attrs["kalos_workbook"]`, so it is visible to a downstream
    provenance report even though no merge happened.
    """
    oriented = {name: _apply_orientation_prepass(df) for name, df in sheets.items()}
    report = analyze_workbook(oriented)

    if report.run_level_sheet is None:
        first_name = next(iter(oriented))
        out = oriented[first_name]
        out.attrs["kalos_workbook"] = {
            "merged": False,
            "reason": "ambiguous workbook: no sheet was confidently classified as the run-level sheet",
            "sheets": {
                name: {"role": c.role, "reason": c.reason} for name, c in report.classifications.items()
            },
        }
        return out

    merged, provenance = merge_workbook(report, max_columns=MAX_COLUMNS)
    merged.attrs["kalos_orientation"] = oriented[report.run_level_sheet].attrs.get("kalos_orientation")
    merged.attrs["kalos_workbook"] = {
        "merged": True,
        "sheet_orientation": {name: df.attrs.get("kalos_orientation") for name, df in oriented.items()},
        **provenance,
    }
    return merged


def _parse_upload(raw: bytes) -> pd.DataFrame:
    """Turn raw upload bytes into a bounded dataframe, or raise `UploadRejected`.

    Guards, in order:
      1. byte-size cap (default 25 MB, env `KALOS_MAX_UPLOAD_MB`);
      2. filetype sniff by MAGIC BYTES, not extension: a zip header (`PK\\x03\\x04`)
         is treated as xlsx/xls, anything else as UTF-8 text/CSV;
      3. shape caps: CSV/TSV rows and a column ceiling, and an xlsx cell-count
         (rows * cols) ceiling to blunt zip-bomb expansion.
    All rejection messages are generic (no parser text, column, or cell echoed).
    """
    if len(raw) > MAX_UPLOAD_BYTES:
        raise UploadRejected(_ERR_TOO_LARGE)

    if raw[:4] == _ZIP_MAGIC:
        # xlsx/xls: enforce the cell-count ceiling BEFORE pd.read_excel fully
        # materializes the frame, so a zip-bomb whose DECLARED sheet dimensions are
        # enormous is rejected without the memory spike of building the DataFrame.
        # A zip header on a corrupt or non-xlsx zip (BadZipFile, openpyxl's
        # InvalidFileException, ValueError) is a client error, not a server one.
        _reject_oversized_xlsx(raw)
        try:
            # sheet_name=None reads EVERY sheet (a dict of name -> DataFrame, in
            # workbook order), not just the first - `pd.read_excel`'s default
            # (`sheet_name=0`) silently dropped every sheet but the first, which
            # is wrong for a multi-sheet batch record (a run-level sheet plus
            # per-topic sheets - see `kalos/normalize/workbook.py`).
            parsed = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        except Exception as exc:  # noqa: BLE001 - normalized to a generic 400 below
            log.warning("rejected an unreadable xlsx upload: %s", type(exc).__name__)
            raise UploadRejected(_ERR_PARSE) from exc
        if not parsed:
            raise UploadRejected(_ERR_PARSE)
        # Belt-and-suspenders: re-check the materialized shape. The pre-read guard
        # uses each sheet's declared dimensions; this catches a mismatch and the
        # column ceiling on the actual parsed frame. The cell cap binds on the SUM
        # across every sheet (see `_reject_oversized_xlsx`'s docstring) - reading
        # multiple sheets must never let a workbook slip past the guard that a
        # single-sheet read already enforced.
        total_cells = 0
        for sheet_df in parsed.values():
            rows, cols = sheet_df.shape
            if cols > MAX_COLUMNS:
                raise UploadRejected(_ERR_TOO_MANY_COLUMNS)
            total_cells += rows * cols
        if total_cells > MAX_XLSX_CELLS:
            raise UploadRejected(_ERR_TOO_LARGE)
        if len(parsed) == 1:
            # Single-sheet xlsx: byte-identical to kalos's pre-wiring behavior -
            # the same single frame, through the same orientation pre-pass, with
            # the multi-sheet workbook tier never running at all.
            (df,) = parsed.values()
            return _apply_orientation_prepass(df)
        return _apply_workbook_prepass(parsed)

    # Otherwise treat as text/CSV. A binary blob that is neither a zip nor valid
    # tabular text will not yield usable numeric columns and is rejected downstream
    # with the same generic parse message.
    text = raw.decode("utf-8", errors="replace")
    if text.strip() == "":
        raise UploadRejected(_ERR_PARSE)
    head = text[:4000]
    sep = "\t" if head.count("\t") > head.count(",") else ","
    # cap columns first (cheap, from the header) before reading the full body
    ncols = pd.read_csv(io.StringIO(text), sep=sep, nrows=0).shape[1]
    if ncols > MAX_COLUMNS:
        raise UploadRejected(_ERR_TOO_MANY_COLUMNS)
    df = pd.read_csv(io.StringIO(text), sep=sep)
    # Fail closed on the row cap rather than silently truncating to the first
    # MAX_CSV_ROWS rows: silent data loss would contradict the safe-errors
    # contract, so an over-cap CSV is rejected like the xlsx cell-cap guard.
    if len(df) > MAX_CSV_ROWS:
        raise UploadRejected(_ERR_TOO_LARGE)
    return _apply_orientation_prepass(df)
