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

from kalos.core.analysis import MAX_FIT_ROWS, UploadRejected, _ERR_TOO_MANY_FIT_ROWS

log = logging.getLogger("kalos.portal")

# --- upload safety limits ---------------------------------------------------- #
_MAX_UPLOAD_MB = float(os.environ.get("KALOS_MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = int(_MAX_UPLOAD_MB * 1024 * 1024)
MAX_CSV_ROWS = 100_000       # rows read from a CSV/TSV upload
MAX_COLUMNS = 512            # columns allowed in any upload (CSV or xlsx)
MAX_XLSX_CELLS = 2_000_000   # rows * cols ceiling for a parsed xlsx (zip-bomb guard)
_ZIP_MAGIC = b"PK\x03\x04"   # xlsx/xls-as-zip start-of-file marker

# MAX_FIT_ROWS / UploadRejected / _ERR_TOO_MANY_FIT_ROWS live in
# `kalos.core.analysis` (imported above) - that cap guards the GP fit itself,
# not the raw upload, so it belongs with the engine, not this untrusted-input
# boundary. Imported here (and re-exported) so existing call sites that reach
# them via `kalos.portal.uploads` keep working unchanged.

# Generic, non-leaking messages. We never echo the parser error, a column name,
# or a cell value back to an unauthenticated caller.
_ERR_PARSE = "Could not parse the uploaded file. Check it is a CSV or Excel run-sheet."
_ERR_TOO_LARGE = "The uploaded file is too large."
_ERR_TOO_MANY_COLUMNS = "The uploaded file has too many columns."
# A numerically-hard-but-valid file (near-duplicate or ill-conditioned rows) that
# defeats even the jittered fit retry gets its OWN message, so it is not confused
# with the generic parse failure.
_ERR_FIT = (
    "The model could not be fit on this data - likely near-duplicate or "
    "ill-conditioned rows."
)


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
    """
    import openpyxl

    try:
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True)
    except Exception as exc:  # noqa: BLE001 - corrupt/non-xlsx zip -> generic 400
        log.warning("rejected an unreadable xlsx upload: %s", type(exc).__name__)
        raise UploadRejected(_ERR_PARSE) from exc
    try:
        for ws in wb.worksheets:
            cols = ws.max_column or 0
            rows = ws.max_row or 0
            if cols > MAX_COLUMNS:
                raise UploadRejected(_ERR_TOO_MANY_COLUMNS)
            if rows * cols > MAX_XLSX_CELLS:
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
            if rows * cols > MAX_XLSX_CELLS:
                raise UploadRejected(_ERR_TOO_LARGE)

    if needs_fallback:
        _reject_oversized_xlsx_via_openpyxl(raw)


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
            df = pd.read_excel(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001 - normalized to a generic 400 below
            log.warning("rejected an unreadable xlsx upload: %s", type(exc).__name__)
            raise UploadRejected(_ERR_PARSE) from exc
        rows, cols = df.shape
        # Belt-and-suspenders: re-check the materialized shape. The pre-read guard
        # uses the sheet's declared dimensions; this catches a mismatch and the
        # column ceiling on the actual parsed frame.
        if cols > MAX_COLUMNS:
            raise UploadRejected(_ERR_TOO_MANY_COLUMNS)
        if rows * cols > MAX_XLSX_CELLS:
            raise UploadRejected(_ERR_TOO_LARGE)
        return df

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
    return df
