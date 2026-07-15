"""Kalos portal — the untrusted-input boundary.

This engine ingests untrusted run sheets from external clients, so the raw
upload and its parsed shape are capped to bound memory and blunt zip-bomb
expansion. Sizes are configurable via env; the CSV/xlsx caps are constants.
"""
from __future__ import annotations

import io
import logging
import os

import pandas as pd

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


def _reject_oversized_xlsx(raw: bytes) -> None:
    """Reject an xlsx whose declared dimensions exceed the cell / column caps,
    BEFORE `pd.read_excel` materializes the frame (the zip-bomb guard).

    Opens the workbook read-only with openpyxl and reads each sheet's declared
    dimension (`max_row * max_column`) without loading cell values. If any sheet's
    declared cell count exceeds `MAX_XLSX_CELLS` (or its column count exceeds
    `MAX_COLUMNS`), raise `UploadRejected` so the caller never allocates the full
    frame. A corrupt or unreadable zip raises `UploadRejected` too (client error).
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
