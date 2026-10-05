"""Kalos portal — JSON-safe serialization helpers."""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd


def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame rows as JSON-safe dicts: NaN/NaT/inf -> None.

    Missing cells arrive from pandas as float NaN; Starlette's JSONResponse
    encodes with allow_nan=False, so an un-sanitized NaN 500s the response.
    Coerce every non-finite float and NaT to JSON null, so both the stored
    payload and the HTTP response are valid JSON. None round-trips back to NaN
    when `_payload_to_frame` rebuilds the DataFrame for the Singleton.
    """
    safe = df.astype(object).where(pd.notna(df), None)
    # column labels are strings here; to_dict types them as Hashable
    records = cast("list[dict[str, Any]]", safe.to_dict(orient="records"))
    for row in records:
        for key, val in row.items():
            if isinstance(val, float) and not np.isfinite(val):
                row[key] = None
    return records
