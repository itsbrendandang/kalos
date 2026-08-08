"""Backward-compatible re-export.

The GP-fit entrypoint (`_analyze` and its helpers) moved to
`kalos.core.analysis` so that `kalos.runner.singleton` (the Singleton runner)
can depend on it without the runner ever importing the portal - previously
`kalos.runner.singleton` imported `_analyze` from THIS module, meaning the
engine/runner transitively depended on the portal package. This module
re-exports the same names so existing portal imports (`kalos.portal.app`, and
any test importing `kalos.portal.analysis` directly) keep working unchanged.
"""
from __future__ import annotations

from kalos.core.analysis import (
    ANALYZE_SEED,
    MAX_FIT_ROWS,
    UploadRejected,
    _ERR_TOO_MANY_FIT_ROWS,
    _GROUP_HINT,
    _ID_HINT,
    _OUTCOME_HINT,
    _TARGET_PREF,
    _analyze,
    _annotate,
    _anonymize_result,
    _dedupe_columns,
    _numeric_cols,
    _seed_everything,
)

__all__ = [
    "ANALYZE_SEED",
    "MAX_FIT_ROWS",
    "UploadRejected",
    "_analyze",
    "_annotate",
]
