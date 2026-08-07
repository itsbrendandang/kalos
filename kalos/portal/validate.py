"""Backward-compatible re-export.

`column_provenance` and its supporting types moved to `kalos.core.provenance`
as part of breaking the portal<->runner circular import: `kalos.runner.singleton`
used to reach the GP-fit entrypoint via `kalos.portal.analysis`, which imported
this module, so the engine (`kalos.core`/`kalos.runner`) transitively depended
on the portal. See `kalos.core.analysis` (where `_analyze` now lives) for the
full picture. This module re-exports the same names so existing portal
imports keep working unchanged.
"""
from __future__ import annotations

from kalos.core.provenance import (
    ColumnProvenance,
    ColumnStatus,
    column_provenance,
    provenance_dicts,
)

__all__ = [
    "ColumnProvenance",
    "ColumnStatus",
    "column_provenance",
    "provenance_dicts",
]
