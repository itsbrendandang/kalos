"""Deterministic run-sheet normalization: units, canonical names, and a plan.

Import-light by design (stdlib + `re` + `pandas` + the anonymizer's identity
rules only) - this package must stay usable anywhere `kalos.kit` is, without
ever pulling in torch/botorch/gpytorch. No LLM call and no network access
happens anywhere in this package; an LLM-authored `NormalizationPlan` is
consumed here, never produced.
"""
from __future__ import annotations

from .apply import NormalizedResult, apply_plan
from .plan import ColumnPlan, ColumnProvenance, NormalizationPlan
from .synonyms import Role, guess_role, snake_canonical
from .units import canonical_suffix, convert, parse_value

__all__ = [
    "parse_value",
    "convert",
    "canonical_suffix",
    "snake_canonical",
    "guess_role",
    "Role",
    "ColumnPlan",
    "NormalizationPlan",
    "ColumnProvenance",
    "NormalizedResult",
    "apply_plan",
]
