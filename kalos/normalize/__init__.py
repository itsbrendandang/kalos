"""Deterministic run-sheet normalization: units, canonical names, and a plan.

Import-light by design (stdlib + `re` + `pandas` + the anonymizer's identity
rules only) - this package must stay usable anywhere `kalos.kit` is, without
ever pulling in torch/botorch/gpytorch. No top-level import of `anthropic`,
`pydantic`, or `typesafe_sdk` happens here either (see `llm.py` and
`typesafe_tier.py`) - those are lazy-imported only
inside the live LLM call, so this package stays usable without the
`normalize` extra installed. `offline_plan`/`propose_plan`'s offline fallback
never touches the network; the live LLM path only activates when
`credentials_available()` is true, and always falls back to the offline path
on any failure.
"""
from __future__ import annotations

from .apply import NormalizedResult, apply_plan
from .config import NormalizeConfig, credentials_available, load_config, typesafe_credentials_available
from .llm import offline_plan, propose_plan
from .payload import build_payload
from .plan import ColumnPlan, ColumnProvenance, NormalizationPlan
from .roles import plan_to_roles
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
    "NormalizeConfig",
    "load_config",
    "credentials_available",
    "typesafe_credentials_available",
    "build_payload",
    "propose_plan",
    "offline_plan",
    "plan_to_roles",
]
