"""Turn a `NormalizationPlan` into the `ColumnRoles` the analyze path accepts.

This is the seam that lets a plan's per-column decisions (from the TypeSafe
tier, an LLM, or the offline heuristics) drive `POST /api/run` instead of
the bioprocess profile's header regexes. It maps RAW column names, because
the analyze path runs on the uploaded sheet as-is; the plan's canonical
names and unit conversions are not applied here.

Torch-free: `kalos.domains` is torch-free, and it is imported lazily anyway
so `kalos.normalize` keeps its import-light contract.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .plan import NormalizationPlan

if TYPE_CHECKING:
    from kalos.domains import ColumnRoles


def plan_to_roles(plan: NormalizationPlan) -> "ColumnRoles | None":
    """The `ColumnRoles` a plan implies, or `None` when it names no target or
    no feature columns. (With no declared features the analyze path would infer
    them from every remaining numeric column - including a numeric group the
    plan kept out of the inputs - so a plan without features does not decide
    roles at all.)

      - `target`: the plan's target column (a valid plan has at most one).
      - `features`: every `feature` column. The analyze path still applies
        its own >=80% numeric gate and reports anything it leaves out.
      - `groups`: the first `group` column; any further group column is
        listed in `ids`, since the analyze path takes one grouping column.
      - `ids`: identity, freetext, and metadata columns plus extra groups -
        everything that must never be modeled as an input.
    """
    from kalos.domains import ColumnRoles

    target = next((c.raw_name for c in plan.columns if c.role == "target"), None)
    if target is None:
        return None
    features = tuple(c.raw_name for c in plan.columns if c.role == "feature")
    if not features:
        return None
    group_cols = [c.raw_name for c in plan.columns if c.role == "group"]
    ids = tuple(
        c.raw_name for c in plan.columns if c.role in ("identity", "freetext", "metadata")
    ) + tuple(group_cols[1:])
    return ColumnRoles(
        target=target,
        features=features,
        groups=group_cols[0] if group_cols else None,
        ids=ids,
    )


__all__ = ["plan_to_roles"]
