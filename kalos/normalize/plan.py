"""A normalization plan: per-column decisions (rename/convert/hash/drop) plus
the provenance of what actually happened when the plan was applied.

A `NormalizationPlan` is the thing an LLM (or the offline fallback in
`synonyms.guess_role`) proposes; `apply.apply_plan` is the thing that actually
executes it deterministically against a dataframe. Keeping the plan as its own
serializable object means the same plan can be inspected, edited, and re-run,
and a client audit can always answer "why does column X look like this" from
committed JSON, not from re-running an LLM.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .synonyms import Role

Action = Literal["renamed", "converted", "hashed", "dropped_identity", "dropped_freetext", "coerced"]


@dataclass(frozen=True)
class ColumnPlan:
    """The decision for one raw column: keep (as what, renamed to what, with
    what unit conversion) or drop (identity/freetext)."""

    raw_name: str
    canonical_name: str | None
    role: Role
    unit_token: str | None = None
    to_base: bool = False
    is_identity: bool = False
    redacted: bool = False
    note: str = ""


@dataclass
class ColumnProvenance:
    """What `apply_plan` actually did to one column, for audit."""

    raw_name: str
    canonical_name: str | None
    role: Role
    action: Action
    n_converted: int = 0
    n_coerced_nan: int = 0


@dataclass
class NormalizationPlan:
    """The full per-column plan for one run sheet, plus who authored it."""

    columns: list[ColumnPlan]
    created_by: Literal["llm", "offline"]
    model: str | None = None

    def validate(self) -> None:
        """Raise `ValueError` on an internally inconsistent plan:
          - more than one column with `role == "target"`.
          - two columns sharing the same non-None `canonical_name`.
          - an identity or freetext column that was still given a
            non-None `canonical_name` (those columns are dropped, so a
            canonical name for them is a contradiction in the plan).
        """
        targets = [c for c in self.columns if c.role == "target"]
        if len(targets) > 1:
            names = [c.raw_name for c in targets]
            raise ValueError(f"more than one target column in plan: {names}")

        seen: dict[str, str] = {}
        for c in self.columns:
            if c.canonical_name is None:
                continue
            if c.canonical_name in seen:
                raise ValueError(
                    f"duplicate canonical_name {c.canonical_name!r} for columns "
                    f"{seen[c.canonical_name]!r} and {c.raw_name!r}"
                )
            seen[c.canonical_name] = c.raw_name

        for c in self.columns:
            if c.role in ("identity", "freetext") and c.canonical_name is not None:
                raise ValueError(
                    f"column {c.raw_name!r} has role {c.role!r} but a non-None "
                    f"canonical_name {c.canonical_name!r} (identity/freetext columns "
                    "are dropped and must not carry a canonical name)"
                )

    def summary(self) -> dict[str, int]:
        """Counts describing the plan: how many raw columns came in, how many
        got a canonical name, how many of each role, how many are dropped
        (identity + freetext), and how many will go through a unit conversion.
        """
        n_dropped = sum(1 for c in self.columns if c.role in ("identity", "freetext"))
        return {
            "n_in": len(self.columns),
            "n_canonical": sum(1 for c in self.columns if c.canonical_name is not None),
            "n_target": sum(1 for c in self.columns if c.role == "target"),
            "n_feature": sum(1 for c in self.columns if c.role == "feature"),
            "n_group": sum(1 for c in self.columns if c.role == "group"),
            "n_dropped": n_dropped,
            "n_unit_conversions": sum(1 for c in self.columns if c.to_base),
        }

    def to_json(self) -> str:
        """Serialize to JSON. Round-trips exactly through `from_json`."""
        return json.dumps(
            {
                "columns": [asdict(c) for c in self.columns],
                "created_by": self.created_by,
                "model": self.model,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, raw: str) -> "NormalizationPlan":
        """Rebuild a `NormalizationPlan` from `to_json`'s output."""
        data: dict[str, Any] = json.loads(raw)
        columns = [ColumnPlan(**col) for col in data["columns"]]
        return cls(columns=columns, created_by=data["created_by"], model=data.get("model"))


__all__ = ["Action", "ColumnPlan", "ColumnProvenance", "NormalizationPlan"]
