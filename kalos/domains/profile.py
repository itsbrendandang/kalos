"""Domain abstraction: declared column roles and mixed design-space specs.

The engine (`kalos.core`) is domain-agnostic: it optimizes numeric arrays and
knows nothing about biology. This module holds the thin, torch-free layer that
maps a tabular problem in any domain onto that engine.

`DomainProfile` carries the fallback hint patterns used to INFER column roles
when the caller does not declare them. A domain's vocabulary (biology's
"titer"/"strain", say) lives in a profile such as
`kalos.domains.bioprocess.BIOPROCESS_PROFILE`, never in the engine.

`ColumnRoles` is an explicit, caller-declared schema. When supplied it skips
inference entirely, so a non-bio user declares what their columns mean instead
of renaming them to match a regex.

`DesignSpace` describes each optimized dimension as continuous (numeric bounds)
or categorical (ordered levels), and encodes/decodes between human labels and
the integer-coded numeric matrix the GP consumes.

Torch-free by contract (numpy and pandas only, both core dependencies) so it
stays importable from the lean `kalos.kit` facade without loading the
torch/botorch/gpytorch stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Pattern

import numpy as np
import pandas as pd

DimensionKind = Literal["continuous", "categorical"]


@dataclass(frozen=True)
class DomainProfile:
    """Fallback regexes for inferring column roles when none are declared.

    Kept as data (not code in the engine) so a new domain is a new profile, not
    an engine edit. `outcome_hint` matches measured-output columns (candidate
    targets, and excluded from features to avoid leakage); `target_pref` picks a
    preferred target among the outcomes; `id_hint` matches identifier columns to
    drop; `group_hint` matches a replicate-grouping column.
    """

    name: str
    outcome_hint: Pattern[str]
    target_pref: Pattern[str]
    id_hint: Pattern[str]
    group_hint: Pattern[str]


@dataclass(frozen=True)
class ColumnRoles:
    """An explicit, caller-declared column schema for the analyze path.

    `target` is the column to maximize (required). `features` are the process
    inputs to model; empty means "infer the features from the profile as
    usual". `groups` is the optional replicate-grouping column. `ids` are
    columns to ignore. `categoricals` names the feature columns whose values are
    unordered labels rather than numbers; their levels are derived from the data.
    """

    target: str
    features: tuple[str, ...] = ()
    groups: str | None = None
    ids: tuple[str, ...] = ()
    categoricals: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict) -> "ColumnRoles":
        """Parse a roles object (e.g. from a JSON request body).

        `target` is required and must be a non-empty string. List fields accept
        a list of strings (or a single string); missing fields default to empty.
        """
        target = data.get("target")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("roles.target must be a non-empty column name")

        def as_tuple(value: object) -> tuple[str, ...]:
            if value is None:
                return ()
            if isinstance(value, str):
                return (value,) if value.strip() else ()
            if isinstance(value, (list, tuple)):
                return tuple(str(v) for v in value if str(v).strip())
            raise ValueError("roles list fields must be a string or list of strings")

        groups = data.get("groups")
        if groups is not None and not isinstance(groups, str):
            raise ValueError("roles.groups must be a column name or null")
        return cls(
            target=target,
            features=as_tuple(data.get("features")),
            groups=(groups or None),
            ids=as_tuple(data.get("ids")),
            categoricals=as_tuple(data.get("categoricals")),
        )


@dataclass(frozen=True)
class Dimension:
    """One optimized dimension: continuous with `[lower, upper]`, or categorical
    with an ordered tuple of `levels` (its integer code is the level index)."""

    name: str
    kind: DimensionKind
    lower: float = 0.0
    upper: float = 1.0
    levels: tuple[str, ...] = ()

    @property
    def is_categorical(self) -> bool:
        return self.kind == "categorical"


@dataclass(frozen=True)
class DesignSpace:
    """A mixed continuous/categorical design space over an ordered set of dims.

    The engine sees an integer-coded numeric matrix (categorical label -> level
    index) and a `(2, d)` bounds box; this class owns the mapping in both
    directions so labels never leak into the GP and integer proposals decode
    back to labels for the client.
    """

    dims: tuple[Dimension, ...]

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.dims]

    @property
    def cat_dims(self) -> list[int]:
        """Indices of the categorical dimensions (BoTorch `cat_dims`)."""
        return [i for i, d in enumerate(self.dims) if d.is_categorical]

    @property
    def cont_indices(self) -> list[int]:
        return [i for i, d in enumerate(self.dims) if not d.is_categorical]

    @property
    def has_categoricals(self) -> bool:
        return bool(self.cat_dims)

    @property
    def cat_cardinalities(self) -> list[int]:
        """Level count per categorical dim, aligned to `cat_dims`."""
        return [len(self.dims[i].levels) for i in self.cat_dims]

    def bounds(self) -> np.ndarray:
        """Full `(2, d)` box: continuous dims use their range, categorical dims
        span `[0, n_levels - 1]` over their integer codes."""
        lo: list[float] = []
        hi: list[float] = []
        for d in self.dims:
            if d.is_categorical:
                lo.append(0.0)
                hi.append(float(max(len(d.levels) - 1, 0)))
            else:
                lo.append(float(d.lower))
                hi.append(float(d.upper))
        return np.vstack([lo, hi])

    def cont_bounds(self) -> np.ndarray:
        """`(2, n_cont)` box over just the continuous dims, for input
        normalization (`Normalize(indices=cont_indices)`)."""
        idx = self.cont_indices
        if not idx:
            return np.empty((2, 0), float)
        lo = [float(self.dims[i].lower) for i in idx]
        hi = [float(self.dims[i].upper) for i in idx]
        return np.vstack([lo, hi])

    def encode_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Encode a dataframe (columns named as in `dims`) to the numeric matrix
        the GP consumes: continuous columns coerced to float, categorical
        columns mapped to their integer level codes."""
        cols: list[np.ndarray] = []
        for d in self.dims:
            if d.is_categorical:
                code = {lvl: i for i, lvl in enumerate(d.levels)}
                mapped = df[d.name].astype(str).map(code)
                cols.append(mapped.to_numpy(float))
            else:
                cols.append(pd.to_numeric(df[d.name], errors="coerce").to_numpy(float))
        if not cols:
            return np.empty((len(df), 0), float)
        return np.column_stack(cols)

    def decode_row(self, x) -> list:
        """Decode one proposal row back to labels: categorical codes rounded and
        clamped to a valid level, continuous coordinates rounded to 3 places."""
        vals = np.asarray(x, float).reshape(-1)
        out: list = []
        for i, d in enumerate(self.dims):
            if d.is_categorical:
                if not d.levels:
                    out.append(None)
                    continue
                k = int(round(float(vals[i])))
                k = max(0, min(k, len(d.levels) - 1))
                out.append(d.levels[k])
            else:
                out.append(round(float(vals[i]), 3))
        return out


def build_design_space(
    df: pd.DataFrame,
    features: list[str],
    categoricals: tuple[str, ...] = (),
    *,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> DesignSpace:
    """Build a `DesignSpace` from a run sheet and a feature list.

    `features` is the ordered set of feature columns. `categoricals` is the
    subset to treat as unordered labels; their levels are the sorted unique
    non-blank values (sorted for determinism). Continuous bounds come from
    `bounds[name]` when provided, else from the column's observed min/max.
    """
    cat = set(categoricals)
    dims: list[Dimension] = []
    for name in features:
        if name in cat:
            values = {
                str(v)
                for v in df[name].dropna().tolist()
                if str(v).strip() != ""
            }
            dims.append(
                Dimension(name, "categorical", levels=tuple(sorted(values)))
            )
        else:
            if bounds is not None and name in bounds:
                lo, hi = bounds[name]
            else:
                col = pd.to_numeric(df[name], errors="coerce")
                lo, hi = float(col.min()), float(col.max())
            dims.append(Dimension(name, "continuous", lower=lo, upper=hi))
    return DesignSpace(tuple(dims))


__all__ = [
    "DimensionKind",
    "DomainProfile",
    "ColumnRoles",
    "Dimension",
    "DesignSpace",
    "build_design_space",
]
