"""A domain-neutral profile for non-bio tabular optimization.

`GENERIC_PROFILE` recognizes only generic column vocabulary: an objective named
`target`/`objective`/`response`/... and obvious identifier/grouping columns. It
carries no industry-specific terms, so it is the template a new domain copies (or
the profile a caller uses together with an explicit `ColumnRoles` schema).
"""
from __future__ import annotations

import re

from .profile import DomainProfile

GENERIC_PROFILE = DomainProfile(
    name="generic",
    outcome_hint=re.compile(
        r"^(target|objective|response|output|outcome|result|score|y)$", re.I
    ),
    target_pref=re.compile(r"^(target|objective|y)$", re.I),
    id_hint=re.compile(
        r"^(id|name|index|run|experiment|round|date|time|group|batch|lot|replicate|notes?)$",
        re.I,
    ),
    group_hint=re.compile(r"^(group|batch|lot|replicate)$", re.I),
)

__all__ = ["GENERIC_PROFILE"]
