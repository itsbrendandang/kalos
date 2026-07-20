"""The bioprocess domain profile: the column-role hint regexes Kalos shipped
with, kept here as data so the engine stays domain-neutral.

`BIOPROCESS_PROFILE` reproduces the exact inference the analyze path used before
the domain abstraction, so an upload with no declared roles behaves identically
to before. It is the default profile in `kalos.portal.analysis`.
"""
from __future__ import annotations

import re

from .profile import DomainProfile

BIOPROCESS_PROFILE = DomainProfile(
    name="bioprocess",
    outcome_hint=re.compile(
        r"titer|titre|yield|conc|purity|lipase|biomass|od\d|product|response|output|score|kda|activity",
        re.I,
    ),
    target_pref=re.compile(r"titer|titre|lipase|yield", re.I),
    id_hint=re.compile(
        r"^(id|name|sample.*|well|index|run|experiment|round|date|time|medium|strain|recipe|batch|campaign|group|lot|notes?)$",
        re.I,
    ),
    group_hint=re.compile(r"medium|strain|recipe|batch|campaign|group|lot", re.I),
)

__all__ = ["BIOPROCESS_PROFILE"]
