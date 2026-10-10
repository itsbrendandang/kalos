"""Canonical column naming + deterministic offline role guessing.

Two independent jobs live here:
  1. `snake_canonical` - turn a messy raw header into a stable snake_case name.
  2. `guess_role` - a fully offline, deterministic fallback for classifying a
     column's role (target/feature/group/identity/freetext/metadata) when no
     LLM is available or its output is being cross-checked. It reuses the
     SAME identity rules as the rest of the repo (imported, not copied - see
     `kalos.data.anonymizer`) and mirrors the SAME outcome/id/group regex
     patterns `kalos.portal.analysis` uses for its (torch-requiring) online
     analysis path, so the offline plan and the live engine never disagree
     about what counts as an id, an outcome, or a group column.
"""
from __future__ import annotations

import re
from typing import Literal, Sequence

from kalos.data.anonymizer import DROP_EXACT, DROP_SUBSTR, HASH_EXACT, HASH_SUBSTR

Role = Literal["target", "feature", "group", "identity", "freetext", "metadata"]

# --- header -> snake_case ------------------------------------------------- #

# A parenthetical unit annotation on a header, e.g. "Titer (g/L)" -> drop the
# "(g/L)" part entirely before snake-casing so units live in the parsed value
# (see units.py), not baked into the canonical column name.
_PAREN_UNIT_RE = re.compile(r"\([^)]*\)")
_NON_ALNUM_RUN_RE = re.compile(r"[^0-9a-zA-Z]+")


def snake_canonical(name: str) -> str:
    """Lowercase, strip parenthetical unit annotations, and snake_case a header.

    Examples:
      "Titer (g/L)" -> "titer"
      "pH_final"    -> "ph_final"
      "DO %"        -> "do"
    """
    stripped = _PAREN_UNIT_RE.sub(" ", name)
    snake = _NON_ALNUM_RUN_RE.sub("_", stripped.strip().lower())
    return snake.strip("_")


# --- synonym prior (conservative, documented) ----------------------------- #

# Canonical name -> known aliases seen on real run sheets. This is only an
# offline HINT used by `guess_role`'s outcome/group detection below - it is
# deliberately small and conservative rather than an exhaustive dictionary, so
# a false-positive rename never silently mislabels a column the client meant
# to keep distinct.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "titer": ("titer", "titre", "product_titer", "final_titer"),
    "temperature": ("temp", "temperature", "temp_c", "culture_temp"),
    "ph": ("ph", "ph_final", "culture_ph"),
    "dissolved_oxygen": ("do", "do_pct", "dissolved_oxygen", "dissolved_o2"),
    "od": ("od", "od600", "optical_density"),
    "feed_rate": ("feed_rate", "feed", "feedrate"),
    "biomass": ("biomass", "biomass_conc", "dcw"),
    # Added while reconciling ports/csv-orientation/NOTES.md's unit-inference
    # table against this file: that table's "speed, rpm, agitation" and
    # "pressure" patterns had no canonical-name entry here despite units.py
    # now recognizing their unit tokens (rpm/bar). "volume, vol" is included
    # as a NAME hint only - units.py deliberately does NOT register a bare
    # "L"/"mL" unit token (a run sheet's "L" is commonly a vessel-size
    # identifier like "5L", not a measurement; see units.py's module
    # docstring), so a volume column's cell values pass through unconverted
    # exactly as they did before this addition.
    "agitation": ("agitation", "agitation_speed", "rpm", "stirrer_rpm"),
    "pressure": ("pressure", "pressure_setpoint"),
    "volume": ("volume", "reactor_volume", "vol"),
}

# --- role guessing (deterministic offline fallback) ------------------------ #

# Mirrored from `kalos.portal.analysis` (`_OUTCOME_HINT`, `_TARGET_PREF`,
# `_ID_HINT`, `_GROUP_HINT`). Not imported directly: those names are module-private
# (leading underscore, no `__all__` entry) in a module that transitively pulls
# in torch on other code paths, and this package must stay import-light (see
# `kalos/normalize/__init__.py`). Keep these patterns byte-identical to the
# source so the offline plan and the live analyze path never disagree.
_OUTCOME_HINT = re.compile(
    r"titer|titre|yield|conc|purity|lipase|biomass|od\d|product|response|output|score|kda|activity|titer",
    re.I,
)
_TARGET_PREF = re.compile(r"titer|titre|lipase|yield", re.I)
_GROUP_HINT = re.compile(r"medium|strain|recipe|batch|campaign|group|lot", re.I)


def guess_role(header: str, *, is_numeric: bool, parse_rate: float) -> Role:
    """Deterministic, offline classification of one column's role.

    Order of decisions (first match wins), all driven only by the header text,
    whether the column parses as numeric, and its numeric parse rate - no
    network, no LLM, fully reproducible given the same three inputs:
      1. Identity: header matches the anonymizer's DROP rules (exact or
         substring) -> "identity" (dropped downstream, never renamed).
      2. Group: header matches the anonymizer's HASH rules, or the
         `_GROUP_HINT` pattern -> "group" (kept but pseudonymized downstream).
      3. Target: header matches `_OUTCOME_HINT` AND the column is numeric ->
         "target".
      4. Feature: numeric and not identity/group/outcome -> "feature".
      5. Freetext: non-numeric with a low parse rate (mostly non-numeric
         text) -> "freetext" (dropped downstream).
      6. Metadata: everything else (non-numeric but structured, e.g. a run
         date or a short code) -> "metadata".

    `parse_rate` is the fraction of non-blank cells that parse as a plain
    number (0.0-1.0), matching the >=0.8 numeric threshold convention used
    elsewhere in the repo (`kalos.portal.analysis._numeric_cols`).
    """
    key = header.strip().lower()
    if key in DROP_EXACT or any(t in key for t in DROP_SUBSTR):
        return "identity"
    if key in HASH_EXACT or any(t in key for t in HASH_SUBSTR) or _GROUP_HINT.search(header):
        return "group"
    if _OUTCOME_HINT.search(header) and is_numeric:
        return "target"
    if is_numeric:
        return "feature"
    if parse_rate < 0.8:
        return "freetext"
    return "metadata"


def preferred_target(candidates: Sequence[str]) -> str:
    """Deterministically pick one target from `candidates`: a non-empty
    sequence of headers `guess_role` called "target", in sheet order.

    Mirrors `kalos.portal.analysis`'s inferred target: the first candidate
    matching `_TARGET_PREF` (titer/titre/lipase/yield), else the first
    candidate. Same headers in the same order always give the same pick.
    """
    return next((c for c in candidates if _TARGET_PREF.search(c)), candidates[0])


__all__ = ["Role", "SYNONYMS", "snake_canonical", "guess_role", "preferred_target"]
