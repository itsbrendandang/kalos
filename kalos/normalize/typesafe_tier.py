"""TypeSafe tier: decide each column's plan from typed System One judgments.

The Anthropic and Ollama providers ask a generative model to write a whole
`NormalizationPlan` as JSON and then validate what comes back. This tier asks
TypeSafe's System One model (Jev, via `typesafe-sdk`) narrow questions
instead, and code composes the plan from the typed answers:

  - `role_<i>`     Choice over the five keepable roles (target, feature,
                   group, metadata, freetext). Its `confidence` gates whether
                   the answer is acted on at all.
  - `identity_<i>` Noul: does this column identify a client, a person, or a
                   specific sample? A separate judgment, not a sixth role,
                   because it is a privacy gate with its own threshold
                   (`_IDENTITY_THRESHOLD`) rather than a preference.
  - `name_<i>`     Choice over candidate canonical names that CODE builds
                   (the `SYNONYMS` keys plus the column's own snake_case
                   name as the no-match outcome) - select, never generate,
                   so a canonical name is always one the codebase knows or
                   the header's own. Asked only when the header is not
                   already an exact alias; exact lookups stay in code.

Unit detection is NOT asked: it is an exact rule (`units.parse_value`) and
reuses the offline plan's helper, so every provider agrees on units.

All questions for a chunk of columns go in one request over the same state
(the identity-screened `build_payload` output - never raw data), so they run
in parallel and none can see another's answer. `name_<i>` is asked
speculatively and consumed only when the column is kept.

Policy is explicit and lives here, not in the model:
  - a role answer below `config.typesafe_min_confidence` is not acted on: the
    deterministic offline guess for that column is kept and the note says so;
  - at most one target: the most probable one wins, any other column the
    model called a target is kept as `metadata` (never a feature - a second
    outcome used as an input would leak the outcome into the model);
  - a canonical-name collision falls back to the column's own name, then a
    numeric suffix, so `NormalizationPlan.validate` always passes.

Every per-column decision carries its probabilities in `ColumnPlan.note`, so
an audit can answer "why is this column a feature" from the committed plan.

`typesafe-sdk` is imported lazily inside `_make_client` only - this module
stays importable without the `typesafe` extra, and `propose_plan` falls back
to the offline plan on any failure here, exactly like the other providers.
The API key is read by the SDK client from `TYPESAFE_API_KEY`; this module
never handles it.
"""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from kalos.data.anonymizer import Anonymizer

from .config import NormalizeConfig
from .llm import (
    _canonical_base_name,
    _merge_dropped_identity,
    _offline_columns,
    _unit_and_parse_rate,
    _with_unit_suffix,
)
from .plan import ColumnPlan, NormalizationPlan
from .synonyms import SYNONYMS, snake_canonical

# A column whose identity probability reaches this is dropped as identity.
# Dropping a real input costs one feature; keeping a client identifier costs
# a privacy breach, so the gate sits at even odds rather than demanding
# confidence before it protects.
_IDENTITY_THRESHOLD = 0.5

# Columns per request. Each column asks at most three questions; chunking keeps
# a wide sheet (up to the portal's 512-column cap) from becoming one oversized
# request. Every chunk still sees the full screened payload as state.
_COLUMNS_PER_REQUEST = 20

_REQUEST_TIMEOUT = 30.0

_TASK = (
    "Normalize the columns of a bioprocess experiment run sheet before modeling. "
    "Each entry in `columns` describes one column: its header, a coarse dtype, how many "
    "cells are filled, and a few sample values. Columns that matched known identity rules "
    "were already removed, and samples of free-text columns are replaced by '<redacted>'."
)

_ROLE_CRITERIA: dict[str, str] = {
    "target": (
        "The measured outcome the experiment is trying to improve, such as product titer, "
        "yield, purity, or activity. It is recorded after the run, not chosen before it."
    ),
    "feature": (
        "A process input or condition that was set or measured during the run and could "
        "explain the outcome, such as temperature, pH, agitation, feed rate, or a media "
        "component concentration."
    ),
    "group": (
        "A label shared by related runs, such as a campaign, batch, lot, strain, medium, or "
        "recipe identifier. It groups replicates or variants rather than measuring anything."
    ),
    "metadata": (
        "Structured bookkeeping that is neither a process input nor the main outcome, such as "
        "a run date, a well or vessel position, an instrument code, or a secondary measured "
        "output that must not be used as an input."
    ),
    "freetext": "Prose written by a person, such as lab notes, comments, or deviation descriptions.",
}

_IDENTITY_CRITERIA = {
    "true": (
        "The column names a client, a person, or one specific sample: an operator or analyst "
        "name, a client sample label, a customer or company name, an email address, or a phone "
        "number."
    ),
    "false": (
        "The column holds measurements, settings, grouping codes, dates, or positions that do "
        "not name a client, a person, or a specific client sample."
    ),
}

_KEEP_OWN_NAME = (
    "None of the other names describes the same quantity; keep the column's own name."
)


def _name_candidates(header: str) -> dict[str, str] | None:
    """Candidate canonical names for one header, or `None` when code already
    knows the answer (the header is an exact `SYNONYMS` alias) or there is
    nothing to choose between."""
    if _canonical_base_name(header) in SYNONYMS:
        return None
    own = snake_canonical(header)
    if not own:
        return None
    candidates = {
        canonical: (
            f"The column measures {canonical.replace('_', ' ')} "
            f"(also written as: {', '.join(aliases)})."
        )
        for canonical, aliases in SYNONYMS.items()
    }
    candidates[own] = _KEEP_OWN_NAME
    return candidates


def build_questions(columns: list[dict[str, Any]], indices: Iterable[int]) -> dict[str, dict[str, Any]]:
    """The question set for the payload columns at `indices`.

    Plain question dictionaries (`{"type": ..., "instructions": ...,
    "criteria": ...}`), which `typesafe-sdk` accepts as-is. Question names are
    for code only and are never sent to the model, so every instruction names
    its column by state path and header.
    """
    questions: dict[str, dict[str, Any]] = {}
    for i in indices:
        header = columns[i]["header"]
        ref = f"`columns[{i}]` (header {header!r})"
        questions[f"role_{i}"] = {
            "type": "choice",
            "instructions": f"What role does the column {ref} play in this run sheet?",
            "criteria": dict(_ROLE_CRITERIA),
        }
        questions[f"identity_{i}"] = {
            "type": "noul",
            "instructions": f"The column {ref} identifies a client, a person, or a specific client sample.",
            "criteria": dict(_IDENTITY_CRITERIA),
        }
        candidates = _name_candidates(header)
        if candidates is not None:
            questions[f"name_{i}"] = {
                "type": "choice",
                "instructions": (
                    f"Which standard name describes the quantity in the column {ref}? "
                    "Choose the column's own name unless a standard name is the same quantity."
                ),
                "criteria": candidates,
            }
    return questions


def build_state(payload: dict[str, Any]) -> dict[str, Any]:
    """The state every request carries: the task framing plus the screened
    payload columns (identity columns absent, free-text samples redacted)."""
    return {"task": _TASK, "columns": payload["columns"]}


def _make_client(config: NormalizeConfig) -> Any:
    """A `typesafe_sdk.TypeSafeClient` for `config`. The only place the SDK is
    imported; it reads `TYPESAFE_API_KEY` from the environment itself."""
    from typesafe_sdk import TypeSafeClient

    return TypeSafeClient(model=config.typesafe_model, timeout=_REQUEST_TIMEOUT)


class _Answers:
    """The answers from every chunk's `SystemOneResponse`, merged by name."""

    def __init__(self) -> None:
        self.choices: dict[str, Any] = {}
        self.nouls: dict[str, Any] = {}

    def add(self, response: Any) -> None:
        self.choices.update(response.choices)
        self.nouls.update(response.nouls)


def ask(client: Any, payload: dict[str, Any]) -> _Answers:
    """Send every column's questions, `_COLUMNS_PER_REQUEST` columns at a time,
    over the same state. Raises on any API error (the caller falls back)."""
    columns = payload["columns"]
    state = build_state(payload)
    answers = _Answers()
    for start in range(0, len(columns), _COLUMNS_PER_REQUEST):
        indices = range(start, min(start + _COLUMNS_PER_REQUEST, len(columns)))
        answers.add(client.system_one(state=state, questions=build_questions(columns, indices)))
    return answers


def plan_from_answers(
    df: pd.DataFrame,
    payload: dict[str, Any],
    dropped_identity: list[str],
    answers: Any,
    config: NormalizeConfig,
    *,
    anonymizer: Anonymizer | None = None,
) -> NormalizationPlan:
    """Compose a `NormalizationPlan` from typed answers. Pure: no network.

    `answers` needs `.choices` / `.nouls` mappings keyed by question name
    (a `SystemOneResponse` or the merged `_Answers`). A missing answer raises
    `KeyError`, which `propose_plan` treats like any other provider failure.
    """
    offline = {
        c.raw_name: c for c in _offline_columns(df, anonymizer=anonymizer, max_sample=config.max_sample)
    }
    min_conf = config.typesafe_min_confidence
    columns: list[ColumnPlan] = []
    p_target: dict[str, float] = {}

    for i, entry in enumerate(payload["columns"]):
        header = entry["header"]
        p_identity = float(answers.nouls[f"identity_{i}"].noul)
        role_answer = answers.choices[f"role_{i}"]
        role_conf = float(role_answer.confidence)
        probs = ", ".join(f"{k}={float(v):.2f}" for k, v in sorted(role_answer.probabilities.items()))

        if p_identity >= _IDENTITY_THRESHOLD:
            columns.append(
                ColumnPlan(
                    raw_name=header,
                    canonical_name=None,
                    role="identity",
                    is_identity=True,
                    note=f"typesafe: identity p={p_identity:.2f}",
                )
            )
            continue

        if role_conf < min_conf:
            fallback = offline[header]
            columns.append(
                ColumnPlan(
                    raw_name=fallback.raw_name,
                    canonical_name=fallback.canonical_name,
                    role=fallback.role,
                    unit_token=fallback.unit_token,
                    to_base=fallback.to_base,
                    is_identity=fallback.is_identity,
                    redacted=fallback.redacted,
                    note=(
                        f"typesafe: role {role_answer.choice!r} confidence {role_conf:.2f} "
                        f"< {min_conf:.2f}; kept offline guess {fallback.role!r} ({probs})"
                    ),
                )
            )
            if fallback.role == "target":
                p_target[header] = float(role_answer.probabilities.get("target", 0.0))
            continue

        role = role_answer.choice
        if role == "freetext":
            columns.append(
                ColumnPlan(
                    raw_name=header,
                    canonical_name=None,
                    role="freetext",
                    redacted=True,
                    note=f"typesafe: freetext confidence {role_conf:.2f} ({probs})",
                )
            )
            continue

        unit_token, to_base, _ = _unit_and_parse_rate(df[header], entry.get("dtype"))
        base_name, name_note = _choose_base_name(header, i, answers, min_conf)
        columns.append(
            ColumnPlan(
                raw_name=header,
                canonical_name=_with_unit_suffix(base_name, unit_token, to_base),
                role=role,
                unit_token=unit_token,
                to_base=to_base,
                note=f"typesafe: {role} confidence {role_conf:.2f} ({probs}){name_note}",
            )
        )
        if role == "target":
            p_target[header] = float(role_answer.probabilities.get("target", 0.0))

    columns = _one_target(columns, p_target)
    columns = _unique_names(columns)
    merged = _merge_dropped_identity(columns, dropped_identity, [str(c) for c in df.columns])
    plan = NormalizationPlan(columns=merged, created_by="typesafe", model=config.typesafe_model)
    plan.validate()
    return plan


def _choose_base_name(header: str, i: int, answers: Any, min_conf: float) -> tuple[str, str]:
    """The base canonical name and a note fragment. An exact alias is resolved
    by code; otherwise the `name_<i>` choice is used when confident enough,
    else the column's own snake_case name."""
    if f"name_{i}" not in answers.choices:
        return _canonical_base_name(header), ""
    name_answer = answers.choices[f"name_{i}"]
    conf = float(name_answer.confidence)
    if conf >= min_conf:
        return str(name_answer.choice), f"; name {name_answer.choice!r} confidence {conf:.2f}"
    own = snake_canonical(header)
    return own, f"; name {name_answer.choice!r} confidence {conf:.2f} < {min_conf:.2f}, kept {own!r}"


def _one_target(columns: list[ColumnPlan], p_target: dict[str, float]) -> list[ColumnPlan]:
    """Keep only the most probable target; demote any other to `metadata`."""
    targets = [c.raw_name for c in columns if c.role == "target"]
    if len(targets) <= 1:
        return columns
    keep = max(targets, key=lambda name: (p_target.get(name, 0.0), -targets.index(name)))
    out: list[ColumnPlan] = []
    for c in columns:
        if c.role == "target" and c.raw_name != keep:
            c = ColumnPlan(
                raw_name=c.raw_name,
                canonical_name=c.canonical_name,
                role="metadata",
                unit_token=c.unit_token,
                to_base=c.to_base,
                note=f"{c.note}; second outcome column ({keep!r} is the target), kept as metadata",
            )
        out.append(c)
    return out


def _unique_names(columns: list[ColumnPlan]) -> list[ColumnPlan]:
    """Resolve canonical-name collisions in column order: the first column
    keeps the name, a later one falls back to its own snake_case name (with
    the same unit suffix), then to a numeric suffix."""
    seen: set[str] = set()
    out: list[ColumnPlan] = []
    for c in columns:
        name = c.canonical_name
        if name is not None and name in seen:
            name = _with_unit_suffix(snake_canonical(c.raw_name) or "column", c.unit_token, c.to_base)
            n = 2
            stem = name
            while name in seen:
                name = f"{stem}_{n}"
                n += 1
            c = ColumnPlan(
                raw_name=c.raw_name,
                canonical_name=name,
                role=c.role,
                unit_token=c.unit_token,
                to_base=c.to_base,
                is_identity=c.is_identity,
                redacted=c.redacted,
                note=f"{c.note}; renamed to avoid a duplicate canonical name",
            )
        if name is not None:
            seen.add(name)
        out.append(c)
    return out


def live_plan_typesafe(
    df: pd.DataFrame,
    config: NormalizeConfig,
    payload: dict[str, Any],
    dropped_identity: list[str],
    *,
    anonymizer: Anonymizer | None = None,
) -> NormalizationPlan:
    """The live TypeSafe call: ask every question, compose the plan. Raises on
    any failure; `propose_plan` is the one place that falls back."""
    if not payload["columns"]:
        raise ValueError("no columns survived the identity pre-screen; nothing to ask")
    with _make_client(config) as client:
        answers = ask(client, payload)
    return plan_from_answers(df, payload, dropped_identity, answers, config, anonymizer=anonymizer)


__all__ = ["build_questions", "build_state", "ask", "plan_from_answers", "live_plan_typesafe"]
