"""Tier-2 deterministic multi-sheet workbook merge: classify the sheets of an
uploaded Excel workbook (one RUN-LEVEL sheet, zero or more MERGEABLE
per-topic sheets, the rest UNMERGEABLE) and left-join the mergeable sheets
onto the run-level sheet by their shared run/batch id column.

Real batch records commonly arrive as a multi-sheet workbook: a batch-summary
sheet (one row per run) plus per-topic sheets - media composition, timing,
assays - that also carry one row per run and share a run/batch id column with
the summary sheet. Before this module, `kalos/portal/uploads.py` handed
`pd.read_excel` no `sheet_name` argument, which defaults to `sheet_name=0` -
only the FIRST sheet was ever read; every other sheet in the workbook was
silently dropped (see `kalos/portal/uploads.py::_parse_upload`, the
`pd.read_excel(io.BytesIO(raw))` call, pre-wiring).

Matches kalos's tier-2 "structured extractor for known formats" pattern
(see `docs/ingestion-architecture.md` in the kalos-transition repo, "Tier 2:
structured extractors for known formats" -> "Multi-sheet batch-ticket +
instrument peak-table merge (pattern only)"): the input shape is known in
advance (a run-level sheet plus per-topic sheets sharing an id column), so
the merge is fully deterministic - no LLM call is ever appropriate here. The
one-row-per-run-per-sheet-then-merge-by-id shape also mirrors
`ports/run-summary-compiler/datacompile.py`'s `merge_all` (several per-topic
summaries, each already one row per batch, left-joined onto a master sheet
by `batch_id`) - this module generalizes that pattern to sheets read out of
a single workbook instead of separate CSV files, and adds the classification
step that pattern didn't need (there, which file was the master and which
were per-topic summaries was a fixed argument, not something to detect).

Never guesses. Like `kalos.normalize.orientation`, a sheet (or an entire
workbook) this heuristic cannot confidently classify is reported ambiguous,
never silently forced one way or the other - see `analyze_workbook`'s and
`merge_workbook`'s docstrings for exactly which situations degrade to
"ambiguous"/"unmergeable" and why, and each `SheetClassification.reason`
for a specific, per-sheet explanation.

Id-shape detection reuses `kalos.normalize.orientation._ID_LIKE_RE` (a
common prefix + separator + counter shape: "Batch-1", "RUN-0007", "bq0031",
a bare integer) rather than inventing a second regex for the same concept -
that pattern is exactly what a per-run identifier column looks like,
regardless of whether the pre-pass using it is orientation detection or
workbook merge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from kalos.normalize.orientation import _ID_LIKE_RE

Role = Literal["run_level", "mergeable", "unmergeable", "ambiguous"]

# Fraction of a candidate id column's non-blank values that must match the
# id shape before that column counts as "id-like" at all. Matches
# `orientation.py`'s own `_ID_FRACTION` threshold and rationale: tolerate a
# few stray labels/blanks while still requiring the column to overwhelmingly
# look like a run identifier, not just occasionally.
_ID_MATCH_RATE = 0.7

# A sheet whose non-id columns are mostly long free-text (mean string length
# at or above this, e.g. a "notes"/"comments" column) reads as a free-text
# notes sheet rather than structured per-run data - one of the three
# UNMERGEABLE reasons this module distinguishes (the others being a
# time-series shape and an empty sheet). Chosen generously above ordinary
# short categorical/label values ("Batch-1", "media A") but well below a
# sentence or two of free-text notes.
_FREE_TEXT_MEAN_LEN = 40.0
# Fraction of a sheet's columns that must read as free text for the sheet
# itself to be called a free-text notes sheet.
_FREE_TEXT_COL_FRACTION = 0.5

_SLUG_RE = re.compile(r"[^0-9a-zA-Z]+")


@dataclass(frozen=True)
class SheetClassification:
    """One sheet's classification verdict, always with a stated `reason` -
    "why did the classifier decide this" must be answerable from the report
    itself, matching `OrientationReport.signals`'s provenance discipline.
    """

    role: Role
    reason: str
    id_column: str | None = None
    id_unique_rate: float | None = None
    n_rows: int = 0
    n_cols: int = 0


@dataclass(frozen=True)
class WorkbookReport:
    """`analyze_workbook`'s verdict for one workbook (a name -> DataFrame
    mapping, in workbook sheet order).

    `id_column` is the shared run/batch id column name, or `None` if no
    column name reached the id-shape threshold in at least two sheets (no
    shared id column could be determined at all - see
    `_find_shared_id_column`). `run_level_sheet` is the name of the sheet
    chosen as the merge anchor, or `None` when classification could not
    confidently pick one (no candidate, or a tie between candidates - see
    `analyze_workbook`). `frames` carries the (already orientation-normalized
    by the caller) input sheets unchanged, so `merge_workbook` has everything
    it needs from the report alone.
    """

    id_column: str | None
    run_level_sheet: str | None
    classifications: dict[str, SheetClassification]
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)


class WorkbookMergeError(ValueError):
    """Raised by `merge_workbook` when called on an ambiguous report (no
    confident run-level sheet). Callers must check
    `report.run_level_sheet is not None` before calling `merge_workbook` -
    see `kalos/portal/uploads.py`'s ambiguous-workbook fallback for what to
    do instead (degrade to the first sheet, same as a plain single-sheet
    upload, never raise this into the client).
    """


def _slug(name: object) -> str:
    """Sanitize a sheet name into a lowercase, underscore-joined column-name
    prefix ("Media Composition" -> "media_composition"). Never empty."""
    s = _SLUG_RE.sub("_", str(name).strip().lower()).strip("_")
    return s or "sheet"


def _nonblank_lower(values: "pd.Series[Any]") -> "pd.Series[Any]":
    """String-cast, stripped, lowercased values with blanks/NaN-as-text
    dropped - the shared cleanup `_id_match_rate` and `_find_shared_id_column`
    both need before matching `_ID_LIKE_RE` or counting uniqueness."""
    s = values.astype(str).str.strip().str.lower()
    return s[(s != "") & (s != "nan")]


def _id_match_rate(values: "pd.Series[Any]") -> tuple[float, "pd.Series[Any]"]:
    """Fraction of non-blank `values` matching the id shape, and the
    non-blank values themselves (the caller needs both: the rate to decide
    id-like-ness, the values to then check uniqueness)."""
    nonblank = _nonblank_lower(values)
    if len(nonblank) == 0:
        return 0.0, nonblank
    matches = sum(bool(_ID_LIKE_RE.match(v)) for v in nonblank)
    return matches / len(nonblank), nonblank


def _is_free_text_sheet(df: pd.DataFrame) -> bool:
    """A sheet whose columns are mostly long free-text strings (notes,
    comments) rather than structured per-run values - see
    `_FREE_TEXT_MEAN_LEN`/`_FREE_TEXT_COL_FRACTION`.

    Uses `pd.api.types.is_string_dtype` rather than `dtype == object`: pandas
    3.x's default inferred string dtype for text columns is no longer plain
    `object`, and `is_string_dtype` covers both that and the legacy `object`
    case a column of mixed/unconverted text can still have.
    """
    if df.shape[1] == 0:
        return False
    long_text_cols = 0
    string_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    if not string_cols:
        return False
    for col in string_cols:
        vals = df[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        if vals.str.len().mean() >= _FREE_TEXT_MEAN_LEN:
            long_text_cols += 1
    return (long_text_cols / len(df.columns)) >= _FREE_TEXT_COL_FRACTION


def _find_shared_id_column(sheets: dict[str, pd.DataFrame]) -> str | None:
    """The column NAME that reads as id-like (>= `_ID_MATCH_RATE`) in the
    most sheets - the workbook's merge key candidate.

    Column names are matched case/whitespace-insensitively (a per-topic
    sheet's header capitalization commonly drifts - "Batch ID" vs
    "batch_id"... though only exact-after-casefold matches count, so a
    genuinely different header is never silently treated as the same
    column); the first-seen casing becomes the canonical name. Requires at
    least two sheets to agree before calling a column name "shared" -
    a name that is id-like in only one sheet is not evidence of a shared
    merge key, and returning it here would let a single sheet dictate a
    merge key nothing else can actually join on. Returns `None` (workbook
    ambiguous) rather than guessing when no name clears that bar.
    """
    counts: dict[str, int] = {}
    canonical: dict[str, str] = {}
    for df in sheets.values():
        seen_in_sheet: set[str] = set()
        for col in df.columns:
            key = str(col).strip().casefold()
            if key in seen_in_sheet:
                continue
            rate, _ = _id_match_rate(df[col])
            if rate >= _ID_MATCH_RATE:
                counts[key] = counts.get(key, 0) + 1
                canonical.setdefault(key, str(col))
                seen_in_sheet.add(key)
    if not counts:
        return None
    best_key = max(counts, key=lambda k: counts[k])
    if counts[best_key] < 2:
        return None
    return canonical[best_key]


def analyze_workbook(sheets: dict[str, pd.DataFrame]) -> WorkbookReport:
    """Classify every sheet in `sheets` (name -> DataFrame, in workbook
    order) as RUN-LEVEL, MERGEABLE, or UNMERGEABLE.

    Detection, in order:
      1. Find the workbook's shared id column (`_find_shared_id_column`). If
         none clears the threshold in at least two sheets, the whole
         workbook is ambiguous: every sheet is classified "ambiguous" with
         that reason, `id_column` and `run_level_sheet` are both `None`.
      2. For each sheet: empty -> unmergeable ("empty sheet"). Missing the
         shared id column -> unmergeable, with the reason distinguishing a
         free-text notes sheet (`_is_free_text_sheet`) from a plain
         "missing the shared id column" case. Id column present but its
         values do not clear `_ID_MATCH_RATE` -> unmergeable. Id column
         present, id-shaped, but with REPEATED values -> unmergeable, the
         "time-series with repeated ids" case (a sheet with one row per
         run cannot have duplicate ids; a sheet that does is a time series
         or otherwise multi-row-per-run, and merging it would duplicate
         rows in the result rather than adding columns).
      3. Every sheet with a fully-unique, id-shaped id column is a
         RUN-LEVEL candidate. The candidate with the widest per-run
         coverage (most distinct id values; ties broken by column count,
         i.e. breadth of per-run data) becomes `run_level_sheet`; every
         other candidate becomes "mergeable". A tie on BOTH criteria
         between two or more candidates is never broken by guessing - all
         tied sheets are reported "ambiguous" and `run_level_sheet` stays
         `None` for the whole workbook.
    """
    id_column = _find_shared_id_column(sheets)
    classifications: dict[str, SheetClassification] = {}

    if id_column is None:
        for name, df in sheets.items():
            classifications[name] = SheetClassification(
                role="ambiguous",
                reason=(
                    "no id-like column is shared (matching the run-id shape) "
                    "across two or more sheets in this workbook"
                ),
                n_rows=df.shape[0],
                n_cols=df.shape[1],
            )
        return WorkbookReport(
            id_column=None, run_level_sheet=None, classifications=classifications, frames=dict(sheets)
        )

    candidates: list[tuple[str, int, int]] = []  # (name, n_unique_ids, n_cols)
    for name, df in sheets.items():
        n_rows, n_cols = df.shape
        if n_rows == 0 or df.dropna(how="all").empty:
            classifications[name] = SheetClassification(
                role="unmergeable", reason="empty sheet", n_rows=n_rows, n_cols=n_cols
            )
            continue

        if id_column not in df.columns:
            reason = (
                "free-text notes sheet (no structured, id-linked data)"
                if _is_free_text_sheet(df)
                else f"missing the shared id column '{id_column}'"
            )
            classifications[name] = SheetClassification(
                role="unmergeable", reason=reason, n_rows=n_rows, n_cols=n_cols
            )
            continue

        rate, nonblank = _id_match_rate(df[id_column])
        if len(nonblank) == 0:
            classifications[name] = SheetClassification(
                role="unmergeable",
                reason=f"the '{id_column}' column is empty in this sheet",
                id_column=id_column,
                n_rows=n_rows,
                n_cols=n_cols,
            )
            continue
        if rate < _ID_MATCH_RATE:
            classifications[name] = SheetClassification(
                role="unmergeable",
                reason=(
                    f"'{id_column}' column values do not consistently match the "
                    f"run-id shape ({rate:.0%} match)"
                ),
                id_column=id_column,
                n_rows=n_rows,
                n_cols=n_cols,
            )
            continue

        n_unique = int(nonblank.nunique())
        unique_rate = n_unique / len(nonblank)
        if unique_rate < 1.0:
            classifications[name] = SheetClassification(
                role="unmergeable",
                reason=(
                    f"'{id_column}' has repeated values ({n_unique} unique of "
                    f"{len(nonblank)} rows) - looks like a time-series/multi-row-"
                    "per-run sheet, would duplicate rows if merged"
                ),
                id_column=id_column,
                id_unique_rate=round(unique_rate, 4),
                n_rows=n_rows,
                n_cols=n_cols,
            )
            continue

        candidates.append((name, n_unique, n_cols))

    if not candidates:
        for name, df in sheets.items():
            if name not in classifications:
                classifications[name] = SheetClassification(
                    role="ambiguous",
                    reason="no sheet has a confidently unique run-id column to anchor a merge",
                    n_rows=df.shape[0],
                    n_cols=df.shape[1],
                )
        return WorkbookReport(
            id_column=id_column, run_level_sheet=None, classifications=classifications, frames=dict(sheets)
        )

    candidates.sort(key=lambda c: (-c[1], -c[2]))
    top_unique, top_cols = candidates[0][1], candidates[0][2]
    tied = [c for c in candidates if c[1] == top_unique and c[2] == top_cols]

    if len(tied) > 1:
        tied_names = {c[0] for c in tied}
        for name, n_unique, n_cols in candidates:
            df = sheets[name]
            if name in tied_names:
                classifications[name] = SheetClassification(
                    role="ambiguous",
                    reason=(
                        f"tied with {len(tied_names) - 1} other sheet(s) for widest "
                        f"per-run coverage ({n_unique} unique ids, {n_cols} columns) "
                        "- run-level sheet not guessed"
                    ),
                    id_column=id_column,
                    id_unique_rate=1.0,
                    n_rows=df.shape[0],
                    n_cols=n_cols,
                )
            else:
                classifications[name] = SheetClassification(
                    role="mergeable",
                    reason="shares the run-id column with one row per run",
                    id_column=id_column,
                    id_unique_rate=1.0,
                    n_rows=df.shape[0],
                    n_cols=n_cols,
                )
        return WorkbookReport(
            id_column=id_column, run_level_sheet=None, classifications=classifications, frames=dict(sheets)
        )

    run_level_name = candidates[0][0]
    for name, n_unique, n_cols in candidates:
        df = sheets[name]
        if name == run_level_name:
            classifications[name] = SheetClassification(
                role="run_level",
                reason=(
                    f"widest per-run coverage ({n_unique} unique ids, {n_cols} columns) "
                    "among sheets with a fully-unique run-id column"
                ),
                id_column=id_column,
                id_unique_rate=1.0,
                n_rows=df.shape[0],
                n_cols=n_cols,
            )
        else:
            classifications[name] = SheetClassification(
                role="mergeable",
                reason="shares the run-id column with one row per run",
                id_column=id_column,
                id_unique_rate=1.0,
                n_rows=df.shape[0],
                n_cols=n_cols,
            )

    return WorkbookReport(
        id_column=id_column, run_level_sheet=run_level_name, classifications=classifications, frames=dict(sheets)
    )


def merge_workbook(
    report: WorkbookReport, *, max_columns: int | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Left-join every MERGEABLE sheet in `report` onto the run-level sheet
    by `report.id_column`, and return `(merged_frame, provenance)`.

    `max_columns`, when given, is the caller's total-column ceiling (kalos's
    upload path passes its existing `kalos.portal.uploads.MAX_COLUMNS`
    constant here rather than this module re-declaring its own copy of that
    number). A sheet whose columns would push the running total past the
    cap is excluded from the merge (role "excluded" in `provenance`, never
    silently truncated column-by-column) rather than raising - the caller
    already has a working (smaller) merged frame either way.

    Hard rules, defense in depth:
      - a sheet's id-column uniqueness is REVALIDATED here, not just trusted
        from `report.classifications` - if it is no longer unique at merge
        time, the sheet is demoted to unmergeable-with-reason instead of a
        many-to-one join silently duplicating rows in the result.
      - a column name collision between a sheet being merged and a column
        already present in the merged frame is resolved by prefixing that
        column with the sheet's name (slugified) - and ONLY on an actual
        collision; every rename actually applied is reported in
        `provenance["sheets"][<name>]["renamed_columns"]`, never silent.

    Raises `WorkbookMergeError` if `report.run_level_sheet` is `None` (an
    ambiguous report) - callers must check that before calling, and degrade
    instead (see `kalos/portal/uploads.py`).
    """
    if report.run_level_sheet is None or report.id_column is None:
        raise WorkbookMergeError(
            "merge_workbook requires a confidently-classified run-level sheet; "
            "this report is ambiguous (see report.classifications for why) - "
            "callers must check report.run_level_sheet before calling merge_workbook."
        )

    id_column = report.id_column
    run_level_name = report.run_level_sheet
    merged = report.frames[run_level_name].copy()
    used_names: set[str] = set(merged.columns.astype(str))

    provenance: dict[str, Any] = {
        "id_column": id_column,
        "run_level_sheet": run_level_name,
        "sheets": {
            run_level_name: {
                "role": "run_level",
                "reason": report.classifications[run_level_name].reason,
                "merged": True,
                "rows_contributed": int(len(merged)),
                "columns_added": [],
                "renamed_columns": {},
            }
        },
    }

    for name, frame in report.frames.items():
        if name == run_level_name:
            continue
        cls = report.classifications[name]
        if cls.role != "mergeable":
            provenance["sheets"][name] = {"role": cls.role, "reason": cls.reason, "merged": False}
            continue

        # Defense in depth: revalidate uniqueness at merge time rather than
        # trusting the classifier's earlier verdict - a merge that would
        # duplicate rows is demoted here too, never silently performed.
        _, nonblank = _id_match_rate(frame[id_column])
        n_unique = int(nonblank.nunique())
        if len(nonblank) == 0 or n_unique < len(nonblank):
            provenance["sheets"][name] = {
                "role": "unmergeable",
                "reason": (
                    f"id column not unique at merge time ({n_unique} unique of "
                    f"{len(nonblank)} rows) - demoted rather than duplicating rows"
                ),
                "merged": False,
            }
            continue

        other_cols = [c for c in frame.columns if c != id_column]
        rename_map: dict[str, str] = {}
        prefix = _slug(name)
        for col in other_cols:
            col_str = str(col)
            if col_str in used_names:
                candidate = f"{prefix}_{col_str}"
                suffix = 2
                while candidate in used_names:
                    candidate = f"{prefix}_{col_str}__{suffix}"
                    suffix += 1
                rename_map[col] = candidate

        final_cols = [rename_map.get(c, c) for c in other_cols]

        if max_columns is not None and len(used_names) + len(final_cols) > max_columns:
            provenance["sheets"][name] = {
                "role": "excluded",
                "reason": (
                    f"merging this sheet's {len(final_cols)} column(s) would exceed "
                    f"the {max_columns}-column cap ({len(used_names)} already used)"
                ),
                "merged": False,
            }
            continue

        to_merge = frame[[id_column, *other_cols]].rename(columns=rename_map)
        merged = merged.merge(to_merge, on=id_column, how="left")
        used_names.update(str(c) for c in final_cols)

        provenance["sheets"][name] = {
            "role": "mergeable",
            "reason": cls.reason,
            "merged": True,
            "rows_contributed": int(frame[id_column].notna().sum()),
            "columns_added": final_cols,
            "renamed_columns": rename_map,
        }

    return merged, provenance


__all__ = [
    "Role",
    "SheetClassification",
    "WorkbookReport",
    "WorkbookMergeError",
    "analyze_workbook",
    "merge_workbook",
]
