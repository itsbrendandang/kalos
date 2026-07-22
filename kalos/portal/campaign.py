"""Kalos portal — the campaign store: one closed optimization loop.

See docs/CAMPAIGN_LOOP.md for the full contract this module implements. A
*campaign* is one optimization target plus a growing base dataset
(`base_rows`) and a list of `pending` runs — recipes started from a proposed
batch, each either *awaiting* a measured outcome (`result: null`) or
*measured* (`result` filled), waiting to be folded into `base_rows` on the
next re-analyze.

Deliberately kept torch-free: this module only persists data. `_analyze`
(`kalos.portal.analysis`) is the science; `kalos.portal.campaign_routes`
calls it, not this store, so importing `kalos.portal.campaign` never pays the
torch/botorch import tax and the store stays trivially unit-testable.

Persistence mirrors the `_LATEST` pattern in `kalos.portal.app`: one JSON file
under `KALOS_STATE_DIR` (default `~/.kalos`), guarded by a lock, written
atomically (temp file + `os.replace`) so a reader never observes a
half-written file. The env var is read independently here (not imported from
`app.py`'s `_STATE_DIR`) to avoid a circular import — `app.py` seeds the
campaign after every upload, so it must import this module, not the reverse.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


class CampaignError(ValueError):
    """A client-facing campaign operation error: unknown run id, non-finite
    result value, or no active campaign. The message is authored here (never
    a raw parser/library error), so it is always safe for a route handler to
    echo back to the caller as-is."""


def _finite_number(value: Any) -> float | None:
    """`value` as a float if it is a real, finite number (excluding bool,
    which is technically an `int` subclass but never a measured outcome),
    else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _best_measured(base_rows: list[dict[str, Any]], target: str) -> float | None:
    """The best (max) MEASURED target value over `base_rows`, or None when no
    row carries a finite numeric target — never a prediction (docs/CAMPAIGN_LOOP.md,
    "Honesty constraints"). Because base rows only ever grow, this is
    non-decreasing round over round, which is what the progress trajectory plots."""
    values = [v for row in base_rows if (v := _finite_number(row.get(target))) is not None]
    return max(values) if values else None


class CampaignStore:
    """Persists exactly one campaign to `<state_dir>/campaign.json`.

    One campaign at a time (single local user), matching
    docs/CAMPAIGN_LOOP.md. Every read and write is guarded by a lock, and
    every write is atomic — written to a temp file in the same directory,
    then `os.replace`'d into place — so a concurrent reader never observes a
    partially-written file.
    """

    def __init__(self, state_dir: Path | None = None) -> None:
        self._dir = Path(state_dir) if state_dir is not None else Path(
            os.environ.get("KALOS_STATE_DIR", Path.home() / ".kalos")
        )
        self._path = self._dir / "campaign.json"
        self._lock = threading.Lock()

    # --- persistence -------------------------------------------------------- #

    def _read_locked(self) -> dict[str, Any] | None:
        """Read the campaign file. Caller must hold `self._lock`."""
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError):  # a corrupt file must not break the API
            return None
        return data if isinstance(data, dict) else None

    def _write_locked(self, state: dict[str, Any]) -> None:
        """Persist `state` atomically, stamping it with a fresh `generation`
        token. Caller must hold `self._lock`.

        Every write mints a new `generation` (a random token), so any
        intervening write — a `seed()` from a concurrent `/api/run`, a
        `set_result`, another `start` — is detectable: a re-analyze captures
        the generation it planned against (`plan_fold`) and refuses to commit
        (`commit_fold`) if it changed underneath. This is what stops a
        concurrent upload from silently destroying just-folded measured data
        (see docs/CAMPAIGN_LOOP.md, "Re-analyze = the loop closing").
        """
        state["generation"] = uuid.uuid4().hex
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(state))
        os.replace(tmp_path, self._path)

    # --- public API ----------------------------------------------------------- #

    def seed(self, df: pd.DataFrame, target: str, features: list[str]) -> None:
        """Start a FRESH campaign, overwriting any existing one.

        Called after every successful `/api/run` upload: a new upload is a
        new base dataset, so any in-flight pending runs from a previous
        campaign are discarded rather than silently folded into the wrong
        dataset. `target`/`features` come from the analysis result
        (`result["target"]`, `result["proposal_features"]`), never guessed
        from column names — see docs/CAMPAIGN_LOOP.md, "Seeding".
        """
        from kalos.portal.serialization import _json_safe_records  # local: avoids a cycle at import time

        with self._lock:
            base_rows = _json_safe_records(df)
            state: dict[str, Any] = {
                "target": target,
                "features": list(features),
                "base_rows": base_rows,
                "pending": [],
                "round": 0,
                # Progress trajectory: best-so-far at each round, starting at
                # round 0 (the uploaded base). Each reanalyze appends a point.
                "history": [{"round": 0, "best": _best_measured(base_rows, target), "n_base": len(base_rows)}],
                "updated_at": time.time(),
            }
            self._write_locked(state)

    def get(self) -> dict[str, Any] | None:
        """The raw campaign state, or None if no campaign has been seeded yet."""
        with self._lock:
            return self._read_locked()

    def summary(self) -> dict[str, Any]:
        """The `GET /api/campaign` response: `{"has_campaign": False}` before
        any upload has ever seeded a campaign, else the full summary
        (docs/CAMPAIGN_LOOP.md, "GET /api/campaign response").

        `best` is `max(target over base_rows)` — the best MEASURED value so
        far, never a prediction (see "Honesty constraints" in the design
        doc) — and is None when there is no base data yet or the target
        column carries no finite numeric value.
        """
        with self._lock:
            state = self._read_locked()
        if state is None:
            return {"has_campaign": False}

        target = state["target"]
        base_rows: list[dict[str, Any]] = state["base_rows"]
        best = _best_measured(base_rows, target)

        pending = [{**run, "awaiting": run["result"] is None} for run in state["pending"]]
        n_awaiting = sum(1 for run in pending if run["awaiting"])
        return {
            "has_campaign": True,
            "target": target,
            "features": state["features"],
            "n_base": len(base_rows),
            "best": best,
            "round": state["round"],
            # Older campaigns (seeded before history existed) fall back to a
            # single current point so the frontend always has something to plot.
            "history": state.get("history")
            or [{"round": state["round"], "best": best, "n_base": len(base_rows)}],
            "pending": pending,
            "n_awaiting": n_awaiting,
            "n_measured": len(pending) - n_awaiting,
        }

    def start(self, recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Append each proposed recipe as a pending, awaiting run. Returns
        the appended runs (each with its assigned `id`)."""
        with self._lock:
            state = self._read_locked()
            if state is None:
                raise CampaignError("no active campaign; upload a run sheet first")
            now = time.time()
            appended: list[dict[str, Any]] = []
            for recipe in recipes:
                # Validate the recipe mapping HERE, at the point of the bad
                # input, so a malformed body 400s cleanly at /api/campaign/start
                # rather than surfacing as an unhandled 500 rounds later inside
                # fold_and_snapshot's `{**recipe, target: result}` expansion.
                recipe_map = recipe.get("recipe")
                if not isinstance(recipe_map, dict) or not recipe_map:
                    raise CampaignError(
                        "each recipe must include a non-empty 'recipe' mapping of feature -> value"
                    )
                run = {
                    "id": uuid.uuid4().hex,
                    "recipe": recipe_map,
                    "pred": recipe.get("pred"),
                    "std": recipe.get("std"),
                    "mode": recipe.get("mode"),
                    "reason": recipe.get("reason"),
                    "result": None,
                    "created_at": now,
                    "measured_at": None,
                }
                state["pending"].append(run)
                appended.append(run)
            state["updated_at"] = now
            self._write_locked(state)
            return appended

    def set_result(self, run_id: str, value: float) -> dict[str, Any]:
        """Set the measured outcome on a pending run. Raises `CampaignError`
        (never silently ignored) if `run_id` is unknown or `value` is not a
        finite number — a NaN/inf result must never be foldable into the
        base dataset on the next re-analyze."""
        if not math.isfinite(value):
            raise CampaignError(f"result value must be a finite number; got {value!r}")
        with self._lock:
            state = self._read_locked()
            if state is None:
                raise CampaignError(f"no active campaign; unknown run id {run_id!r}")
            for run in state["pending"]:
                if run["id"] == run_id:
                    now = time.time()
                    run["result"] = value
                    run["measured_at"] = now
                    state["updated_at"] = now
                    self._write_locked(state)
                    return run
            raise CampaignError(f"unknown pending run id {run_id!r}")

    def plan_fold(self) -> tuple[pd.DataFrame, str, str]:
        """Compute the folded base dataset WITHOUT persisting anything.

        Returns `(DataFrame(folded_base_rows), target, generation)` where the
        DataFrame is `base_rows` plus every measured pending run folded in, and
        `generation` is the token of the campaign state this plan was built
        from. The caller (`kalos.portal.campaign_routes`) runs `_analyze` on
        the DataFrame and, only if that succeeds, calls `commit_fold(generation)`
        to make the fold durable.

        Nothing is mutated or written here, so a `_analyze` failure — or a
        concurrent `seed()` from a fresh upload — leaves the campaign exactly
        as it was: no lost round, no half-folded base (docs/CAMPAIGN_LOOP.md,
        "Re-analyze = the loop closing"). Raises `CampaignError` if there is no
        active campaign or no measured run to fold (re-analyzing with nothing
        new would only inflate `round` and the progress trajectory).
        """
        with self._lock:
            state = self._read_locked()
            if state is None:
                raise CampaignError("no active campaign; upload a run sheet first")
            target = state["target"]
            folded = list(state["base_rows"])
            measured = 0
            for run in state["pending"]:
                if run["result"] is not None:
                    folded.append({**run["recipe"], target: run["result"]})
                    measured += 1
            if measured == 0:
                raise CampaignError("no measured runs to fold; log at least one result first")
            return pd.DataFrame(folded), target, state["generation"]

    def commit_fold(self, generation: str) -> dict[str, Any]:
        """Make the fold planned by `plan_fold` durable, but ONLY if the
        campaign has not changed since (its `generation` still matches).

        Folds every measured pending run into `base_rows`, drops them from
        `pending` (awaiting runs stay), increments `round`, appends the
        progress-trajectory point, and persists. Returns the updated state.

        Raises `CampaignError` if the campaign was reseeded or otherwise
        written underneath the in-flight re-analysis (`generation` mismatch):
        the analysis the caller just computed describes a base dataset that no
        longer exists, so committing it — and letting the caller overwrite
        `/api/latest` with it — would silently destroy the newer data. The
        caller turns this into a "please retry" 400 and leaves `/api/latest`
        untouched.
        """
        with self._lock:
            state = self._read_locked()
            if state is None or state.get("generation") != generation:
                raise CampaignError(
                    "the campaign changed during re-analysis (a new upload or result "
                    "landed); nothing was committed — please re-analyze again"
                )
            target = state["target"]
            still_pending: list[dict[str, Any]] = []
            for run in state["pending"]:
                if run["result"] is not None:
                    state["base_rows"].append({**run["recipe"], target: run["result"]})
                else:
                    still_pending.append(run)
            state["pending"] = still_pending
            state["round"] += 1
            # Record the new best-so-far for the progress trajectory. base_rows
            # only grew, so this point is >= the previous one.
            history = state.setdefault("history", [])
            history.append(
                {
                    "round": state["round"],
                    "best": _best_measured(state["base_rows"], target),
                    "n_base": len(state["base_rows"]),
                }
            )
            state["updated_at"] = time.time()
            self._write_locked(state)
            return state


# --- module-level singleton, mirroring kalos.portal.experiments.get_store --- #
_STORE: CampaignStore | None = None


def get_campaign_store() -> CampaignStore:
    """The module-level `CampaignStore` at the default
    `~/.kalos/campaign.json`.

    Tests override this via `app.dependency_overrides[get_campaign_store]`
    (same pattern as `kalos.portal.experiments.get_store`), pointing at a
    fresh `CampaignStore(tmp_path)` so they never touch the real
    `~/.kalos/campaign.json`.
    """
    global _STORE
    if _STORE is None:
        _STORE = CampaignStore()
    return _STORE
