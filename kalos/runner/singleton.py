"""The Singleton runner (`docs/M2_INTEGRATION.md`, "Singleton runner").

Pulls `READY` experiment JSON through a `BackendAdapter`, runs the existing BO
engine (`kalos.portal.app._analyze`, called verbatim - this module adds
orchestration, not new science), pushes the processed result back, and marks
the experiment `DONE` (or `FAILED` with an actionable message). A PID/lock
file at `~/.kalos/runner.lock` (path injectable) guarantees only one runner
instance processes experiments at a time, with stale-lock detection so a
crashed runner does not wedge the system.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kalos.core.replicates import noise_report
from kalos.portal.app import UploadRejected, _analyze
from kalos.runner.adapter import BackendAdapter
from kalos.store.models import Status

try:
    from kalos import __version__ as _PACKAGE_VERSION
except ImportError:  # pragma: no cover - kalos always importable in practice
    _PACKAGE_VERSION = "unknown"

DEFAULT_LOCK_PATH = Path.home() / ".kalos" / "runner.lock"

# A lock older than this is assumed abandoned by a crashed/killed runner even
# if its pid happens to be reused by an unrelated process by the time we look.
_DEFAULT_STALE_SECONDS = 3600.0

def _err_zero_variance_target(target: str) -> str:
    return (
        f"The target column {target!r} has no variance (all values are "
        "identical); it cannot be modeled. Check you selected the "
        "measured-output column."
    )


def _engine_version() -> str:
    return _PACKAGE_VERSION


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, just owned by someone else - still alive from our perspective.
        return True
    return True


class SingletonLock:
    """A PID/lock file at `path` guarding single-instance execution.

    `acquire()` returns `False` (never raises) if another live, non-stale
    runner already holds the lock. A lock is stale if its recorded pid is no
    longer alive, or if it is older than `stale_after` seconds - either way a
    fresh `acquire()` reclaims it.
    """

    def __init__(self, path: str | Path | None = None, *, stale_after: float = _DEFAULT_STALE_SECONDS) -> None:
        self.path = Path(path) if path is not None else DEFAULT_LOCK_PATH
        self.stale_after = stale_after
        self._held = False

    def _is_stale(self) -> bool:
        try:
            info = json.loads(self.path.read_text())
            pid = int(info["pid"])
            started_at = float(info["started_at"])
        except (OSError, ValueError, KeyError, TypeError):
            return True  # unreadable/malformed lock file: treat as abandoned
        if not _pid_alive(pid):
            return True
        return (time.time() - started_at) > self.stale_after

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self._is_stale():
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as fh:
            json.dump({"pid": os.getpid(), "started_at": time.time()}, fh)
        self._held = True
        return True

    def release(self) -> None:
        if self._held:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self._held = False

    def __enter__(self) -> "SingletonLock":
        if not self.acquire():
            raise RuntimeError(f"another runner instance holds the lock at {self.path}")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


@dataclass
class RunResult:
    """Per-experiment outcome of a run attempt, returned by `run_one`/`run_ready`."""

    id: str
    status: str  # "DONE" | "FAILED" | "SKIPPED"
    error: str | None = None


def _payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Build the run-sheet DataFrame `_analyze` expects from the Experiment
    payload contract: `{"columns": [...], "rows": [{col: val, ...}, ...]}`."""
    columns = payload.get("columns") or None
    rows = payload.get("rows") or []
    return pd.DataFrame(rows, columns=columns)


def _compute_noise_floor(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, float | None] | None:
    """When `config.replicate_group_by` is set, run the EXISTING replicate
    noise-floor path (`kalos.core.replicates.noise_report`) over those columns
    against the target, verbatim - this is a pass-through, not a second
    aggregation implementation. Returns `None` when the config does not
    specify replicate columns, when they (or the target) are missing from the
    payload, or when nothing in the sheet is actually replicated.
    """
    group_cols = config.get("replicate_group_by")
    target = config.get("target")
    if not group_cols or not target:
        return None
    missing = [c for c in group_cols if c not in df.columns]
    if missing or target not in df.columns:
        return None
    X = df[group_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if X.shape[0] < 2:
        return None
    report = noise_report(X, y)
    if report["n_replicated"] == 0:
        return None
    icc = report["icc"]
    noise_var = report["noise_var"]
    sigma = float(np.sqrt(noise_var)) if noise_var == noise_var else None  # NaN check
    return {"icc": None if icc != icc else float(icc), "sigma": sigma}


def _check_target_variance(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Reject a payload whose target column has no variation before it ever
    reaches the GP fit. `_analyze` does not itself guard this (a
    constant-target sheet silently fits with a NaN Spearman rather than
    raising), so the Singleton adds this one pre-flight check using the same
    exception type (`UploadRejected`) and generic-actionable-message pattern
    the portal's own upload guards already use - not a second analysis path,
    just an input-validity guard in front of the existing one.
    """
    target = config.get("target")
    columns = df.columns
    if target is None or target not in columns:
        return  # let `_analyze` pick + validate the target itself
    y = pd.to_numeric(df[target], errors="coerce").dropna()
    if len(y) >= 2 and float(y.std()) <= 1e-12:
        raise UploadRejected(_err_zero_variance_target(str(target)))


def _run_one_locked(adapter: BackendAdapter, exp_id: str, *, force: bool = False) -> RunResult:
    """`run_one`'s body, assuming the Singleton lock is already held."""
    exp = adapter.fetch(exp_id)
    status = exp.status

    if status == Status.PROCESSING:
        # Idempotent: never double-process an experiment that is (or still
        # looks) in flight.
        return RunResult(id=exp_id, status="SKIPPED")
    if status == Status.DONE:
        if not force:
            raise ValueError(
                f"experiment {exp_id!r} is already DONE; pass force=True to re-run it"
            )
        adapter.set_status(exp_id, Status.READY, force=True)
    elif status == Status.FAILED:
        adapter.set_status(exp_id, Status.READY)
    elif status == Status.DRAFT:
        adapter.set_status(exp_id, Status.READY)
    elif status != Status.READY:  # pragma: no cover - exhaustive over Status
        raise ValueError(f"experiment {exp_id!r} has unknown status {status!r}")

    adapter.set_status(exp_id, Status.PROCESSING)

    try:
        df = _payload_to_frame(exp.payload)
        config = exp.config
        _check_target_variance(df, config)
        result = _analyze(
            df,
            config.get("target"),
            anonymize=bool(config.get("anonymize", False)),
        )
        provenance: dict[str, Any] = {
            "seed": result.get("seed"),
            "engine_version": result.get("engine_version") or _engine_version(),
            "processed_at": _now_iso(),
        }
        noise_floor = _compute_noise_floor(df, config)
        if noise_floor is not None:
            provenance["noise_floor"] = noise_floor
        adapter.push_result(exp_id, result, provenance)
        return RunResult(id=exp_id, status="DONE")
    except UploadRejected as rej:
        message = str(rej)
    except Exception as exc:  # noqa: BLE001 - any analysis failure must become FAILED, not a crash
        message = str(exc)
    adapter.set_status(exp_id, Status.FAILED, error=message)
    return RunResult(id=exp_id, status="FAILED", error=message)


def run_one(
    adapter: BackendAdapter,
    exp_id: str,
    *,
    force: bool = False,
    lock_path: str | Path | None = None,
) -> RunResult:
    """Run exactly one experiment now, on demand (CLI `--id`).

    Acquires the Singleton lock for the duration of the run. Raises
    `RuntimeError` if another runner instance already holds it.
    """
    with SingletonLock(lock_path):
        return _run_one_locked(adapter, exp_id, force=force)


def run_ready(
    adapter: BackendAdapter,
    *,
    lock_path: str | Path | None = None,
) -> list[RunResult]:
    """Run every `READY` experiment now (CLI `--once`/`--watch`).

    Acquires the Singleton lock once for the whole batch. If the lock is
    already held by another runner, returns `[]` immediately (a no-op, not an
    error) rather than racing it.
    """
    lock = SingletonLock(lock_path)
    if not lock.acquire():
        return []
    try:
        ids = adapter.list_ready()
        return [_run_one_locked(adapter, exp_id) for exp_id in ids]
    finally:
        lock.release()
