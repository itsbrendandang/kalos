"""Session-wide sandbox for the kalos home directory.

Kalos resolves every piece of its on-disk state from `Path.home() / ".kalos"`:

- `kalos/portal/app.py` `_STATE_DIR` / `_LATEST_DIR` (per-tenant
  `latest/<tenant>.json`), bound ONCE at import time from
  `os.environ["KALOS_STATE_DIR"]` or `Path.home() / ".kalos"`.
- `kalos/portal/campaign.py` `CampaignStore.__init__` (`portal.db`), resolved
  per instance from the same env var or `Path.home() / ".kalos"`.
- `kalos/store/sqlite_store.py` `DEFAULT_DB_PATH` (`experiments.db`), a module
  constant bound at import time.
- `kalos/runner/singleton.py` `DEFAULT_LOCK_PATH` (`runner.lock`), likewise.

`Path.home()` is therefore the single resolver every entry point funnels
through, so this module rebinds it to a per-session temp directory BEFORE any
kalos module is imported (conftest is imported ahead of test collection, which
is what makes the import-time constants above bind to the sandbox rather than
to the developer's real home).

That rebinding is a plain module-level assignment, not a `monkeypatch`, which
is the whole point: a leaked function-scoped `monkeypatch.undo()` - the exact
failure mode that let 25 test campaigns land in a real `~/.kalos/portal.db` -
reverts a test's own patches back to their import-time values, and those values
are now inside the sandbox. Session state cannot be undone by a function-scoped
fixture.

On top of the redirect, the low-level filesystem sinks are wrapped so that any
path under the REAL `~/.kalos` (only reachable now by hardcoding it) raises
`RealHomeViolation` and is recorded; an autouse fixture then fails the offending
test by name, even when the access happened on a worker thread or inside an
`except OSError: pass` handler.
"""
from __future__ import annotations

import builtins
import io
import os
import pathlib
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from typing import Any

import pytest

# The developer's REAL home, captured before anything is redirected.
REAL_HOME: pathlib.Path = pathlib.Path(os.path.expanduser("~"))
REAL_KALOS: pathlib.Path = REAL_HOME / ".kalos"
_REAL_KALOS_STR: str = str(REAL_KALOS)

# The sandbox that stands in for it for the whole session.
SANDBOX_HOME: pathlib.Path = pathlib.Path(tempfile.mkdtemp(prefix="kalos-test-home-"))


class RealHomeViolation(BaseException):
    """A test touched a path inside the developer's real `~/.kalos`.

    Derives from `BaseException`, not `Exception`, on purpose: kalos swallows
    broad exception groups around its best-effort disk writes (e.g.
    `_save_latest`'s `except (OSError, TypeError, ValueError): pass`), and a
    guard that can be swallowed is not a guard.
    """


# Violations are recorded as well as raised, so an access on a worker thread
# (where the raise cannot reach the test) still fails the test at teardown.
_VIOLATIONS: list[str] = []


def _is_real_home_path(candidate: Any) -> bool:
    """True if `candidate` names a path at or under the real `~/.kalos`."""
    if not isinstance(candidate, (str, bytes, os.PathLike)):
        return False  # file descriptors, None, sqlite ":memory:" handles, ...
    try:
        text = os.fsdecode(candidate)
    except (TypeError, ValueError):
        return False
    if ".kalos" not in text:  # cheap reject for the overwhelming majority
        return False
    resolved = os.path.abspath(os.path.expanduser(text))
    return resolved == _REAL_KALOS_STR or resolved.startswith(_REAL_KALOS_STR + os.sep)


def _check(operation: str, *candidates: Any) -> None:
    for candidate in candidates:
        if _is_real_home_path(candidate):
            detail = f"{operation}({os.fsdecode(candidate)!r}) targets the real {REAL_KALOS}"
            _VIOLATIONS.append(detail)
            raise RealHomeViolation(
                f"Blocked: {detail}. Tests must never read or write the developer's "
                f"real kalos home; point the path at tmp_path (see tests/conftest.py)."
            )


# Path-parameter name(s) for each `os` sink, keyed by attribute name. A
# keyword-only call - `os.mkdir(path="...")` - uses the sink's own parameter
# name, which differs per function, so each entry lists exactly the keyword(s)
# that can carry a path for that sink. `rename`/`replace` list two: source and
# destination.
_OS_SINK_PATH_KEYWORDS: dict[str, tuple[str, ...]] = {
    "open": ("path",),  # os.open, the low-level fd open - not builtins.open
    "mkdir": ("path",),
    "makedirs": ("name",),
    "remove": ("path",),
    "unlink": ("path",),
    "rmdir": ("path",),
    "rename": ("src", "dst"),
    "replace": ("src", "dst"),
}


def _guarded(
    operation: str, real: Callable[..., Any], path_keywords: tuple[str, ...] = ()
) -> Callable[..., Any]:
    """Wrap `real` so its path argument(s) are checked whether passed
    positionally or by keyword.

    `path_keywords` names exactly the keyword argument(s) that can carry a
    path for this sink (e.g. `("file",)` for `open`, `("src", "dst")` for
    `os.rename`) - deliberately narrow, so an unrelated string keyword (a
    mode, a flag) is never mistaken for a path and flagged.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        keyword_candidates = (kwargs[key] for key in path_keywords if key in kwargs)
        _check(operation, *args[:2], *keyword_candidates)
        return real(*args, **kwargs)

    return wrapper


def _install_sandbox() -> None:
    """Redirect `Path.home()` and wrap the filesystem sinks. Runs at import."""
    pathlib.Path.home = classmethod(lambda cls: SANDBOX_HOME)  # type: ignore[method-assign]

    # `builtins.open` and `io.open` are separate references to the same
    # function; `Path.open`/`read_text`/`write_text` go through `io.open`.
    # Its path parameter is named `file`.
    guarded_open = _guarded("open", builtins.open, ("file",))
    builtins.open = guarded_open  # type: ignore[assignment]
    io.open = guarded_open  # type: ignore[assignment]

    # `Path.mkdir`/`Path.unlink`/`os.makedirs` delegate to these `os` globals.
    for name, path_keywords in _OS_SINK_PATH_KEYWORDS.items():
        setattr(os, name, _guarded(f"os.{name}", getattr(os, name), path_keywords))

    # sqlite opens its database file in C, bypassing every wrapper above.
    # Its path parameter is named `database`.
    sqlite3.connect = _guarded("sqlite3.connect", sqlite3.connect, ("database",))  # type: ignore[assignment]


_install_sandbox()


class HomeGuard:
    """Handle on the sandbox, exposed to tests through the `home_guard` fixture."""

    real_home: pathlib.Path = REAL_HOME
    real_kalos: pathlib.Path = REAL_KALOS
    sandbox: pathlib.Path = SANDBOX_HOME
    violation: type[BaseException] = RealHomeViolation

    def pop_violations(self) -> list[str]:
        """Drain the recorded violations so a test that provoked one on
        purpose does not fail at teardown."""
        drained = list(_VIOLATIONS)
        _VIOLATIONS.clear()
        return drained


@pytest.fixture(scope="session", autouse=True)
def kalos_home_sandbox() -> Iterator[pathlib.Path]:
    """Hold the redirected kalos home for the whole session and clean it up."""
    assert pathlib.Path.home() == SANDBOX_HOME, "the kalos home sandbox was not installed"
    yield SANDBOX_HOME
    shutil.rmtree(SANDBOX_HOME, ignore_errors=True)


@pytest.fixture(autouse=True)
def fail_on_real_home_access(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail the test that touched the real `~/.kalos`, naming it.

    The raise inside the sink is the immediate signal; this is the backstop for
    the cases where the raise cannot reach the test - a worker thread, or a
    swallowed exception.
    """
    _VIOLATIONS.clear()
    yield
    if _VIOLATIONS:
        detail = "; ".join(_VIOLATIONS)
        _VIOLATIONS.clear()
        pytest.fail(f"{request.node.nodeid} touched the real kalos home: {detail}")


@pytest.fixture
def home_guard() -> HomeGuard:
    return HomeGuard()
