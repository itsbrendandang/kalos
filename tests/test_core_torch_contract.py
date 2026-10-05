"""Guard: `kalos.core` modules stay torch-free unless explicitly allowlisted.

`kalos.core.__init__` documents that the package is torch-free except for
`surrogate.py`, `optimize.py`, and `multiobjective.py` - the three modules
that actually define the GP surrogate / acquisition machinery and therefore
need torch/botorch/gpytorch at import time. Everything else must defer any
such import to inside a function body, because `import kalos.core` (which
every sibling submodule runs first) must not drag in the ~220 MB torch/
botorch/gpytorch RSS before a fit is ever requested - that is what keeps the
idle `--watch` poller and portal boot cheap.

This is a static check, not a runtime one (contrast `test_kit_torch_free.py`,
which asserts `kalos.kit` does not load torch as a *runtime* side effect of
import, in a subprocess). Parsing with `ast` instead of grepping means a
string that merely mentions "torch" in a docstring or comment does not trip
this test, and an import nested inside a function body - the exact pattern
that keeps a module's callers lazy - correctly does NOT count as top-level,
because it never appears in `ast.Module.body`.
"""
from __future__ import annotations

import ast
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent.parent / "src" / "kalos" / "core"

# Deliberate exceptions. Adding a file here is a deliberate decision about
# the ~220 MB torch budget, not a quick fix for a failing test - only do it
# if the module genuinely needs torch/botorch/gpytorch at import time, and
# make sure its callers still import it lazily (see kalos/core/__init__.py
# and kalos/__init__.py) so the rest of the contract holds.
ALLOWED_TORCH_MODULES = {"surrogate.py", "optimize.py", "multiobjective.py"}

_TORCH_ROOTS = {"torch", "botorch", "gpytorch"}


def _top_level_torch_import(path: Path) -> str | None:
    """Return the offending import name if `path` imports torch/botorch/
    gpytorch at module top level, else None.

    Only `tree.body` (the module's direct top-level statements) is checked.
    An import inside a function or method body lives in that function's own
    body list, not the module's, so it is never visited here - which is
    correct: a nested import only runs (and only pays the torch tax) when
    the function is actually called.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _TORCH_ROOTS:
                    return alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.split(".")[0] in _TORCH_ROOTS:
                return node.module
    return None


def test_core_modules_outside_allowlist_defer_torch_imports() -> None:
    offenders: dict[str, str] = {}
    for path in sorted(CORE_DIR.glob("*.py")):
        if path.name in ALLOWED_TORCH_MODULES:
            continue
        offending = _top_level_torch_import(path)
        if offending is not None:
            offenders[path.name] = offending

    assert not offenders, (
        f"kalos/core/ modules outside the allowlist {sorted(ALLOWED_TORCH_MODULES)} "
        "must not import torch/botorch/gpytorch at module top level. Doing so "
        "drags the ~220 MB torch/botorch/gpytorch stack into `import kalos.core` "
        "and everything that transitively imports it - including the idle "
        "`--watch` poller and portal boot - even when no GP fit is ever "
        f"requested. Found top-level imports in: {offenders}. If a new module "
        "genuinely needs torch at import time, adding it to ALLOWED_TORCH_MODULES "
        "here is a deliberate decision about that budget, not a fix for this "
        "test - and its callers must keep importing it lazily so the contract "
        "still holds in practice."
    )


def test_allowlist_entries_still_exist() -> None:
    # Keeps the allowlist honest: if an allowlisted module is renamed or
    # removed, the allowlist should be updated in the same change rather
    # than silently going stale.
    existing = {p.name for p in CORE_DIR.glob("*.py")}
    stale = ALLOWED_TORCH_MODULES - existing
    assert not stale, f"ALLOWED_TORCH_MODULES references files that no longer exist: {stale}"
