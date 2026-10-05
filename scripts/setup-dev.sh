#!/usr/bin/env bash
# Build (or refresh) the kalos development environment in ./.venv.
#
# One script for every place a fresh machine needs the environment back:
#   - the devcontainer (.devcontainer/devcontainer.json runs it on create),
#   - a Claude Code cloud environment's setup script (paste
#     `bash scripts/setup-dev.sh` there; the container is cached afterwards),
#   - a laptop: `bash scripts/setup-dev.sh && source .venv/bin/activate`.
#
# Installs kalos[ml,portal,dev,typesafe] - the set CI installs, plus the
# TypeSafe tier - with CPU-only torch first so pip never resolves the
# multi-GB CUDA build. Idempotent: an existing .venv is reused and only
# missing packages are installed. Non-interactive.
#
#   KALOS_EXTRAS   override the extras (default: ml,portal,dev,typesafe)
#   KALOS_VENV     override the venv path (default: .venv)
#   KALOS_LEADGENE=1  also install experiments/leadgene_pipeline/requirements.txt
set -euo pipefail

cd "$(dirname "$0")/.."

VENV="${KALOS_VENV:-.venv}"
EXTRAS="${KALOS_EXTRAS:-ml,portal,dev,typesafe}"

if command -v uv >/dev/null 2>&1; then
  [ -x "$VENV/bin/python" ] || uv venv -q "$VENV" --python 3.12 || uv venv -q "$VENV"
  pip_install() { uv pip install -q --python "$VENV/bin/python" "$@"; }
else
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  pip_install() { "$VENV/bin/python" -m pip install -q "$@"; }
fi

if [[ ",$EXTRAS," == *",ml,"* ]] && ! "$VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
  # Some networks (including locked-down cloud sandboxes) block PyTorch's own
  # index; PyPI's default torch wheel is larger but works the same on CPU.
  pip_install torch --index-url https://download.pytorch.org/whl/cpu \
    || { echo "setup-dev: PyTorch CPU index unreachable; installing torch from PyPI" >&2; pip_install torch; }
fi
pip_install -e ".[$EXTRAS]"

if [ "${KALOS_LEADGENE:-0}" = "1" ]; then
  pip_install -r experiments/leadgene_pipeline/requirements.txt
fi

"$VENV/bin/python" - <<'PY'
import importlib.util
import kalos

found = {m: importlib.util.find_spec(m) is not None for m in ("torch", "botorch", "fastapi", "typesafe_sdk", "pytest")}
print("kalos", kalos.__file__)
print("  " + "  ".join(f"{m}={'yes' if ok else 'no'}" for m, ok in found.items()))
PY
echo "ready: source $VENV/bin/activate  (then: python -m pytest -q)"
