# Common tasks. `make help` lists them. Every target runs inside .venv
# (`make setup` builds it), so nothing touches the system Python.
PY ?= .venv/bin/python

.PHONY: help setup lint typecheck test check run image up clean

help: ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup: ## build .venv with kalos[ml,portal,dev,typesafe] (scripts/setup-dev.sh)
	bash scripts/setup-dev.sh

lint: ## ruff over the package, tests, and examples (same as CI)
	$(PY) -m ruff check src/ tests/ examples/

typecheck: ## mypy (config in pyproject.toml)
	$(PY) -m mypy

test: ## the full pytest suite (slow: GP benchmark sweeps)
	$(PY) -m pytest -q

check: lint typecheck test ## everything CI runs before the image build

run: ## serve the portal at http://127.0.0.1:8050
	$(PY) -m kalos.portal

image: ## build the engine image (Docker; Podman: add --format docker)
	docker build -f deploy/engine/Containerfile -t kalos-engine:local .

up: ## run the full stack (fill in deploy/.env first; see deploy/README.md)
	docker compose -f deploy/compose.yaml up -d --build

clean: ## remove caches and build output
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -not -path './.venv/*' -exec rm -rf {} +
