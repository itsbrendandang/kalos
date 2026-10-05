# Kalos

**Kalos tells a bioprocess lab which experiments to run next.** Upload a
spreadsheet of past runs (recipe settings and the titer each one produced).
Kalos checks the data, learns how the settings drive the outcome, says how far
that model can be trusted, and proposes the next batch of recipes most likely
to improve it. Then you run them, log the results, and it proposes again.

Under the hood it is Bayesian optimization on BoTorch (a Gaussian-process
surrogate plus an acquisition function), with leakage-safe evaluation that
reports a weak model as weak instead of overselling it.

## How it works

| Step | What happens | Code |
| --- | --- | --- |
| 1. Ingest | Parse CSV / TSV / Excel under hard size caps | `portal/uploads.py` |
| 2. Validate | Eleven data-quality checks, unit conversion, physically possible bounds | `validation/` |
| 3. Read the columns | Decide target, features, groups, and ids — header rules by default; TypeSafe or an LLM on request | `normalize/`, `domains/` |
| 4. Model and evaluate | GP surrogate; grouped cross-validation with confidence intervals; signed drivers; conformal bands; an XGBoost baseline the GP must beat | `core/`, `portal/analysis.py` |
| 5. Propose | The next batch (qLogNEI), avoiding recipes likely to fail and runs already in flight | `core/optimize.py` |
| 6. Loop | Log measured results, re-analyze on the grown dataset, propose again | `portal/campaign.py` |

Paths are under `src/kalos/`. The module map is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

```bash
make setup                  # .venv with the engine, portal, dev tools (scripts/setup-dev.sh)
source .venv/bin/activate
make run                    # portal at http://127.0.0.1:8050
curl -F file=@examples/synthetic_bioprocess/synthetic_bioprocess.csv http://127.0.0.1:8050/api/run
```

Or install only what you need:

| Install | Gives you |
| --- | --- |
| `pip install -e .` | the torch-free toolkit (`kalos.kit`: leakage-safe splits, drivers, conformal intervals, promotion gates) |
| `pip install -e ".[ml,portal]"` | the BoTorch engine and the web portal |
| `".[xgboost]"` · `".[typesafe]"` · `".[normalize]"` · `".[protein]"` | the XGBoost baseline · TypeSafe column decisions · the Anthropic normalize tier · real ESM-2 protein embeddings |

Every API key is optional; [.env.example](.env.example) lists them. Nothing
loads `.env` automatically: `set -a; . ./.env; set +a` exports it.

## Project layout

```
src/kalos/       the package: core/ (GP, acquisition, evaluation), portal/ (FastAPI), normalize/,
                 validation/, domains/, store/, runner/, scale/, bench/, providers/, kit/
tests/           pytest suite (the CI gate)
examples/        runnable demos and a synthetic dataset
experiments/     research prototypes, not shipped
docs/            architecture, API, roadmap, design notes
deploy/          container images (one directory per image) and the compose stack
scripts/         dev-environment setup
.devcontainer/   VS Code / Codespaces dev container
Makefile         setup, lint, typecheck, test, run, image (`make help`)
```

## Development

```bash
make check      # lint + typecheck + tests, what CI runs (the full suite takes a while: GP benchmark sweeps)
make lint       # ruff
make typecheck  # mypy
make test       # pytest
```

CI also builds the engine image and boots it until its healthcheck passes. The
same environment rebuilds in the dev container or a cloud session from
`scripts/setup-dev.sh`.

## Deploy

```bash
cp deploy/.env.example deploy/.env   # fill in the required keys
mkdir -p deploy/backups && sudo chown 10001:10001 deploy/backups   # the backup sidecar runs as uid 10001
make up                              # docker compose -f deploy/compose.yaml up -d --build
```

[deploy/README.md](deploy/README.md) lists every container file;
[deploy/RUNBOOK.md](deploy/RUNBOOK.md) covers upgrades, backups, and security.

## Documentation

| Doc | For |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | module map, experiment store, campaign loop, other domains |
| [docs/API.md](docs/API.md) | `POST /api/run`: upload guarantees, `roles=auto`, the XGBoost baseline |
| [docs/TYPESAFE.md](docs/TYPESAFE.md) | TypeSafe column decisions and their settings |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | does the optimizer beat a space-filling design? |
| [docs/ROADMAP.md](docs/ROADMAP.md) | what to work on next |
| [docs/README.md](docs/README.md) | every other doc |
| [CHANGELOG.md](CHANGELOG.md) | what changed, newest first |

No proprietary data lives in this repo; the demos use synthetic data and public
protein sequences.
