# deploy/ — container files

Every container file in the repo, what it builds, and from which context.
Operating the stack (deploy, upgrade, rollback, backup drill, security
checklist) is in [RUNBOOK.md](RUNBOOK.md).

## Production stack (`docker compose -f deploy/docker-compose.yaml ...`)

| File | Builds | Build context | Notes |
| --- | --- | --- | --- |
| [docker-compose.yaml](docker-compose.yaml) | the three-service stack: `engine`, `web`, `backup` | — | only `web` publishes a host port; engine state in the `kalos-state` volume; backups to a host dir outside it |
| [Dockerfile.engine](Dockerfile.engine) | `kalos-engine` — FastAPI/uvicorn portal, `kalos[ml,portal,typesafe]`, CPU torch | repo root (filtered by [../.dockerignore](../.dockerignore)) | multi-stage, non-root, `HEALTHCHECK` on `/healthz`; CI builds and boot-tests it |
| [Dockerfile.web](Dockerfile.web) | `kalos-web` — Next.js front end | the sibling `kalos-web` repo (`../../kalos-web`) | needs `output: "standalone"` in kalos-web's `next.config.ts`; not built in CI |
| [backup/Dockerfile](backup/Dockerfile) | `kalos-backup` — SQLite `.backup` sidecar running [backup/backup.sh](backup/backup.sh) | `deploy/backup/` | reads the state volume read-only |
| [.env.example](.env.example) | — | — | copy to `deploy/.env` (never committed); every key documented inline |

```bash
cp deploy/.env.example deploy/.env       # fill in the REQUIRED keys
docker compose -f deploy/docker-compose.yaml build
docker compose -f deploy/docker-compose.yaml up -d
docker compose -f deploy/docker-compose.yaml ps   # engine/web report healthy
```

## Development

| File | Builds | Notes |
| --- | --- | --- |
| [../.devcontainer/Dockerfile](../.devcontainer/Dockerfile) | the dev image: Python 3.12, uv, sqlite3, shellcheck | tools only — no project code baked in |
| [../.devcontainer/devcontainer.json](../.devcontainer/devcontainer.json) | VS Code / Codespaces container | runs [../scripts/setup-dev.sh](../scripts/setup-dev.sh); `.venv` in a named volume; forwards port 8050 |
| [../scripts/setup-dev.sh](../scripts/setup-dev.sh) | `.venv` with `kalos[ml,portal,dev,typesafe]` | the same script on a laptop, in the dev container, or in a cloud session's setup script |

The dev and production images share nothing on purpose: the dev container
mounts the working tree and installs editable, while `Dockerfile.engine`
installs a fixed copy of `kalos/` into a venv and runs as an unprivileged user.

## Checks that guard these files

- `tests/test_deploy_config.py` — compose structure, required env keys, no
  secret-looking literals, non-root users, healthchecks, the backup script's
  syntax, and that `.dockerignore` keeps what `Dockerfile.engine` copies
  while excluding `.venv`, `.git`, and `.env`.
- CI `docker-build` job (`.github/workflows/ci.yml`) — builds the engine image
  and waits for its own `HEALTHCHECK` to report healthy.
