# deploy/ — container files

One directory per image, each with a `Containerfile` (Docker and Podman both
read it), plus one `compose.yaml` for the stack. Operating it — deploy,
upgrade, rollback, backup drill, Podman notes, security checklist — is in
[RUNBOOK.md](RUNBOOK.md); its "Container files" section is the detailed
version of this page.

## Production stack

| Path | Image | Build context |
| --- | --- | --- |
| [compose.yaml](compose.yaml) | the stack: `engine`, `web`, `backup` (only `web` publishes a port) | — |
| [engine/Containerfile](engine/Containerfile) | `kalos-engine` — FastAPI portal, `kalos[ml,portal,typesafe,xgboost]`, CPU torch; non-root, `HEALTHCHECK` on `/healthz` | repo root, filtered by the [../.containerignore](../.containerignore) allowlist |
| `Containerfile` in the kalos-web repo | `kalos-web` — Next.js front end, proxies to the engine with a bearer token | the sibling `kalos-web` checkout |
| [backup/Containerfile](backup/Containerfile) | `kalos-backup` — SQLite `.backup` sidecar running [backup/backup.sh](backup/backup.sh) | `deploy/backup/` |
| [.env.example](.env.example) | — copy to `deploy/.env` (never committed); required keys documented inline, optional provider keys listed and documented in the repo-root `.env.example` | — |

```bash
cp deploy/.env.example deploy/.env            # fill in the REQUIRED keys
mkdir -p deploy/backups && sudo chown 10001:10001 deploy/backups   # writable by the backup sidecar (uid 10001)
docker compose -f deploy/compose.yaml up -d --build
docker compose -f deploy/compose.yaml ps      # engine/web report healthy
# Podman: podman compose -f deploy/compose.yaml up -d --build  (see RUNBOOK, "Podman")
```

## Development

| Path | What it is |
| --- | --- |
| [../.devcontainer/Containerfile](../.devcontainer/Containerfile) | the dev image: Python 3.12, uv, sqlite3, shellcheck — tools only, no project code |
| [../.devcontainer/devcontainer.json](../.devcontainer/devcontainer.json) | VS Code / Codespaces container: runs [../scripts/setup-dev.sh](../scripts/setup-dev.sh), keeps `.venv` in a named volume, forwards port 8050 |
| [../scripts/setup-dev.sh](../scripts/setup-dev.sh) | `.venv` with `kalos[ml,portal,dev,typesafe]` — the same script on a laptop, in the dev container, or in a cloud session |

The dev and production images share nothing on purpose: the dev container
mounts the working tree and installs editable; `engine/Containerfile`
installs a fixed copy of `kalos/` into a venv and runs unprivileged. The dev
container builds from `.devcontainer/` as its context, so the repo-root
allowlist does not apply to it.

## Checks that guard these files

- `tests/test_deploy_config.py` — compose structure, required env keys, no
  secret-looking literals, non-root users, healthchecks, the backup script's
  syntax, and the build-context allowlist (it must re-include every path
  `engine/Containerfile` copies, and `.dockerignore` must stay a link to
  `.containerignore`).
- CI `docker-build` job (`.github/workflows/ci.yml`) — builds the engine image
  and waits for its own `HEALTHCHECK` to report healthy.
