# kalos deployment runbook

Covers the stack in `deploy/docker-compose.yaml`: the kalos engine
(FastAPI/uvicorn, `python -m kalos.portal`), kalos-web (Next.js), and a
SQLite backup sidecar.
Read `deploy/docker-compose.yaml`'s header comment first if you have not -
it explains why this stack has two host-published front doors instead of
one, and why the engine needs real auth configured to start at all.

This is a v0 pack: builds are unverified (no image has actually been built
- see "What was verified" at the bottom), and TLS termination is explicitly
out of scope, delegated to a host-level reverse proxy placed in front of
both published ports. Nothing here does TLS itself.


## Verified quickstart (first real deployment, 2026-09-10)

The whole stack, from a clean machine with Docker running:

```
cd kalos/deploy
cp .env.example .env        # fill in KALOS_AUTH_TOKENS, KALOS_ENGINE_TOKEN, KALOS_ANON_SALT
docker compose up -d --build
```

Then open http://localhost:3000 - the ONLY published port. Verified numbers
from the reference run (M-series Mac, Docker VM 11 CPU / 8 GB): engine image
1.82 GB, web 314 MB; first analyze on the 160-run demo sheet completed through
the authed proxy in ~19s; a concurrent second upload got the honest 503 busy
answer in 0.1s.

Env knobs that exist because this deployment found the need for them:
- `KALOS_ENGINE_TIMEOUT_MS` (web) - proxy timeout for engine calls, default
  120000. The original 30s guess 502'd mid-fit on a cold container.
- `KALOS_TORCH_THREADS` (engine) - the compose passes 0 (= do not pin) by
  default: the bare-metal 4-thread pin measured a >12x pathological slowdown
  (47s -> 600s+) on this image's linux-aarch64 torch/OpenBLAS build.
- One analysis at a time is enforced engine-side (HTTP 503 + Retry-After when
  busy; kalos/portal/busy.py): a client that gives up cannot stack orphaned
  fits against its own retry.

## Prerequisites

- Docker with Compose v2 (`docker compose version`, not the standalone
  `docker-compose` v1 binary).
- This repo (`kalos-engine`) and `kalos-web` checked out as SIBLING directories,
  e.g. both under `~/GitHub/`. `deploy/docker-compose.yaml`'s `web.build`
  section resolves `kalos-web`'s path as `../../kalos-web` relative to this
  file - if your checkout layout differs, that one path is the thing to
  edit, nothing else in this pack assumes a particular layout.

## Deploy

```bash
cd kalos/deploy

# 1. Real environment, never committed.
cp .env.example .env
# Edit .env: fill in KALOS_AUTH_TOKENS, KALOS_ANON_SALT, KALOS_CORS_ORIGINS,
# NEXT_PUBLIC_API_URL at minimum. See .env.example's inline comments for what
# each does and how to generate a token/salt.

# 2. The backup sidecar writes to a host-mounted directory (deliberately
# outside the named `kalos-state` volume - see docker-compose.yaml) that
# Docker will otherwise auto-create as root-owned, which the backup
# container (uid 10001, unprivileged) cannot then write into.
mkdir -p backups
chown -R 10001:10001 backups   # or: sudo chown -R 10001:10001 backups

# 3. Build all three images.
docker compose build

# 4. Bring the stack up.
docker compose up -d

# 5. Confirm the engine actually started with auth enforced (it exits
# non-zero instead if KALOS_AUTH_TOKENS/KALOS_ANON_SALT are missing - see
# "Security checklist" below).
docker compose logs engine | grep "security posture"
# Expect: "kalos portal security posture: auth=enforced, cors=allowlist(1)"
# auth=OPEN or cors=dev-localhost here means .env is not actually filled in.

# 6. Health.
docker compose ps                    # both engine and web should show "healthy"
curl -f http://localhost:${KALOS_ENGINE_PORT:-8050}/healthz
curl -f http://localhost:${KALOS_WEB_PORT:-3000}/
```

## Upgrade

```bash
cd kalos/deploy
git -C .. pull                       # or however the engine repo updates
git -C ../../kalos-web pull          # kalos-web separately

docker compose build
docker compose up -d                 # recreates only the services whose image changed
docker compose logs -f engine web    # watch both healthchecks go green
```

The `kalos-state` volume is untouched by a rebuild - it is a named volume,
not baked into the image, so an upgrade never loses `experiments.db`,
`portal.db`, or `runner.lock`.

## Rollback

This pack does not push to a registry (out of scope - see "What was
verified"), so rollback here means rebuilding the previous commit's image
under a distinct tag, not pulling one back down:

```bash
cd kalos/deploy
git -C .. checkout <previous-good-sha>
git -C ../../kalos-web checkout <previous-good-sha>   # if it changed too

KALOS_IMAGE_TAG=rollback-$(date +%Y%m%d) docker compose build
KALOS_IMAGE_TAG=rollback-$(date +%Y%m%d) docker compose up -d
```

Data is not rolled back with the code - `kalos-state` is shared across
versions. If the previous version's schema is genuinely incompatible with
what the bad version wrote, restore from a backup (below) instead of just
rolling back the image.

## Backup / restore drill

The backup sidecar (`deploy/backup/`) runs `sqlite3 <db> ".backup <dest>"`
against `experiments.db` and `portal.db` on `BACKUP_INTERVAL_SECONDS`
(default daily), keeping `BACKUP_KEEP_DAYS` (default 7) of history under
`deploy/backups/{experiments,portal}/`. See `deploy/backup/backup.sh`'s
header comment for exactly why `.backup` and not `cp`/`tar` of the live
files: kalos's stores run in SQLite's default rollback-journal mode (not
WAL), so a plain file copy can land mid-write and produce a backup that
opens fine but is silently missing the in-flight transaction. `.backup`
uses SQLite's Online Backup API, which is safe against a concurrently
writing engine process.

**Verify a backup is actually happening:**

```bash
docker compose logs backup --tail 20
ls -la deploy/backups/experiments/ deploy/backups/portal/
```

**Restore drill** (run this periodically against a scratch stack, not only
when something is actually on fire - a backup nobody has ever restored from
is a hope, not a backup):

```bash
cd kalos/deploy

# 1. Stop the engine so nothing writes to the live db while restoring.
docker compose stop engine

# 2. Pick the backup to restore and copy it into the live volume, under the
#    live filenames. `docker compose exec` won't work with the engine
#    stopped, so use `docker run` against the same named volume directly.
LATEST_EXP=$(ls -t deploy/backups/experiments/*.db | head -1)
LATEST_CAMPAIGN=$(ls -t deploy/backups/portal/*.db | head -1)
docker run --rm \
  -v kalos-state:/state \
  -v "$(pwd)/backups:/backups:ro" \
  alpine:3.20 sh -c "
    cp /backups/experiments/$(basename "$LATEST_EXP") /state/experiments.db &&
    cp /backups/portal/$(basename "$LATEST_CAMPAIGN") /state/portal.db &&
    chown 10001:10001 /state/experiments.db /state/portal.db
  "

# 3. Bring the engine back up and confirm the restored data is visible.
docker compose start engine
docker compose logs -f engine
curl -sf -H "Authorization: Bearer <a real provisioned token>" \
  http://localhost:${KALOS_ENGINE_PORT:-8050}/api/experiments | head -c 300
```

The `latest/*.json` per-tenant analysis cache is intentionally NOT part of
this drill - it is explicitly best-effort persistence in the source itself
(`kalos/portal/app.py`'s `_save_latest`) and regenerates the moment a tenant
re-runs an analysis, so it does not need restore-grade rigor. The backup
script still copies it opportunistically under
`deploy/backups/latest-cache/` if you want it anyway.

## Log locations

- `docker compose logs engine` - uvicorn access/error output plus kalos's
  own logger (`kalos.portal`), including the startup security-posture line
  and the per-request "refused a remote request" warning if it ever fires.
- `docker compose logs web` - Next.js server output.
- `docker compose logs backup` - one line per backup attempt (success with
  the file written and its size, or an explicit `FAILED` line that leaves
  prior backups untouched rather than clobbering them with a partial file).
- Nothing is written to a host log directory by default; add a `logging:`
  driver block per service in `docker-compose.yaml` if centralized log
  shipping is needed later - out of scope for v0.

## Health checks

Both `engine` and `web` carry an image-level `HEALTHCHECK` (see
`Dockerfile.engine` / kalos-web's `Dockerfile` for exactly what each checks and
why). The engine's hits `GET /healthz` (wave-2 fix; an earlier pass of this
deploy pack used `GET /` as a stand-in because the route did not exist yet
- see git history for that reasoning). `/healthz` is unauthenticated by
design, returns a fixed `{"status": "ok"}`, and never touches the database,
so it stays a pure liveness check. `docker compose ps` shows `healthy` /
`unhealthy` / `starting` per service; `web`'s startup is ordered on
`engine` reaching `healthy` via `depends_on: condition: service_healthy`.

## Security checklist

- [ ] `KALOS_AUTH_TOKENS` (or `KALOS_AUTH_TOKENS_FILE`, if you switch to a
      mounted file for rotation without an image rebuild) set to at least
      one real, freshly generated token - not the placeholder hash.
- [ ] `KALOS_ANON_SALT` set to a real random secret - not the placeholder.
- [ ] `KALOS_CORS_ORIGINS` set to the exact origin(s) the browser loads
      kalos-web from.
- [ ] `docker compose logs engine | grep "security posture"` shows
      `auth=enforced` and `cors=allowlist(N)`, not `OPEN` / `dev-localhost`.
- [ ] TLS termination is delegated to a host-level reverse proxy (nginx,
      Caddy, your cloud LB) placed in front of the single published port.
      This compose stack serves plain HTTP on its front door; that is a
      deliberate v0 scope cut, not an oversight - wiring TLS in is the
      first thing to add before this stack sees a network it does not
      fully trust.
- [x] **Auth between web and engine - CLOSED (wave 2)**: kalos-web now ships
      a server-side proxy (`app/api/engine/[...path]/route.ts`). The browser
      calls same-origin `/api/engine/...`; the Next server forwards to the
      engine over the compose-internal network with `Authorization: Bearer
      $KALOS_ENGINE_TOKEN`. The token is server-only env (never NEXT_PUBLIC_,
      never in the client bundle), and the engine no longer publishes a host
      port at all - one front door. Setup: mint a `KALOS_AUTH_TOKENS` entry
      (subject e.g. "kalos-web-service") and set `KALOS_ENGINE_TOKEN` in
      `deploy/.env` to its PLAINTEXT counterpart. (`.env.example` could not be
      auto-edited - this workspace hard-denies `.env*` writes - add the
      `KALOS_ENGINE_TOKEN=` line there by hand.) The old direct-browser mode
      survives as an explicit dev escape hatch: set `NEXT_PUBLIC_API_URL` and
      re-publish the engine port in the compose. The previous option of
      `KALOS_ALLOW_OPEN_ACCESS=1` + network-perimeter control remains the
      guard's own documented escape hatch for fully-controlled networks, but
      is no longer needed to make the UI functional.
- [ ] Only ONE port is published: `KALOS_WEB_PORT` (default 3000). The
      engine is compose-internal only (wave 2 - the web server proxies to
      it). Confirm with `docker compose ps` - no other service should show
      a `PORTS` column entry.
- [ ] Exactly one `engine` replica. `kalos/runner/singleton.py`'s
      `runner.lock` and the store's single-writer assumption
      (`kalos/store/sqlite_store.py`'s own docstring: "there is never more
      than one writer at a time by design") are not safe under
      `docker compose up --scale engine=N` for N>1. This compose file does
      not set `deploy.replicas`, so this is the default, but it is worth
      stating as a hard constraint rather than an accident of the current
      config.

## Changing the engine URL

Because `NEXT_PUBLIC_API_URL` is compiled into kalos-web's client bundle at
build time (see kalos-web's `Dockerfile`), moving the engine to a new host/port
means:

```bash
# edit NEXT_PUBLIC_API_URL in deploy/.env, then:
docker compose build web
docker compose up -d web
```

A plain `docker compose restart web` or `docker compose up -d` with only
the environment changed does NOT pick this up - the value is already baked
into the JS files on disk inside the image.

## What was verified vs. not

**Verified:**
- `deploy/docker-compose.yaml` parses and resolves correctly:
  `docker compose config` (Compose v2, installed in this environment)
  succeeds against a filled-in `.env`, with all three build contexts
  resolving to the expected paths (this repo's root for `engine`, the
  sibling `kalos-web` checkout for `web`, `deploy/backup` for `backup`) and
  all `${VAR}` substitutions resolving as intended.
- Every fact this pack's comments assert about the kalos source was
  confirmed by reading the actual files, not assumed, as of this task:
  `kalos/portal/__main__.py`'s uvicorn entrypoint and `assert_safe_exposure`
  call; `kalos/portal/config.py`'s bind/CORS/salt logic; `kalos/portal/
  auth.py`'s token model; `kalos/portal/app.py`'s registered routes (no
  `/healthz` at the time) and the `_refuse_open_remote_access` middleware;
  `kalos/store/sqlite_store.py` and `kalos/portal/campaign.py`'s
  `KALOS_STATE_DIR` handling (and the `SqliteStore`/`runner.lock` gap where
  it was then ignored). **Both gaps were closed in a later wave**:
  `kalos/portal/app.py` now has `GET /healthz`, and `SqliteStore`/
  `SingletonLock` now read `KALOS_STATE_DIR` too - see the "Health checks"
  section above and this Dockerfile's `KALOS_STATE_DIR` comment for the
  current state. `kalos/runner/singleton.py`'s single-writer lock;
  `kalos-web/package.json` and `next.config.ts` (no `output: "standalone"`
  today); `kalos-web/
  lib/api.ts` and its callers (all `"use client"`, no `Authorization`
  header sent anywhere).
- `deploy/backup/backup.sh` runs cleanly under `sh -n` (syntax check) and
  was reasoned through by hand for the failure modes it claims to handle
  (missing db file yet, a failed `.backup` call, pruning). `shellcheck` was
  not available in this environment (neither the binary nor a working
  Docker daemon to run its image) to run a real lint pass - worth doing
  before this script sees production.

**NOT verified (stated plainly, not glossed over):**
- No image was actually built (`docker build`/`docker compose build`) -
  this task's brief explicitly excludes that as heavy; a real build could
  still surface a missed system package, a pip resolution conflict, or (for
  `web`) the `output: "standalone"` gap actually failing the build as
  documented.
- No container was actually run; the healthchecks, the auth-refusal
  startup path, and the backup/restore drill are reasoned from reading the
  source, not exercised end-to-end.
- `deploy/backup/backup.sh` was not executed against a real SQLite file.
