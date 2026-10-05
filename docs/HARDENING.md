# Kalos production hardening (in-house engineering track)

This is the living plan for taking the Kalos portal from a single-user local tool to a multi-tenant, durable, operable service.
It covers the **behind-the-scenes engineering** only.
Regulatory and certification work (21 CFR Part 11, GAMP 5 validation, SOC 2 / ISO 27001) is handled **offline, out of this repo**, and is intentionally out of scope here.

The readiness assessment that motivates this work rated the science and honesty layer as ready, and the surrounding system as not.
This track closes the surrounding-system gaps in dependency order.

## Principles

- **Backward compatible by default.** Every hardening step ships behind configuration and defaults to the current behavior, so the local dev loop and pilot deployments never break mid-migration.
- **In-house, self-hosted.** No dependency on an external identity provider or SaaS control plane. Auth, storage, and secrets are provisioned and owned by the operator.
- **Fail closed when configured, open when not.** With no auth configured the API stays open (dev/pilot); the moment tokens are provisioned it enforces them. There is no half-authenticated state.
- **Reuse what exists.** The experiments path already has a `SqliteStore` (`src/kalos/store/`) and a constant-time bearer-token check (`src/kalos/portal/experiments.py`). New work generalizes those rather than inventing parallel mechanisms.

## Sequence

The phases are ordered by hard dependency: you cannot isolate data per tenant without an identity to isolate by, and you cannot make a service reliable before it has a durable store.

### Phase 1 - Foundation: identity and durable, tenant-scoped state

**1a. Authentication + authorization (in progress).**
A general in-house token layer (`src/kalos/portal/auth.py`): a bearer token resolves to a `Principal` (subject, tenant, scopes).
Tokens are provisioned by the operator and stored as SHA-256 hashes (never plaintext, never logged); the presented token is hashed and constant-time compared.
With no tokens configured the layer returns an anonymous principal on a `default` tenant with read+write scope, preserving today's behavior and logging a startup warning.
Mutating endpoints (`/api/run`, `/api/campaign/*`) require the `write` scope; the existing runner-token channel is subsumed by the same mechanism over time.

**1b. Tenant-scoped persistence (done).**
Every store read/write is keyed by the caller's `Principal.tenant`.
The campaign is now a SQLite row per tenant in `<state_dir>/portal.db` (`campaigns(tenant, state, updated_at)`), replacing the single `~/.kalos/campaign.json`; the transactional generation token carries over inside the per-tenant `state` blob unchanged.
`/api/latest` is now keyed by tenant too (an in-memory map plus a per-tenant best-effort JSON cache under `<state_dir>/latest/`), replacing the single global `_LATEST`.
So two tenants can never see or overwrite each other's campaign or analysis.
Follow-up: move the `latest` cache into the same SQLite store as a row, and add an `owner`/`created_by` column for finer-grained authorization.

**1c. Transport + secrets (in progress).**
CORS is now an explicit allowlist when `KALOS_CORS_ORIGINS` is set (the production posture) and the permissive localhost regex otherwise (`src/kalos/portal/config.py`); the effective security posture (auth + CORS) is logged once at startup, with a warning when the portal is not locked down.
TLS is terminated by a reverse proxy in front of the app (deployment note below); the app speaks plain HTTP on the loopback to the proxy.
Follow-up: a secrets-provider abstraction so `KALOS_*` secrets can come from a file/Vault instead of the environment, and tightening CORS methods/headers from `*` to the minimum the client needs.

**Deployment (transport):** run the app behind a TLS-terminating reverse proxy (nginx/Caddy/an ALB). The proxy holds the certificate and forwards to the app over loopback; set `KALOS_CORS_ORIGINS` to the exact browser origin(s) the proxy serves.

### Phase 2 - Reliability and scale

- Durable store already survives restarts; add backups and a documented restore.
- Move the CPU-bound `_analyze` off the request path onto an async job queue so a long GP fit never holds a worker; surface job status.
- High-availability deployment notes (stateless app tier + shared store).
- Revisit the exact-GP `MAX_FIT_ROWS=2000` cap with a sparse/variational surrogate for large historical datasets.

### Phase 3 - Operability

- Structured, correlatable request logging (never logging secrets or raw client data).
- Health/readiness endpoints, metrics, and alerting hooks.
- A documented deployment (container + config) and runbook.

## Configuration reference

| Variable | Meaning | Default |
| --- | --- | --- |
| `KALOS_AUTH_TOKENS_FILE` | Path to a JSON file of provisioned principals (see below). Takes precedence over the env var. | unset |
| `KALOS_AUTH_TOKENS` | Inline JSON array of provisioned principals. | unset |
| `KALOS_RUNNER_TOKEN` | Legacy single-token gate for the machine-to-machine `/api/experiments/{id}/result` channel. Subsumed by the token layer over time. | unset |
| `KALOS_CORS_ORIGINS` | Comma-separated explicit CORS allowlist (e.g. `https://app.acme.com`). When set, replaces the permissive localhost default. | unset (dev localhost) |

Optional external-provider keys (`ANTHROPIC_API_KEY`, `TYPESAFE_API_KEY`, `NVIDIA_API_KEY`, `BENCHLING_API_KEY`) follow the same rule as the secrets above: environment only, never logged, never returned. `GET /api/providers` reports each one's presence by variable name, not value. The model-backed normalize tiers (`KALOS_LLM_PROVIDER`) only ever receive the identity-screened payload built by `src/kalos/normalize/payload.py`; the full list of variables is in `.env.example`.

When neither `KALOS_AUTH_TOKENS_FILE` nor `KALOS_AUTH_TOKENS` is set, the API runs in **open mode** (anonymous `default` tenant, read+write, admin withheld) and logs a warning at startup.

### Provisioned-principal shape

```json
[
  {
    "token_sha256": "9f86d0818188...b0f00a08",
    "subject": "svc-lab-ingest",
    "tenant": "acme-bio",
    "scopes": ["read", "write"]
  }
]
```

Generate a token and its hash out of band, store only the hash here, and hand the raw token to the caller once:

```bash
TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
printf '%s' "$TOKEN" | sha256sum          # store this hex under token_sha256
echo "give this to the caller once: $TOKEN"
```

## Next backbone (ranked)

A review of progress against the readiness scorecard surfaced that campaign + latest were tenant-scoped but the **experiments store was still global and ungated** - a live cross-tenant leak. That is fixed (P1 below). Remaining engineering backbone, in dependency order:

- **P1 - Tenant-scope + auth-gate the experiments store: DONE** (`feat/hardening-experiments-tenancy`). `src/kalos/store/sqlite_store.py` gains a `tenant` column (+ index + one-time backfill migration) and every query filters by it; `src/kalos/portal/experiments.py` routes require `read`/`write` and pass `Principal.tenant`; `LocalStoreAdapter` is tenant-bound so the runner only touches the caller's experiments. Remote-runner `/result` still operates on the `default` tenant (machine channel, off by default) - multi-tenant remote runners are a follow-up.
- **P2 - Observability**: `/healthz` (liveness) + `/readyz` (`SELECT 1` against the DBs), a request-id + JSON logging middleware that never logs bodies/tokens, optional `/metrics`. Moves the Operational-maturity dimension off BLOCKING; low cost, high review value.
- **P3 - Async job queue for `_analyze`**: a `jobs(id, tenant, kind, status, result_ref, ...)` table + submit/poll API so `/api/run` and `/reanalyze` enqueue and return a job id, taking the CPU-bound GP fit off the request path and making in-flight work restart-durable. Reuse the Singleton runner pattern.
- **P4 - Move `_LATEST` into SQLite**: a `latest(tenant, state, updated_at)` row in `portal.db`, retiring the per-tenant file cache; lets the reanalyze latest-write carry a generation stamp in one transaction, closing the documented sub-ms race. Prerequisite for P5 (one DB file = complete tenant state).
- **P5 - Backups + restore/DR**: scheduled SQLite online `.backup()` into `KALOS_BACKUP_DIR` + a documented restore runbook. Do P4 first so a snapshot captures all state.

## Status log

- Phase 1a authentication + authorization: **done** (merged).
- Phase 1b tenant-scoped persistence: **done** (merged) - campaign -> SQLite rows per tenant; `/api/latest` per tenant; every store call keyed by `Principal.tenant`.
- Phase 1c transport/CORS: **done** on `feat/hardening-1c` - CORS allowlist via `KALOS_CORS_ORIGINS`, startup security-posture logging, TLS-via-proxy deployment note. Secrets-provider abstraction deferred.
- P1 experiments store tenant-scope + auth-gate: **done** on `feat/hardening-experiments-tenancy` - closes the cross-tenant leak the review found; the tenancy guarantee now covers all three stores (campaign, latest, experiments).
- P2 observability: **partly done** - `/healthz` (liveness, never touches the DB) and `/readyz` (a read-only `store.list()` against `experiments.db`; it does not probe `portal.db`) exist in `src/kalos/portal/app.py`; the request-id + JSON logging middleware and `/metrics` do not yet.
- P5 backups: **partly done** - the `deploy/backup` sidecar takes scheduled SQLite `.backup` snapshots to a host directory outside the state volume, with a restore drill in `deploy/RUNBOOK.md`; the drill itself has not been run as written (RUNBOOK, "Still NOT verified"). The per-tenant latest cache is still files, which is why P4 should come first.
- P3 async job queue, P4 latest->SQLite: not started (`_analyze` runs in a threadpool under the single analysis slot, `src/kalos/portal/busy.py`; `_LATEST` is an in-memory map over per-tenant files).
- The prioritized cross-cutting backlog (engine, ops, docs) lives in [ROADMAP.md](ROADMAP.md).
