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
- **Reuse what exists.** The experiments path already has a `SqliteStore` (`kalos/store/`) and a constant-time bearer-token check (`kalos/portal/experiments.py`). New work generalizes those rather than inventing parallel mechanisms.

## Sequence

The phases are ordered by hard dependency: you cannot isolate data per tenant without an identity to isolate by, and you cannot make a service reliable before it has a durable store.

### Phase 1 - Foundation: identity and durable, tenant-scoped state

**1a. Authentication + authorization (in progress).**
A general in-house token layer (`kalos/portal/auth.py`): a bearer token resolves to a `Principal` (subject, tenant, scopes).
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
CORS is now an explicit allowlist when `KALOS_CORS_ORIGINS` is set (the production posture) and the permissive localhost regex otherwise (`kalos/portal/config.py`); the effective security posture (auth + CORS) is logged once at startup, with a warning when the portal is not locked down.
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
| `KALOS_RUNNER_TOKEN` | Legacy single-token gate for the machine-to-machine `/api/experiments/{id}/result` channel. Carries no identity, so it reaches the `default` tenant only. Superseded by a `runner`-scoped principal (see `KALOS_RUNNER_API_TOKEN`); subsumed by the token layer over time. | unset |
| `KALOS_RUNNER_API_TOKEN` | Runner-side, for `python -m kalos.runner` with `KALOS_BACKEND=http`: the plaintext of a provisioned principal with `read` and `runner` scopes on the runner's tenant (add `write` only if you use `--id` to re-queue experiments). Sent as the bearer on every runner call. Required whenever the portal has tokens provisioned. | unset |
| `KALOS_RUNNER_ALLOW_INSECURE_HTTP` | Runner-side. The runner refuses to send any token over plain `http` to a non-loopback `KALOS_BACKEND_URL`; set to `1` to accept that risk knowingly on a network you trust. | unset (refuse) |
| `KALOS_CORS_ORIGINS` | Comma-separated explicit CORS allowlist (e.g. `https://app.acme.com`). When set, replaces the permissive localhost default. | unset (dev localhost) |

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

Scopes are `read`, `write`, `admin`, and `runner`.
`runner` is the remote Singleton runner's machine privilege: the runner-only status transitions (`READY -> PROCESSING`, `PROCESSING -> FAILED`) and the result push, on the principal's tenant.
Grant it only to the runner's own principal, never to a client; like `admin`, it is never granted in open mode.
A polling runner (`--once`/`--watch`) needs only `["read", "runner"]`; add `write` only if you use `--id` to re-queue DRAFT, FAILED, or DONE experiments.

Generate a token and its hash out of band, store only the hash here, and hand the raw token to the caller once:

```bash
TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
printf '%s' "$TOKEN" | sha256sum          # store this hex under token_sha256
echo "give this to the caller once: $TOKEN"
```

### Security notes: the remote runner

Threat model: the runner is a trusted machine on the operator's side, the portal may be reachable from the internet, and this repository is public, so assume an attacker has read every line of this code.

**What the `runner` scope grants.**
A runner principal can claim (`READY -> PROCESSING`), fail (`PROCESSING -> FAILED`), and push the result for any experiment in its own tenant.
That includes the power to record a fabricated result, which is inherent to the role, so treat the runner token like a database credential.
Provision one runner principal per tenant, and never give `runner` to a human or a browser client.

**What it does not grant.**
No upload, no READY flip, and no campaign mutation (those need `write`).
No other tenant: a cross-tenant id reads as 404, never 403, so tenants cannot even probe each other's ids.
No `DONE` through `PATCH` for anyone; `DONE` is only reachable through the result push, which carries the result.
Open mode never grants `runner`, so a portal with no tokens provisioned cannot be driven by a remote runner at all.

**Transport.**
The runner sends its token on every request, so `get_adapter()` refuses to start rather than send it anywhere unsafe.
It rejects any `KALOS_BACKEND_URL` that is not `http(s)` with a host, and any URL with `user:password@` embedded.
It refuses plain `http` to a non-loopback host while a token is configured; `KALOS_RUNNER_ALLOW_INSECURE_HTTP=1` is the deliberate override for a private network you trust.
It never follows a redirect: stock urllib re-sends `Authorization` to wherever a 3xx points, including another host or an `https` to `http` downgrade (`tests/test_m2_adapter.py` proves this against real sockets).
Errors name the scheme and host only, never the full URL or a token.

**The legacy token.**
`KALOS_RUNNER_TOKEN` is a shared secret with no identity: whoever holds it can push results into the `default` tenant.
Prefer a runner principal, and once every runner has moved over, unset `KALOS_RUNNER_TOKEN` on the portal.
With neither it nor auth tokens configured, `/result` does not exist (404).
It is compared as bytes in constant time, so a wrong guess cannot be timed and a non-ASCII header is a 401, not a 500.

**Storage and rotation.**
The portal stores only the SHA-256 hash; the plaintext lives only on the runner host.
To rotate, add the new hash next to the old one, switch the runner to the new token, then remove the old hash.
The portal re-reads its token configuration on every request, so none of these steps needs a restart.
Environment variables are visible to anything running as the same user that can read the process environment; a file-based secrets provider is the deferred Phase 1c follow-up.

**Known limits.**
A remote runner cannot reclaim an orphaned `PROCESSING` experiment: `HttpBackendAdapter` has no `list_processing`, and `PATCH` refuses `PROCESSING -> READY` for everyone.
A runner that crashes mid-analysis leaves that experiment stuck until an operator resets it in the store.
The runner trusts the responses of the portal it authenticates to over TLS and does not cap their size.

## Next backbone (ranked)

A review of progress against the readiness scorecard surfaced that campaign + latest were tenant-scoped but the **experiments store was still global and ungated** - a live cross-tenant leak. That is fixed (P1 below). Remaining engineering backbone, in dependency order:

- **P1 - Tenant-scope + auth-gate the experiments store: DONE** (`feat/hardening-experiments-tenancy`). `kalos/store/sqlite_store.py` gains a `tenant` column (+ index + one-time backfill migration) and every query filters by it; `kalos/portal/experiments.py` routes require `read`/`write` and pass `Principal.tenant`; `LocalStoreAdapter` is tenant-bound so the runner only touches the caller's experiments. ~~Remote-runner `/result` still operates on the `default` tenant (machine channel, off by default) - multi-tenant remote runners are a follow-up.~~ Done in P1.1.
- **P1.1 - Authenticate the remote runner: DONE** (`fix/runner-api-auth`). P1 gated the experiments routes but `HttpBackendAdapter` still sent no credential on `list_ready`/`fetch`/`set_status`, so any deployment with tokens provisioned 401'd the runner's first poll; past that, the `PATCH` client allowlist 409'd the runner's own `READY -> PROCESSING` claim, and `/result` only looked on the `default` tenant. The runner now authenticates as a least-privilege provisioned principal (`KALOS_RUNNER_API_TOKEN`, `["read", "runner"]`) holding a new `runner` scope, which unlocks exactly the runner transitions and a tenant-aware `/result`; `KALOS_RUNNER_TOKEN` keeps working for the `default` tenant. The HTTP transport gained a 30s timeout, one retry when the connection cannot be opened, a refusal to follow redirects, and a refusal to send tokens over cleartext `http` off-box. See "Security notes: the remote runner" above.
- **P2 - Observability**: `/healthz` (liveness) + `/readyz` (`SELECT 1` against the DBs), a request-id + JSON logging middleware that never logs bodies/tokens, optional `/metrics`. Moves the Operational-maturity dimension off BLOCKING; low cost, high review value.
- **P3 - Async job queue for `_analyze`**: a `jobs(id, tenant, kind, status, result_ref, ...)` table + submit/poll API so `/api/run` and `/reanalyze` enqueue and return a job id, taking the CPU-bound GP fit off the request path and making in-flight work restart-durable. Reuse the Singleton runner pattern.
- **P4 - Move `_LATEST` into SQLite**: a `latest(tenant, state, updated_at)` row in `portal.db`, retiring the per-tenant file cache; lets the reanalyze latest-write carry a generation stamp in one transaction, closing the documented sub-ms race. Prerequisite for P5 (one DB file = complete tenant state).
- **P5 - Backups + restore/DR**: scheduled SQLite online `.backup()` into `KALOS_BACKUP_DIR` + a documented restore runbook. Do P4 first so a snapshot captures all state.

## Status log

- Phase 1a authentication + authorization: **done** (merged).
- Phase 1b tenant-scoped persistence: **done** (merged) - campaign -> SQLite rows per tenant; `/api/latest` per tenant; every store call keyed by `Principal.tenant`.
- Phase 1c transport/CORS: **done** on `feat/hardening-1c` - CORS allowlist via `KALOS_CORS_ORIGINS`, startup security-posture logging, TLS-via-proxy deployment note. Secrets-provider abstraction deferred.
- P1 experiments store tenant-scope + auth-gate: **done** on `feat/hardening-experiments-tenancy` - closes the cross-tenant leak the review found; the tenancy guarantee now covers all three stores (campaign, latest, experiments).
- P1.1 remote runner authentication: **done** on `fix/runner-api-auth` - a remote runner completes a round trip against a portal with tokens provisioned, on any tenant.
- P2 observability, P3 async job queue, P4 latest->SQLite, P5 backups: planned, not started (see "Next backbone" above).
