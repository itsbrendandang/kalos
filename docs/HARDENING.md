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

**1b. Tenant-scoped persistence (next).**
Add a `tenant` (and `owner`) column to the stores and key every read/write by the caller's `Principal.tenant`.
Replace the single global in-memory `_LATEST` and the single `~/.kalos/campaign.json` with per-tenant rows in the durable store, so two tenants can never see or overwrite each other's analysis or campaign.
The `campaign.json` generation-token transactional design carries over as a row-versioning column.

**1c. Secrets + transport.**
Move `KALOS_*` secrets behind a secrets provider abstraction (env for dev, file/e.g. Vault for prod).
Document TLS termination (reverse proxy) and tighten CORS from "any localhost" to an explicit allowlist when auth is on.

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

## Status log

- Phase 1a authentication + authorization: **in progress** on `feat/hardening-auth`.
- All later phases: planned, not started.
