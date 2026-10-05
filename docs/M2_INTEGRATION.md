# M2 - Integrate the BO engine into the Voyager Platform

Maps the "Integrate into Voyager Platform" milestone (Voyager ML Implementation Plan, 5/28/2026) onto the itsbrendandang stack.
Target decided 2026-07-14: build the experiment store, the ready-to-process flag, and the Singleton runner inside `kalos` (itsbrendandang), behind a config-driven backend adapter so the same Singleton can later point at the real Voyager portal without a rewrite.
No Bioqore-org repo is touched.

## Plan tasks -> where they land

| Plan task | Where |
| --- | --- |
| 1. Status flag in the Experiment table for "ready to process" | `Experiment.status` enum in the store (`READY` is the flag) |
| 2. Singleton process that pulls/pushes JSON, auto-runs flagged experiments, and can run any experiment on demand | `kalos/runner/singleton.py` + `BackendAdapter` |
| 3. Call the process manually on the server | `python -m kalos.runner` CLI + `POST /api/experiments/{id}/run` |
| 4. OPTIONAL: button to run "all ready" | `POST /api/experiments/run-ready` |
| 5. OPTIONAL: per-experiment run button that replaces existing output | `POST /api/experiments/{id}/run?force=true` |

## Architecture (maps to the plan's diagram)

```
Polaris Front End      = kalos-web (Voyager surface)
Back End API           = kalos FastAPI portal (/api/experiments/*)
Database               = experiment store (SQLite, ~/.kalos/experiments.db)
ML Process (Bayesian)  = kalos core (existing BO engine) driven by the Singleton
Data Moat              = bioqore-data (wired in M3, not M2)
```

Data flow the Singleton implements: pull unprocessed (READY) experiment JSON -> run BO -> push processed JSON back -> mark DONE.

## The Experiment contract (the JSON that crosses the seam)

This is the stable interface. Both the local store and any future HTTP backend serialize to exactly this shape.

```jsonc
{
  "id": "exp_<uuid>",
  "name": "string",
  "status": "DRAFT | READY | PROCESSING | DONE | FAILED",
  "created_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "config": {
    "target": "column name of the measured objective",
    "outcomes": ["optional multi-objective columns"],
    "replicate_group_by": ["optional columns that define a recipe (for replicate-averaging)"],
    "anonymize": false
  },
  "payload": {
    // the run sheet: the exact structure kalos.portal._analyze already consumes
    "columns": ["Media_A", "...", "target"],
    "rows": [ { "Media_A": 1.0, "...": 0.0, "target": 0.03 } ]
  },
  "result": null,          // populated on DONE: the analysis object /api/latest already returns
  "provenance": {          // populated on DONE
    "seed": 0,
    "engine_version": "kalos x.y.z",
    "processed_at": "ISO-8601",
    "noise_floor": { "icc": 0.26, "sigma": 0.008 }  // when replicate structure is present
  },
  "error": null            // populated on FAILED with the actionable UploadRejected message
}
```

Status lifecycle (only these transitions are legal):
`DRAFT -> READY -> PROCESSING -> (DONE | FAILED)`; `FAILED -> READY` (retry); `DONE -> READY` only with `force` (re-run, discards prior `result`); `PROCESSING -> READY` only with `force`, and only ever issued by the Singleton's own orphan recovery (`reclaim_stale`) - never reachable by a client (the `PATCH /api/experiments/{id}` endpoint rejects any request against a `PROCESSING` experiment outright, regardless of `force`).

`PATCH /api/experiments/{id}` additionally enforces a narrower CLIENT allowlist on top of the transitions above: the only status a client may set is `READY` (from `DRAFT`/`FAILED` freely, from `DONE` only with `force`). A client can never PATCH an experiment directly to `PROCESSING`, `DONE`, or `FAILED` - those are set by the runner.

## The seam: `BackendAdapter`

A `Protocol` so the Singleton never knows which backend it is talking to.

```python
class BackendAdapter(Protocol):
    def list_ready(self) -> list[str]: ...            # ids of READY experiments
    def fetch(self, exp_id: str) -> Experiment: ...   # pull one (raises if missing)
    def set_status(self, exp_id: str, status: Status) -> None: ...
    def push_result(self, exp_id: str, result: dict, provenance: dict) -> None: ...
```

- `LocalStoreAdapter` (default): backed by the SQLite store. Ships in M2.
- `HttpBackendAdapter` (documented stub): base-URL + token from config; the four methods map to REST calls on a future portal. Not wired to a live server in M2; it exists so the seam is real, has a typed signature, and has a contract test against a mock. Since the P1 tenancy hardening (docs/HARDENING.md) every experiments route is scope-gated, so all four methods send the runner's provisioned API token as `Authorization: Bearer <token>`: a `KALOS_AUTH_TOKENS` principal with `read` and `runner` scopes on the runner's tenant (plus `write` only for `--id` re-queues). `push_result` falls back to the legacy `/result` token when no API token is set (see "Security note" below).

Selection via config: `KALOS_BACKEND=local` (default) or `http`, `KALOS_BACKEND_URL=...`, the API token from `KALOS_RUNNER_API_TOKEN`, and the legacy `/result` token from `KALOS_RUNNER_TOKEN` (falling back to the already-documented `KALOS_BACKEND_TOKEN`). One factory `get_adapter()`.

## Singleton runner

`kalos/runner/singleton.py`:
- Single-instance guard: a PID/lock file at `~/.kalos/runner.lock` so two runners cannot process the same experiment (the plan says "Singleton"). Stale-lock detection on start.
- `run_one(exp_id, *, force=False)`: DRAFT/READY -> PROCESSING -> run `kalos.portal._analyze` on the payload -> push result -> DONE. On any `UploadRejected` or exception: FAILED with the message. `force` allows re-running a DONE experiment (discards prior result first).
- `run_ready()`: reclaims any stale `PROCESSING` experiments back to `READY` first (`reclaim_stale`, orphan recovery - see below), then `list_ready()` then `run_one` for each; returns a per-experiment summary. Per-experiment resilient: one experiment raising unexpectedly is recorded as `FAILED` in the summary, not allowed to abort the batch.
- Orphan recovery: the Singleton lock guarantees single-instance execution, so any experiment still `PROCESSING` when a runner acquires the lock cannot belong to a live run - it is an orphan from a run that crashed mid-analysis. `reclaim_stale` re-queues each one to `READY` (a `force`-gated, recovery-only `PROCESSING -> READY` edge in `legal_transition`) so the same `run_ready()` pass picks it back up. This edge is NOT reachable through the client-facing `PATCH /api/experiments/{id}` endpoint, which rejects any request whose current status is `PROCESSING` regardless of `force`.
- Reuses the existing engine (`_analyze`, replicate aggregation, noise floor) verbatim - M2 adds orchestration, not new science.

CLI: `python -m kalos.runner --once` (process all READY and exit), `--watch N` (poll every N s), `--id exp_x [--force]` (one experiment).

## Portal API additions (kalos FastAPI)

Thin wrappers over store + runner; every existing endpoint stays unchanged.

| Method + path | Does |
| --- | --- |
| `GET /api/experiments` | list `{id, name, status, updated_at}`; optional `?status=` filters to one status (omitted = all) |
| `POST /api/experiments` | create from an uploaded run sheet -> DRAFT |
| `GET /api/experiments/{id}` | full experiment incl. `result` |
| `PATCH /api/experiments/{id}` | set status - client allowlist: only `READY` is settable (from `DRAFT`/`FAILED` freely, from `DONE` only with `force`); any other target, or any request while the experiment is `PROCESSING`, is a 409. A `runner`-scoped principal may also make the runner transitions `READY -> PROCESSING` and `PROCESSING -> FAILED`; `DONE` is never PATCH-able |
| `POST /api/experiments/{id}/run` | run one now (`?force=true` to replace output) |
| `POST /api/experiments/run-ready` | run all READY now |
| `POST /api/experiments/{id}/result` | ingest `{result, provenance}` and mark DONE (only legal from `PROCESSING`, and rejected 409 if a result is already present) - the push endpoint `HttpBackendAdapter.push_result` targets. Gated and off by default: 404 unless `KALOS_RUNNER_TOKEN` or auth tokens are configured on the portal, then requires `Authorization: Bearer` with either a `runner`-scoped principal (its tenant) or the matching `KALOS_RUNNER_TOKEN` (the `default` tenant only); 401 for a missing/wrong token, 403 for a principal without `runner` |

## kalos-web (Polaris front end)

Replace the Voyager surface's client-side localStorage mock **data source** with these endpoints (keep the store shape and UI). The status flag, the per-experiment Run button, and Run-all-ready become real calls. localStorage stays only as an offline/optimistic cache. This turns the M2 UI from mock into the real integration.

## Non-goals for M2 (explicit)

- No Data Moat context retrieval / update-on-completion (that is M3).
- No auth on the portal beyond what exists (single-tenant local), with one exception: `POST /api/experiments/{id}/result` is token-gated (`KALOS_RUNNER_TOKEN`) and off by default (404) - see the endpoints table above and "Security note" below. Every other endpoint is unauthenticated, same as before.
- `HttpBackendAdapter` is a typed, contract-tested stub, not a live client to the Bioqore portal.

### Security note: `/result` is a privileged write, closed by default

`/result` marks a `PROCESSING` experiment `DONE` from a client-supplied `{result, provenance}` body - unlike `PATCH .../{id}`, it is not restricted to the `READY` flag, so an unauthenticated version of it would let anyone race a fabricated result in ahead of (or instead of) the genuine one.
The local M2 loop (`LocalStoreAdapter`) never calls this endpoint at all - it writes results to the store directly - so it is disabled (404) unless an operator explicitly configures a credential for it, which only a remote/HTTP runner (`HttpBackendAdapter`) needs.
Two credentials are accepted: a provisioned principal holding the `runner` scope, which writes to its own tenant, and the legacy `KALOS_RUNNER_TOKEN` (constant-time compare), which carries no identity and so reaches the `default` tenant only.
A missing or wrong token is a 401, and a principal without `runner` (such as a plain read+write client token) is a 403, so a client can never fabricate a result.
The `runner` scope is never granted in open mode.
It also refuses to overwrite an experiment that already has a non-null `result` (409), independent of the `PROCESSING`-only rule, so a stray or racing push can never silently clobber a genuine result.

## Acceptance criteria

1. Create an experiment from the synthetic run sheet -> it is `DRAFT`; flip to `READY`.
2. `python -m kalos.runner --once` processes it: `READY -> PROCESSING -> DONE`, `result` populated with the same analysis `/api/latest` returns, provenance stamped.
3. A zero-variance-target experiment goes to `FAILED` with the actionable message (reuses the upload-robustness guard), not a crash.
4. The Singleton lock prevents a second concurrent runner from double-processing.
5. `HttpBackendAdapter` passes its contract test against a mock transport.
6. kalos-web drives the whole loop from the UI against the live portal.
7. `pytest` green; front end `tsc`/`lint`/`build`/`vitest` green.
