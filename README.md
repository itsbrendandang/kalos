# Kalos

A Bayesian-optimization platform for bioprocess development, built on **BoTorch**
(a real GP surrogate + acquisition) with **NVIDIA BioNeMo / ESM-2** protein
features. Honest by construction: leakage-controlled cross-validation and
fail-closed promotion gates, on a production BoTorch core rather than a
hand-rolled GP + EI.

> Genuine, classical/probabilistic + deep-learning ML — not marketing. The BO
> loop runs here today, and ESM-2 embeds protein sequences on the Mac's GPU.

## The pipeline at a glance

One uploaded run sheet flows through these stages. Every stage is a plain
module you can call on its own; the portal (`POST /api/run`) and the Voyager
runner just chain them. (Roles are normally inferred inside the analysis, after
validation; with `roles=auto` stage 3 is decided first and handed to it.)

| # | Stage | What it decides | Where |
| --- | --- | --- | --- |
| 1 | **Ingest** | parse CSV/TSV/Excel under hard size caps, sniff the filetype | `portal/uploads.py` |
| 2 | **Validate** | eleven data-quality checks; unit conversion; physically-possible bounds | `validation/` |
| 3 | **Normalize + roles** | which column is the target, a feature, a group, an id; canonical names | `normalize/` (offline, LLM, or **TypeSafe** tier), `domains/` |
| 4 | **Analyze** | grouped-CV reliability, signed drivers, conformal bands, promotion verdict | `portal/analysis.py` over `core/` |
| 5 | **Propose** | the next batch (qLogNEI / qLogNEHVI), feasibility-gated, in-flight-aware | `core/optimize.py`, `core/multiobjective.py`, `core/feasibility.py` |
| 6 | **Loop** | log measured results, re-analyze on the grown dataset, re-propose | `portal/campaign.py` ([docs/CAMPAIGN_LOOP.md](docs/CAMPAIGN_LOOP.md)) |
| 7 | **Store + run** | persist experiments, run READY ones, push results | `store/`, `runner/` ([docs/M2_INTEGRATION.md](docs/M2_INTEGRATION.md)) |

Stage 3 is where TypeSafe makes typed decisions (see
[TypeSafe decisions](#typesafe-decisions-for-the-normalize-and-roles-stage)); the
statistics in stages 4-5 stay in code, where the honesty guarantees live.

## What's here

```
kalos/
  core/
    surrogate.py       BoTorch SingleTaskGP (input-normalized, output-standardized)
    optimize.py        qLogNEI acquisition + optimize_acqf -> next batch (in-flight runs as X_pending,
                       feasibility-gated in log space, optional outcome-constraint floor)
    multiobjective.py  qLogNEHVI for two+ objectives (titer AND purity) + Pareto front
    evaluation.py      grouped CV Spearman (leakage-controlled, repeated partitions) + interval calibration + LOGO/top-k/group-mean baselines
    splits.py          group-aware CV + a leakage tripwire (assert_no_group_leakage)
    drivers.py         signed Spearman drivers with bootstrap confidence intervals (tested per recipe, not per row)
    conformal.py       split-conformal prediction intervals (distribution-free)
    gates.py           fail-closed promotion gates
  kit/
    __init__.py        torch-free facade re-exporting splits/drivers/conformal/gates/anonymizer
  data/
    anonymizer.py      strip identity, hash grouping keys
  store/
    models.py          Experiment record + Status lifecycle (DRAFT -> READY -> PROCESSING -> DONE|FAILED)
    sqlite_store.py    SQLite experiment store (~/.kalos/experiments.db)
  runner/
    singleton.py       Voyager Singleton: pull READY experiments -> run BO -> push result
    adapter.py         BackendAdapter seam (local store today, http portal later)
  features/
    protein.py         ESM-2 embeddings (transformers, MPS) + KmerEmbedder stand-in
  scale/
    features.py        physics-informed scale features (P/V, vs, kLa, hydrostatic proxies; cited, fittable)
    transfer.py        scale-up transfer over the Surrogate (+ opt-in physics_mean / multi_fidelity candidates)
    candidates.py      the evidence-attached alternates (physics-informed mean, scale-as-fidelity)
    evaluation.py      leave-one-scale-out with extrapolation direction reported separately
  domains/
    profile.py         torch-free ColumnRoles / DesignSpace + DomainProfile (declared roles, mixed spaces)
    bioprocess.py      the default bioprocess role-hint profile (legacy behavior)
    generic.py         a domain-neutral profile for non-bio tabular optimization
  validation/
    bounds.py          physically-possible ranges per measurement dimension (physics vs convention, each justified)
    checks.py          the eleven data-quality checks (units, bounds, duplicates, missingness, informative missingness, outliers, provenance, replicates, controls, constants, constant-within-group)
    runner.py          validate_frame: runs every check, never raises, + apply_unit_conversions
    report.py          Finding / UnitConversion / ValidationReport, JSON-safe serialization
  normalize/
    units.py           the canonical unit registry: parse "34.6 C", convert to a base unit
    synonyms.py        deterministic header -> canonical column mapping
    payload.py         the privacy boundary: identity columns removed, free text redacted before any model sees it
    llm.py             offline plan + optional LLM tier (Anthropic or self-hosted Ollama); offline fallback always works
    typesafe_tier.py   optional TypeSafe tier: per-column typed judgments (role / identity / canonical name)
    roles.py           plan -> ColumnRoles, so a plan can drive /api/run (roles=auto)
    apply.py           execute a plan deterministically (drop, hash, rename, convert)
  providers/
    registry.py        external-provider credential slots (Anthropic / TypeSafe / BioNeMo / Benchling), all keyless-by-default
  bench/               closed-loop benchmarks with known optima (docs/BENCHMARK.md)
  portal/
    app.py             FastAPI app: live-engine views + /api/experiments + the "run your own data" upload path
    analysis.py        `_analyze`: the stage-4/5 pipeline behind /api/run and campaign re-analysis
    campaign.py        the closed loop's state + re-analysis (campaign_routes.py mounts /api/campaign)
    validate.py        ingestion preflight + per-column provenance (what was kept/dropped and why)
examples/            demos + run_on_media_data.py
experiments/         off-path research prototypes; NOT part of the shipped package or its guarantees (see experiments/README.md)
tests/               pytest (CI gate)
docs/                design, benchmark, campaign-loop, M2, and hardening write-ups (docs/README.md is the index)
deploy/              production images, compose stack, backup sidecar (deploy/README.md indexes them; RUNBOOK.md operates them)
scripts/setup-dev.sh one idempotent dev-environment setup for laptops, the devcontainer, and cloud sessions
.devcontainer/       VS Code / Codespaces dev container: Dockerfile (tools) + devcontainer.json (runs scripts/setup-dev.sh)
.dockerignore        keeps the engine image's root build context to pyproject.toml, README.md, kalos/
```

A lot of the honest-evaluation and data machinery was carried over from a prior
lean engine (`voyager-brain-rebuild`): the leakage-controlled splits, the
bootstrap-Spearman drivers, the data anonymization, and the push-feed ingestion.

## Experiment store and the Voyager loop

Run data flows through the Experiment store. A run sheet is uploaded as an
Experiment (`POST /api/experiments`), flagged `READY`, and the Singleton runner
pulls it, runs the BO engine, and writes the result back
(`DRAFT -> READY -> PROCESSING -> DONE|FAILED`). The store is SQLite at
`~/.kalos/experiments.db`. See `docs/M2_INTEGRATION.md` for the full contract.

Client/strain identity can be stripped and grouping keys hashed on upload via the
anonymizer (`kalos/data/anonymizer.py`), so a run stays groupable for
leakage-safe CV without carrying who it belongs to.

## The campaign loop (`/api/campaign*`)

The closed optimization loop behind the kalos-web `/decide` surface: propose a
batch, run the recipes, log the measured outcome, re-propose on the grown
dataset. A campaign is one target plus a growing dataset of (recipe -> measured
outcome) rows; `POST /api/campaign/reanalyze` folds logged results into that
dataset and re-runs the same leakage-controlled `_analyze`, so each round stays
as honest as the first (grouped-CV reliability, conformal bands, "not modeled"
callouts).

Runs that were started but have not been measured yet are handed to the
acquisition as in-flight points (`X_pending`). They have no outcome, so they
never join the fit; but without them a mid-round re-proposal treats a recipe
currently in the incubator as unexplored and proposes it again. On a 4-factor
design at `q=5` over 10 seeds, a blind re-proposal repeated a mean of 2.1 of its
5 recipes against work already running; with the pending block it repeats none
(`python -m kalos.bench --pending`). See `docs/CAMPAIGN_LOOP.md` for the full
contract.

## Uploading a run sheet (`POST /api/run`)

The portal accepts an uploaded CSV / TSV / Excel run sheet and returns the analysis
(auto-detected target, leakage-safe features, honest grouped-CV, signed drivers, a proposed next
batch). Because the sheet comes from an external client, the upload path is guarded:

- **Size + shape caps.** A raw-byte cap (default 25 MB, override with `KALOS_MAX_UPLOAD_MB`), a
  512-column ceiling, a 100k-row CSV cap, and a 2M-cell xlsx cap (a zip-bomb guard). Every cap
  fails closed: an over-cap upload is REJECTED with a 400, never silently truncated. The xlsx
  cell cap is enforced from the workbook's declared dimensions BEFORE the frame is materialized,
  so a zip-bomb is rejected without the memory spike. Filetype is sniffed by magic bytes
  (`PK\x03\x04` -> Excel, else text/CSV), not by extension.
- **Safe errors.** A bad upload returns a generic HTTP 400 ("Could not parse the uploaded file.
  Check it is a CSV or Excel run-sheet."). This holds even when the failure is a GP fit error
  (`torch.linalg.LinAlgError`) or a leakage-guard assertion, not just a parser error: a catch-all
  normalizes any such failure to the same `{error}` JSON envelope. Parser details, column names,
  cell values, paths, and stack traces are logged server-side and never returned to the caller.
- **Provenance.** The response includes a `provenance` list: for every column, its status
  (`kept_feature`, `target`, `dropped_id`, `dropped_output`, `dropped_constant`,
  `dropped_constant_on_fitted_rows`, `dropped_sparse`, `dropped_all_blank`) and how many cells had
  to be coerced from non-numeric text (units like "34.6 C"). A feature that varies over the full
  sheet but is constant on the target-present rows the GP actually fits is dropped and flagged
  (`dropped_constant_on_fitted_rows`), never silently pinned to a zero-width bound. No more silent
  column drops.
- **Data validation gate.** Before any column is typed or dropped, the sheet runs through
  `kalos.validation`: eleven checks covering unit consistency, physical bounds, duplicates vs
  replicates, missingness, informative missingness, outliers, provenance metadata, replicate
  adequacy, controls presence, constant columns, and columns constant within a group. The report is returned as `validation`, with `status` one of `pass`,
  `pass_with_warnings`, or `fail`. An **error** is a physics violation (pH 40, a negative titer, one
  column mixing g/L and mg/mL); a **warning** is possible but operationally suspect. Row lists are
  capped at 20 with the true total in `detail.n_rows_affected`, so a finding never implies its list
  is complete. `KALOS_VALIDATION_MODE=strict` refuses a `fail` upload; the default `warn` analyzes
  it anyway and returns the findings, so adding the gate did not change what the API accepts.
- **Units are converted, not discarded.** A column written `"34.6 C"` parses at 0% as a bare number,
  so it used to fail the >=80% numeric test and come back `dropped_sparse` - a real process input
  silently thrown away. Single-unit columns are now converted to their base unit before feature
  selection and reported in `validation.conversions` with a human label (`"Celsius"`). Only units the
  registry can actually convert are rewritten: a vessel column of `"5L"`/`"500L"` uses an
  unrecognized token, and rewriting it would turn an identifier into a measurement, so it is left
  alone and the client is told.
- **Proposals stay physically possible.** The design box for each continuous feature is built from
  physically valid observations only, so one `-999` sensor sentinel can no longer widen the search
  space into impossible recipes (it previously produced proposals at -422 C, below absolute zero).
  Any narrowing is reported in `design_box_exclusions`, never silent. This holds in every validation
  mode - it does not depend on the client reading the report.
- **Reproducibility.** The analyze path is seeded, so the same upload yields identical proposals;
  the response carries `seed`, `timestamp`, and `engine_version`. Protein embeddings pin the ESM-2
  checkpoint to an immutable commit, so upstream cannot change model features without a diff here.
- **Opt-in anonymization.** Pass the form field `anonymize=true` to pseudonymize identifier-type
  columns in the response. Feature and target names are kept as-is (the owner UI legitimately shows
  drivers like "Methanol").
- **Decided roles (`roles=auto`).** Pass the form field `roles=auto` to have the normalize tier
  decide target / features / group / ids instead of the bioprocess header patterns (a JSON
  `roles` object still declares them by hand). The response gains a `role_decision` block: which
  tier decided (`typesafe`, `llm`, or `offline`), whether the decision was applied, and each
  column's role with its rationale and probabilities - column names only, never cell values. If no
  consistent plan names a target, roles are inferred exactly as without the field. An explicit
  `target` field always wins over the decided one.

## TypeSafe decisions for the normalize and roles stage

[TypeSafe](https://typesafe.ai)'s System One model (Jev) returns typed judgments with
probabilities instead of generated text. Kalos uses it where ordinary code needs semantic
understanding - reading what a client's column *means* - and nowhere else:

| Question per column | TypeSafe primitive | How code uses the answer |
| --- | --- | --- |
| What role does it play: target, feature, group, metadata, free text? | **Choice** | acted on only above `KALOS_TYPESAFE_MIN_CONFIDENCE` (default 0.6); below it the offline guess is kept |
| Does it identify a client, a person, or a sample? | **Noul** | dropped as identity at p >= 0.5 - a privacy gate, not a preference |
| Which known canonical name is it (or its own)? | **Choice** over names code builds | select, never generate; exact aliases are resolved in code and never asked |

Code keeps the rules: units are an exact registry lookup, at most one target survives (a second
"outcome" becomes metadata, never an input), name collisions are resolved deterministically, and
every decision's probabilities are written into the plan's `note` for audit. The model only ever
sees the identity-screened payload from `normalize/payload.py`. Any failure (no key, network,
malformed answer) falls back to the offline plan - TypeSafe can improve the decision, never block
an upload.

```bash
python -m pip install -e ".[typesafe]"     # typesafe-sdk
export KALOS_LLM_PROVIDER=typesafe TYPESAFE_API_KEY=...
python -c "import pandas as pd; from kalos.normalize import propose_plan; print(propose_plan(pd.read_csv('sheet.csv')).to_json())"
curl -F file=@sheet.csv -F roles=auto http://127.0.0.1:8050/api/run   # TypeSafe-decided roles
```

`GET /api/providers` reports whether the key is present. The Claude Code plugin
(`typesafe@typesafe-ai`) carries TypeSafe's own design guidance for extending this tier.

## Install / run

The core install is **torch-free**: `numpy` / `pandas` / `scikit-learn` / `scipy`
only, enough to use `kalos.kit` (leakage-controlled splits, signed Spearman
drivers, split-conformal intervals, promotion gates, the anonymizer). The GP
surrogate, acquisition, and grouped-CV evaluation need the `ml` extra
(torch / botorch / gpytorch).

```bash
python -m pip install -e .              # core, torch-free (kalos.kit primitives only)
python -m pip install -e ".[ml]"        # + the BoTorch engine (surrogate, optimize, evaluation)
python -m pip install -e ".[protein]"   # + real ESM-2 (transformers)

python examples/run_bo_loop.py          # BoTorch closed-loop demo (needs [ml])
python examples/run_multiobjective.py   # multi-objective (titer + purity) Pareto demo (needs [ml])
python examples/run_protein_embed.py    # ESM-2 embedding demo (downloads a small model)
python -m pytest                        # tests (KALOS_TEST_ESM=1 to include ESM-2)

python -m pip install -e ".[ml,portal]" # the web portal (FastAPI) drives the live GP, so it needs [ml] too
python -m kalos.portal                # -> http://127.0.0.1:8050  (live BO + Pareto view)

python -m pip install -e ".[typesafe]"  # + the TypeSafe normalize/roles tier (typesafe-sdk)
python -m pip install -e ".[normalize]" # + the Anthropic LLM normalize tier
```

### Development environment (survives new sessions)

`scripts/setup-dev.sh` rebuilds the full environment - `.venv` with
`kalos[ml,portal,dev,typesafe]`, CPU torch first - in one idempotent command, so a fresh machine
or a fresh session is ready in one step:

| Where | How |
| --- | --- |
| Laptop | `bash scripts/setup-dev.sh && source .venv/bin/activate` |
| VS Code / Codespaces | open in the dev container (`.devcontainer/`); it runs the script, keeps `.venv` in a named volume, and forwards the portal port 8050 |
| Claude Code cloud session | put `bash scripts/setup-dev.sh` in the cloud environment's setup script; the container is cached after it runs |
| Production | not this script - build the images in `deploy/` (see `deploy/README.md` and `deploy/RUNBOOK.md`) |

`KALOS_EXTRAS=ml,portal,dev` narrows the install; `KALOS_LEADGENE=1` also installs the
leadgene experiment's requirements. Copy `.env.example` to `.env` for optional keys.

### Development checks

CI runs the same three gates on every push and pull request (`.github/workflows/ci.yml`).
Reproduce them from a clean `.[ml,portal,dev]` install (or `bash scripts/setup-dev.sh`):

```bash
python -m pip install -e ".[ml,portal,dev]"  # + ruff, mypy, and type stubs
ruff check kalos/ tests/ examples/           # lint
mypy                                          # types (config in pyproject [tool.mypy])
python -m pytest -q                           # tests
```

## Device

BoTorch GPs run in float64 for numerical stability, and Apple's MPS backend is
float32-only, so the **GP runs on CPU** (fast for the small-sample regime BO
targets). The **ESM-2 embedder runs on MPS** when available. No GPU is required
to use the platform locally.

## Domains: reusing the engine beyond bioprocess

The optimization engine (`core/`), the experiment store (`store/`), the runner
(`runner/`), and the upload/validate pipeline are domain-neutral: they operate on
numeric arrays and know nothing about biology. Biology lives in the `domains/`
layer as data, not code in the engine.

- **Declared column roles.** By default the analyze path infers roles (target,
  features, ids, group) from the bioprocess profile's name hints. A caller in any
  other domain can instead declare a `ColumnRoles` schema (target, features,
  categoricals, groups, ids) so they state what their columns mean rather than
  renaming them to match a regex. `POST /api/run` accepts an optional `roles`
  JSON field for this; with it the analysis runs under the neutral generic
  profile.
- **Mixed continuous + categorical design spaces.** A `DesignSpace` marks each
  dimension continuous or categorical. When categoricals are present the engine
  fits a BoTorch `MixedSingleTaskGP` and proposes with `optimize_acqf_mixed`, and
  proposals decode integer level codes back to labels. The continuous-only path
  is unchanged.

To target a new domain: declare a `ColumnRoles` (or copy `domains/generic.py`
into a new profile) and, if it has discrete choices, list them under
`categoricals`. Nothing in `core/` changes. The `mixed_bump` objective in
`bench/` exercises the mixed loop end to end.

## Roadmap

- ~~Multi-objective (qNEHVI) for titer + purity together.~~ **Done** (`core/multiobjective.py`).
- ~~Mixed continuous/categorical inputs (carbon source, medium) via BoTorch.~~ **Done**
  (`domains/`, `core` mixed GP + `optimize_acqf_mixed`); large categorical spaces fall
  back to `optimize_acqf_mixed_alternating`.
- The push-feed ingestion loop (platform streams runs in; brain proposes the next
  batch; anonymized on read).
- ~~Feasibility classifier + gated acquisition.~~ **Done** (`core/feasibility.py`,
  `bo_feas` / `bo_feas_clean` strategies in `bench/pool.py`).
- ~~Calibration (ECE) for the promotion gates.~~ **Done**
  (`core/evaluation.py::interval_calibration`, reported as `reliability.calibration`).
- ~~Wire `check_gates` into the analysis path.~~ **Done**
  (`core/feasibility.py::feasibility_cv_report` supplies AUC/Brier/classifier-ECE;
  `_analyze` reports the fail-closed verdict as the top-level `promotion` block.
  Reported, never enforced: the verdict cannot reject an upload.)
- ~~Replicate-aware aggregation + assay noise-floor estimation + optional
  fixed-noise GP.~~ **Done** (`core/replicates.py`, `Surrogate.fit(..., noise=...)`,
  `pool_from_frame(..., aggregate=True)`) — the SNR lever from `docs/BENCHMARK.md`.
- ~~Semantic column-role decisions beyond header regexes.~~ **Done** (`normalize/typesafe_tier.py`,
  `roles=auto` on `/api/run`). Next: evaluate the confidence threshold on real client sheets.
- Larger ESM-2 / full NVIDIA BioNeMo backend behind the same embedder interface.

## Documentation

| Doc | Read it for |
| --- | --- |
| [docs/README.md](docs/README.md) | the index of everything below |
| [docs/CAMPAIGN_LOOP.md](docs/CAMPAIGN_LOOP.md) | the propose -> run -> log -> re-propose contract |
| [docs/M2_INTEGRATION.md](docs/M2_INTEGRATION.md) | the experiment store, runner, and Voyager seam |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | does BO beat space-filling designs, and why noise decides it |
| [docs/HARDENING.md](docs/HARDENING.md) | auth, tenancy, CORS, and the production track |
| [docs/DESIGN.md](docs/DESIGN.md) | the engine portal's visual design system |
| [docs/ROADMAP.md](docs/ROADMAP.md) | what to work on next: known bugs, refactors, product gaps, ops, CI |
| [deploy/README.md](deploy/README.md) | every container file (production + dev) and its build context |
| [deploy/RUNBOOK.md](deploy/RUNBOOK.md) | deploying, upgrading, backing up |
| [experiments/README.md](experiments/README.md) | off-path research prototypes |
| [CHANGELOG.md](CHANGELOG.md) | what changed, newest first |

No proprietary data lives in this repo; the demos use synthetic data and public
protein sequences.
