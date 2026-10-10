# Kalos

[![CI](https://github.com/itsbrendandang/kalos/actions/workflows/ci.yml/badge.svg)](https://github.com/itsbrendandang/kalos/actions/workflows/ci.yml)

A Bayesian-optimization platform for bioprocess development, built on **BoTorch** (a real GP surrogate + acquisition) with **NVIDIA BioNeMo / ESM-2** protein features.
Honest by construction: leakage-controlled cross-validation and fail-closed promotion gates, on a production BoTorch core rather than a hand-rolled GP + EI.
The front end lives in [kalos-web](https://github.com/itsbrendandang/kalos-web).

## Contents

- [Getting started](#getting-started)
  - [Install](#install)
  - [Run the demos and the portal](#run-the-demos-and-the-portal)
  - [Development checks](#development-checks)
- [Repository layout](#repository-layout)
- [HTTP API](#http-api)
  - [Uploading a run sheet (`POST /api/run`)](#uploading-a-run-sheet-post-apirun)
  - [The campaign loop (`/api/campaign*`)](#the-campaign-loop-apicampaign)
  - [The Scale-Up Readout (`POST /api/scale/readout`)](#the-scale-up-readout-post-apiscalereadout)
- [Running it](#running-it)
  - [Hardware](#hardware)
  - [Deployment](#deployment)
- [Extending to other domains](#extending-to-other-domains)
- [Further reading](#further-reading)

## Getting started

### Install

The core install is **torch-free**: `numpy` / `pandas` / `scikit-learn` / `scipy` only, enough to use `kalos.kit` (leakage-controlled splits, signed Spearman drivers, split-conformal intervals, promotion gates, the anonymizer).
The GP surrogate, acquisition, and grouped-CV evaluation need the `ml` extra (torch / botorch / gpytorch).

```bash
python -m pip install -e .              # core, torch-free (kalos.kit primitives only)
python -m pip install -e ".[ml]"        # + the BoTorch engine (surrogate, optimize, evaluation)
python -m pip install -e ".[protein]"   # + real ESM-2 (transformers)
python -m pip install -e ".[ml,portal]" # the web portal (FastAPI) drives the live GP, so it needs [ml] too
```

### Run the demos and the portal

```bash
python examples/run_bo_loop.py          # BoTorch closed-loop demo (needs [ml])
python examples/run_multiobjective.py   # multi-objective (titer + purity) Pareto demo (needs [ml])
python examples/run_protein_embed.py    # ESM-2 embedding demo (downloads a small model)
python -m kalos.portal                  # -> http://127.0.0.1:8050 (KALOS_HOST / KALOS_PORT to change)
```

### Development checks

CI runs the same three gates on every push and pull request (`.github/workflows/ci.yml`).
Reproduce them from a clean `.[ml,portal,dev]` install:

```bash
python -m pip install -e ".[ml,portal,dev]"  # + ruff, mypy, and type stubs
ruff check kalos/ tests/ examples/           # lint
mypy                                         # types (config in pyproject [tool.mypy])
python -m pytest -q                          # tests (KALOS_TEST_ESM=1 to include ESM-2)
```

## Repository layout

```
kalos/
  core/
    surrogate.py       BoTorch SingleTaskGP (input-normalized, output-standardized)
    optimize.py        qLogNEI acquisition + optimize_acqf -> next batch (in-flight runs as X_pending,
                       feasibility-gated in log space, optional outcome-constraint floor)
    multiobjective.py  qLogNEHVI for two+ objectives (titer AND purity) + Pareto front
    feasibility.py     feasibility classifier + gated acquisition, and its CV report for the gates
    replicates.py      replicate-aware aggregation + assay noise-floor estimation
    evaluation.py      grouped CV Spearman (leakage-controlled, repeated partitions) + interval calibration + LOGO/top-k/group-mean baselines
    splits.py          group-aware CV + a leakage tripwire (assert_no_group_leakage)
    drivers.py         signed Spearman drivers with bootstrap confidence intervals (tested per recipe, not per row)
    conformal.py       split-conformal prediction intervals (distribution-free)
    gates.py           fail-closed promotion gates
  kit/
    __init__.py        torch-free facade re-exporting splits/drivers/conformal/gates/anonymizer
  data/
    anonymizer.py      strip identity, hash grouping keys
  features/
    protein.py         ESM-2 embeddings (transformers, MPS) + KmerEmbedder stand-in
  scale/
    features.py        physics-informed scale features (P/V, vs, kLa, hydrostatic proxies; cited, fittable)
    transfer.py        scale-up transfer over the Surrogate (+ opt-in physics_mean / multi_fidelity candidates)
    candidates.py      the evidence-attached alternates (physics-informed mean, scale-as-fidelity)
    evaluation.py      leave-one-scale-out with extrapolation direction reported separately
  domains/
    profile.py         torch-free ColumnRoles / DesignSpace + DomainProfile (declared roles, mixed spaces)
    bioprocess.py      the default bioprocess role-hint profile
    generic.py         a domain-neutral profile for non-bio tabular optimization
  validation/
    bounds.py          physically-possible ranges per measurement dimension (physics vs convention, each justified)
    checks.py          the eleven data-quality checks (units, bounds, duplicates, missingness, informative missingness, outliers, provenance, replicates, controls, constants, constant-within-group)
    runner.py          validate_frame: runs every check, never raises, + apply_unit_conversions
    report.py          Finding / UnitConversion / ValidationReport, JSON-safe serialization
  normalize/
    units.py           the canonical unit registry: parse "34.6 C", convert to a base unit
    synonyms.py        deterministic header -> canonical column mapping
    llm.py             optional LLM-assisted mapping (needs ANTHROPIC_API_KEY; offline fallback always works)
  providers/
    registry.py        external-provider credential slots (Anthropic / BioNeMo), all keyless-by-default
  portal/
    app.py             FastAPI app: live-engine views + the "run your own data" upload path
    validate.py        ingestion preflight + per-column provenance (what was kept/dropped and why)
    campaign*.py       the campaign loop routes and store
    scale_routes.py    the Scale-Up Readout
  bench/               benchmark harness (python -m kalos.bench)
deploy/              container images, docker compose, and the operator runbook
docs/                contracts and evidence behind the API (see Further reading)
examples/            demos, synthetic data sets, run_on_media_data.py
experiments/         off-path research prototypes; NOT part of the shipped package or its guarantees
tests/               pytest
```

## HTTP API

### Uploading a run sheet (`POST /api/run`)

The portal accepts an uploaded CSV / TSV / Excel run sheet and returns the analysis: auto-detected target, leakage-safe features, honest grouped-CV, signed drivers, and a proposed next batch.
Because the sheet comes from an external client, the upload path is guarded:

- **Size + shape caps.**
  A raw-byte cap (default 25 MB, override with `KALOS_MAX_UPLOAD_MB`), a 512-column ceiling, a 100k-row CSV cap, and a 2M-cell xlsx cap (a zip-bomb guard).
  Every cap fails closed: an over-cap upload is REJECTED with a 400, never silently truncated.
  The xlsx cell cap is enforced from the workbook's declared dimensions BEFORE the frame is materialized, so a zip-bomb is rejected without the memory spike.
  Filetype is sniffed by magic bytes (`PK\x03\x04` -> Excel, else text/CSV), not by extension.
- **Safe errors.**
  A bad upload returns a generic HTTP 400 ("Could not parse the uploaded file. Check it is a CSV or Excel run-sheet.").
  This holds even when the failure is a GP fit error (`torch.linalg.LinAlgError`) or a leakage-guard assertion, not just a parser error: a catch-all normalizes any such failure to the same `{error}` JSON envelope.
  Parser details, column names, cell values, paths, and stack traces are logged server-side and never returned to the caller.
- **Provenance.**
  The response includes a `provenance` list: for every column, its status (`kept_feature`, `target`, `dropped_id`, `dropped_output`, `dropped_constant`, `dropped_constant_on_fitted_rows`, `dropped_sparse`, `dropped_all_blank`) and how many cells had to be coerced from non-numeric text (units like "34.6 C").
  A feature that varies over the full sheet but is constant on the target-present rows the GP actually fits is dropped and flagged (`dropped_constant_on_fitted_rows`), never silently pinned to a zero-width bound.
- **Data validation gate.**
  Before any column is typed or dropped, the sheet runs through `kalos.validation`: eleven checks covering unit consistency, physical bounds, duplicates vs replicates, missingness, informative missingness, outliers, provenance metadata, replicate adequacy, controls presence, constant columns, and columns constant within a group.
  The report is returned as `validation`, with `status` one of `pass`, `pass_with_warnings`, or `fail`.
  An **error** is a physics violation (pH 40, a negative titer, one column mixing g/L and mg/mL); a **warning** is possible but operationally suspect.
  Row lists are capped at 20 with the true total in `detail.n_rows_affected`, so a finding never implies its list is complete.
  `KALOS_VALIDATION_MODE=strict` refuses a `fail` upload; the default `warn` analyzes it anyway and returns the findings.
- **Units are converted, not discarded.**
  Single-unit columns such as `"34.6 C"` are converted to their base unit before feature selection and reported in `validation.conversions` with a human label (`"Celsius"`), instead of failing the numeric test and coming back `dropped_sparse`.
  Only units the registry can actually convert are rewritten: a vessel column of `"5L"`/`"500L"` uses an unrecognized token, and rewriting it would turn an identifier into a measurement, so it is left alone and the client is told.
- **Proposals stay physically possible.**
  The design box for each continuous feature is built from physically valid observations only, so one `-999` sensor sentinel cannot widen the search space into impossible recipes (it once produced proposals at -422 C, below absolute zero).
  Any narrowing is reported in `design_box_exclusions`, never silent, in every validation mode.
- **Reproducibility.**
  The analyze path is seeded, so the same upload yields identical proposals; the response carries `seed`, `timestamp`, and `engine_version`.
  Protein embeddings pin the ESM-2 checkpoint to an immutable commit, so upstream cannot change model features without a diff here.
- **Opt-in anonymization.**
  Pass the form field `anonymize=true` to strip client/strain identity and hash grouping keys (`kalos/data/anonymizer.py`), so a run stays groupable for leakage-safe CV without carrying who it belongs to.
  Feature and target names are kept as-is (the owner UI legitimately shows drivers like "Methanol").

### The campaign loop (`/api/campaign*`)

The closed optimization loop behind the kalos-web `/decide` surface: propose a batch, run the recipes, log the measured outcome, re-propose on the grown dataset.
A campaign is one target plus a growing dataset of (recipe -> measured outcome) rows.
`POST /api/campaign/reanalyze` folds logged results into that dataset and re-runs the same leakage-controlled analysis, so each round stays as honest as the first (grouped-CV reliability, conformal bands, "not modeled" callouts).

Runs that were started but have not been measured yet are handed to the acquisition as in-flight points (`X_pending`).
They have no outcome, so they never join the fit; but without them a mid-round re-proposal treats a recipe currently in the incubator as unexplored and proposes it again.
On a 4-factor design at `q=5` over 10 seeds, a blind re-proposal repeated a mean of 2.1 of its 5 recipes against work already running; with the pending block it repeats none (`python -m kalos.bench --pending`).
See `docs/CAMPAIGN_LOOP.md` for the full contract.

### The Scale-Up Readout (`POST /api/scale/readout`)

Upload a multi-scale run sheet and a target scale; get back one printable page that either issues a predicted level at the target scale with an approximate-coverage interval, or refuses and says why.
Before it predicts, kalos backtests itself on your own scale ladder (fit on the smaller scales, predict the next one up).
It only issues a number when that backtest beats two naive baselines and the requested step up is within 2x of a step it has already won (warning up to 5x, refusal beyond).
Nothing is stored server-side; the page prints the SHA-256 of the upload, the engine version, every threshold, and the full normalize plan.
It predicts a level, not a ranking of recipes at scale.
Development decision support only; not a GMP or regulatory record.
A synthetic 9-scale demo sheet ships in `examples/synthetic_scaleup/`.

## Running it

### Hardware

BoTorch GPs run in float64 for numerical stability, and Apple's MPS backend is float32-only, so the **GP runs on CPU** (fast for the small-sample regime BO targets).
The **ESM-2 embedder runs on MPS** when available.
No GPU is required to use the platform locally.

### Deployment

The deploy pack lives in `deploy/`: container images for the engine and the web app, a `docker compose` file, and a SQLite backup sidecar.
`deploy/RUNBOOK.md` has the quickstart, the security checklist, and the backup/restore drills.
The portal binds to loopback by default and refuses to serve unauthenticated on any other interface.

## Extending to other domains

The optimization engine (`core/`) and the upload/validate pipeline are domain-neutral: they operate on numeric arrays and know nothing about biology.
Biology lives in the `domains/` layer as data, not code in the engine.

- **Declared column roles.**
  By default the analyze path infers roles (target, features, ids, group) from the bioprocess profile's name hints.
  A caller in any other domain can instead declare a `ColumnRoles` schema (target, features, categoricals, groups, ids), stating what their columns mean rather than renaming them to match a regex.
  `POST /api/run` accepts an optional `roles` JSON field for this; with it the analysis runs under the neutral generic profile.
- **Mixed continuous + categorical design spaces.**
  A `DesignSpace` marks each dimension continuous or categorical.
  When categoricals are present the engine fits a BoTorch `MixedSingleTaskGP` and proposes with `optimize_acqf_mixed`, and proposals decode integer level codes back to labels.
  The continuous-only path is unchanged.

To target a new domain, declare a `ColumnRoles` (or copy `domains/generic.py` into a new profile) and list any discrete choices under `categoricals`.
Nothing in `core/` changes.
The `mixed_bump` objective in `bench/` exercises the mixed loop end to end.

## Further reading

| Doc | What it covers |
| --- | --- |
| [`docs/CAMPAIGN_LOOP.md`](docs/CAMPAIGN_LOOP.md) | The campaign API contract |
| [`docs/SCALE_READOUT.md`](docs/SCALE_READOUT.md) | Scale-Up Readout contract, gate checks, and decision table |
| [`docs/SCALE_EVIDENCE.md`](docs/SCALE_EVIDENCE.md) | The evidence behind the scale model |
| [`BENCHMARK.md`](BENCHMARK.md) | Whether the optimizer beats a space-filling design |
| [`deploy/RUNBOOK.md`](deploy/RUNBOOK.md) | Deploying and operating the stack |

No proprietary data lives in this repo; the demos use synthetic data and public protein sequences.
