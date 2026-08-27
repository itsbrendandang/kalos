# Kalos

A Bayesian-optimization platform for bioprocess development, built on **BoTorch**
(a real GP surrogate + acquisition) with **NVIDIA BioNeMo / ESM-2** protein
features. Honest by construction: leakage-controlled cross-validation and
fail-closed promotion gates, on a production BoTorch core rather than a
hand-rolled GP + EI.

> Genuine, classical/probabilistic + deep-learning ML — not marketing. The BO
> loop runs here today, and ESM-2 embeds protein sequences on the Mac's GPU.

## What's here

```
kalos/
  core/
    surrogate.py       BoTorch SingleTaskGP (input-normalized, output-standardized)
    optimize.py        qLogNEI acquisition + optimize_acqf -> next batch (in-flight runs as X_pending)
    multiobjective.py  qLogNEHVI for two+ objectives (titer AND purity) + Pareto front
    evaluation.py      grouped cross-validated Spearman (leakage-controlled)
    splits.py          group-aware CV + a leakage tripwire (assert_no_group_leakage)
    drivers.py         signed Spearman drivers with bootstrap confidence intervals
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
  domains/
    profile.py         torch-free ColumnRoles / DesignSpace + DomainProfile (declared roles, mixed spaces)
    bioprocess.py      the default bioprocess role-hint profile (legacy behavior)
    generic.py         a domain-neutral profile for non-bio tabular optimization
  validation/
    bounds.py          physically-possible ranges per measurement dimension (physics vs convention, each justified)
    checks.py          the nine data-quality checks (units, bounds, duplicates, missingness, outliers, provenance, replicates, controls, constants)
    runner.py          validate_frame: runs every check, never raises, + apply_unit_conversions
    report.py          Finding / UnitConversion / ValidationReport, JSON-safe serialization
  normalize/
    units.py           the canonical unit registry: parse "34.6 C", convert to a base unit
    synonyms.py        deterministic header -> canonical column mapping
    llm.py             optional LLM-assisted mapping (needs ANTHROPIC_API_KEY; offline fallback always works)
  providers/
    registry.py        external-provider credential slots (Anthropic / BioNeMo / Benchling), all keyless-by-default
  portal/
    app.py             FastAPI app: live-engine views + /api/experiments + the "run your own data" upload path
    validate.py        ingestion preflight + per-column provenance (what was kept/dropped and why)
examples/            demos + run_on_media_data.py
experiments/         off-path research prototypes; NOT part of the shipped package or its guarantees (e.g. missingness_indicator/)
tests/               pytest
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
  `kalos.validation`: nine checks covering unit consistency, physical bounds, duplicates vs
  replicates, missingness, outliers, provenance metadata, replicate adequacy, controls presence,
  and constant columns. The report is returned as `validation`, with `status` one of `pass`,
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
```

### Development checks

CI runs the same three gates on every push and pull request (`.github/workflows/ci.yml`).
Reproduce them from a clean `.[ml,portal,dev]` install:

```bash
python -m pip install -e ".[ml,portal,dev]"  # + ruff, mypy, and type stubs
ruff check kalos/                            # lint
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
  `bo_feas` / `bo_feas_clean` strategies in `bench/pool.py`). Calibration (ECE/Brier)
  for the promotion gates is still pending.
- ~~Replicate-aware aggregation + assay noise-floor estimation + optional
  fixed-noise GP.~~ **Done** (`core/replicates.py`, `Surrogate.fit(..., noise=...)`,
  `pool_from_frame(..., aggregate=True)`) — the SNR lever from `BENCHMARK.md`.
- Larger ESM-2 / full NVIDIA BioNeMo backend behind the same embedder interface.

No proprietary data lives in this repo; the demos use synthetic data and public
protein sequences.
