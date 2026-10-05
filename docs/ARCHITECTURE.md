# Architecture

How the `kalos` package is organized, and the two loops it serves. The
[root README](../README.md) has the short version; this is the map.

## Module map

```
src/kalos/
  core/
    surrogate.py       BoTorch SingleTaskGP (input-normalized, output-standardized)
    optimize.py        qLogNEI acquisition + optimize_acqf -> next batch (in-flight runs as X_pending,
                       feasibility-gated in log space, optional outcome-constraint floor)
    multiobjective.py  qLogNEHVI for two+ objectives (titer AND purity) + Pareto front
    evaluation.py      grouped CV Spearman (leakage-controlled, repeated partitions) + interval calibration + LOGO/top-k/group-mean
                       baselines + the XGBoost baseline the GP is scored against (paired, same folds)
    splits.py          group-aware CV + a leakage tripwire (assert_no_group_leakage)
    drivers.py         signed Spearman drivers with bootstrap confidence intervals (tested per recipe, not per row)
    conformal.py       split-conformal prediction intervals (distribution-free)
    gates.py           fail-closed promotion gates
    feasibility.py     producer/non-producer classifier that gates acquisition away from recipes likely to fail
    replicates.py      replicate-aware aggregation, assay noise floor, heteroscedasticity diagnosis
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
    llm.py             offline plan (no network or key) + optional LLM tier (Anthropic or self-hosted Ollama)
    typesafe_tier.py   optional TypeSafe tier: per-column typed judgments (role / identity / canonical name)
    roles.py           plan -> ColumnRoles, so a plan can drive /api/run (roles=auto)
    apply.py           execute a plan deterministically (drop, hash, rename, convert)
  providers/
    registry.py        external-provider credential slots (Anthropic / TypeSafe / BioNeMo / Benchling), all keyless-by-default
  bench/               closed-loop benchmarks with known optima (BENCHMARK.md)
  portal/
    app.py             FastAPI app: live-engine views + /api/experiments + the "run your own data" upload path
    uploads.py         upload guards: size/shape caps, magic-byte filetype sniffing, safe parsing
    auth.py            bearer-token principals, scopes, tenants (open mode when no tokens are configured)
    analysis.py        `_analyze`: the stage-4/5 pipeline behind /api/run and campaign re-analysis
    campaign.py        the closed loop's state + re-analysis (campaign_routes.py mounts /api/campaign)
    validate.py        ingestion preflight + per-column provenance (what was kept/dropped and why)
```


A lot of the honest-evaluation and data machinery was carried over from a prior
lean engine (`voyager-brain-rebuild`): the leakage-controlled splits, the
bootstrap-Spearman drivers, the data anonymization, and the push-feed ingestion.

## Experiment store and the Voyager loop

Run data flows through the Experiment store. A run sheet is uploaded as an
Experiment (`POST /api/experiments`), flagged `READY`, and the Singleton runner
pulls it, runs the BO engine, and writes the result back
(`DRAFT -> READY -> PROCESSING -> DONE|FAILED`). The store is SQLite at
`~/.kalos/experiments.db`. See [M2_INTEGRATION.md](M2_INTEGRATION.md) for the full contract.

Client/strain identity can be stripped and grouping keys hashed on upload via the
anonymizer (`src/kalos/data/anonymizer.py`), so a run stays groupable for
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
(`python -m kalos.bench --pending`). See [CAMPAIGN_LOOP.md](CAMPAIGN_LOOP.md) for the full
contract.

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

## Device

BoTorch GPs run in float64 for numerical stability, and Apple's MPS backend is
float32-only, so the **GP runs on CPU** (fast for the small-sample regime BO
targets). The **ESM-2 embedder runs on MPS** when available. No GPU is required
to use the platform locally.
