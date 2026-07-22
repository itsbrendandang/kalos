# Changelog

Newest first.

## 2026-07-21 (even later)

### Changed - the SNR lever ships in production: replicate-aware proposals (`kalos/portal/analysis.py`)

BENCHMARK.md established that BO only beats random on the real zero-inflated media data when it optimizes the replicate-averaged (reproducible) titer with the measured assay noise floor fed to the GP - single measurements reward lucky noise spikes (ICC ~0.26, roughly three quarters of titer variance is assay noise).
That fix lived only in the benchmark harness; the production analysis path still fit the GP on raw single measurements.

- `kalos/portal/analysis.py` (`_analyze`): when the fitted rows have replicated recipes (>= 2 replicated, >= 6 distinct recipes, positive noise floor), the proposal surrogate is now fit on `aggregate_replicates()` means with per-recipe fixed observation variance `sigma^2 / n_reps` (via `Surrogate.fit(noise=...)`), and the proposed batch optimizes that reproducible objective; the shown incumbent is the reproducible best.
  Non-replicated sheets fall through to the unchanged raw fit.
- Every analysis result now carries a `noise` block: `n_recipes`, `n_replicated`, `replicate_aware`, `icc`, `noise_sd`, `signal_sd`, `best_single`, `best_reproducible` - the honest signal-to-noise picture and the reproducible ceiling, not just the lucky spike.
  Diagnostics (grouped-CV reliability, drivers) stay on the raw rows - they are already replicate-grouped for leakage and describe the as-measured signal.
- `examples/benchmark_media_pool.py` (new): a committed, runnable reproduction of the real-data pool retrospective (`KALOS_MEDIA_DATA=/path python examples/benchmark_media_pool.py`), racing BO / feasibility-gated BO / random on the reproducible objective (BO leads) and the single-measurement objective (the artifact). No client data is committed.
- `tests/test_replicate_aware_analysis.py` (new): replicate-aware fit triggers + honest noise report; non-replicated unchanged; deterministic.

## 2026-07-21 (later)

### Added - campaign loop: the closed optimization loop (`kalos/portal/campaign.py`, `/api/campaign*`)

Until now the portal was a one-shot analysis viewer: upload a run sheet, get a batch, done.
This adds the loop that makes it a product - propose, run, log the measured outcome, re-propose - reusing the existing `_analyze` path with no new engine capability.
Design and full contract in `docs/CAMPAIGN_LOOP.md`.

- `kalos/portal/campaign.py` (new, torch-free): `CampaignStore` persists one campaign (a target plus a growing `base_rows` dataset and a list of started `pending` runs) to `~/.kalos/campaign.json`, lock-guarded and written atomically (temp file + `os.replace`), mirroring the `_LATEST` state pattern.
  `best` is the max MEASURED target over `base_rows`, never a prediction; a non-finite result is rejected; only runs with a real measured outcome are ever folded into the dataset.
- `kalos/portal/campaign_routes.py` (new): `GET /api/campaign` (summary for the `/decide` view), `POST /api/campaign/start` (append proposed recipes as awaiting runs), `POST /api/campaign/result` (log a measured outcome), `POST /api/campaign/reanalyze` (fold measured runs into the dataset, re-run `_analyze` via a worker thread, persist as the new `/api/latest`, increment the round).
  `reanalyze` returns the same shape `GET /api/latest` does (including `dataset` and `updated`), so the frontend can swap it straight into its `PopulatedResult` state.
- `kalos/portal/app.py`: a fresh `/api/run` upload now seeds a fresh campaign from `(df, target, proposal_features)` (best-effort - a seeding failure never breaks the upload); the campaign router is mounted beside the experiments router.
- Honest by construction each round: re-analyze routes through the same leakage-controlled grouped-CV `_analyze`, so reliability, conformal bands, and the "not modeled" callouts stay first-class every cycle.
- Progress trajectory: the campaign records a `history` of `{round, best, n_base}` points (round 0 at seed, one per re-analyze) so the frontend can plot best-so-far converging. `best` is measured and base rows only grow, so the trajectory is non-decreasing.

`tests/test_campaign.py`: 12 tests (seed, summary states, start, result validation, and the fold-and-re-analyze round-trip that grows the base, increments the round, and extends the history trajectory).

### Fixed - campaign re-analyze is now transactional (no data loss, no phantom rounds)

Review of the closed loop surfaced a persist-then-validate ordering defect: `fold_and_snapshot` committed the fold (round++, measured runs merged into `base_rows`) to disk *before* `_analyze` ran, so an analysis failure permanently advanced the round with no rollback, and a concurrent `/api/run` upload during the multi-second analysis could silently destroy the just-folded measured data and clobber `/api/latest` with stale numbers.

- `kalos/portal/campaign.py`: `fold_and_snapshot` is split into `plan_fold()` (computes the folded dataset in memory, mutating nothing) and `commit_fold(generation)` (persists only if the campaign is unchanged).
  Every write now stamps a fresh `generation` token, so a re-analysis that was planned against one campaign refuses to commit if a `seed()`/`set_result`/`start` landed underneath it.
- `kalos/portal/campaign_routes.py`: `reanalyze` now plans the fold, runs `_analyze`, and only then commits - so a failed analysis leaves the campaign untouched (a retry is meaningful), and a campaign reseeded mid-analysis returns `409` without ever calling `_save_latest`, so `campaign.json` is never overwritten with stale results.
  Re-analyzing with no newly measured run is rejected up front (it would only inflate the round and the progress trajectory).
- `/api/latest` is a separate resource (own lock, taken in the opposite order by the upload path), so its final write is guarded best-effort: the route re-checks the campaign generation immediately before `_save_latest` and skips the stale write if a fresh upload reseeded in between.
  This eliminates the multi-second race across `_analyze` and the upload-lands-after-commit case; a sub-millisecond residual window is documented in `docs/CAMPAIGN_LOOP.md` (fully sealing it needs an ordered stamp on `_LATEST`).
- `kalos/portal/campaign.py`: `plan_fold` reads `generation` with `.get()` so a pre-token `campaign.json` re-analyzes without crashing (`KeyError` -> `500`); the migration is self-healing (the commit re-stamps a real token).
- `kalos/portal/campaign.py`: `start()` now validates each `recipe` is a non-empty mapping and raises `CampaignError` at the point of the bad input, instead of surfacing as an unhandled `500` rounds later inside the fold.
- `kalos/portal/campaign_routes.py`: documented the deliberate unauthenticated auth posture for `/api/campaign*` (browser-facing, localhost-bound, same as `/api/run`).
- `docs/CAMPAIGN_LOOP.md`: added Mermaid diagrams (the closed loop, the pending-run lifecycle, the transactional re-analyze sequence) and documented the transactional design and the residual `/api/latest` window.
- `pyproject.toml`: added `httpx>=0.27,<1` to the `dev` extra. `starlette.testclient.TestClient` (used by every portal test) needs `httpx`, but it is not pulled in transitively, so CI's `pip install -e ".[ml,portal,dev]"` left it absent and the whole portal suite errored at import (`RuntimeError: ... requires the httpx2 package`). This was red on `feat/campaign-loop` before this branch; the fix lands with the merge.

`tests/test_campaign.py`: +8 tests - malformed-recipe rejection (×4), no-measured-runs rejection, `_analyze`-failure leaves round/base untouched then a retry folds normally, `commit_fold` aborting when the campaign is reseeded underneath it, and re-analyze on a legacy (pre-`generation`) `campaign.json` not crashing.

## 2026-07-21

### Changed - CI now gates ruff + mypy + the portal tests, and the type layer is clean

CI ran `pytest` only, and it installed `.[ml,dev]` without the `portal` extra, so the portal tests (which `importorskip("fastapi")`) were silently skipped on every run.
Lint and type checking were never enforced at all.
This wires the quality gates that were only ever run by hand.

- `.github/workflows/ci.yml`: installs `.[ml,portal,dev]` and runs `ruff check kalos/`, `mypy`, then `pytest`.
  The portal is a shipped surface, so its tests now actually execute in CI instead of skipping.
- `pyproject.toml`: `dev` extra gains `ruff`, `mypy`, `pandas-stubs`, and `scipy-stubs`, so the checks are reproducible from a clean `.[dev]` install rather than depending on a globally installed tool.
- `pyproject.toml`: added a `[tool.mypy]` block targeting Python 3.12 (the CI and dev interpreter) over the `kalos` package.
  `torch` / `botorch` / `gpytorch` / `linear_operator` are set to `follow_imports = skip`; following their full typed surface pushed a cold-cache run into minutes, and we typecheck kalos's own code, not theirs.

### Fixed - 12 real mypy errors + a `rounds=0` edge case in `/api/multi`

With the type stubs installed, mypy surfaced twelve genuine typing gaps.
None changed runtime behavior except the last.

- `kalos/core/gates.py`: `_finite` now returns `TypeGuard[float]`, so mypy narrows `stats.get(key)` from `Any | None` to `float` inside the guarded branch (the `float(v)` / `v < lo` comparisons were untyped before).
- `kalos/core/splits.py`, `kalos/core/drivers.py`: array-like parameters (`y`, `groups`, `signal`) are typed `numpy.typing.ArrayLike` instead of `Sequence`, since they are called with numpy arrays and immediately go through `np.asarray`.
- `kalos/portal/serialization.py`, `kalos/portal/analysis.py`, `kalos/core/drivers.py`: narrowed a few `object`-typed values (pandas `to_dict` records, driver dicts, `feature_names`) with `cast` / explicit annotations so the downstream `float(...)` / indexing typechecks.
- `kalos/portal/app.py`: `run_multi` now clamps `rounds`/`q` to at least 1 and pre-binds `last_batch`.
  A `?rounds=0` request previously hit `last_batch` unbound and raised `NameError`; it now returns one honest round.
  Added an `assert s.model is not None` after `fit()` to document the invariant mypy could not otherwise see.

## 2026-07-20

### Added - domain-neutral core: declared column roles + mixed continuous/categorical design spaces (`kalos/domains/`)
Kalos was branded bioprocess-only, but the engine (`core/`), store, runner, and upload pipeline
operate on numeric arrays and are domain-agnostic. This change makes that reusable without touching
the engine, and adds categorical-parameter support so the platform fits industries with discrete
process choices (which catalyst, which resin), not just continuous recipes.

- `kalos/domains/` (new, torch-free): `ColumnRoles` (an explicit target/features/groups/ids/
  categoricals schema), `DomainProfile` (fallback role-hint regexes as data, not engine code),
  `DesignSpace` + `Dimension` (per-dimension continuous/categorical spec with integer encoding and
  label decoding), and `build_design_space`. Ships `BIOPROCESS_PROFILE` (the legacy hints, so the
  default path is unchanged) and `GENERIC_PROFILE` (domain-neutral). Importing `kalos.domains` never
  loads torch (`tests/test_domains.py`).
- `kalos/core/surrogate.py`: `Surrogate.fit(..., cat_dims=)` fits a BoTorch `MixedSingleTaskGP`
  (CategoricalKernel on the categorical dims, Matern on the continuous, continuous dims normalized to
  the box) when categoricals are present; the continuous `SingleTaskGP` path is unchanged.
- `kalos/core/optimize.py`: `propose(..., cat_dims=, cat_cardinalities=)` uses `optimize_acqf_mixed`
  (enumerating categorical assignments, exact) for small categorical spaces and falls back to
  `optimize_acqf_mixed_alternating` past `MAX_MIXED_COMBOS`. Continuous path unchanged.
- `kalos/core/evaluation.py`: `grouped_cv_report(..., cat_dims=)` threads the mixed GP through
  leakage-controlled CV so the reported number matches the deployed model.
- `kalos/portal/analysis.py`: `_analyze(..., roles=, profile=)`. With a declared `ColumnRoles` the
  roles are used directly (generic profile); with none, the bioprocess profile infers them exactly as
  before. Drivers are computed over continuous features only (a Spearman "driver" for a nominal
  category is not meaningful). Proposals carry a decoded `recipe` (`{feature: value}`), and the
  response adds `categorical_features`.
- `kalos/portal/app.py`: `POST /api/run` accepts an optional `roles` JSON form field; when present the
  upload is analyzed domain-neutrally.
- `kalos/portal/validate.py`: `column_provenance(..., declared=)` records each column's role `source`
  (`declared` vs `inferred`) so the audit trail is honest about who decided the role.
- `kalos/bench/`: `MixedObjective` + `mixed_bump` + `run_mixed_one` exercise the mixed loop; a test
  confirms mixed BO reaches far lower simple regret than random choice.
- No behavior change on the bioprocess path: the default profile and continuous engine branch are
  byte-for-byte the prior code paths (verified field-by-field against `main`); the analyze response
  only gains additive fields (`categorical_features`, `proposal_optimizer`, per-proposal `recipe`,
  per-column `source`). Guarded by the existing portal/hardening/bench tests.

### Fixed - provenance honesty for declared roles + surfaced mixed optimizer (review follow-ups)
An adversarial review of the change above found the modeling core correct but flagged honesty gaps in
the new declared-roles/provenance layer (no correctness bugs). Addressed:
- `kalos/portal/analysis.py`: a declared feature / categorical / group / id name that does not match a
  sheet header now raises a clear `ValueError` instead of being silently dropped (a typo previously
  vanished with no signal, so a client believed a column was honored when it was not; a typo'd id in
  particular used to leave the real id column in as a feature).
- `kalos/portal/validate.py`: `column_provenance` gained `declared_ids` and `declared_features`. A
  declared id now reports as `dropped_id` with `source="declared"` (not the misleading
  `dropped_sparse`/`inferred`), and a column declared as a continuous feature but holding text now
  reports as the new `dropped_non_numeric` status - an honest "you likely meant to mark this
  categorical" - instead of `dropped_sparse`.
- `kalos/portal/analysis.py`: the analyze response now carries `proposal_optimizer`
  (`continuous` | `mixed_exact` | `mixed_alternating`) so the client can tell when a large categorical
  space fell back from exact enumeration to the alternating heuristic, alongside the existing
  seed/timestamp/engine_version audit fields.

A second review round found and closed further honesty gaps:
- `kalos/portal/analysis.py`: a self-contradictory schema (the target also declared an id or
  categorical) now raises instead of silently dropping one role.
- `kalos/portal/analysis.py`: declared continuous features must clear the same >=80% numeric-parse
  gate inference mode uses; a mostly-text column declared as a feature is now reported
  `dropped_non_numeric` rather than silently zero-filled into the model as a `kept_feature`.
- `kalos/portal/analysis.py`: blank categorical cells are no longer an ordinary proposable level -
  they are excluded from the levels and their rows dropped from the fit (an unknown categorical can
  neither be modeled nor recommended). The response reports `n_dropped_incomplete`, and `n` is the
  row count actually fit.
- Test coverage added for all of the above plus previously-untested paths: the alternating optimizer
  (large cardinality), all-categorical design spaces, single-level categorical drop, declared group
  columns, and mixed-BO run reproducibility.
- Known follow-up (not yet done): there is no per-level replicate-count warning for a categorical
  level too sparse to identify - the recommended materials guardrail before running on real small-n
  formulation data.

## 2026-07-18

### Changed - torch/botorch/gpytorch are now optional (`kalos[ml]`); new `kalos.kit` torch-free facade
Phase 1 of engine consolidation: a sibling repo (`voyager-brain-rebuild`, deliberately torch-free)
is meant to import Kalos's leakage-controlled splits, driver analysis, conformal intervals,
promotion gates, and anonymizer instead of keeping its own copies. That only works if installing
`kalos` does not drag in a ~220 MB torch/botorch/gpytorch stack.

- `pyproject.toml`: `torch`, `botorch`, `gpytorch` moved out of core `dependencies` into a new
  `ml` extra. Core install (`pip install kalos`) is now torch-free: `numpy` / `pandas` /
  `scikit-learn` / `scipy` only. The `portal` extra still needs the live GP, so it is installed as
  `kalos[ml,portal]`.
- `kalos/kit/__init__.py` (new): a thin re-export facade over the already-torch-free
  `kalos.core.splits`, `kalos.core.drivers`, `kalos.core.conformal`, `kalos.core.gates`, and
  `kalos.data.anonymizer`. Nothing moved - existing imports like
  `from kalos.core.drivers import ...` are unchanged. `import kalos.kit` is guaranteed to never
  load torch (`tests/test_kit_torch_free.py`).
- No behavior changes: `kalos/__init__.py` and `kalos/core/__init__.py` were already lazy-loading
  the torch-dependent surrogate/optimize/evaluation exports (PEP 562 `__getattr__`) from a prior
  commit; this change only reorganizes the install metadata and adds the `kit` facade on top of
  that existing lazy-load boundary.

## 2026-07-06

### Added - Replicate-aware aggregation + assay noise floor + fixed-noise GP (`kalos/core/replicates.py`)
`BENCHMARK.md`'s SNR write-up found the real media DoE is heavily replicated (96 rows over 27
distinct recipes, up to 14 reps per recipe) with an ICC of ~0.26 - roughly 74% of titer variance
is assay noise, not recipe-to-recipe signal. This adds the tooling to act on that: aggregate
replicates into a reproducible per-recipe objective, estimate the assay noise floor from the
replicate spread, and optionally hand that noise estimate to the GP directly instead of making it
re-infer noise from a handful of points.

- `kalos/core/replicates.py` (new): `aggregate_replicates(X, y)` groups rows by identical
  rounded feature vectors and returns `(X_unique, y_mean, y_var, n_reps)` in deterministic
  first-occurrence order. `estimate_noise_floor(X, y)` pools the within-group sample variance
  over replicated groups into a single assay noise variance (`nan` if nothing is replicated).
  `noise_report(X, y)` adds `n_rows` / `n_recipes` / `n_replicated` / `signal_var` / `icc` on top,
  for a one-call summary of how much of the variance is real signal.
- `kalos/core/surrogate.py`: `Surrogate.fit(X, y, bounds, *, noise=None)` gains an optional fixed
  observation-noise variance (`None` default, a scalar, or a per-point array, all in the target's
  original units). When given, the GP is built with `train_Yvar` alongside
  `outcome_transform=Standardize` - BoTorch scales `Yvar` through the standardization internally
  and `SingleTaskGP` auto-selects a `FixedNoiseGaussianLikelihood`, so no extra likelihood
  plumbing was needed. Confirmed working on the installed BoTorch 0.18.1 / GPyTorch 1.15.2 before
  wiring it in. `noise=None` is byte-for-byte the previous inferred-noise behavior.
- `kalos/bench/pool.py`: `run_pool_one` / `run_pool` take a `noise` parameter, forwarded to every
  `Surrogate.fit(...)` call in the BO branches (default `None`, bo/random unchanged).
  `pool_from_frame(df, target, *, aggregate=False)` can collapse replicate rows to per-recipe
  means before returning `(X, y, feats)`; `feats` is unaffected, default `False` is unchanged.
- +10 tests (`tests/test_replicates.py`): known-duplicate aggregation, rounding-based near-duplicate
  merging, noise-floor recovery against an injected variance, ICC ballpark on synthetic
  signal/noise, fixed-noise `Surrogate.fit` (scalar and per-point array) end to end, `run_pool`
  with fixed noise producing a finite monotone trajectory, and `pool_from_frame(..., aggregate=True)`.

### Added - Feasibility classifier + gated acquisition (`kalos/core/feasibility.py`)
BENCHMARK.md's finding was that BO loses to random on the real media DoE because titer is
zero-inflated (~21% non-producers) and the GP over-exploits a noisy incumbent on a spiky
feasible/infeasible surface. Rather than change the acquisition function, feasibility (producer
vs non-producer) is now modeled as a separate binary classifier that gates EI, so the fix is
composable with the existing GP.

- `FeasibilityClassifier`: `StandardScaler` + `LogisticRegression(class_weight="balanced")` on
  binary labels. Cold-start safe: fewer than 2 classes or fewer than 3 minority-class examples in
  `fit` skips sklearn entirely and `predict_proba` returns all-ones (no gating, defers to EI) -
  this is what keeps the pool loop from crashing on sklearn's single-class fit error before enough
  non-producers have been observed.
- `feasible_labels(y, threshold=0.0)`: feasible iff `y > threshold` (strict).
- `feasibility_cv_auc(...)`: pooled out-of-fold CV-AUC for the feasibility classifier, using
  `StratifiedGroupKFold` when `groups` is given and `StratifiedKFold` otherwise; reduces
  `n_splits` to the minority class count and returns `nan` (never raises) when a stratified split
  or a well-defined AUC isn't possible.
- `kalos/bench/pool.py`: two new pool strategies, `bo_feas` (GP fit on all evaluated points, EI
  gated by predicted P(feasible)) and `bo_feas_clean` (GP fit only on feasible evaluated points,
  falling back to all points below 3 feasible examples, EI gated the same way). `bo` and `random`
  are unchanged. +6 tests (`tests/test_feasibility.py`, `tests/test_pool.py`) including a
  zero-inflated synthetic pool comparison.

### Update - Honest benchmark result (`BENCHMARK.md`)
Feasibility is highly learnable on the real media DoE (grouped-CV AUC 0.891), but gating EI
with it does not rescue BO (`bo_feas` matches plain `bo`; `bo_feas_clean` still loses to
random, 0.046 vs 0.082). The lever remains replicates / signal-to-noise, not the classifier.

## 2026-07-05

### Added - Closed-loop benchmark (`kalos/bench/`, `BENCHMARK.md`)
An honest answer to "does the optimizer beat a space-filling design?". `python -m kalos.bench`
races the BO loop against Latin Hypercube and random on synthetic surfaces with known optima,
sweeping observation noise. Finding: BO dominates when the signal is clean (reaches LHS's
end-value ~11-14 experiments sooner, near-zero regret) but its edge shrinks to marginal-or-nil
at 15% measurement noise - which is why the real, noisy media data shows only a weak ~0.37-0.52
signal. The lever is data quality (replicates, signal-to-noise), not the algorithm. Full write-up
and reproduction steps in `BENCHMARK.md`; +4 tests.

### Fixed - unify the anonymization scrub lists
`kalos/ingest/feed.py` kept its own second copy of the identity-scrub rules, which had drifted from
the canonical lists in `kalos/data/anonymizer.py`: the feed copy was missing the `subject` and `mrn`
drop-substrings and only hashed `campaign_id` / `campaign`, so `lot`, `batch_id`, `experiment`, and
`batch` columns in an uploaded run sheet passed through un-hashed on the live ingestion path
(`FileDataFeed.read` -> `anonymize_frame`, reached from `propose_next_batch`).

- `anonymize_frame` now imports `DROP_EXACT`, `DROP_SUBSTR`, `HASH_EXACT`, `HASH_SUBSTR` from
  `kalos.data.anonymizer` (single source of truth) and checks both the exact and substring hash sets,
  matching `Anonymizer.anonymize_meta`.
- `anonymize_frame`'s `salt` now defaults to `None` and resolves via `default_salt()`, so it honors
  `KALOS_ANON_SALT` instead of silently hardcoding the dev salt literal.

### Changed - Wave A1.2: production hardening
Go-live hardening from a memory / data-volume / BoTorch review. Response contract preserved.

- **Concurrency** (`kalos/portal/app.py`): the CPU-bound `/api/run` body (parse + GP fit + save)
  now runs via `run_in_threadpool` instead of blocking the single event loop, so concurrent
  uploads no longer hang the whole service. Added `torch.set_num_threads` (`KALOS_TORCH_THREADS`)
  to avoid CPU oversubscription, and a `threading.Lock` around the `_LATEST` read-modify-write.
- **GP-training-row cap** (`kalos/portal/app.py`): a `MAX_FIT_ROWS` guard (`KALOS_MAX_FIT_ROWS`,
  default 2000) rejects oversized fits before building the O(n^2) exact-GP kernel, separately from
  the raw-upload row cap. Rejects rather than silently subsamples.
- **Anonymization salt** (`kalos/data/anonymizer.py`): the salt now reads from `KALOS_ANON_SALT`
  and warns loudly on the dev fallback, so barcodes are not a fixed, dictionary-attackable pseudonym.
- **BoTorch robustness** (`kalos/core/surrogate.py`, `optimize.py`): `fit_gpytorch_mll` is retried
  with escalating Cholesky jitter and raises a distinct `FitError` on persistent failure, mapped to
  its own honest 400 (not the generic parse message). The single-objective acquisition moved from
  `qLogExpectedImprovement` to `qLogNoisyExpectedImprovement` (titer is noisy; matches the
  multi-objective side). `bounds` is now a required argument on the surrogate fits.
- Tests: `tests/test_production_hardening.py` covers the row cap, the env salt, and the FitError 400.

### Fixed - Wave A1.1: review fixes for the upload + optimization path
Follow-up fixes to Wave A1 from a code review of the FastAPI Bayesian-optimization engine.
No architecture change; the `/api/run` and `/api/latest` response contract is preserved and only
extended (a new `dropped_constant_on_fitted_rows` provenance status).

- **Multi-objective bounds-sanity** (`kalos/core/multiobjective.py`): the multi-objective path now
  mirrors the single-objective degenerate-bounds guard.
  `MultiObjectiveSurrogate.fit` sanitizes bounds (via `sanitize_bounds`) before building the
  `Normalize` box, and `propose_multiobjective` sanitizes before `optimize_acqf` and clips returned
  proposals into the observed box.
  This closes the constant-feature (e.g. fixed `Culture_Volume`) NaN / out-of-range blow-up class
  on qLogNEHVI proposals, the same ~33,000,000 regression already fixed on the single-objective side.
- **Error hygiene: fit-time failures stay in the JSON envelope** (`kalos/portal/app.py`, `/api/run`):
  a catch-all `except Exception` was added after the narrow parser catch.
  A `torch.linalg.LinAlgError` from a GP fit or an `AssertionError` from the leakage guard is not a
  plain `ValueError`, so it previously escaped to FastAPI's default text/plain HTTP 500 and broke the
  `{error}` JSON contract the frontend parses.
  It now returns the same generic 400 envelope; the full traceback is logged server-side and no
  parser text, exception message, or stack trace reaches the client.
- **Zip-bomb: xlsx cell cap enforced before materialization** (`kalos/portal/app.py`,
  `_reject_oversized_xlsx`): the xlsx cell-count ceiling now runs on the workbook's declared
  dimensions (opened read-only with openpyxl) BEFORE `pd.read_excel` materializes the frame, so a
  zip-bomb is rejected without the memory spike.
  The post-read shape check is kept as a belt-and-suspenders re-check.
- **CSV row cap fails closed** (`kalos/portal/app.py`, `_parse_upload`): an over-cap CSV is now
  REJECTED with a 400 (`MAX_CSV_ROWS`), matching the xlsx cell-cap behavior, instead of silently
  truncating to the first 100k rows.
  Silent data loss contradicted the safe-errors contract.
- **Honest constant-on-fitted-rows check** (`kalos/portal/app.py` `_analyze`,
  `kalos/portal/validate.py`): the varying-feature filter is recomputed on the target-present rows
  the GP actually fits, not the full column.
  A feature that varies over the whole sheet but is constant on the fitted rows would collapse its
  design box to zero width; it is now dropped and flagged in provenance as
  `dropped_constant_on_fitted_rows`, never silently pinned.

## 2026-07-02 (later)

### Added - Wave A1: safety + product-readiness hardening for the upload path
The client-facing `/api/run` upload path was hardened for external, untrusted run sheets.
No architecture change (still no auth/tenancy); the `/api/run` and `/api/latest` response
contract is preserved and only extended.

- **Upload guards** (`kalos/portal/app.py`, `_parse_upload`): a byte-size cap on the raw upload
  (default 25 MB, env `KALOS_MAX_UPLOAD_MB`), a filetype sniff by MAGIC BYTES (`PK\x03\x04` zip
  header -> xlsx/xls, otherwise UTF-8 text/CSV), a column ceiling (`MAX_COLUMNS=512`), a CSV row
  cap (`MAX_CSV_ROWS=100000`), and an xlsx cell-count ceiling (`MAX_XLSX_CELLS=2,000,000`, a
  zip-bomb guard). Any guard trip returns HTTP 400 with a generic message.
- **Error hygiene**: the catch-all `except Exception: return {"error": str(exc)}` was replaced by
  a narrow catch (`pandas.errors.*`, `ValueError`, `UnicodeError`) that returns a single generic
  message ("Could not parse the uploaded file. Check it is a CSV or Excel run-sheet.") and logs
  the full traceback server-side via the `logging` module. No parser text, column name, cell
  value, path, or stack trace ever reaches the client.
- **Ingestion provenance** (`kalos/portal/validate.py`): a new typed `column_provenance` returns a
  per-column status (`kept_feature`, `target`, `dropped_id`, `dropped_output`, `dropped_constant`,
  `dropped_sparse`, `dropped_all_blank`) plus a non-numeric `coerced_cells` count, surfaced as a
  new `provenance` field on the analyze result. This fixes the silent-column-drop problem: clients
  now see exactly what was used and what was dropped and why. Duplicate column labels are
  de-duplicated (`X`, `X.1`) so both stay visible.
- **Privacy**: raw column names and cell values never appear in logs or client error messages.
  Feature/target names are NOT force-anonymized (the owner UI legitimately shows drivers like
  "Methanol"); instead `/api/run` gains an opt-in `anonymize: bool = False` form field that
  pseudonymizes identifier-type columns only (stable, irreversible hash via `data/anonymizer`).
- **Reproducibility + audit**: the analyze path seeds `torch.manual_seed` + `np.random.seed`, so
  the same upload yields identical proposals, and the result now carries `seed`, `timestamp`
  (unix int), and `engine_version` (from `kalos.__version__`).
- **Bounds-sanity** (`kalos/core/surrogate.py` `sanitize_bounds`, applied in `Surrogate.fit` and
  `core/optimize.propose`): non-finite bounds are repaired and zero-width (constant-feature)
  intervals are widened, and every proposed coordinate is clamped into the observed
  `[min, max]` box. This closes the historical out-of-range blow-up (a constant `Culture_Volume`
  proposing ~= 33,000,000) at its root: a degenerate normalization box no longer NaN-poisons the
  GP fit, and no proposal can escape the observed range. Locked with a regression test.
- +14 tests (`tests/test_hardening.py`): oversized/wrong-magic-bytes/too-many-columns rejections,
  provenance on a messy sheet (units-in-cells, %-strings, a duplicate column, a constant column),
  error-hygiene (malformed bytes -> generic 400, no stack trace/path in the body), bounds-sanity,
  and seed reproducibility. 32 passing, 1 skipped (the ESM-2 test stays behind its flag).

## 2026-07-02

### Added — conformal band + honest reliability in the analyze output
`_analyze` (and therefore `/api/run` and `/api/latest`) now also returns:
- `conformal_q`: a distribution-free +/- half-width from the pooled out-of-fold residuals
  (`core/conformal.q_from_residuals`, alpha=0.1). An honest, often-wider alternative to the
  surrogate's own posterior std, which tends to be overconfident on small bioprocess datasets.
  Coverage is exact for iid split-conformal; treat it as approximate under grouped CV.
- `reliability`: `{spearman, ci95, spearman_floor: 0.20, clears_floor, ci_excludes_zero,
  unmodeled}`. Only what this path can actually assess; the spearman floor mirrors
  `GatesConfig.min_spearman`. It deliberately does NOT assert feasibility or calibration gates
  (listed under `unmodeled`), which this path does not measure. Consumed by the kalos-web
  Voyager "Phase 2 (live)" view so the UI shows a truthful uncertainty band and trust signal
  instead of a fabricated success probability or scale-up curve.

### Added — /api/latest: the Overview runs on real data, not the demo
The portal persists the most recent uploaded analysis (in-memory, plus best-effort JSON at
`$KALOS_STATE_DIR/latest_analysis.json`, default `~/.kalos`) on every `/api/run`, and serves it
at `GET /api/latest` (`{has_data, dataset, updated, ...analysis}`). This lets the kalos-web
Overview reflect the last dataset a user actually uploaded - real reliability, drivers, and
proposed experiments - instead of the synthetic `_titer`/`_purity` demo objective. `has_data`
is `false` until the first upload, so the home shows an upload prompt rather than pretending
there is data. A serialization or disk error can never fail an upload (persistence is
best-effort). +1 test.

### Fixed — portal upload path routed through the honest CV kit
The `/api/run` analyzer had its own inline CV loop and grouping. Rewired it to the one
leakage-checked path from `core`:
- Grouping now uses `row_hash_groups` on the RAW (NaN-preserving) feature values, so rows
  missing different components are not merged into one replicate group by the zero-fill.
- CV now uses `grouped_cv_report`, so the portal returns a pooled out-of-fold Spearman with a
  group-bootstrap 95% CI (`cv_ci95`) and the effective group count (`cv_n_groups`) — the honest
  signal, not a bare point estimate.
- Added the portal's first test (`tests/test_portal.py`): other measured outputs stay excluded
  from features (anti-leakage) and the honest CV fields are returned. On the raw Anagram upload
  the honest number is -0.23 [-0.42, 0.10] (23 features / 21 groups) — no reliable ranking at
  that feature/group ratio, the truthful "reduce features / collect more data" message rather
  than a fake positive.

### Fixed — leakage + honesty in grouped cross-validation (independent Codex + multi-agent review)
An independent review (OpenAI Codex plus a multi-agent modeling audit) found the reported CV
skill could be both contaminated and overstated at small n. Fixed the confirmed items:
- **NaN grouping leak** (`core/splits.py`): `row_hash_groups` collapsed missing values to 0.0,
  so rows missing different features hashed into one replicate group and leaked across every
  fold. NaN now gets a distinct token, so a real 0.0 and a missing value are different groups.
- **Degenerate split removed** (`core/splits.py`): `make_splits` returned a train==validation
  dummy when too few groups remained. It now returns no split (CV unavailable), so a model is
  never scored against itself.
- **One splitter** (`core/evaluation.py`): deleted the weaker round-robin `grouped_folds` /
  `_row_groups`; all grouped CV routes through the single leakage-checked `splits.make_splits`
  (GroupKFold + a no-overlap assertion). `grouped_folds` stays as a thin compatible wrapper.
- **Train/serve normalization consistency**: the CV loop fits every fold under one fixed
  normalization box (design bounds, else the observed range), matching the deployed model
  instead of each fold's own training envelope.
- **Honest reporting** (`core/evaluation.py`): new `grouped_cv_report` pools out-of-fold
  predictions and returns a group-level bootstrap 95% CI plus `n_oof` / `n_groups` / `n_folds`.
  `grouped_cv_spearman` now pools OOF too (was: a mean of tiny per-fold rhos that can only be
  +/-1 at this n).
- 16 tests pass (3 new regression tests: NaN-distinct grouping, categorical grouping, no dummy
  split); the ESM-2 model test stays skipped behind its flag.

## 2026-06-26

### Fixed — research-backed design-critique priorities (4-lens multi-agent review)
- A multi-agent critique (user-research + two research-currency lenses citing 2022-2025 HCI/data-viz papers + the design framework) found the portal was a half-step behind current human-in-the-loop BO practice. Applied the top three:
  - **Uncertainty + "why" on every proposal.** Each proposed experiment now carries a predicted value, a predictive uncertainty (GP posterior sigma), and an explore/exploit tag with a one-line reason ("predicted high (+1.6 vs best)" / "reduce model uncertainty here (±0.9)"). This is what HITL-BO and trust-calibration research (Microsoft G11; arXiv 2402.07632) say a recommendation must carry, and it finally binds the emerald/clay explore-exploit metaphor to a computed quantity. Applies to single, multi (predicted titer/purity), and the upload path.
  - **Newcomer first-10-seconds.** A plain-language value line ("Kalos suggests which experiment to run next..."), "BoTorch + ESM-2" demoted to a "powered by" footnote, "Example data" badges on the demo panels, panel titles in plain words, and "Held-out rank corr" reframed as "Model reliability: Moderate" with a tooltip.
  - **Accessibility (WCAG 2.2).** The drop zone is now a real keyboard-operable button (role/tabindex/Enter-Space); added focus-visible rings; the Pareto + predicted-vs-measured charts gained legends, an open-circle marker channel, darker gray, and a labeled y=x line (fixes color-only encoding); the target dropdown clears the 24px target size; clay text darkened to #8F5418 for 4.5:1; charts gained aria-label summaries; the Run button keeps its icon during the running state.
- 13 tests still pass; verified the new payloads on real data.

### Renamed cultivar -> Kalos + ported the best of the lean engine
- Renamed the repo / package / references from `cultivar` to `kalos` (a Pokemon region; deliberately unconnected to the prior Voyager/Bioqore branding).
- Compared against the lean engine (`voyager-brain-rebuild`, also itsbrendandang) with a 4-agent review and ported the high-value pieces it had and Kalos lacked:
  - **Barcode data registry** (`kalos/data/`): every dataset gets a `KAL-DS-*` barcode and every run a stable content-derived `KAL-*` barcode; identity is stripped and grouping keys hashed on ingest (anonymizer ported from the data moat). Queryable (get/filter), exportable (to_dataframe), persistable (one JSON). `examples/organize_data.py` turns a folder of CSV/TSV into one registry — verified on the real media data: 166 runs across 3 datasets, `Sample Name` dropped, `Experiment` hashed, `Medium` kept.
  - **Core gems** (`kalos/core/`): `splits.py` (group-aware CV + a leakage tripwire), `drivers.py` (signed bootstrap-Spearman with CIs), `conformal.py` (distribution-free split-conformal intervals).
  - **Ingestion** (`kalos/ingest/`): the push DataFeed + ProposalSink + decoupled IngestionLoop, anonymized on read, adapted to the BoTorch surrogate.
- +4 tests (13 passing). Next wave (flagged): LLM data-normalization with identity stripping, the multi-task GP, and TuRBO.

### Added — "Run your own data" drop zone in the portal
- New panel + `POST /api/run` endpoint: drag-and-drop (or pick) a CSV / TSV / Excel run sheet and the real engine analyzes it — auto-detects the target (the value to maximize, override via a dropdown), uses process INPUTS as features (other measured outputs are excluded to prevent output-to-output leakage), runs honest grouped cross-validation, ranks signed drivers, and proposes the next batch. Renders a drivers bar (emerald up / clay down), a predicted-vs-measured out-of-fold scatter, KPIs, and a recommendation table, all in the Calm Growth language.
- Leakage guard verified on the Anagram TSV: naive "treat every number as a feature" gave a fake rank corr 0.97 (it was predicting titer from other outputs); excluding outcome columns drops it to the honest ~0.03 with the real drivers (Methanol +0.46, KOH -0.37, Ammonia +0.33). Sparse component columns (blank = absent) are now kept and zero-filled. Added `python-multipart` + `openpyxl` to the portal extra.

### Fixed — design-quality pass (4-lens adversarial critique)
- A parallel design critique (hierarchy, color/a11y, typography, data-viz) scored the reskin 7/6/8/5 and found real issues; fixed them: the demo objective now produces an HONEST positive titer (0-22 mg/L) and purity (0-100%) instead of a negative squared-distance, so the hero no longer reads "-0.000" and "emerald = good" stays coherent. WCAG AA: the emerald hero went to the deeper `#0E7C57` with white label+value, muted text darkened to `#586A5E`, and the soft-fill chip text uses the deep emerald. Header subtitle moved off mono (mono is numbers only), added mg/L / % units, a Pareto legend caption, and uniform card padding. DESIGN.md tokens updated to match. Re-verified: best titer climbs 14.1 -> 21.9 mg/L.

### Added — Kalos design language ("Calm Growth") + portal reskin
- New `DESIGN.md`: a fresh, standalone design system, deliberately unlike the prior dark clinical-console look. Warm paper surfaces, white rounded cards with soft elevation, one emerald accent (#15A375, cultivation = growth) plus a single clay second hue for explore/warning, Sora display + Plus Jakarta Sans body + JetBrains Mono numbers only. The memorable thing: a calm, optimistic tool about growth.
- Reskinned `kalos/portal/index.html` in that language: emerald hero KPI card, soft rounded panels, emerald convergence line with a low-opacity area fill, re-themed Plotly to the palette and mono axis labels. Same live-engine wiring. Verified in-browser at http://127.0.0.1:8050.

### Added — `examples/run_on_media_data.py`: engine on real media data
- Runs the engine on real Anagram runs (media + pH -> lipase titer; data read from `KALOS_MEDIA_DATA`, not vendored): grouped-CV of the surrogate, a proposed next batch, and the multi-objective titer-vs-purity Pareto front. First run (82 defined-media runs, grouped by medium): **grouped-CV Spearman 0.53** (vs ~0.25 from the lean engine on the same data — the BoTorch GP extracts more signal), the promotion gate fails closed on missing feasibility/calibration, the proposed batch pushes Methanol to its observed max (the known top driver), and the Pareto front exposes a real titer-vs-purity tradeoff (max titer 0.022 at only 2% purity vs 0.0167 at 100%).

### Added — web portal (`kalos/portal/`)
- A FastAPI viewer over the LIVE engine (not mock data): it runs the real BoTorch single- and multi-objective optimization on request and charts the convergence curve, the titer-vs-purity Pareto front, and the proposed next batch. Dark, cobalt-accented, Plotly. `pip install -e ".[portal]"` then `python -m kalos.portal` -> http://127.0.0.1:8050. Verified in-browser: both charts render, the single-objective loop converges to [0.696, 0.301, 0.491] (optimum [0.7,0.3,0.5]) and the Pareto front shows 7-9 tradeoff points.

### Added — multi-objective optimization (qLogNEHVI)
- `core/multiobjective.py`: `MultiObjectiveSurrogate` (one GP per objective via `ModelListGP`) + `propose_multiobjective` using q-Log Noisy Expected Hypervolume Improvement. Optimizes two or more objectives at once (e.g. titer AND purity), with a Pareto-front helper and an auto reference point. Verified: the demo grows the Pareto front from 4 to 7 points and traces the real titer-vs-purity tradeoff curve. +1 test (9 passing). `python examples/run_multiobjective.py`.

### Fixed — review pass (ML engineer + scientist agent)
- A rigorous review verified the BoTorch integration is correct (`best_f` is in original target units given the outcome transform; qLogEI and `optimize_acqf` are used properly; the loop converges). Applied the real findings:
  - Input normalization is now tied to the **design bounds** (`Surrogate.fit(X, y, bounds=...)`) instead of the training-data envelope, so the GP normalizes to the fixed search box. The loop converges to the synthetic optimum (gap ~0.03 over repeated runs).
  - `fit()` rejects empty / NaN / mismatched inputs with clear errors, so `best_f` can no longer be silently NaN-poisoned.
  - The `examples/` run standalone via a `sys.path` shim — `python examples/run_bo_loop.py` works without installing.
  - The ESM-2 tokenizer truncates to the model context (1024) so long sequences degrade instead of crashing.
  - +3 tests (bad-input rejection, propose-before-fit, bounds-normalized fit). 8 passing.

### Added — clean platform on BoTorch + ESM-2 (initial scaffold)
- New personal repo: a clean Bayesian-optimization engine on a production BoTorch stack.
- `core/surrogate.py` — BoTorch SingleTaskGP (input-normalized, output-standardized), fit by exact marginal likelihood. Runs on CPU (GPs need float64; MPS is float32-only).
- `core/optimize.py` — qLogEI acquisition + `optimize_acqf` to propose the next batch within bounds. Verified end to end: the demo loop climbs to the synthetic optimum.
- `core/evaluation.py` — grouped cross-validated Spearman of the surrogate (leakage-controlled).
- `core/gates.py` — fail-closed promotion gates carried over from the lean engine (a missing/NaN required metric blocks promotion).
- `features/protein.py` — real ESM-2 embeddings via Hugging Face transformers (runs on Apple MPS, verified: 320-dim vectors distinguish HSA vs Herceptin) plus a dependency-free `KmerEmbedder`. The NVIDIA BioNeMo family behind one interface.
- pyproject packaging, pytest suite (5 passing + ESM-2 test behind a flag), CI, examples.
