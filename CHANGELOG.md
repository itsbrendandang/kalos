# Changelog

Newest first.

## 2026-10-10 (one request can no longer stall the engine)

### Fixed - the demo routes did unbounded work for anyone

`GET /api/single` and `/api/multi` took `rounds` and `q` straight from the URL
with no limit and no token, outside the one-analysis slot - and kalos-web's
proxy forwards them with the query string. `?rounds=100000&q=64` queued hours
of GP fits; `rounds=0` crashed with a 500. Now `rounds` is 1-12 and `q` is 1-4
(422 otherwise, defaults unchanged), and both need the `read` scope like every
other data route (open mode still serves them on loopback).

### Fixed - uploads were read whole before the size check

`/api/run` and `/api/scale/readout` read the entire body into memory, then
rejected anything over the cap (`KALOS_MAX_UPLOAD_MB`, 25 MB by default). A multi-GB upload could
exhaust the single engine process for every tenant. They now read at most one
byte past the cap, which is all the check needs.

## 2026-09-23 (the Scale-Up Readout)

You can now hand a prospect one page that answers "what will this process do at the next scale,
and should I believe the number?" - and that says no when the data cannot support an answer.

### Added - `POST /api/scale/readout`

Upload a multi-scale run sheet plus a target scale and get a self-contained, print-ready HTML
readout (page 1 is the decision; a provenance appendix follows). Kalos first backtests itself on
the sheet's own scale ladder against two naive baselines, then issues a prediction with an
approximate-coverage interval, a prediction with named warnings, or a refusal with every reason.
The license to extrapolate comes only from backtest steps the model actually won. The route is
stateless: nothing from the upload is written to disk or the database. See `docs/SCALE_READOUT.md`.
The gate no longer requires a minimum run count per scale, so a sheet with many bench runs and only
one or two runs at each large scale is read honestly instead of rejected outright. When the evidence
is too thin (too few pooled residuals, or no rung with enough runs to license a reference step
ratio), the readout still returns a 200 page showing everything it could compute, plus a "What it
would take" section stating in plain terms how many more runs, and at what scale, would change that.
`agitation_rpm`/`airflow_L_per_min` are now optional on the sheet and target: when a sheet does not
record one or both, the readout falls back to a scale-only feature set instead of a 422, and the
page names exactly what is not modeled as a result.

The extrapolation license comes only from backtest steps near the scale being extrapolated from (within one decade of the largest trained scale), so a step won at bench scale cannot license a plant-scale step.

### Added - a synthetic multi-scale demo sheet

`examples/synthetic_scaleup/`: 9 scales from 1 to 5000 L, 10 runs each, seeded and pinned by a test
so the committed CSV cannot drift. On it, a 1.5x step (7500 L) issues a prediction whose backtest
MAE is 0.263 against 0.699 and 0.729 for the naive baselines.

### Changed - the scale-model evidence is now reconciled

The scale evaluation's docstring said v0 did not clearly beat the naive baselines on upward
extrapolation; rerunning it on the synthetic mab-scaleup dataset shows it does on MAE (0.533 vs
1.510 and 1.382, n=5), while its ranking of recipes at the larger scale stays unmeasured at that n.
Docstring and `docs/SCALE_EVIDENCE.md` now say the same thing, with the command to reproduce it.

### For contributors

`leave_one_scale_out_report` gains `include_oof` (per-row out-of-fold arrays) and `held_out_scales`
(evaluate only the named folds, skipping the pooled LOGO pass). Both default off, and the default
output is pinned byte-identical by a regression test. `deploy/backups/` is now gitignored.

## 2026-09-10 (the deploy pack meets reality)

The wave-2 deploy pack was written and config-tested but its images were never
built (disk-deferred). Building and actually running them surfaced three real
production defects - each invisible to every unit and CI gate, each now fixed
with a regression test.

### Fixed - set-but-empty env crashed the boot

Compose/k8s templates pass `KALOS_MAX_UPLOAD_MB=${KALOS_MAX_UPLOAD_MB:-}`,
which arrives SET BUT EMPTY - `os.environ.get`'s default never applies and
`float("")` killed the engine at import, on the very first container start.
The sizing knobs (upload MB, fit rows, torch threads) now treat empty as
unset, the same convention kalos/providers/ always followed.

### Fixed - one analysis at a time, honestly (kalos/portal/busy.py)

`_analyze` is CPU-bound for tens of seconds and `run_in_threadpool` work
cannot be cancelled: a client that gives up (a proxy timeout, a closed tab)
orphans a thread that keeps computing. With no admission control, every retry
contended with the ghosts of its predecessors - measured as 598% engine CPU
with ZERO connected clients. A single analysis slot now guards /api/run and
campaign reanalyze: the second request gets HTTP 503 + Retry-After (answered
in 0.1s in the live test), and the slot is released by the WORKER THREAD when
the fit truly finishes, so an orphan keeps holding it and the 503 stays a
true statement about the machine.

### Fixed - the torch thread pin is pathological in the container

The bare-metal default (pin intra-op threads to 4) took the demo analyze from
47s to over 600s on the image's linux-aarch64 torch/OpenBLAS build - a >12x
slowdown from the pin itself, isolated by timing the identical computation
pinned vs unpinned in the same container. `KALOS_TORCH_THREADS=0` now means
"do not pin" and the compose passes it by default; bare metal keeps the tuned
4. (The old code floored 0 to a 1-thread pin - the worst possible reading.)

### Changed - Dockerfile layering, and the verified numbers

Torch (the 700-900 MB layer) now installs before any source COPY, so a
one-line engine change rebuilds in ~40s instead of re-downloading torch -
found the first time the image needed a real fix. Verified footprint: engine
1.82 GB (inside its own honest 1.3-1.8 GB estimate), web 314 MB, first
authed proxied analyze ~19s warm. kalos-web's proxy timeout became
env-tunable (`KALOS_ENGINE_TIMEOUT_MS`, default 120s) after the 30s guess
502'd a cold container fit mid-computation.

## 2026-08-25 (wave 2: acted-on diagnostics, one front door, and a vindication)

Five agents plus orchestrator integration; every number below is from a
reported run or a locked regression test.

### The finding of the wave - ScaleBridge v0 was vindicated, not fixed

Wave 1 reported v0 ranking upward-extrapolated runs backwards (Spearman -0.7)
and framed it as "regressing toward a scale-adjusted mean". The mandatory
rank-crossing check told a different story: the real dataset shows NO
detectable recipe-x-scale interaction (F(3,47)=0.917, p=0.44), the exact
permutation p-value for rho=-0.7 at n=5 is 0.233 (noise), and on a harder
synthetic with a PLANTED rank crossing, all three candidates - including
unmodified v0 - recover it (Spearman ~0.88, and v0 with the best MAE). The
precise conclusion: v0's extrapolation ranking is UNMEASURED at this power,
not demonstrated-bad. No candidate promoted; `physics_mean` (physics-informed
mean function) and `multi_fidelity` (scale as a fidelity dimension,
SingleTaskMultiFidelityGP verified against installed botorch) ship as
documented opt-in alternates on ScaleUpTransferModel, evidence attached.

### Added - the heteroscedasticity diagnosis is finally acted on

`alternative_scale`: when (and only when) the noise report's
`suggests_transform` fires, a second labeled evaluation runs on
log(y + offset) - the exact transform icc_log already uses, NOT literal
log1p, which the report's own history records as wrong at titer scale - and
reports log-scale CV, calibration, and ICC beside the raw numbers.
Proposals and every other number stay on the raw scale; the block is proven
purely additive by a deep-equality test, and it is computed LAST so its GP
fit cannot advance the shared seeded RNG before the proposal path consumes
it (running it earlier would have silently changed proposed batches whenever
the diagnostic fired). Measured cost ~0.12s, only when triggered.

### Fixed - state-path split-brain and the missing health surface

SqliteStore and the runner lock now honor KALOS_STATE_DIR (call-time reads,
matching campaign.py's convention; unset env is byte-identical to before).
/healthz returns exactly {"status": "ok"} unauthenticated - verified the
version string is NOT otherwise public, so it is not leaked here - and
/readyz does a read-only store probe, 503-with-reason on failure. Deploy
healthchecks now hit /healthz; the stale "engine ignores KALOS_STATE_DIR"
comments are gone.

### Changed - one front door

With kalos-web's server-side proxy (kalos-web #26) attaching the bearer
token, the compose no longer publishes the engine's port at all: the browser
calls same-origin /api/engine, the web SERVER forwards over the internal
network. KALOS_ENGINE_URL/KALOS_ENGINE_TOKEN are runtime env on the web
service (never in the client bundle); the runbook's known-gap bullet is
replaced with setup instructions. Direct-browser mode survives as an
explicit dev escape hatch.

### Evidence recorded - MAP-SAAS: a clean negative result

The trial the research memo motivated ran in full
(kalos/bench/saas_experiment.py, 8 (n, d) cells x 5 seeds, both models fed
the same known noise): SAAS never beats the plain SingleTaskGP outside its
own noise floor, costs 1.9-19x the fit time, and proposes worse in 6 of 8
cells - including at kalos's actual low-d operating point. NOT wired;
surrogate.py untouched by construction. Banked for any future revisit:
gp_shape's lengthscale reader would silently read 1 of SAAS's 4 additive
kernels (wrong relevance ranking), and SAAS's constructor draws from the
global RNG outside propose()'s seed fork.

## 2026-08-25 (wave 1: the engine practices what its benchmark proved)

Six parallel agents; every measured claim below is from a locked regression
test or a reported run.

### Added - production proposals are finally feasibility-gated

BENCHMARK.md proved plain BO loses to random search on zero-inflated titers
and that feasibility-gated EI fixes it - and the fix lived only in the bench
harness while production proposed ungated. Now the acquisition itself is
gated: log(EI x p_feasible) = logEI + log p_feasible, with the fitted
StandardScaler+LogisticRegression rebuilt exactly as a differentiable torch
module (gradients verified end to end), q-batch feasibility composed as a
product (conservative under zero-inflation, documented as an approximation).

Gate POLICY, all three conditions reported and none silent: the classifier
actually fit (not its cold-start fallback), n_infeasible >= 5, and CV
feasibility AUC >= the SAME 0.65 floor the promotion gate already uses - one
number, one meaning. Any condition failing reproduces the ungated path
byte-identically (asserted at the propose() call level, not assumed).
Measured: on a feature-dependent zero-inflated sheet, gated batches score
mean p_feasible 0.953 vs 0.389 ungated on the identical classifier; through
the production propose() path on an 85%-infeasible pool, gated best-found
9.084 vs ungated 8.884 over 8 seeds. The response reports `proposal_gating`
and every proposal row carries `p_feasible`.

### Added - the decided constrained behavior exists: maximize titer s.t. a floor

`_analyze(..., constraint={"column": "purity_pct", "floor": 95.0})` fits a
second GP on the constraint outcome and threads it through BoTorch's native
constrained qLogNEI (ModelListGP + objective index + less-than-zero callable;
verified against the installed 0.18.1 source, including that prune_baseline
respects the constraints). Composes with the feasibility gate. Falls back to
unconstrained - reported, never crashing - on a missing/sparse column or fit
failure. The constraint column is stripped from features unconditionally,
including against a declared-roles bypass; both leakage paths are tested.
Measured: on an anti-correlated titer/purity design, constrained batches
predict purity 94.5 vs 89.6 unconstrained.

### Added - ScaleBridge v0 (kalos/scale/): physics-informed scale transfer

Geometry/power/gas-velocity/kLa/hydrostatic proxies with every constant
exposed as a fittable parameter and cited (van't Riet 1979, Rushton 1950,
Doran); transfer as feature engineering over the proven Surrogate;
leave-one-scale-out evaluation with extrapolation direction reported
separately. Measured on the 55-run synthetic scale-up set: pooled MAE 0.66 vs
naive 1.77/1.42, Spearman 0.68 vs ~0. Stated plainly: on upward extrapolation
to 2000 L the model wins on MAE (0.53 vs 1.4-1.5) but has NOT yet
demonstrated correct ranking (Spearman -0.7 on an n=5 coarse grid) - v0
regresses toward a scale-adjusted mean; the v1 path (physics-informed mean
function, more scales) is documented in the module.

### Added - ingestion grows two tiers

- Orientation pre-pass (kalos/normalize/orientation.py): transposed run
  sheets (rows = parameters) are detected and normalized before anything else
  sees them, traced via frame metadata and surfaced as `orientation` in the
  response. Hardened beyond the ported heuristic: a first column that is
  itself numeric can never be read as parameter labels, and shape signals
  alone never suffice - the port's raw heuristic false-positived on kalos's
  own standard tall/narrow fixtures, caught by the existing hardening tests.
- Open-source LLM provider seam (KALOS_LLM_PROVIDER: anthropic | ollama |
  none): the Ollama path speaks the real API (health via /api/tags,
  generate with format json, temperature 0), validates against the identical
  schema the Anthropic path uses, and falls back deterministically on any
  failure - the LLM proposes, deterministic code validates, unchanged.
  Stdlib HTTP only; "none" never touches the network. Units gained rpm, bar,
  g/kg, L/min + agitation/pressure synonyms; bare L/mL stays deliberately
  excluded (vessel-size labels, not measurements - the existing test that
  guards this exclusion caught the naive addition).

### Added - deploy/ - the stack becomes deployable

Multi-stage non-root images (CPU-only torch), compose with health checks and
env-driven secrets satisfying the bind-safety guard by construction, and a
SQLite backup sidecar using the Online Backup API (a cp of a live rollback-
journal database can silently lose the in-flight transaction), retention
windowed, backups outside the compose volumes. RUNBOOK.md carries deploy /
upgrade / rollback / restore-drill / security checklist. Found and
documented: SqliteStore and the runner lock hardcode ~/.kalos and ignore
KALOS_STATE_DIR (compose works around it; engine fix queued), and kalos-web
sends no Authorization header yet (both options documented in the runbook).

### Fixed / smaller

- FeasibilityClassifier gained a public `fitted` accessor and
  `torch_gate_params()` so callers never reach into sklearn internals.
- The research memo (scratchpad, feeding wave 2) verified kalos is clean of
  the upstream-DELETED HeteroskedasticSingleTaskGP and already on the
  recommended train_Yvar pattern.

## 2026-08-23 (the review's remainder, built by a Sonnet fleet)

Five agents implemented the rest of the 2026-08-22 engine review plus portable
findings mined from the owner's earlier engine (voyager-brain-rebuild). Each
change below carries its own tests; the full suite, ruff, and mypy are green on
the combined result.

### Added - the CV confidence interval now covers partition variance

`make_splits` takes the unshuffled GroupKFold branch for a continuous target, so
the group-bootstrap CI on the headline `cv_spearman` was conditional on ONE fixed
partition - and the splits.py docstring admitted the damage (0.44 to 0.71 across
n_splits 3 to 8) while nothing computed the sweep it recommended.

`grouped_cv_report` now supports repeated grouped CV (`n_repeats`): repeat 0 is
the exact historical unshuffled partition, later repeats shuffle the group
assignment deterministically, the bootstrap runs per repeat, and `ci95` pools all
draws. Every point estimate and OOF-derived quantity (`cv_spearman`,
`conformal_q`, calibration, the `oof` scatter) stays anchored to repeat 0, so
nothing moved except the interval - the pinned-number tests pass unmodified. The
portal runs `CV_N_REPEATS = 2` (3 measured over the runtime budget) and reports
`cv_n_repeats` + `cv_spearman_per_repeat` so the spread is visible. On an
engineered partition-sensitive sheet the CI width grows 0.769 -> 0.833; leave-
one-group-out sheets collapse to 1 effective repeat because only one partition
exists.

### Added - the promotion gate finally has real numbers to read

`check_gates` had zero callers on the analysis path and `feasibility_cv_auc`
zero callers outside tests, so the fail-closed gate guarded nothing.
`feasibility_cv_report` now returns AUC, Brier, and the classifier's 10-bin ECE
from the same pooled-OOF CV (`feasibility_cv_auc` is a thin wrapper over it), and
`_analyze` assembles all four gate stats and reports `check_gates` as a new
top-level `promotion` verdict. Reported, never enforced: the verdict cannot
reject an upload, because changing what the API accepts is a product decision.
On an all-producer sheet the feasibility metrics are unmeasurable and the verdict
fails closed with "unmeasured" failures - correct semantics, and the block
carries a `meaning` string so that reads as "not yet shown fit to promote", not
as a rejection.

Named distinction, because two metrics share a name: the gate's `ece` is the
FEASIBILITY CLASSIFIER's calibration (is P(feasible) honest); the
`reliability.calibration` ece is the regression interval calibration (are the
titer error bars honest). Different questions, same scale, both reported.

### Added - three evaluation-hygiene anchors from voyager-brain-rebuild

All three earned their place on the owner's real study data before porting:

- **`cv_logo`** (leave-one-group-out): on the real clone funnel, LOSO Spearman
  was -0.12 where shuffled 5-fold read 0.91 - the gap IS the cross-campaign
  generalization claim. Computed when a declared group column exists and the
  group count is within `LOGO_MAX_GROUPS = 12` (measured cost ~0.5s); skipped
  with a stated reason otherwise, never silently.
- **`cv_topk`**: top-5 overlap between true and predicted ranking on the pooled
  OOF - the direct form of "which N do I advance", which Spearman can obscure on
  zero-inflated targets. Ties resolve by stable sort, documented.
- **`cv_group_mean_baseline`**: predict each row by the mean of the OTHER rows in
  its recipe group and report that Spearman. On the owner's real data a
  campaign-mean-only anchor captured ~0.75 of a model's ~0.86 headline; a model
  that does not clearly beat this floor may be recognizing recipes, not modeling
  the process. The interpretation ships in the payload.

### Added - two validation checks the real data demanded

- **Informative missingness (MNAR).** On the real Cytena funnel, per-row
  missingness vs titer ran Spearman -0.835: "no measurement" was the culling
  decision, not benign absence - and kalos zero-fills blanks before the GP fit.
  `check_informative_missingness` correlates row-level missing fraction (and
  per-sparse-column missingness indicators) with the target and warns when |rho|
  clears 0.35. Warning severity only; strict mode cannot start rejecting sheets
  over it. The engine's own `experiments/missingness_indicator/RESULTS.md` posed
  exactly this question and could not answer it; the check now answers it per
  upload. Running it against the real media DoE still needs a `BIOQORE_DATA`
  checkout, which this machine does not have.
- **Constant-within-group.** 4 of 6 clone features on the same data had exactly
  one unique value within every campaign - group identity in disguise, inflating
  within-population scores to ~0.86 when the campaign-mean anchor alone scored
  ~0.75. `check_constant_within_group` flags features that vary globally but are
  frozen within every group, which the global constant-column check cannot see.

### Added - a non-proprietary benchmark fixture

`examples/synthetic_bioprocess/`: 160 seeded rows, 10 continuous bioprocess
inputs in real units, a known optimum, a feasibility gate zeroing ~44% of runs.
Ported from voyager-brain-rebuild with its generator (verified to reproduce the
CSV byte-for-byte). Fills the gap between `bench/`'s abstract math surfaces and
the uncommittable client data: a domain-realistic regression fixture that is safe
to commit.

### Fixed - four smaller defects

- `propose()` and `propose_multiobjective()` accept an optional `seed`. Applied
  inside `torch.random.fork_rng()`, so a seeded call is reproducible WITHOUT
  clobbering the caller's global RNG state. Caught during testing: qLogNEI with
  `prune_baseline=True` draws posterior samples at construction time, so the
  acquisition must be built inside the fork too - the first attempt leaked.
  Default `None` is byte-identical to the historical caller-seeds-globally
  contract.
- `_sanitize_pending`'s wholesale width-mismatch rejection is logged at warning
  level instead of silent. Without it, a caller bug that disables the in-flight
  guard is indistinguishable from "nothing running" - both read
  `n_pending_considered == 0`.
- `bootstrap_spearman` recorded a degenerate resample draw as rho = 0.0, pulling
  the bootstrap distribution toward zero for exactly the noisiest features. NaN
  draws are now excluded via nan-aware reductions; a fully degenerate feature
  falls back to the old all-zeros output so the JSON response stays finite.
- Each objective in `MultiObjectiveSurrogate` gets its own `Normalize` instance
  instead of sharing one stateful module across the `ModelListGP` - numerically
  identical today, and no longer one `learn_bounds=True` away from cross-coupling
  the objectives.

### Recorded - validated negatives from voyager-brain-rebuild, so they are not re-attempted

- A scalar per-campaign offset multi-task GP is rank-frozen for held-out
  campaigns by construction; a real transfer test needs a coregionalization
  kernel, and probably more than 3 campaigns.
- `averageTotalTiter` is the row-mean of its per-stage components; any
  titer-derived feature "predicts" it mechanically. Check derived targets for
  circularity before trusting a high score.
- Hardcoded scale-up projectors and a DirichletCalibrator on a ~20-row tail were
  both rejected there for inflation reasons that still apply here.

## 2026-08-22 (the unit of analysis is the recipe)

### Fixed - the driver panel counted replicates as independent evidence

Every other statistic in the engine knows that three wells of one recipe are not
three independent runs. The CV groups by recipe, the noise floor is estimated
within recipe, the proposal is fit on the replicate-averaged objective. The
driver panel was the one place that still counted rows.

That is not a cosmetic inconsistency. Both tests behind `significant` - the
bootstrap CI and the Benjamini-Hochberg correction - ask how surprising an
association is given how much independent evidence stands behind it. Testing
rows on a replicated sheet computes every p-value against an `n` the sheet does
not have, so BH stops correcting anything, and BH is what the 2026-08 multiplicity
fix rests on.

Re-running that same simulation (30 pure-noise features, target independent of
all of them, 300 reports) at the replication depth a media DoE actually ships
with:

| sheet | reports containing a "significant" driver |
| --- | --- |
| 60 independent rows | **7.3%** |
| 20 recipes x 3 replicates | **87.0%** |
| 20 recipes x 3 replicates, averaged first | **8.0%** |

A cluster bootstrap alone does not fix it (82.5%): the row-level p-values are
what BH reads, so the aggregation has to happen before the test rather than
around it. The panel now runs on replicate-averaged rows keyed by the same
`recipe_key` the CV grouping and the noise floor already use, and
`driver_selection` states its unit (`unit`, `n_units`, `n_rows`) instead of
leaving a client to infer it from `n_tested`.

End to end, on a 60-row sheet whose target is independent of every column, the
engine reported three significant drivers at p=0.0006, p=0.0034 and p=0.0036. It
now reports none. On a sheet with no replicates every group is a singleton and
the aggregation is an exact no-op, so unreplicated uploads are unchanged.

### Fixed - `feasibility_cv_auc` raised on the data it exists to score

The guard capped `n_splits` by the minority-class count but not by the number of
groups, and those are different numbers: a replicated sheet can hold six
non-producing rows across only four recipes. sklearn refuses to make more folds
than there are groups, so the function raised `ValueError` against a docstring
that promised it never raises.

The value feeds the fail-closed promotion gate, and an exception is not a closed
gate - it is a crash that skips the verdict. Unmeasurable now returns `nan`,
which the gate blocks on.

### Fixed - the shape report was anchored to the luckiest single well

`gp_shape_report` sweeps each feature through the incumbent, the row with the
best measured target, and states that its `X`/`y` are the rows the surrogate was
fit on. On the replicate-aware path they were not: the surrogate is fit on
replicate-averaged rows and the raw sheet was passed, so every sweep was anchored
at `best_single` - a row that GP never saw, selected by taking a max over assay
noise - while the proposals in the same response were scored against
`best_reproducible`. Two incumbents, one response.

Stated plainly, because it was measured rather than assumed: this is a
consistency fix and not an accuracy win. Against a known interior optimum over 40
seeds, mean absolute error in the reported optimum moved 0.177 -> 0.188 at an
assay sd of 1.5 and 0.283 -> 0.270 at 3.5. The case for it is that the report now
describes the model that produced the proposals, and no longer selects its anchor
by maximizing noise.

### Added - calibration is measured instead of disclaimed

A model can rank held-out runs correctly and still quote every interval at half
its true width, and the scientist reading "predicted 4.2 +/- 0.3" is acting on the
0.3. `reliability.unmodeled` listed "calibration (ECE)" for a mechanical reason:
the grouped CV computed a held-out posterior sd on every fold and then discarded
it, keeping only the mean.

It is kept now, and `interval_calibration` scores it: `ece` is the mean gap
between nominal and empirical coverage across four central intervals, on the same
0-to-1 scale `GatesConfig.max_ece` is written against, and `z_std` says which way
a miscalibrated model errs. It is reported as `reliability.calibration` beside the
verdict, never folded into `clears_floor` - tightening the verdict changes which
uploads the API accepts, which is a product decision.

WHICH band is scored turned out to matter more than the metric. `Surrogate.
posterior` returns the LATENT band by default: uncertainty about the response
surface, which is what the acquisition reasons over. A held-out value is a
measurement and carries assay noise on top of that, so scoring observations
against the latent band under-covers by construction. On a replicated sheet with
an assay sd of 1.0 that read as z_std=3.20, ece=0.48 - a catastrophically
overconfident model that was nothing of the kind. Against the predictive band
(`Surrogate.posterior(..., observation_noise=True)`, new keyword, default
unchanged) the same fit measures z_std=1.32, ece=0.09: mildly overconfident,
which is true and useful. Assay noise is zero-mean, so no existing number moved -
`cv_spearman` and `conformal_q` are asserted unchanged.

### Fixed - the deferred-torch design had been quietly defeated

`kalos/portal/analysis.py` documents that `kalos.core.evaluation`, `kalos.core.
optimize` and torch are imported lazily, so the idle `--watch` poller and the
portal do not pay a ~220 MB import for an analysis they may never run. A single
top-level `from kalos.core.evaluation import producer_only_spearman`, added for
one call site inside `_analyze`, had broken it: `evaluation` imports `surrogate`,
so `import kalos.portal.analysis` cost 1.2s and pulled the whole stack. The
intent lived only in a comment; `tests/test_portal.py` now asserts it in a
subprocess.

## 2026-08-19 (in-flight runs are not proposed again)

### Fixed - the acquisition now knows what is already running

A campaign round is not atomic. Five recipes are proposed, the scientist starts them, some assays come back before the others, and the loop is re-analyzed on what has landed so far. The runs still incubating have no outcome, so they cannot join the fit - and the acquisition was never told about them separately, so it treated their region of the design space as unexplored and proposed them again.

That is budget spent twice for one point of information, and it is invisible in a regret curve, because regret only counts what was measured.

`kalos/bench/pending.py` measures it (`python -m kalos.bench --pending`). On a 4-factor design at `q=5` over 10 seeds:

| re-proposal | duplicated recipes per round (of 5) | worst seed |
| --- | --- | --- |
| blind | **2.10** | 4 of 5 |
| pending-aware | **0.00** | 0 of 5 |

Roughly 40% of a mid-round batch was repeat work. The surface is a smooth single optimum on purpose: that is the case where a blind re-proposal is *least* likely to collide, because the acquisition's own q-batch diversity already spreads one batch out.

**The plumbing.**

- `kalos/core/optimize.py`: `propose(..., pending=)` feeds BoTorch's `X_pending`, which integrates over the unknown outcomes of started runs. Pending points get the same discipline the returned proposals already get - clamped into the design box, categorical coordinates snapped to integer level codes - so an in-flight recipe recorded slightly outside the box still marks the right neighborhood as taken. A malformed row is dropped individually rather than raising: failing to propose anything at all is a worse outcome than proposing without the pending penalty.
- `kalos/portal/analysis.py`: `_analyze(..., pending=)` encodes in-flight recipes into the same design as the fitted rows (continuous components zero-filled like `Xc_zf`, categoricals through the fitted level codes). A recipe naming a level that does not exist in this design is dropped rather than placed at a guessed coordinate, and `n_pending_considered` reports how many actually reached the optimizer - what was used, not what was offered.
- `kalos/portal/campaign.py`: `awaiting_recipes(tenant)` returns the started-but-unmeasured runs. Read outside the plan/commit transaction on purpose: a run started in that gap is simply absent from the list, which costs the acquisition one in-flight point of information and nothing else.
- `kalos/portal/campaign_routes.py`: `reanalyze` passes them to `_analyze`.

**The honesty constraint holds.** An awaiting run informs the *acquisition*, as a point already taken. It never informs the *fit* - it has no measured outcome, and a run without an outcome must never become a data point.

**Not to be confused with the feasibility result.** `BENCHMARK.md` already tested gating acquisition by a feasibility classifier on the real media DoE and found it changed nothing (`bo_feas` was identical to `bo`). This is not a claim about finding better recipes. It is a claim about not paying twice for the same experiment, and it is the kind of waste that only appears once the loop is run in rounds rather than benchmarked in one shot.

## 2026-08-09 (GP-native response shapes)

### Added - `kalos/core/gp_shape.py`: interior optima and feature relevance from the GP that proposes

`kalos.core.drivers` reports a signed Spearman rho, which is univariate and monotonic, so it is structurally blind to the shape bioprocess responses usually have. A simulated titer peaking at pH 7.0 gives rho = +0.164 - below the trust floor - so the driver panel reports `significant: false` for the single most important variable on the sheet, and the sign of rho points the process the wrong way.

The same sheet now also returns:

```
GP SHAPES  swept_at=incumbent
  pH        rel=0.663 rho=+0.164 interior_optimum  opt=7.000  missed_by_spearman=True
  Methanol  rel=0.299 rho=+0.768 monotonic_up      opt=-      missed_by_spearman=False
  noise     rel=0.038 rho=+0.002 flat              opt=-      missed_by_spearman=False
```

The optimum is recovered at exactly 7.000, and both views ship side by side so a scientist sees the rank correlation and the shape together rather than having to trust one.

**Read off the surrogate the engine already fits, not a second model.** Three things follow, and each was a problem with the gradient-boosted-tree prototype this replaces (parked on `feat/xgboost-drivers`, which segfaults: xgboost and torch each carry an OpenMP runtime and the second into a parallel region crashes the process, uncatchably):

- **One authority.** The shapes describe the same posterior that produced `proposals`, so the report cannot rank features differently from the model being optimized.
- **Uncertainty is free.** `Surrogate.posterior` returns a mean AND a standard deviation, so an interior optimum is claimed only when the peak clears the better endpoint by at least `PEAK_SD_MULTIPLE` combined posterior sds. `peak_gain` and `peak_separation_sd` are both reported so a stricter bar can be applied without re-running. A point-predicting tree cannot make this check at all.
- **Relevance is free.** The Matern kernel is fitted with ARD, one lengthscale per dimension. A short lengthscale means the response moves fast along that axis, which is per-feature relevance already paid for during the fit.

**Swept at the incumbent, not the medians.** Sweeping one feature means fixing the rest, and the usual choice - column medians - is a recipe that may never have been run, putting the whole profile where the GP has no data and reporting the shape of its prior. The sweep is taken around the best observed row instead: a real recipe, near data, and the question a scientist is actually asking. `swept_at` names this in the response rather than leaving it implied.

**Gated on out-of-fold skill, which the negative control forced.** Posterior separation alone is NOT sufficient: on a pure-noise target the GP fits small wiggles, and near the training data its posterior sd is tiny, so a meaningless bend scores a large separation ratio. The first version confidently claimed interior optima on noise. The shapes are now conditioned on `cv_spearman`, the same out-of-fold measurement and `RELIABILITY_SPEARMAN_FLOOR` the reliability verdict already uses, so a model that has not shown it can predict held-out runs reports no shapes at all. When no skill measure is supplied the report says so in `unmodeled` rather than implying the shapes were validated.

Also: the 0.20 floor was a bare literal in two places in `analysis.py` and is now the single `RELIABILITY_SPEARMAN_FLOOR`, so the verdict and the shapes can never be held to different bars.

`tests/test_gp_shape.py` (+15): the headline interior optimum where Spearman is blind, ARD ranking the real driver above noise, monotonic features not flagged, an ignored feature reported `flat` rather than as a trend, no interior claim on a monotonic response, the noise-target refusal, refusing an unfitted surrogate (it must describe the deployed model, never fit its own), mismatched names, a constant column, determinism, and strict-JSON safety.

Suite: 402 passed, 1 skipped. ruff and mypy clean.
## 2026-08-09 (driver multiplicity + stated coverage)

### Fixed - the driver panel manufactured false process insights

The panel tests every continuous feature against the target, then ships the strongest by `|rho|`, and until now applied no multiplicity correction at all. Measured over 400 simulated reports on 30 pure-noise features against an independent target:

```
reports containing >=1 false "significant" driver
  uncorrected per-feature 95% CI:   78.8%
  Benjamini-Hochberg at q = 0.05:    3.5%
```

Nearly four in five reports would show a fabricated driver. Selection makes it worse than the raw rate implies, because ranking by `|rho|` preferentially surfaces exactly the flukes, and the frontend then labelled them "confirmed driver" with causal verbs ("raises", "lowers"). This is the same failure class as the `run_number` rho = 1.0 incident: a statistic that is technically computed correctly and scientifically meaningless.

Demonstrated on one realistic sheet (1 real driver, 14 noise columns, n=60): `noise_1` (p=0.032) and `noise_10` (p=0.047) both had bootstrap CIs excluding zero and would both have shipped as significant. Both are now correctly rejected, leaving only the real driver.

- `kalos/core/drivers.py`: `benjamini_hochberg(pvals, q)`, a step-up FDR procedure. BH rather than Bonferroni because it controls the expected PROPORTION of false findings, which is the right error rate when a scientist acts on several drivers; Bonferroni at 30 collinear media-DoE features would suppress the real drivers too. The docstring states the positive-dependence assumption and that surviving does not make a driver causal.
- `kalos/portal/analysis.py`: `significant` now requires BOTH the bootstrap CI excluding zero AND surviving BH. Applied over ALL tested features before the top-k cut, never after - correcting for the 8 shipped when 30 were tested would understate the multiplicity it exists to control. Each driver also carries `p`, `ci_excludes_zero` and `survives_fdr` so a reviewer can see which test a borderline feature failed.
- New `driver_selection` block reports `n_tested`, `n_reported`, `top_k`, `fdr_method`, `fdr_q`, `n_bootstrap` and `ranked_by`. Selecting the strongest of many tested features is itself a statistical act, and the client needs to see that it happened rather than being shown eight drivers as though eight were tested.

Known limitation, stated rather than hidden: `n_bootstrap` stays at 200, so the 2.5% CI percentile rests on roughly the 5th order statistic and is coarse. Raising it to 2000 was measured at 5.1s (n=55) and 15.8s (n=2000), which would triple analyze latency, and BH on exact p-values is now the primary gate with the CI as a secondary check. The count is reported so the coarseness is visible.

### Fixed - the response now states the coverage it actually computed

The band is computed at `alpha=0.1`, a 90% band. The frontend had drifted to labelling that same band "95%" on `/decide` and "90%" on the Voyager surface. A stated coverage number that is wrong on screen is an overclaim, not a hedge.

- `kalos/portal/analysis.py`: `CONFORMAL_ALPHA` is now a named constant and the response carries `conformal_coverage` (0.9), so no consumer has to hardcode a percentage. Coverage is a property of the method, not the data, so it is reported even when the band itself is `None` for want of out-of-fold residuals.

`tests/test_driver_multiplicity.py` (+15): the BH primitive (step-up behaviour, carry-along, order alignment, empty/NaN), only-the-real-driver-is-significant, the specific rescue of a noise feature whose CI excludes zero by chance, `significant` as a strict conjunction, selection disclosure, FDR applied over all tested features, and coverage reported with and without a band.

Suite: 400 passed, 1 skipped. ruff and mypy clean.

## 2026-08-08 (identifier columns: privacy + spurious drivers)

### Fixed - a numeric identifier column was being modeled as a process input

Given a sheet carrying `run_number` and `batch_id`, the engine fit on both, reported both as significant drivers at rho = 1.0, and proposed a recipe instructing the scientist to **"set batch_id = 100.037"**. A run index rises monotonically with time, so it correlates with any drift or learning trend across a campaign and will almost always rank as a top driver. That is a spurious correlation presented as process insight, which is the failure mode this codebase works hardest everywhere else to prevent.

### Fixed - identifier column names survived `anonymize=True`

The same root cause: `DomainProfile.id_hint` is anchored to whole tokens (`^(id|name|run|batch|campaign|lot|...)$`), which matches a column called exactly `run` but not `run_number`, `batch_id`, `campaign_id` or `lot_number` - and compound names are what real run sheets use. So `anonymize=True` published those names verbatim while claiming to pseudonymize identifier columns. The codebase already contradicted itself here: the metadata scrubber's `HASH_EXACT` does list `campaign_id`, so one id was hashed as metadata and published as a column name in the same response.

- `kalos/portal/analysis.py`: `identifier_pattern` unions a profile's own `id_hint` with the compound-identifier pattern, and feature selection, provenance, and anonymization all now resolve through it, so they cannot disagree about what an identifier is. Union, never intersection - it can only classify more names as identifiers, never fewer. `id_hint` itself is untouched, because it also drives role classification and narrowing it would discard real measurements like `batch_titer`.
- `_anonymize_result` now covers the `validation` block and `design_box_exclusions`, not just `group_col` and `provenance`. A finding names its column twice - a `column` field and the sentence built around it - so both are rewritten together. Aliasing the structured field while leaving the name in the prose beside it would have been anonymization worth nothing.
- Precision holds in the other direction: `Methanol`, `pH`, `scale_L`, `lipase_titer`, `batch_titer` and `run_duration_days` are all still treated as real columns.

`tests/test_identifier_privacy.py` (+29): the pattern in both directions, identifiers never becoming features / drivers / recipe entries, provenance agreeing, no identifier surviving `anonymize=True` (including inside message prose), real feature names preserved, pseudonyms stable, and opt-in behavior unchanged when `anonymize=False`.

Also recorded as a test rather than silently changed: `id_hint` carries a `sample.*` glob, so a genuine measurement named `sample_volume_L` is classified as an identifier and dropped. That is pre-existing behavior and narrowing it is a scientific decision, not a drive-by edit inside a privacy fix.

Suite: 385 passed, 1 skipped. ruff and mypy clean.

## 2026-08-08 (cleanup: remove the barcode registry)

### Removed - `kalos.data.barcode_registry` and its demo

Barcoding is explicitly off the roadmap (2026-07-14: keep the data model simple), and the registry had no caller in the product path - only a re-export, one test, and a demo script entirely about it. Removed rather than left to rot:

- `kalos/data/barcode_registry.py` (145 lines) and `examples/organize_data.py`.
- `kalos/data/__init__.py`: dropped the `BarcodeRegistry` / `RunRecord` re-exports.
- `kalos/data/anonymizer.py`: `run_barcode`, `dataset_barcode`, and the `_stable_payload` helper went with it - the registry was their only caller, so keeping them would have left dead code behind the thing that was removed to avoid dead code. The `Anonymizer` class itself stays; it is used across `kalos.normalize`, the `kit` facade, and the analyze path.

**Coverage was preserved, not dropped.** `anonymize_meta` had no test of its own - its only exercise rode along inside the barcode-registry test. Since it is the function standing between a client's identity and anything the engine persists, it now has a direct test asserting the full contract: identity keys (client/strain/operator) dropped entirely, grouping ids (campaign/lot) kept but irreversibly hashed so CV can still group by them, ordinary metadata passed through, hashing deterministic under a fixed salt and salt-dependent across tenants.

Audited the other dead-code candidates before touching anything and kept them all: `core/gates.py` (public API with tests, though still not wired into the portal - a real follow-up), `core/feasibility.py` (used by the benchmark sweep), `features/protein.py` (ESM-2, on hold but planned), `core/multiobjective.py` (used by the portal).

Suite: 356 passed, 1 skipped. ruff and mypy clean (65 source files, down from 66).

## 2026-08-08 (campaign store schema repair)

### Fixed - an incompatible `campaigns` table silently discarded every campaign write

`CREATE TABLE IF NOT EXISTS` is not a migration. It matches on table NAME and ignores shape, so a `campaigns` table with the wrong columns or key was accepted at startup, and the failure only surfaced per-write, as a `sqlite3.OperationalError` ("ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE constraint") deep inside `_write_locked`. `/api/run` catches and logs that while continuing, so the portal looked healthy while persisting nothing - indefinitely, because nothing ever repaired the table.

Observed, not hypothetical: a development database carried `PRIMARY KEY (tenant, campaign_id)` from an unmerged multi-campaign branch. `tenant` is *in* that key but is not the sole key, and SQLite's `ON CONFLICT(tenant)` requires exactly the latter. Every campaign write on `main` had been failing. The table was found empty, which is the proof - nothing had ever been persisted.

- `kalos/portal/campaign.py`: `_ensure_campaigns_schema` checks the real shape via `PRAGMA table_info` (every required column present, and `tenant` the sole primary key) and repairs a mismatch. Repair is **non-destructive**: the old table is renamed to `campaigns_backup_N` and left in place, a correct table is created, and salvageable state is copied across. Where the old shape held several campaigns per tenant, the most recently updated one wins, since that is the campaign the single-campaign API would have been serving. Logged at WARNING, because an operator needs to know a table was renamed under them and where the rows went.
- The backup name is the first unused `campaigns_backup_N`, so an upgrade/downgrade ping-pong never clobbers an earlier backup.
- `tests/test_campaign_schema_migration.py` (+12): the exact composite-key shape found in the wild, newest-per-tenant selection, non-destructiveness, repeated migration, an unsalvageable same-name table, and both happy paths (a fresh database and reopening a correct one must not migrate or churn backups).

Audited the sibling `kalos/store/sqlite_store.py` for the same hazard: it is sound, its `ALTER TABLE` tenant backfill having applied correctly.

## 2026-08-08 (bioprocess data validation gate)

### Fixed - the optimizer could propose physically impossible recipes

Reproduced end-to-end through the real `_analyze` path, not hypothesized.
A 12-row sheet with one `-999` "sensor offline" sentinel in its temperature column returned these recommendations:

```
recipe: {'feed_mL_h': -0.6,   'temp_C': -422.948, 'pH': 40.0}
recipe: {'feed_mL_h': 0.75,   'temp_C': -350.659, 'pH': 36.759}
```

Bioreactor temperatures below absolute zero, negative pump rates, and an impossible pH, each with a confidence interval attached.
The cause is short: the design box handed to the acquisition optimizer was the raw observed `[min, max]` of every feature, so a single sentinel widened the search space to `[-999, 37]`.
Per-column provenance made it worse by reporting `coerced_cells=0` on all four columns - an active clean bill of health.

- `kalos/portal/analysis.py`: `_physical_range` builds each continuous dimension's bounds from physically valid observations only, using the hard bounds in `kalos/validation/bounds.py`. The same sheet now proposes `temp_C` 37.3, `pH` 6.9, `feed_mL_h` 0.636. Narrowing is never silent: `design_box_exclusions` reports every affected column and how many cells were excluded.
- This safety property holds in every validation mode. It does not depend on a client reading the report.

### Added - `kalos/validation`: a nine-check ingest gate wired into `/api/run`

`kalos/normalize/` (a complete, tested unit registry) had **zero imports from the product path** - so a titer column mixing g/L and mg/mL across rows passed ingest silently, as did a pH of 40. The registry is now live.

- `kalos/validation/{report,bounds,checks,runner}.py` (new): units consistency, physical bounds, duplicates vs replicates, missingness, MAD-based outliers, provenance metadata, replicate adequacy, controls presence, constant columns. Errors are physics violations; warnings are operationally suspect but possible.
- `kalos/portal/analysis.py`: the gate runs on the sheet as uploaded, before any column is typed or dropped, so it can still see `"34.6 C"` and a mixed-unit column. The report is served as `validation` in the `/api/run` response.
- `KALOS_VALIDATION_MODE` = `warn` (default) or `strict`. Default is `warn` deliberately: adding a gate must not start rejecting data clients push through today. `strict` refuses an upload whose status is `fail`.
- Unit-tagged columns are now **converted rather than discarded**. `"34.6 C"` parses at 0% as a bare number, so a real temperature input used to fail the >=80% numeric test and come back `dropped_sparse`. It is now a usable feature in Celsius, with the conversion reported explicitly.
- Only units the registry can actually convert are rewritten. A vessel column of `"5L"`/`"500L"` uses an unrecognized token, and converting it would turn an identifier into a measurement; it is left untouched and the client is told. Backed by `kalos/normalize/units.py::is_known_unit`, which exists because `canonical_suffix` returns `""` for both an unknown unit and a genuinely dimensionless one (pH, OD).

### Fixed - three defects found by running the gate against a real 55-batch mAb scale-up dataset

- **Outliers on a designed scale ladder.** MAD z-scores assume a unimodal linear distribution. On a bioreactor ladder (0.01 to 2000 L) the median is 10 and the MAD about 10, so every run at 200 L or above scored `|z| > 5`: 20 of 55 rows flagged, plus every scale-proportional column. A strictly-positive column spanning >= 2 orders of magnitude is now assessed on log10, which is how a multiplicative quantity is actually distributed. False positives went from 11 warnings to 6, and all 6 survivors are correct - the 4 remaining flagged batches are the intentionally degraded runs, coherent across viability, monomer, aggregate and endotoxin at once.
- **Provenance false alarm.** The check reused `DomainProfile.id_hint`, which is anchored (`^(...|batch|lot)$`) because `_analyze` uses it to decide whether a column is *entirely* an identifier. Asking "does an identifier exist" needs to match compound headers, so a sheet whose first two columns were `batch_id` and `run_number` was told it had no traceability. `_RUN_ID_RE` now handles compound names while still rejecting measurements like `batch_titer`.
- **`inf` broke the response.** Unbounded dimensions carry `hard_hi = inf`, which reached `detail["hard_range"]`; Starlette's `JSONResponse` correctly refuses non-finite floats, so the gate's own finding turned a good analysis into a 400. `report_dict` sanitizes at the serialization boundary, mapping non-finite floats to `null` ("no bound").
- `infer_dimension` coverage went from 7/25 to 22/25 against the real column vocabulary (glucose, lactate, glutamine, ammonium, sodium, osmolality, pCO2, VCD/TCD), and the pH hint no longer matches `phosphate`, `morphology`, `sulphate` or `phase` - a `phosphate_g_L` feed column was being given pH's 0-14 range and reported as impossible.

### Added - `kalos/providers`: credential slots with keyless fallbacks

Every provider is unavailable and every feature works without a key. No speculative client code for unwired services.

- Anthropic (LLM column/unit normalization; falls back to the deterministic offline mapper), NVIDIA BioNeMo (a documented SLOT - there is no BioNeMo client in this codebase; the public non-gated ESM-2 checkpoint needs no token), Benchling (a SLOT; the honest source of the run-id/operator/date provenance the gate can currently only warn about).
- `GET /api/providers` reports status. `ProviderStatus` carries env var NAMES only, never values.
- `.env.example` documents every environment variable the codebase reads.

### Fixed - reproducibility and CI

- `kalos/features/protein.py`: `ESM2Embedder` pinned to an immutable commit (`DEFAULT_ESM2_REVISION`). `from_pretrained` with no `revision` resolves to a mutable branch, so upstream could change every embedding with no diff in our code - breaking reproducibility in the worst way, silently, under a fixed seed.
- `pyproject.toml`: `anthropic` added to the `dev` extra. CI installs `.[ml,portal,dev]`, so the two mocked-live tests in `tests/test_normalize_llm.py` failed on a fresh checkout. Chosen over a skip guard so CI keeps genuinely covering that path.

### Tests

`tests/test_validation.py` (+25, checks in isolation) and `tests/test_validation_gate.py` (+14, the live path, led by the absolute-zero regression). Suite: 344 passed, 1 skipped. ruff and mypy clean.

## 2026-07-22 (P1: experiments tenancy)

### Fixed - production hardening P1: the experiments store is now tenant-scoped and auth-gated

A progress review found that campaign + latest were tenant-scoped (Phase 1b) but the **experiments store was still global and its endpoints entirely ungated** - a live cross-tenant leak once multiple tenants exist. Closed:

- `kalos/store/sqlite_store.py`: `experiments` gains a `tenant` column (with an index and a one-time `ALTER TABLE ... DEFAULT 'default'` backfill migration for pre-existing databases); every `get`/`list`/`set_status`/`save_result` filters by tenant and `create` records it. A cross-tenant id reads as not-found, so tenants can't probe each other's ids.
- `kalos/portal/experiments.py`: every route now requires the `read` or `write` scope and passes `Principal.tenant` to the store.
- `kalos/runner/adapter.py`: `LocalStoreAdapter` is bound to a tenant, so the Singleton runner only ever sees and mutates the calling tenant's experiments. (The remote-runner `/api/experiments/{id}/result` channel still operates on the `default` tenant - a documented follow-up.)
- `tests/test_tenant_isolation.py`: +3 tests - experiments isolated per tenant (list/get/mutate), the legacy backfill to `default`, and HTTP proof that the endpoints require auth and one tenant never sees another's experiments. Existing M2/experiments suites green (default tenant unchanged).

## 2026-07-22 (even later still)

### Added - production hardening Phase 1c: CORS allowlist + startup security posture (`kalos/portal/config.py`)

- `kalos/portal/config.py` (new): centralizes the portal's deployment security config. CORS is an explicit allowlist when `KALOS_CORS_ORIGINS` is set (the production posture) and the permissive localhost regex otherwise (dev/pilot, unchanged). `log_security_posture` logs auth+CORS state once at startup and warns when the portal is not locked down.
- `kalos/portal/app.py`: the CORS middleware now uses `cors_config()`; the posture is logged at startup.
- `docs/HARDENING.md`: `KALOS_CORS_ORIGINS` config reference and a TLS-via-reverse-proxy deployment note.
- `tests/test_config.py` (new): 9 tests - allowlist parsing, dev-default regex vs configured allowlist, and the posture warning.

## 2026-07-22 (later)

### Changed - production hardening Phase 1b: per-tenant persistence (`docs/HARDENING.md`)

Every portal read/write is now keyed by the caller's tenant (`Principal.tenant` from the auth layer), so two tenants can never see or overwrite each other's campaign or analysis. Backward compatible: in open mode everything maps to the `default` tenant, so the dev/pilot loop is unchanged.

- `kalos/portal/campaign.py`: the `CampaignStore` is now backed by **SQLite** - one `campaigns(tenant, state, updated_at)` row per tenant in `<state_dir>/portal.db`, replacing the single `~/.kalos/campaign.json`. Each method takes a `tenant` (default `"default"`); each write is one transaction. The transactional generation-token logic is unchanged (it lives inside the per-tenant `state` blob).
- `kalos/portal/app.py`: `_LATEST` is now a per-tenant map plus a per-tenant best-effort cache file under `<state_dir>/latest/<tenant>.json` (tenant sanitized for the filename), replacing the single global. `/api/latest` requires the `read` scope and returns the caller's tenant's analysis; `/api/run` seeds the campaign and saves latest under `Principal.tenant`.
- `kalos/portal/campaign_routes.py`: `GET /api/campaign` requires `read`; every store call (`summary`/`start`/`set_result`/`plan_fold`/`commit_fold`/`get`) and the reanalyze `_save_latest`/`_load_latest` pass `principal.tenant`.
- `tests/test_tenant_isolation.py` (new): 4 tests - campaigns and `/api/latest` isolated per tenant at the store level, reseeding one tenant never touches another, and an HTTP end-to-end proof that one tenant's campaign is invisible and untouchable by another through the authenticated API. Existing portal/campaign fixtures updated for the SQLite + per-tenant-latest shape.

## 2026-07-22

### Added - production hardening Phase 1a: in-house API authentication (`kalos/portal/auth.py`)

First step of the behind-the-scenes hardening track (`docs/HARDENING.md`) toward a multi-tenant, operable service. Compliance/certification work is handled offline and is out of scope here.

- `kalos/portal/auth.py` (new): a self-hosted token layer. A bearer token resolves to a `Principal` (subject, tenant, scopes); tokens are provisioned as SHA-256 hashes (never plaintext, never logged) and constant-time compared. Config is read at request time from `KALOS_AUTH_TOKENS_FILE` (preferred) or `KALOS_AUTH_TOKENS`, so rotation needs no restart.
  FastAPI dependencies `require_principal` / `require_scope(scope)` gate endpoints.
- **Backward compatible.** With no tokens configured the API runs in *open mode* - every request gets an anonymous `default`-tenant principal with read+write (today's behavior) and a startup warning is logged; `admin` is never granted without a real token. The moment tokens are provisioned, enforcement turns on.
- Gated the mutating endpoints on the `write` scope: `POST /api/run` (`kalos/portal/app.py`) and `POST /api/campaign/{start,result,reanalyze}` (`kalos/portal/campaign_routes.py`). `GET` reads stay open for now; per-tenant data isolation is the next slice.
- `docs/HARDENING.md` (new): the in-house hardening plan and configuration reference (auth -> tenant-scoped persistence -> reliability -> operability).
- `tests/test_auth.py` (new): 27 tests - open mode, valid/invalid/missing/malformed tokens, tenant isolation, file-over-env precedence, scope gating, and HTTP end-to-end that `/api/campaign/start` is 401 unauthenticated, 403 for a read-only token, and clears the gate with a valid write token. Full portal suite green (open mode unchanged).

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
