# Roadmap — what would benefit kalos next

Compiled 2026-10-05 from a read-through of the engine, portal, deploy pack, CI,
and docs. Every item names where it lives. Ordered by value for effort; the
"Next three PRs" at the bottom is the recommended starting point. When an item
ships, strike it here and record it in [CHANGELOG.md](../CHANGELOG.md).

## P0 — correctness bugs (small, do first)

1. **`offline_plan` crashes on two outcome columns.** `guess_role` marks e.g.
   titer *and* purity as `target`, and `NormalizationPlan.validate` rejects the
   plan (`src/kalos/normalize/llm.py`, `offline_plan`). The TypeSafe tier and
   `roles=auto` work around it; direct `offline_plan`/`propose_plan` calls
   without a live tier still raise. Fix: keep one target deterministically
   (mirror the profile's `target_pref`), demote the rest to `metadata`.
2. **The HTTP runner cannot authenticate its reads.** `HttpBackendAdapter`
   (`src/kalos/runner/adapter.py`) sends no `Authorization` on `list_ready`,
   `fetch`, `set_status`; since the P1 tenancy work those routes require
   `read`/`write` scope (`src/kalos/portal/experiments.py`), so a remote runner
   against a hardened engine gets 401. Its docstring still says they are not
   token-gated. Fix: a provisioned API token for the runner, a request timeout,
   and a retry.
3. **Column-role patterns have drifted.** `src/kalos/normalize/synonyms.py` copies
   the outcome/group regexes "byte-identical" from the bioprocess profile, but
   they differ (a trailing `|titer`), and `src/kalos/bench/pool.py` keeps its own
   different id/output hints. `src/kalos/domains` is torch-free now, so import
   `BIOPROCESS_PROFILE` instead of copying.

## P1 — make the pipeline a real module

4. **Split `_analyze`.** It is ~1,000 lines (`src/kalos/portal/analysis.py`,
   `_analyze`) and almost none of it is portal-specific; `src/kalos/runner/singleton.py`
   reaches into this private function. Extract stages — *design* (unit
   conversion, design box, design space), *evaluate* (grouped CV, conformal,
   calibration, baselines, drivers, noise), *propose* (acquisition,
   feasibility, constraint) — into `src/kalos/pipeline/` with one public entry
   point, leaving the portal a thin serializer. Guard with a golden test that
   the result dict is byte-identical before/after.
5. **One source per threshold.** `RELIABILITY_SPEARMAN_FLOOR` restates
   `GatesConfig.min_spearman`; `CONSTRAINT_MIN_ROWS` repeats a bare `6`;
   `src/kalos/portal/validate.py` re-describes `_analyze`'s column rules.

## P2 — product gaps behind "done" roadmap lines

6. **Multi-objective on uploads.** qLogNEHVI exists (`src/kalos/core/multiobjective.py`)
   but only the synthetic `/api/multi` demo uses it; an uploaded titer+purity
   sheet is optimized single-objective.
7. **Outcome constraints over HTTP.** `_analyze(constraint=...)` works as a
   library call only; neither `/api/run` nor the campaign routes accept it.
8. **ScaleBridge is unwired.** `src/kalos/scale/` is imported by nothing outside
   itself; its extrapolation ranking is unmeasured at current power.
9. **Calibrate the TypeSafe tier.** Build a labeled set of real headers ->
   roles, measure TypeSafe vs offline accuracy and the role Choice's
   calibration, then set `KALOS_TYPESAFE_MIN_CONFIDENCE` from data. Natural
   next TypeSafe question: continuous vs categorical per feature, to fill
   `ColumnRoles.categoricals` (today `roles=auto` leaves it empty).
10. **Sparse categorical levels.** No warning when a categorical level has too
    few replicates to model (CHANGELOG follow-up).

Also open, from the original README roadmap:

- **Push-feed ingestion loop:** the platform streams runs in, the engine
  proposes the next batch, anonymized on read. No code yet.
- **Larger ESM-2 / full NVIDIA BioNeMo backend** behind the same embedder
  interface (`src/kalos/features/protein.py`).
- **Act on the XGBoost verdict:** if `cv_xgboost_baseline` reports
  `xgboost_better` on real sheets, evaluate a tree-ensemble surrogate that
  carries uncertainty (e.g. quantile or bootstrap ensembles) behind the same
  `Surrogate` interface - plain XGBoost cannot drive acquisition.

## P3 — operations ([HARDENING.md](HARDENING.md) backbone)

11. **P4 latest -> SQLite, then P3 job queue.** Closes the documented
    latest-write race and takes the GP fit off the request path.
12. **P2 observability, the missing half:** request-id + JSON logging (no
    bodies or tokens), `/metrics`.
13. **Compose hardening:** `cap_drop: [ALL]`, `security_opt:
    [no-new-privileges:true]`, `read_only` where possible, CPU/memory limits.
14. **Salt placeholder check:** `salt_configured()` accepts any non-empty
    value, including `REPLACE_ME_WITH_A_RANDOM_SECRET`.
15. **Run the backup/restore drill** as written and `shellcheck backup.sh`
    (both open in [RUNBOOK](../deploy/RUNBOOK.md), "Still NOT verified").

## P4 — CI, tests, dependencies

16. **Faster CI.** No pip or Docker layer cache (every run re-downloads
    torch); add `concurrency`, `timeout-minutes`, `permissions`.
17. **A fast lane.** The full suite takes tens of minutes on 4 cores, dominated
    by benchmark sweeps (`tests/test_feasibility.py`, `test_pool.py`,
    `test_bench.py`, `test_saas_surrogate.py`). Mark them `slow`, run them
    nightly or on main, and keep a sub-few-minute PR gate.
18. **Python versions.** `requires-python >= 3.10` but CI tests 3.12 only: add
    3.10/3.11 to a matrix or raise the floor.
19. **Reproducible builds — already biting.** Nothing consumes `uv.lock` (CI
    and the engine image install from `pyproject.toml`'s lower bounds). As of
    2026-10-05, `tests/test_feasibility_gated_proposals.py::test_gated_vs_ungated_best_found_so_far_not_worse_on_zero_inflated_pool`
    fails on **unmodified main** with today's resolution (torch 2.14.1, numpy
    2.5.3, scipy 1.18.1; gated mean 8.81 vs ungated 8.88) though main's CI was
    green on 2026-09-16. Not botorch (fails on the locked 0.16.1 too) and not
    scikit-learn (fails on the locked 1.7.2); torch, numpy, or scipy remain.
    Pin via a constraints file exported from the lock for CI +
    `deploy/engine/Containerfile`, then decide whether the gate's "never worse" claim
    needs a tolerance or a production fix. Let Dependabot watch the lock and the
    `python:3.12-slim` base.
20. **Security gates:** `pip-audit`, `shellcheck`, an image scan.
21. **Direct unit tests** for `src/kalos/portal/validate.py::column_provenance` and
    `GENERIC_PROFILE` (today exercised only through `_analyze`), and for
    `src/kalos/core/conformal.py::conformal_interval` (no test calls it).
    `spearman_driver_matrix` and `q_from_residuals` have only a one-assert smoke
    check in `tests/test_kit_torch_free.py`; the rest of `drivers.py` and
    `conformal.py` is tested directly (`tests/test_data_and_core.py`,
    `tests/test_driver_multiplicity.py`, `tests/test_core.py`).
22. **Widen ruff** past the four classic rule groups (noted in
    `pyproject.toml`; mostly `--fix`-able, own branch).

## P5 — experiments

23. **leadgene pipeline:** pin `requirements.txt`, replace the hand-rolled UCB
    in `propose.py` with kalos/BoTorch batch acquisition, move permutation
    importance out-of-fold, and run its tests as an optional CI job
    ([TODO.md](../experiments/leadgene_pipeline/TODO.md)).

## Docs debt

24. ~~`deploy/.env.example` labels `NEXT_PUBLIC_API_URL` REQUIRED while compose
    treats it as a dev escape hatch.~~ **Fixed 2026-10-05:** it is commented out
    and documented as the escape hatch; `KALOS_ENGINE_TOKEN` is the required key
    (template, RUNBOOK, and `tests/test_deploy_config.py` agree).
25. `CHANGELOG.md` is ~1,200 lines; consider moving entries before 2026-08 to
    `docs/changelog/` with a link.

## Shipped

Kept here so the backlog above stays short; details are in the CHANGELOG.

- Multi-objective qLogNEHVI (titer + purity) with a Pareto front — `core/multiobjective.py`.
- Mixed continuous/categorical design spaces — `domains/`, the mixed GP + `optimize_acqf_mixed`.
- Feasibility classifier and gated acquisition — `core/feasibility.py`.
- Interval calibration (ECE) and the fail-closed promotion verdict, reported (never enforced) — `core/evaluation.py`, `core/gates.py`.
- Replicate-aware aggregation, assay noise floor, optional fixed-noise GP — `core/replicates.py`.
- Semantic column roles beyond header regexes: the TypeSafe tier and `roles=auto` (2026-10-05).
- An XGBoost baseline the GP is scored against, paired on the same folds (2026-10-05).
- `src/` layout, a Makefile, the dev container, and one setup script (2026-10-05).

## Next three PRs

1. **P0 bugs + pin dependencies** (items 1-3, 19): small, independent, each
   with a regression test; pinning first turns CI green again.
2. **CI fast lane + caching** (items 16-17): makes every later PR cheaper.
3. **Extract the pipeline** (item 4): unlocks items 5-8 and gives the runner a
   public API instead of a private import.
