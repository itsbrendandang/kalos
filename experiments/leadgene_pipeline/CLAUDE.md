# CLAUDE.md — repo guide for ml-leadgene-pipeline

Terminal-driven ML tool that trains on featurized CSVs and predicts **titer** for a
cohort of wells, ranking them by predicted titer + cross-model confidence. The
modeling core is ported from `Leadgene_Clone_Picker` (`clone_ranking` + `clone_select`);
this repo is the clean, config-driven consolidation of it.

## Workflow (two steps, persisted artifact between)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pipeline ingest    # raw_training_data/ + raw_prediction_data/ -> featurized CSVs
.venv/bin/python -m pipeline train   --config config/config.leadgene.yaml
.venv/bin/python -m pipeline predict --config config/config.leadgene.yaml
.venv/bin/python -m pytest -q
```

## Architecture (module map)

- `pipeline/feature_extraction.py` — **pure** timecourse→features: `extract_features`
  (~50 features/channel) + `phase_and_rate_features` (phase-thirds mean/slope,
  rate_std, oscillation_count, threshold-crossing — the handful `extract_features`
  doesn't already cover). No config coupling; unit-testable.
- `pipeline/ingest.py` — raw single-purpose CSVs → the combined featurized tables
  `training_data/` / `prediction_data/` expect. Owns the joins (plate map → pool →
  titer) and bioprocess-specific engineering (fed-batch growth/IVCD/qP, seed
  context, duration normalization) that `feature_extraction.py` deliberately
  doesn't know about.
- `pipeline/data.py` — `TrainingPool.from_dir` (pools every CSV in `training_data/`,
  unions columns, raw-titer target) + `feature_columns` (numeric non-meta) +
  `load_predict_cohort`.
- `pipeline/preprocess.py` — `build_preprocessor`, `reduce_numeric_features`,
  `duration_robust_filter` (+ `DURATION_DEPENDENT_DEFAULT`), `cohort_novelty`.
- `pipeline/features.py` — `FeatureSelector` (dynamic cap `max(3, n//4)`).
- `pipeline/models.py` — 5 regressors sharing `fit`/`predict_mean_std` +
  `HierarchicalShrinkageModel` (empirical-Bayes reference+correction).
- `pipeline/evaluation.py` — group-aware CV + bootstrap-CI Spearman trust metric
  (`fit_method`, `fit_hierarchical`), `permutation_importance_spearman`.
- `pipeline/blend.py` — CV-weighted blend → `predicted_titer` + confidence tier.
- `pipeline/propose.py` — `propose_batch`: acquisition over scored candidates
  (UCB `mean + beta*uncertainty` when validated, uncertainty-only space-filling
  when not) → a diversified next batch. Verdict-gated. Stand-in for kalos/BoTorch BO.
- `pipeline/sim.py` — clean-room mechanistic CHO fed-batch simulator (Monod growth,
  lactate inhibition/diauxie, Luedeking-Piret growth-decoupled production). Maps
  controllable process params → endpoint titer; `sample_doe` (LHS), `true_optimum`.
  Synthetic-data + benchmark surface, NOT a validated predictor.
- `examples/closed_loop_benchmark.py` — in-silico validation of the loop: train →
  `propose` → simulate → measure → repeat, BO vs random over a fixed candidate library.
- `pipeline/pipeline.py` — `TrainPipeline` (fit + build reference block) /
  `PredictPipeline` (score + carry cohort/reference into `Report`).
- `pipeline/analysis.py` — compute (`diagnose`, `within_plate_contrast`, `novelty`)
  **separated** from render (`write_analysis` md, `write_visualizations` figures).
- `pipeline/artifact.py` — joblib bundle, `ARTIFACT_VERSION` (retrain on mismatch).

## Key concepts

- **Reference model vs blend.** `reference_model` (config, default `point_gb`) is the
  single model whose fit + CV Spearman + permutation importance drive the two figures
  (`fig1_ranking.png`, `fig2_drivers.png`) — this matches Leadgene's `ProductivityModel`
  path. The 5-model **blend** produces the `predicted_titer` column + confidence tier.
- **Target is raw titer** (intentional divergence from Leadgene's percentile-rank).
- **Duration caveat (important).** MP cultures ≈ 4 days, SF ≈ 10–14 days; longer
  cultivation ⇒ more titer. `exclude_duration_dependent: true` drops run-length/sampling
  features, but does NOT fix the target. **A reference model must be trained on the same
  measurement/duration class as the cohort it ranks** — enforce this with
  `data.include_sources: [<source_dataset value>]` (+ `columns.source_col`), which
  restricts `TrainingPool.from_dir` to matching rows. Confirmed effective on real data:
  pooling MP+SF gave `NOT VALIDATED` (CV Spearman +0.54, CI crossing 0) and an
  implausible predicted-titer range; restricting to `MP_24w_Passage` (the cohort's own
  duration class) gave `USABLE` (+0.67, CI excludes 0) and a plausible range. See `TODO.md`.
- **Honesty.** CV Spearman is always shown with its bootstrap 95% CI + `NOT VALIDATED`
  verdict; permutation importance is **in-sample** (labelled as such); `cohort_novelty`
  warns when the cohort is off-distribution. Small n ⇒ often not validated; that is the
  correct signal, not a bug.

## Data layout (gitignored)

- `raw_training_data/` — one subfolder **per dataset** (any name), classified by
  which files it contains: a **24-well plate** folder holds its own
  `raw_passage24w_do.csv`, `raw_passage24w_ph.csv`, `raw_passage24w_titer.csv`,
  `raw_passage24w_vcd.csv`, `raw_plate_map.csv` (wells are only unique *within* a
  plate, so each plate needs its own well→pool_name map); a **fed-batch run**
  folder holds its own `raw_fedbatch_titer.csv`. `pipeline ingest` discovers every
  subfolder of either kind, engineers each independently, and namespaces
  `run_id` by folder name so identical pool names / run ids across folders never
  collide, then pools everything into `training_data/combined_ml_training_table.csv`.
  Adding another plate or fed-batch run is just adding another folder.
- `raw_prediction_data/` — the cohort's raw DO/pH (+ optional plate map), same
  shape as one plate's DO/pH files, no titer.
- `training_data/` — any number of featurized CSVs (row=well; `titer`, `well_id`,
  optional `clone` group, numeric feature cols). Pooled at train; different feature
  columns are unioned + imputed. Can also be dropped in directly if already featurized.
- `prediction_data/predict.csv` — one CSV, same features, no titer.
- `outputs/`, `artifacts/` — derived; regenerate, never commit.

## Conventions

- pip install into project-local `.venv/`; never system Python.
- Feature columns are channel-prefixed (`DO_`, `pH_`, `VCD_`); ids/labels auto-excluded.
- Plain-dict config (validated in `config.py`), not Pydantic — keep it that way.
- Bump `ARTIFACT_VERSION` when the bundle shape changes; `load` rejects stale bundles.
- **Any future ML-output/functionality improvement goes in `TODO.md`**, not silently
  into scope. Consult the `/ml` and `/architect` skills on substantive changes.
