# ml-leadgene-pipeline

A terminal-driven ML tool that **trains on featurized CSV data and predicts titer
for a cohort of wells**, ranking them by predicted titer, cross-model confidence,
and each model's individual output. The modeling core (five regressors,
cross-validated for trustworthiness and blended) is ported from
`Leadgene_Clone_Picker`'s `clone_ranking` package; the I/O layer is config-driven,
and training and prediction are two steps with a persisted artifact in between.

Pure Python / NumPy / scikit-learn — no GPU or deep-learning framework.

> **Read [TODO.md](TODO.md) before trusting any output.** It tracks what's been
> tried, what worked, and what would most improve accuracy next — the pipeline is
> honest when a result isn't statistically validated; check the verdict, not just
> the numbers.

---

## 1. Install

Standalone, in this directory:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Or inside the kalos repo's shared dev environment (one `.venv` at the repo root,
rebuilt the same way on a laptop, in the dev container, or in a cloud session):

```bash
KALOS_LEADGENE=1 bash ../../scripts/setup-dev.sh   # kalos[ml,portal,dev,typesafe] + requirements.txt
../../.venv/bin/python -m pytest -q                # run from this directory
```

This pipeline is an off-path experiment (see [../README.md](../README.md)): kalos's CI
does not run its tests, so run them yourself after changing it.

## 2. Data layout

**No data is stored in this repo.** The four *input* folders exist as empty
skeletons (each holds a `.gitkeep` so the folder itself is tracked), but
everything *inside* them is gitignored (`.gitignore` uses a `folder/*` +
`!folder/.gitkeep` pattern per folder) — a fresh clone has them ready to receive
data with nothing to create by hand. `artifacts/` and `outputs/` are pure derived
output: not tracked at all (not even empty), since `pipeline train`/`predict`
create them on demand.

| folder | contents | produced by | tracked (empty) |
|---|---|---|---|
| `raw_training_data/` | raw, unfeaturized CSVs (see below) | you (pre-cleaned exports) | yes |
| `raw_prediction_data/` | the cohort's raw DO/pH CSVs | you (pre-cleaned exports) | yes |
| `training_data/` | featurized training CSV(s) | `pipeline ingest`, or drop in your own | yes |
| `prediction_data/` | featurized `predict.csv` | `pipeline ingest`, or drop in your own | yes |
| `artifacts/` | trained model bundle (`.joblib`) | `pipeline train` | no |
| `outputs/` | predictions, analysis, figures | `pipeline predict` | no |

### `raw_training_data/`: one subfolder per dataset (any name)

`pipeline ingest` classifies each subfolder by which files it contains, so
adding a new plate or fed-batch run is just adding a new folder:
- a **24-well plate** folder holds that plate's own `raw_passage24w_do.csv`,
  `raw_passage24w_ph.csv`, `raw_passage24w_titer.csv`, `raw_passage24w_vcd.csv`,
  `raw_plate_map.csv` — no feature engineering, one measurement per file. Wells
  (`B2`, `C3`, …) are only unique *within* a plate, so each plate carries its
  own well→pool_name map.
- a **fed-batch run** folder holds that run's own `raw_fedbatch_titer.csv`.

IDs are namespaced by folder name so identical pool names / run ids across
folders never collide; everything pools into
`training_data/combined_ml_training_table.csv`.
```
raw_training_data/
  raw_24w_passage/                # a 24-well plate folder (any name)
    raw_passage24w_do.csv
    raw_passage24w_ph.csv
    raw_passage24w_titer.csv
    raw_passage24w_vcd.csv
    raw_plate_map.csv
  raw_sf_fedbatch/                 # a fed-batch run folder (any name)
    raw_fedbatch_titer.csv
  # additional plates / runs just need their own folder, e.g.:
  # raw_24w_passage_2/, raw_sf_fedbatch_2/, ...
```

### The other folders

- **`raw_prediction_data/`** — the cohort's raw DO/pH (+ optional plate map), same
  shape as one plate's DO/pH files, no titer. Flat (one cohort), not subfoldered.
- **`training_data/`** — any number of featurized CSVs. Each row is one
  well; columns are a `titer` target, a `well_id`, an optional group column
  (`clone`), and any numeric feature columns. Files may have **different feature
  columns** — they're unioned by name and the gaps are imputed, so heterogeneous
  shapes pool cleanly. Every `*.csv` here is pooled at train time; drop or remove a
  file to control what the model learns from. Can be built by `pipeline ingest`, or
  dropped in directly if you already have featurized CSVs.
- **`prediction_data/predict.csv`** — a single CSV with the same feature columns and
  **no titer**. These are the wells to score.
- **`artifacts/`** — the joblib model bundle `pipeline train` writes to
  `artifact_path` and `pipeline predict` loads back. Namespace `artifact_path`
  per config (e.g. `leadgene_without_sf.joblib`) if you keep more than one.
- **`outputs/`** — everything `pipeline predict` writes, namespaced to
  `output_csv`'s basename (see [§3 Outputs](#outputs-all-namespaced-to-output_csvs-basename)
  below). Point `data.output_csv` at a subfolder (e.g. `outputs/without_sf/predictions.csv`)
  to keep multiple runs' results side by side for comparison instead of overwriting.

### Building featurized CSVs from raw data

```bash
.venv/bin/python -m pipeline ingest   # raw_training_data/ + raw_prediction_data/ -> featurized CSVs
```

This is the canonical path: [pipeline/ingest.py](pipeline/ingest.py) discovers every
plate/fed-batch subfolder under `raw_training_data/`, engineers features per folder,
and writes `training_data/combined_ml_training_table.csv` (and the analogous
`prediction_data/predict.csv` from `raw_prediction_data/`). It owns the *joins*
(plate map → pool name → titer) and the *bioprocess-specific* engineering:

- **Per-well DO/pH** (Step 1): merges the already-ported ~50-feature extractor
  ([pipeline/feature_extraction.py](pipeline/feature_extraction.py) `channel_features`
  — level/shape stats, regression slopes, derivatives, timing, peaks, autocorr,
  rolling stats) with `phase_and_rate_features` — the handful of features it doesn't
  already cover: **phase-thirds** mean/slope (early/mid/late), **rate_std** /
  **oscillation_count** (dispersion and sign-changes of the point-to-point rate),
  and **threshold-crossing time** (DO<20%, pH<6.8). No duplicate columns between
  the two.
- **Fed-batch run-level** (Step 2): peak VCD + day of peak, specific growth rate
  (μ, exponential-phase log-linear fit), viability decline rate, **IVCD** (integral
  of viable cell density — a standard bioprocess productivity proxy), and
  **specific productivity (qP)**.
- **Duration normalization** (Step 3): `titer_per_day_mg_L = titer_final_mg_L /
  culture_duration_days` — the **recommended target** for any model trained across
  both the 4-day (MP) and 10–14-day (SF) datasets; raw titer and duration are kept
  alongside it since duration itself may carry signal.
- **Seed-culture context** (Step 4): the parent seed culture's Day 0/Day 4 VCD and
  viability, broadcast onto every 24-well row (shared starting condition, not
  per-well variation — matters once multiple passages are compared).

You can also drop in your own already-featurized CSVs directly and skip
ingestion entirely.

## 3. Run

Everything is driven by [config/config.leadgene.yaml](config/config.leadgene.yaml)
(copy [config/config.example.yaml](config/config.example.yaml) for a new dataset):

```bash
# raw CSVs -> featurized training_data/ + prediction_data/ (skip if already featurized)
.venv/bin/python -m pipeline ingest

# fit every configured model, cross-validate each, save the artifact
.venv/bin/python -m pipeline train   --config config/config.leadgene.yaml

# load the artifact and predict titer for the cohort
.venv/bin/python -m pipeline predict --config config/config.leadgene.yaml

# rank a pool of untested conditions and pick the next batch to run
.venv/bin/python -m pipeline propose --config config/config.leadgene.yaml \
  --candidates prediction_data/candidates.csv --q 5
```

`train` prints each model's CV Spearman + bootstrap 95% CI + verdict and writes
`artifact_path`. `predict` prints the blend weights + top wells and writes the
outputs below. Overrides: `--train-dir`, `--predict-csv`, `--artifact`, `--output`.

`propose` is the active-learning step: it scores a candidate pool (untested
conditions in the same feature schema, no `titer`) and ranks them by an
acquisition score, then selects a diversified next batch.
The strategy is gated on the honest verdict.
When at least one model validated, it ranks by an upper-confidence bound
(`mean + beta * uncertainty`) to climb toward the optimum (exploit).
When nothing validated, the predicted titers are not trustworthy, so it ranks by
uncertainty alone for space-filling to gather data that can validate the model
(explore).
It writes `<output>_proposals.csv` (per-candidate `pred_titer`, `uncertainty`,
`acq_score`, `selected`).
Knobs live under `propose:` in the config (`q`, `beta`, `diversity`) or via
`--q` / `--beta`.
This is a deliberately simple stand-in for a real Bayesian-optimization
acquisition; wiring it to kalos / BoTorch qEI is the next step (see
[TODO.md](TODO.md)).

### Synthetic data + closed-loop benchmark

`pipeline/sim.py` is a clean-room mechanistic CHO fed-batch simulator (Monod growth,
lactate inhibition and diauxie, Luedeking-Piret growth-decoupled production) that
maps controllable process parameters to an endpoint titer.
It is a synthetic-data and benchmark surface, not a validated predictor of any real
process, and must never be presented as one.

`examples/closed_loop_benchmark.py` uses it to validate the acquisition loop in
silico - train, `propose`, "run" the batch against the simulator, measure, repeat -
comparing `propose` against random selection over a fixed candidate library:

```bash
python -m examples.closed_loop_benchmark
```

### Config surface

```yaml
data:
  train_dir: training_data              # every *.csv here is pooled for training
  predict_csv: prediction_data/predict.csv
  output_csv: outputs/predictions.csv   # manifest/analysis/figures follow this basename
  include_sources: [MP_24w_Passage]     # optional: restrict training to matching source_col
                                         # values -- keeps the model in the same measurement/
                                         # duration class as the cohort it ranks (see below)
columns:
  target: titer                         # column to predict (raw titer units)
  id_col: well_id                       # per-row label carried into the output
  group_col: clone                      # optional; group-aware CV. omit -> each row its own group
  source_col: source_dataset            # column tagging each row's source (used by
                                         # data.include_sources above; also needed for 'hierarchical')
  # reference_source: MP_24w_Passage    # enable 'hierarchical' by naming its reference pool
features:
  exclude: []
  exclude_duration_dependent: true      # drop run-length/sampling features (mixed durations)
models: [point_gb, bootstrap_ensemble, gaussian_process, bayesian_ridge, copula_augmented]
reference_model: point_gb               # its fit + CV Spearman + importance drive the figures
outputs:
  analysis: true                        # write <output>_analysis.md
  visualize: true                       # write the two figures below
  n_top: 5                              # top wells highlighted in the ranking figure
  top_features: 7                       # features shown in the driver figure
```

### `data.include_sources`: keep training in the cohort's duration class

MP passage titer (~40–67, 4-day cultures) and SF fed-batch titer (~1672, 10–14-day
cultures) are the same units at very different scales/durations — pooling both lets a
model learn "which source" instead of real signal. On real data, training on both
(n=11) gave `NOT VALIDATED` (CV Spearman +0.54, CI crossing 0) and an implausible
predicted-titer range (27–1076) for a passage-scale cohort. Restricting to
`include_sources: [MP_24w_Passage]` (n=10, dropping the one mismatched SF row) gave
`USABLE` (+0.67, CI excludes 0) and a plausible range (43–61). Set it to whichever
`source_col` value matches the cohort you're predicting.

### The reference model vs the blend

Two things run together (matching Leadgene's two pipelines, unified here):
- **The blend** (all configured models, CV-weighted) produces `predicted_titer` +
  `confidence_tier` in `predictions.csv`.
- **The reference model** (`reference_model`, a single regressor) is what drives the
  two figures and the CV-Spearman headline — mirroring Leadgene's `ProductivityModel`.

### Outputs (all namespaced to `output_csv`'s basename)

- **`predictions.csv`** — one row per cohort well: `well_id`, **`predicted_titer`**
  (blended, titer units), **`blend_rank`**, `confidence_tier`, and each model's
  `<model>_pred` / `<model>_std`.
- **`predictions_manifest.json`** — per-model CV Spearman, CI, verdict, collapse flag,
  blend weights.
- **`predictions_analysis.md`** — reference-model CV Spearman (with CI), per-model trust
  table, predicted-titer spread, off-distribution (`cohort_novelty`) warnings, and a
  plain-language verdict.
- **`predictions_clone_ranking.csv`** — reference-model ranking (`well_id, pred_titer, rank`).
- **`predictions_drivers.csv`** — permutation-importance table (`feature, importance`).
- **`predictions_figures/`** — the two figures:
  - **`fig1_ranking.png`** — wells ranked by reference-model predicted titer, top-N
    highlighted, dashed "reference mean" line, `CV Spearman` in the title.
  - **`fig2_drivers.png`** — (a) permutation importance ("what the model weights",
    in-sample) and (b) within-plate-SD contrast ("why these picks stand out").

`confidence_tier` is cross-model agreement, **not** statistical validation. The CV
Spearman is shown with its bootstrap CI and a `NOT VALIDATED` verdict when the CI
includes 0 — trust the verdict, not the point estimate.

---

## 4. How it works (ported logic)

- **Feature extraction** (`feature_extraction.py`) — pure `(t, y) -> ~50 features`,
  ported verbatim from Leadgene.
- **Models** (`models.py`) — five regressors sharing one `fit` / `predict_mean_std`
  interface (gradient-boosted point, bootstrap ensemble, Gaussian Process, Bayesian
  Ridge, copula-augmented), plus an optional `HierarchicalShrinkageModel` (empirical-Bayes
  reference+correction; enable via `columns.reference_source`). All regress raw titer.
- **Feature selection** (`features.py`, `preprocess.py`) — variance filter + correlation
  pruning + `duration_robust_filter`, capped at `min(max_numeric_features, n_train // 4)`.
- **Evaluation** (`evaluation.py`) — group-aware CV (`GroupKFold` / `LeaveOneGroupOut`)
  with a bootstrap CI on the CV Spearman as the trust metric; `permutation_importance_spearman`
  (in-sample) for the driver figure.
- **Reference model** — the `reference_model` config names which fitted model's CV Spearman
  + permutation importance + training titer mean are persisted to drive the figures
  (mirrors Leadgene's `ProductivityModel`).
- **Blend** (`blend.py`) — weight each model by `max(0, cv_spearman)`, zero out any that
  collapse on the cohort, combine into `predicted_titer` + a confidence tier; equal-weight
  fallback if nothing validates. `cohort_novelty` flags off-distribution cohorts.

Train computes the CV weights, fits the final models, and builds the reference block
(persisted in the versioned joblib artifact with fitted preprocessors — no train/serve
skew). The cohort-dependent collapse check, confidence tiers, novelty, and within-plate
contrast are computed at predict time.

See [CLAUDE.md](CLAUDE.md) for the module map/conventions and [TODO.md](TODO.md) for
ideas to improve accuracy further.

## 5. Test

```bash
.venv/bin/python -m pytest -q
```
`tests/test_ingest.py` builds synthetic raw CSVs matching the ingestion schema and
checks the joins (plate map → pool → titer, including dropped unmatched pools),
Steps 1–4's engineered features, duration normalization, and column ordering.
`tests/test_train_predict.py` builds a synthetic featurized dataset (two training
files with different feature columns), runs train → predict end-to-end, and checks
the output shape, titer range, and determinism.
