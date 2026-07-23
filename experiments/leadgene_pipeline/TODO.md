# TODO — improving model accuracy

Current state: `config.leadgene.yaml` (MP-only, duration-normalized target) clears
`USABLE` (CV Spearman +0.67, CI excludes 0) on real data — directional, not
production-grade. **More paired training rows is the single biggest lever** (n=10
today; the feature cap is `max(3, n // 4)`) — but assuming that's in progress
elsewhere, here's what else would move accuracy further, roughly in order of
leverage:

1. **Add per-well VCD/viability to the 24-well plate data.** Today MP passage
   wells only get *seed-level* VCD (shared across all wells); the SF row's rich
   per-well growth features (peak VCD, IVCD, qP) aren't available for any MP well.
   If per-well VCD becomes measurable, wire it into `pipeline/ingest.py` the same
   way DO/pH are — likely stronger signal than DO/pH alone.
2. **Try a percentile-rank target instead of titer_per_day_mg_L.** Per-day
   normalization didn't make MP and SF comparable (SF is a different cultivation
   scale, not just longer — see `outputs/comparison_with_vs_without_sf.md`).
   Leadgene's original within-source percentile rank might let both sources
   contribute without one dominating.
3. **Learn a plate↔SF scale correlation.** If MP-plate titer can be calibrated
   against SF titer, the SF row (and future SF runs) becomes usable signal for
   plate-scale predictions instead of being excluded entirely.
4. **Sweep preprocessing hyperparameters.** `correlation_prune_threshold`,
   `max_numeric_features`, `drop_low_variance` in `config/*.yaml` are untuned
   defaults — a small grid search at the current n could improve CV Spearman
   without any new data.
5. **Move permutation importance out-of-fold.** Current importance is in-sample
   (parity with Leadgene) — out-of-fold would give an honest signal for which
   features are actually worth engineering further.
6. **Try `hierarchical`** (`pipeline/models.py`, already implemented but unused) —
   needs enough same-source rows to split into reference + correction pools
   meaningfully; revisit once row count grows.
7. **Reference-data parity check.** Get Leadgene's original moat dataset and
   compare numbers directly — isolates whether any remaining gap is data or method.

# Code-quality backlog (2026-07-23 review)

A review after vendoring this pipeline into kalos `experiments/` fixed the
honesty-layer issues (the blend weight, confidence tier, and analysis
Interpretation now gate on the bootstrap-CI verdict, not the raw CV Spearman
point estimate; the collapse check is scale-invariant; `predictions.csv` carries
a `blend_validated` flag). The leakage audit was clean (preprocessing and feature
selection are fold-safe). The following lower-severity items were deferred:

1. **Pin `requirements.txt`.** Unpinned deps undercut the reproducibility the
   verdict layer promises; pin to the versions this lab was validated against.
2. **Defensively exclude `source_col` from features.** Today it is only kept out
   of the model incidentally, via dtype; exclude it by name so a numeric-coded
   source can never leak in as a feature.
3. **Guard `culture_duration_days == 0`** in the duration-normalized target
   (`ingest.py`) so a bad row fails with a clear message, not a cryptic later error.
4. **Remove the dead categorical-preprocessing path** (`preprocess.py`): the
   pipeline is numeric-only, so the categorical branch never runs.
5. **Guard the `.iloc[0]` seed-context reads** (`ingest.py`) for runs missing a
   VCD Day 0/Day 4 measurement.
6. **Out-of-fold permutation importance** (also item 5 above) would make the
   driver figure an honest generalization signal rather than an in-sample one.
