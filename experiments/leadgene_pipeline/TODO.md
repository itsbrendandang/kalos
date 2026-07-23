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
