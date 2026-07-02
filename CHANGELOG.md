# Changelog

Newest first.

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
