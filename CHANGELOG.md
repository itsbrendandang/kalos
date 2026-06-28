# Changelog

Newest first.

## 2026-06-26

### Added — `examples/run_on_media_data.py`: engine on real media data
- Runs the engine on real Anagram runs (media + pH -> lipase titer; data read from `CULTIVAR_MEDIA_DATA`, not vendored): grouped-CV of the surrogate, a proposed next batch, and the multi-objective titer-vs-purity Pareto front. First run (82 defined-media runs, grouped by medium): **grouped-CV Spearman 0.53** (vs ~0.25 from the lean engine on the same data — the BoTorch GP extracts more signal), the promotion gate fails closed on missing feasibility/calibration, the proposed batch pushes Methanol to its observed max (the known top driver), and the Pareto front exposes a real titer-vs-purity tradeoff (max titer 0.022 at only 2% purity vs 0.0167 at 100%).

### Added — web portal (`cultivar/portal/`)
- A FastAPI viewer over the LIVE engine (not mock data): it runs the real BoTorch single- and multi-objective optimization on request and charts the convergence curve, the titer-vs-purity Pareto front, and the proposed next batch. Dark, cobalt-accented, Plotly. `pip install -e ".[portal]"` then `python -m cultivar.portal` -> http://127.0.0.1:8050. Verified in-browser: both charts render, the single-objective loop converges to [0.696, 0.301, 0.491] (optimum [0.7,0.3,0.5]) and the Pareto front shows 7-9 tradeoff points.

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
