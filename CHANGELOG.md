# Changelog

Newest first.

## 2026-06-26

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
- New personal repo: a clean rebuild of the Voyager engine on a production BO stack.
- `core/surrogate.py` — BoTorch SingleTaskGP (input-normalized, output-standardized), fit by exact marginal likelihood. Runs on CPU (GPs need float64; MPS is float32-only).
- `core/optimize.py` — qLogEI acquisition + `optimize_acqf` to propose the next batch within bounds. Verified end to end: the demo loop climbs to the synthetic optimum.
- `core/evaluation.py` — grouped cross-validated Spearman of the surrogate (leakage-controlled).
- `core/gates.py` — fail-closed promotion gates carried over from the lean engine (a missing/NaN required metric blocks promotion).
- `features/protein.py` — real ESM-2 embeddings via Hugging Face transformers (runs on Apple MPS, verified: 320-dim vectors distinguish HSA vs Herceptin) plus a dependency-free `KmerEmbedder`. The NVIDIA BioNeMo family behind one interface.
- pyproject packaging, pytest suite (5 passing + ESM-2 test behind a flag), CI, examples.
