# Changelog

Newest first.

## 2026-06-26

### Added — clean platform on BoTorch + ESM-2 (initial scaffold)
- New personal repo: a clean rebuild of the Voyager engine on a production BO stack.
- `core/surrogate.py` — BoTorch SingleTaskGP (input-normalized, output-standardized), fit by exact marginal likelihood. Runs on CPU (GPs need float64; MPS is float32-only).
- `core/optimize.py` — qLogEI acquisition + `optimize_acqf` to propose the next batch within bounds. Verified end to end: the demo loop climbs to the synthetic optimum.
- `core/evaluation.py` — grouped cross-validated Spearman of the surrogate (leakage-controlled).
- `core/gates.py` — fail-closed promotion gates carried over from the lean engine (a missing/NaN required metric blocks promotion).
- `features/protein.py` — real ESM-2 embeddings via Hugging Face transformers (runs on Apple MPS, verified: 320-dim vectors distinguish HSA vs Herceptin) plus a dependency-free `KmerEmbedder`. The NVIDIA BioNeMo family behind one interface.
- pyproject packaging, pytest suite (5 passing + ESM-2 test behind a flag), CI, examples.
