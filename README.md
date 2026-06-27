# Voyager Engine

A Bayesian-optimization platform for bioprocess development, built on **BoTorch**
(a real GP surrogate + acquisition) with **NVIDIA BioNeMo / ESM-2** protein
features. A clean rebuild of the Voyager engine on a production BO stack: the
lean engine's honest parts (leakage-controlled cross-validation, fail-closed
promotion gates) carry over; the hand-rolled GP + EI is replaced by BoTorch.

> Genuine, classical/probabilistic + deep-learning ML — not marketing. The BO
> loop runs here today, and ESM-2 embeds protein sequences on the Mac's GPU.

## What's here

```
voyager/
  core/
    surrogate.py     BoTorch SingleTaskGP (input-normalized, output-standardized)
    optimize.py      qLogEI acquisition + optimize_acqf -> next batch
    evaluation.py    grouped cross-validated Spearman (leakage-controlled)
    gates.py         fail-closed promotion gates (carried from the lean engine)
  features/
    protein.py       ESM-2 embeddings (transformers, MPS) + KmerEmbedder stand-in
examples/            runnable demos (BO loop, protein embedding)
tests/               pytest
```

## Install / run

```bash
python -m pip install -e .              # core (torch / botorch / gpytorch)
python -m pip install -e ".[protein]"   # + real ESM-2 (transformers)

python examples/run_bo_loop.py          # BoTorch closed-loop demo
python examples/run_protein_embed.py    # ESM-2 embedding demo (downloads a small model)
python -m pytest                        # tests (VOYAGER_TEST_ESM=1 to include ESM-2)
```

## Device

BoTorch GPs run in float64 for numerical stability, and Apple's MPS backend is
float32-only, so the **GP runs on CPU** (fast for the small-sample regime BO
targets). The **ESM-2 embedder runs on MPS** when available. No GPU is required
to use the platform locally.

## Roadmap

- Multi-objective (qNEHVI) for titer + purity together.
- Mixed continuous/categorical inputs (carbon source, medium) via Ax or BoTorch.
- The push-feed ingestion loop (platform streams runs in; brain proposes the next
  batch; anonymized on read).
- Feasibility classifier + calibration so the promotion gates have AUC/ECE/Brier.
- Larger ESM-2 / full NVIDIA BioNeMo backend behind the same embedder interface.

No proprietary data lives in this repo; the demos use synthetic data and public
protein sequences.
