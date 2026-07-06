# Kalos

A Bayesian-optimization platform for bioprocess development, built on **BoTorch**
(a real GP surrogate + acquisition) with **NVIDIA BioNeMo / ESM-2** protein
features. Honest by construction: leakage-controlled cross-validation and
fail-closed promotion gates, on a production BoTorch core rather than a
hand-rolled GP + EI.

> Genuine, classical/probabilistic + deep-learning ML — not marketing. The BO
> loop runs here today, and ESM-2 embeds protein sequences on the Mac's GPU.

## What's here

```
kalos/
  core/
    surrogate.py       BoTorch SingleTaskGP (input-normalized, output-standardized)
    optimize.py        qLogEI acquisition + optimize_acqf -> next batch
    multiobjective.py  qLogNEHVI for two+ objectives (titer AND purity) + Pareto front
    evaluation.py      grouped cross-validated Spearman (leakage-controlled)
    splits.py          group-aware CV + a leakage tripwire (assert_no_group_leakage)
    drivers.py         signed Spearman drivers with bootstrap confidence intervals
    conformal.py       split-conformal prediction intervals (distribution-free)
    gates.py           fail-closed promotion gates
  data/
    anonymizer.py      strip identity, hash grouping keys, mint stable barcodes
    barcode_registry.py  the organized home for all run data (barcode -> run, queryable, persistable)
  ingest/
    feed.py            push DataFeed + ProposalSink, anonymized on read
    runner.py          decoupled closed loop: read feed -> fit -> propose -> sink
  features/
    protein.py         ESM-2 embeddings (transformers, MPS) + KmerEmbedder stand-in
  portal/
    app.py             FastAPI app: live-engine views + the "run your own data" upload path
    validate.py        ingestion preflight + per-column provenance (what was kept/dropped and why)
examples/            demos + run_on_media_data.py + organize_data.py (folder -> barcode registry)
tests/               pytest
```

A lot of the honest-evaluation and data machinery was carried over from a prior
lean engine (`voyager-brain-rebuild`): the leakage-controlled splits, the
bootstrap-Spearman drivers, the data anonymization, and the push-feed ingestion.

## Data registry (barcodes)

All run data is organized through a barcode registry: every dataset gets a
`KAL-DS-*` barcode and every run a stable, content-derived `KAL-*` barcode, with
client/strain identity stripped and grouping keys hashed on the way in. Look a
run up by barcode, filter by dataset, or export a training table.

```bash
python examples/organize_data.py <data_dir> registry.json   # folder of CSV/TSV -> one registry
```

## Uploading a run sheet (`POST /api/run`)

The portal accepts an uploaded CSV / TSV / Excel run sheet and returns the analysis
(auto-detected target, leakage-safe features, honest grouped-CV, signed drivers, a proposed next
batch). Because the sheet comes from an external client, the upload path is guarded:

- **Size + shape caps.** A raw-byte cap (default 25 MB, override with `KALOS_MAX_UPLOAD_MB`), a
  512-column ceiling, a 100k-row CSV cap, and a 2M-cell xlsx cap (a zip-bomb guard). Every cap
  fails closed: an over-cap upload is REJECTED with a 400, never silently truncated. The xlsx
  cell cap is enforced from the workbook's declared dimensions BEFORE the frame is materialized,
  so a zip-bomb is rejected without the memory spike. Filetype is sniffed by magic bytes
  (`PK\x03\x04` -> Excel, else text/CSV), not by extension.
- **Safe errors.** A bad upload returns a generic HTTP 400 ("Could not parse the uploaded file.
  Check it is a CSV or Excel run-sheet."). This holds even when the failure is a GP fit error
  (`torch.linalg.LinAlgError`) or a leakage-guard assertion, not just a parser error: a catch-all
  normalizes any such failure to the same `{error}` JSON envelope. Parser details, column names,
  cell values, paths, and stack traces are logged server-side and never returned to the caller.
- **Provenance.** The response includes a `provenance` list: for every column, its status
  (`kept_feature`, `target`, `dropped_id`, `dropped_output`, `dropped_constant`,
  `dropped_constant_on_fitted_rows`, `dropped_sparse`, `dropped_all_blank`) and how many cells had
  to be coerced from non-numeric text (units like "34.6 C"). A feature that varies over the full
  sheet but is constant on the target-present rows the GP actually fits is dropped and flagged
  (`dropped_constant_on_fitted_rows`), never silently pinned to a zero-width bound. No more silent
  column drops.
- **Reproducibility.** The analyze path is seeded, so the same upload yields identical proposals;
  the response carries `seed`, `timestamp`, and `engine_version`.
- **Opt-in anonymization.** Pass the form field `anonymize=true` to pseudonymize identifier-type
  columns in the response. Feature and target names are kept as-is (the owner UI legitimately shows
  drivers like "Methanol").

## Install / run

```bash
python -m pip install -e .              # core (torch / botorch / gpytorch)
python -m pip install -e ".[protein]"   # + real ESM-2 (transformers)

python examples/run_bo_loop.py          # BoTorch closed-loop demo
python examples/run_multiobjective.py   # multi-objective (titer + purity) Pareto demo
python examples/run_protein_embed.py    # ESM-2 embedding demo (downloads a small model)
python -m pytest                        # tests (KALOS_TEST_ESM=1 to include ESM-2)

python -m pip install -e ".[portal]"    # the web portal (FastAPI)
python -m kalos.portal                # -> http://127.0.0.1:8050  (live BO + Pareto view)
```

## Device

BoTorch GPs run in float64 for numerical stability, and Apple's MPS backend is
float32-only, so the **GP runs on CPU** (fast for the small-sample regime BO
targets). The **ESM-2 embedder runs on MPS** when available. No GPU is required
to use the platform locally.

## Roadmap

- ~~Multi-objective (qNEHVI) for titer + purity together.~~ **Done** (`core/multiobjective.py`).
- Mixed continuous/categorical inputs (carbon source, medium) via Ax or BoTorch.
- The push-feed ingestion loop (platform streams runs in; brain proposes the next
  batch; anonymized on read).
- ~~Feasibility classifier + gated acquisition.~~ **Done** (`core/feasibility.py`,
  `bo_feas` / `bo_feas_clean` strategies in `bench/pool.py`). Calibration (ECE/Brier)
  for the promotion gates is still pending.
- ~~Replicate-aware aggregation + assay noise-floor estimation + optional
  fixed-noise GP.~~ **Done** (`core/replicates.py`, `Surrogate.fit(..., noise=...)`,
  `pool_from_frame(..., aggregate=True)`) — the SNR lever from `BENCHMARK.md`.
- Larger ESM-2 / full NVIDIA BioNeMo backend behind the same embedder interface.

No proprietary data lives in this repo; the demos use synthetic data and public
protein sequences.
