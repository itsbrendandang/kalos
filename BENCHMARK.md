# Does the optimizer actually beat a space-filling design?

This is the one question a bioprocess-optimization product must answer honestly before it is sold:
given a fixed experiment budget, does the Bayesian-optimization loop reach a good recipe in **fewer experiments** than just running a Latin Hypercube (LHS) design or sampling at random?

`kalos/bench/` answers it on synthetic surfaces with **known optima** (so true simple regret is measurable and the whole thing is reproducible from committed code), while sweeping observation noise.
The noise sweep is the honest link to real data: on the real media DoE the held-out signal is weak (grouped-CV Spearman ~0.37-0.52), and measurement noise is a large part of why.

Reproduce:

```bash
python -m kalos.bench          # full sweep (2 surfaces x 2 noise levels, 10 seeds)
python -m kalos.bench --quick  # fast check (4 seeds)
```

Every trial starts every strategy from the **same** seeded LHS init design (a fair race), then spends the budget one experiment at a time.
The surrogate only ever sees **noisy** observations, like a real assay; regret is judged on the noiseless truth.

## Results (n_init=5, budget=18, 10 seeds)

Simple regret = distance from the true optimum (lower is better).

| Surface | Noise | BO regret | LHS regret | Random regret | BO vs LHS |
| --- | --- | --- | --- | --- | --- |
| Gaussian bump (4d, smooth) | 0% | **0.014** | 2.247 | 3.171 | reaches LHS's end-value ~11 experiments sooner |
| Gaussian bump (4d, smooth) | 15% | **1.676** | 2.247 | 2.592 | ~2 experiments sooner |
| Ackley (4d, rugged) | 0% | **1.489** | 3.625 | 3.696 | ~14 experiments sooner |
| Ackley (4d, rugged) | 15% | **3.572** | 3.625 | 3.850 | ~tied |

## The honest read

**1. The BO engine is genuinely good.**
With clean measurements it does not just win, it dominates: near-zero regret on the smooth surface and a large lead on the rugged one, reaching the value the static LHS design ends at roughly 11-14 experiments sooner.
This is not a broken wrapper over a GP; the optimization machinery works.

**2. Observation noise is the decisive factor, not the algorithm.**
At 15% measurement noise the advantage shrinks to modest on the smooth surface and essentially disappears on the rugged one.
Real assay noise is frequently at or above this level, which is exactly why the real media data shows a fragile ~0.37-0.52 held-out signal: the optimizer is fine, the data is noisy relative to the effect being chased.

**3. What this means for the product.**
The core promise - "reach your target in fewer experiments" - is real, but only when the signal-to-noise is high enough.
The highest-leverage moves to make BO beat a scientist's own LHS on real data are therefore about the data, not the model:

- collect **technical/biological replicates** so a real assay-noise floor can be estimated and fed to the surrogate, and so we know how much of the ceiling is even reachable;
- prioritize campaigns and objectives with enough signal relative to their noise;
- keep the honesty layer (leakage-controlled CV, conformal intervals) front and center, because it is what tells a client honestly whether their data is in the regime where the loop helps.

Positioning follows from this: sell the rigor and the honest decision layer, and be explicit that the loop's value grows with replicate discipline - do not oversell "our AI optimizes your process" on a single noisy campaign.

## On the real media data (pool-based retrospective)

The synthetic sweep says BO helps when the signal is clean and stops helping as noise rises.
So the honest test is the real one.
On the actual combined media DoE (96 runs, target `Lipase_g.L`, via `kalos/bench/pool.py`): does BO pick the best recipes from the pool in fewer experiments than random selection?

`run_pool` starts from a seeded random subset, then each strategy picks the next real experiment to "run", revealing its measured (already noisy) titer.
BO scores every remaining candidate by Expected Improvement from the surrogate and picks the best; random picks any untested one.

Result, mean over 20 seeds (pool max titer 0.111):

| after n picks | BO best-found | random best-found |
| --- | --- | --- |
| init + 10 | 0.029 | 0.034 |
| init + 20 | 0.031 | 0.054 |
| init + 47 | 0.035 | 0.077 |

**On this dataset, BO does worse than random.**
It ends at 0.035 vs random's 0.077, and beats random in only 7 of 20 seeds.
A client would have found higher-titer media faster by randomly trying untested recipes than by following the model.

This is not the harness cheating.
The identical harness on a clean synthetic pool of the same size shows BO winning (BO 6.33 vs random 5.60).
It is the data: the real titers are tiny and zero-inflated - **21% of runs are non-producers (titer near 0)**, max only 0.111 - so the GP fits a spiky feasible/infeasible surface poorly and over-exploits a noisy incumbent, while random keeps exploring and stumbles onto the good rows.

**Bluntly:** the core promise ("BO finds better recipes in fewer experiments") is real on clean signal but is NOT yet supported on this real dataset.
The lever is not a better acquisition function, it is the data: a **feasibility classifier** to model the zero-inflation (the non-producers), and **replicates / higher signal-to-noise** so the surface is learnable at all.
This validates the roadmap - feasibility labels and noise/replicates are the real work, not model tuning - and it means the product must not claim BO superiority on data in this regime.

## Update: we built the feasibility classifier and tested the hypothesis

The section above proposed two levers: a feasibility classifier for the zero-inflation, and replicates / higher signal-to-noise.
We built the first one (`kalos/core/feasibility.py`: a calibrated, cold-start-safe producer/non-producer classifier) and gated the pool-based acquisition with it (`bo_feas` gates EI by predicted P(feasible); `bo_feas_clean` also fits the GP on producer-only points).
Then we tested it on the same real dataset, with leakage-safe features (9 media + pH inputs; the measured outputs `Size`, `Conc.`, `% Purity` are excluded), 30 seeds.

Two things came back, and together they are decisive.

**Feasibility is highly learnable here.**
Grouped-CV AUC for producer vs non-producer is **0.891**.
The classifier is excellent at telling a producing recipe from a non-producer.

**But gating on it does not rescue BO.**

| strategy | final best-found (mean) | win-rate vs random | speed (AUC of best-found curve) |
| --- | --- | --- | --- |
| bo | 0.031 | 10% | 0.236 |
| bo_feas | 0.031 | 10% | 0.236 |
| bo_feas_clean | 0.046 | 20% | 0.247 |
| random | **0.082** | - | **0.439** |

`bo_feas` is identical to plain `bo`, and `bo_feas_clean` improves only marginally and still loses to random by roughly 2x.

**Why an AUC of 0.89 does not help: the bottleneck is not feasibility.**
The GP already avoids the zeros on its own, which is exactly why gating changes nothing (`bo_feas` equals `bo`): its posterior mean over a zero-inflated response already down-weights the non-producers.
What the GP cannot do is rank the producers, because among producing recipes the titer signal is too weak relative to the measurement noise to tell a 0.03 recipe from a 0.11 one.
Random exploration stumbles onto the high-titer rows faster than Expected Improvement, which over-exploits a noisy incumbent.
Perfect feasibility discrimination cannot fix an unlearnable ranking signal.

**Conclusion.**
The feasibility classifier is correct and worth keeping (it now feeds the `feasibility_auc` promotion gate with a real number, and `bo_feas_clean` is a small honest improvement), but it is **not** the lever for the real-data loss.
The remaining lever is the data: **replicates and higher signal-to-noise** so the producer-titer surface is learnable at all.
The product must not claim BO superiority on data in this regime, and the next real work is noise/replicate modeling, not a better acquisition function or classifier.

## Caveats

- The synthetic surfaces measure whether the optimization machinery beats space-filling in a controlled setting; they do not claim a specific number of experiments saved on real data.
- The real-data pool benchmark uses client data that is NOT committed; `kalos/bench/pool.py` takes a DataFrame, so the code stays reproducible and data-free. Point it at a run sheet to reproduce the numbers above.
- The real-data numbers in the update section are reproducible from the private `bioqore-data` store (referenced via the `BIOQORE_DATA` env var); no client data is committed to this repo.
- Noise (synthetic) is expressed as a fraction of each surface's output scale, so it is comparable across surfaces with different units.
