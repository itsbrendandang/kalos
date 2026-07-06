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

## Caveats

- These are synthetic surfaces, so they measure whether the optimization machinery beats space-filling in a controlled setting; they do not claim a specific number of experiments saved on any real dataset.
- A pool-based retrospective benchmark on committed real data is a natural follow-up once a client dataset is checked into the repo under an appropriate data agreement.
- Noise is expressed as a fraction of each surface's output scale, so it is comparable across surfaces with different units.
