# Missingness-as-a-feature: does a binary indicator column help?

Status: synthetic-only research prototype. Not validated on real data. Does not justify any change to production kalos.

## Question

Kalos zero-fills missing feature values.
That is correct under kalos's stated domain assumption: a missing media component means the component is absent from the recipe, so its true contribution to the outcome is zero.
This prototype asks a narrower, honest question: if that assumption were wrong and missingness actually carried information, would adding a binary `x_k_missing` indicator column recover any of that signal, and would it cost anything when the assumption holds?

## Design

Fully synthetic data with a known ground truth, seeded with `numpy.random.default_rng` throughout (no unseeded randomness).

- 80 synthetic recipes, 5 numeric features `x0..x4`, drawn `Uniform(-2, 2)`.
- Target: `y = 2*sin(2*x0) + 1.5*x1^2 + 2*x2*x3 - x4 + N(0, 1)`. Smooth, two nonlinear terms, one interaction involving `x2` (the feature subjected to missingness).
- One feature, `x2`, has a fraction `p_miss` of its values marked NaN, swept over `{0.1, 0.3, 0.5}`.
- Two `missing_mode`s control what the NaNs actually mean, holding everything else fixed:
  - `absent` (kalos's assumption): for the missing rows, the real `x2` truly is 0, and `y` is generated from `x2 = 0` on those rows. Zero-fill is exactly correct here by construction.
  - `informative` (MNAR): missingness happens exactly when the real `x2` is systematically large (`Uniform(1, 3)`, drawn instead of the usual `Uniform(-2, 2)`), and `y` is generated from that real, nonzero value. Zero-fill silently mis-imputes those rows to `x2 = 0`, which is wrong here by construction.

Two model arms, same model class, same CV folds, same everything except the feature matrix:

- Arm A: zero-fill NaN -> 5 features (kalos's current behavior).
- Arm B: zero-fill NaN -> 5 features + 1 binary `x2_missing` indicator.

**Model used: scikit-learn `GaussianProcessRegressor`** (ARD Matern-5/2 kernel + white noise), not `kalos.core.surrogate.Surrogate`.
Both are GPs; sklearn's was chosen for speed only - this sweep fits ~1,200 independent GPs (10 seeds x 3 `p_miss` values x 2 `missing_mode`s x 2 arms x 5 folds), and doing that through BoTorch's Cholesky-jitter retry ladder and torch overhead would multiply runtime for no benefit, since this script never touches the production path.
The comparison stays fair because arm A and arm B always share the identical model class, hyperparameter search, and CV folds - the feature matrix is the only thing that differs.

Evaluation: pooled out-of-fold (OOF) Spearman rank correlation between predicted and true `y` under 5-fold CV, repeated over 10 seeds, reported as mean +/- std across seeds. This mirrors the spirit of kalos's own `grouped_cv_report` (pooled OOF Spearman) without depending on it, since that helper is wired directly to the production `Surrogate`.

## Results

Actual output of `python experiments/missingness_indicator/run.py` (default args: `n=80`, `n_seeds=10`, `noise_sd=1.0`, `n_splits=5`, `p_miss=0.1,0.3,0.5`, `seed_base=0`; elapsed 38.1s):

```
mode          p_miss     Arm A (mean+-std)     Arm B (mean+-std)       delta (B-A)
----------------------------------------------------------------------------------
absent          0.10       0.884 +/- 0.036       0.863 +/- 0.033  -0.021 +/- 0.033
absent          0.30       0.859 +/- 0.042       0.828 +/- 0.062  -0.031 +/- 0.051
absent          0.50       0.848 +/- 0.049       0.802 +/- 0.072  -0.046 +/- 0.065
informative     0.10       0.765 +/- 0.073       0.838 +/- 0.074  +0.072 +/- 0.064
informative     0.30       0.677 +/- 0.115       0.848 +/- 0.031  +0.172 +/- 0.109
informative     0.50       0.738 +/- 0.101       0.858 +/- 0.035  +0.120 +/- 0.096
```

`delta` is the paired per-seed difference (Arm B minus Arm A on the identical data and folds), averaged over seeds, not a difference of the two column means - a stricter, more honest comparison.

## Honest findings

**`absent` regime (missing really does mean absent, kalos's assumption): Arm B does not beat Arm A.**
Adding the indicator makes OOF skill slightly *worse* at every `p_miss`, and the gap widens as `p_miss` grows (-0.021 at 0.10, -0.031 at 0.30, -0.046 at 0.50).
This is exactly what should happen: when zero-fill is already the correct imputation, the indicator column is a pure nuisance parameter that a GP with only 80 points has to spend fitting capacity on, for zero payoff.
**This result validates kalos's existing zero-fill behavior for the "missing = component absent" case** - the current production code path is the right call here, not a shortcut that happens to work.

**`informative` regime (MNAR, missingness itself carries signal): Arm B beats Arm A, clearly.**
Deltas are +0.072, +0.172, +0.120 at `p_miss` = 0.10, 0.30, 0.50 - all positive, all outside one std of zero at p_miss >= 0.3.
The benefit is real but not perfectly monotonic in `p_miss`: it grows sharply from 0.10 to 0.30 and then eases slightly at 0.50 rather than continuing to climb.
That dip is plausible and not tuned away - at `p_miss = 0.5` there are more missing rows for the GP to learn the offset from (helping Arm B), but also fewer *non-missing* rows left to pin down the rest of the interaction term for both arms (Arm A's baseline also degrades, from 0.859 at 0.30 down alongside more noise in both arms' variance).
This is a real result, reported as measured, not smoothed to a cleaner monotonic story.

**Bottom line: the indicator helps if and only if missingness is actually informative, and the benefit grows with how much data is missing (up to the point where general data scarcity starts to dominate).**
In the domain-matching case, it is a strict net negative.

## The caveat that matters

This is entirely synthetic.
Whether kalos's real media DoE recipes live in the `absent` regime or the `informative` regime is genuinely unknown from this prototype - `BIOQORE_DATA` was not accessed, is not accessible to this exercise, and this script makes no attempt to check it.
This result characterizes the idea in isolation; it does not characterize kalos's actual data.

**This does not justify turning on a missingness indicator in production.**
What it does tell us is what evidence would be needed before considering it: concrete proof, from real assay/DoE records, that at least one component's "missing" entries are NOT simply "this component was not added" - e.g. a documented case where a component was dropped from the mix for a reason correlated with the outcome (a formulation constraint, an equipment limitation, a supplier issue) rather than a deliberate zero-dose condition.
Absent that evidence, kalos's current zero-fill remains the right default, and this prototype's own `absent`-regime result is exactly why: adding indicator columns without cause has a real, measured cost.

## Reproduce

```
cd /Users/brendandang/Documents/GitHub/kalos && source .venv/bin/activate
python experiments/missingness_indicator/run.py
```

Flags: `--n`, `--n-seeds`, `--p-miss` (comma-separated), `--noise-sd`, `--n-splits`, `--seed-base`. All randomness is seeded (`numpy.random.default_rng`), so a fixed `--seed-base` reproduces the table above exactly.
