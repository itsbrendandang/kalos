# Scale-up evidence: reconciling the v0 `extrapolate_up` claim

This note exists because two committed sources disagreed about the same
result. The commit message for `ef509a7` (PR #44, "ScaleBridge v0 -
physics-informed scale transfer, honestly evaluated") states that on upward
extrapolation to 2000 L, v0 beat both naive baselines on MAE (0.53 vs
1.4-1.5), while `kalos/scale/evaluation.py`'s module docstring (the
"HONESTY CLAUSE") said v0 did NOT clearly beat the naive baselines on
`extrapolate_up`.

This note reruns the evaluation the commit message describes, records the
actual measured numbers, and is the thing both the docstring and the
Scale-Up Readout design doc now point to. It is a Dependency of that design
doc (see its "Dependencies" section, and Code Quality Review finding 2).

## Dataset

`itsbrendandang/kalos-data`, `datasets/mab-scaleup-synthetic/` (private repo;
`KALOS_DATA` env var contract, same as `tests/test_scale_integration.py` and
`tests/test_scale_v1_real_data.py`). Fully **synthetic** - generated
2026-06-23 by an internal simulator, no client data (see that dataset's own
`DATA.md`). 55 fed-batch CHO mAb production batches spanning 0.01 L to
2000 L working volume across 11 distinct scales:
`[0.01, 0.05, 0.25, 1.0, 5.0, 10.0, 50.0, 200.0, 500.0, 1000.0, 2000.0]`.

Dataset commit at the time of this run: `3b298dd` (2026-08-25).

## Candidate, features, bounds

- Candidate: **v0** (`kalos.core.surrogate.Surrogate`, the default model
  `leave_one_scale_out_report` fits when `model_factory` is not passed).
- Process columns: `["ph_setpoint", "dissolved_oxygen_pct",
  "temperature_C"]` (`agitation_rpm` and `airflow_L_per_min` are derived as
  the per-batch time-average over each batch's sensor series, matching
  `tests/test_scale_integration.py`'s `_load_real_dataset`).
- Physics scale features: `kalos.scale.transfer.build_feature_matrix`'s
  default set (`DEFAULT_SCALE_FEATURE_CONFIG`).
- Bounds: **not passed** (`bounds=None`) - the function's default box, the
  min/max envelope of every surviving row (train and held-out together),
  same as every other call in this repo's test suite that does not pass
  `bounds` explicitly.
- `n` per bucket: 55 rows total across 11 scales; the `extrapolate_up`
  bucket is the single largest scale (2000 L), so **n=5** (5 replicate runs
  at 2000 L) - see "Known limitation" below.

## Measured result

Reproduced exactly on this run (`kalos` at `83b4390`, the tip of
`origin/main` this branch is based on; `torch==2.14.0`,
`botorch==0.18.1`, `gpytorch==1.15.2`):

| Bucket | v0 MAE | naive_mean MAE | naive_nn MAE | v0 beats naive_mean | v0 beats naive_nn |
|---|---|---|---|---|---|
| `extrapolate_up` (n=5, 2000 L) | 0.5334 | 1.5100 | 1.3820 | **True** | **True** |
| `overall` (pooled, n=55) | 0.6625 | 1.7740 | 1.4195 | **True** | **True** |

Ranking (Spearman), `extrapolate_up`: v0 = **-0.7**, at **n=5**. The exact
permutation-null p-value for this statistic at n=5 (120 permutations of the
held-out order) is **0.2333** - indistinguishable from noise. This matches
the commit message's stated `-0.7` and `p=0.233` exactly.

**Conclusion, stated the way `ef509a7`'s later "v0 vindicated by the data"
sub-commit already reframed it**: v0 measurably beats both naive baselines
on MAE at the actual extrapolation distance the product sells (2000 L), on
this dataset, with these bounds. Its ranking at that same bucket is
**unmeasured, not demonstrated bad** - n=5 is too few points for a rank
correlation to be a measurement (`p=0.233` is consistent with pure chance),
and a mandatory rank-crossing check on this same (synthetic) dataset (see
`tests/test_scale_v1_real_data.py`) independently found no statistically
detectable recipe-by-scale interaction to rank in the first place
(`F(3,47)=0.917, p=0.44`). The original module docstring's "v0 did NOT
clearly beat the naive baselines on extrapolate_up" was wrong about the MAE
comparison specifically; it has been corrected (see below).

## Known limitation this note does not paper over

`extrapolate_up` is, by construction, always exactly the single largest
trained scale's own replicates (every other held-out scale has a larger
training scale, so it falls in `interpolate` or `extrapolate_down`). On this
dataset that is 5 runs at 2000 L. Five points is enough to measure an MAE
comparison against two naive baselines, but not enough to measure a rank
correlation - which is exactly why the ranking claim stays "unmeasured"
rather than "good" or "bad" for either number. This is also why the Scale-Up
Readout design's ladder backtest (`leave_one_scale_out_report(...,
include_oof=True, held_out_scales=...)`) pools residuals across multiple
scale-transition rungs rather than relying on any single bucket like this
one.

## Reproduce this

```bash
export KALOS_DATA=/path/to/itsbrendandang/kalos-data   # checkout root
.venv/bin/python -c "
import os
import pandas as pd
from pathlib import Path
from kalos.scale.evaluation import leave_one_scale_out_report

dataset_dir = Path(os.environ['KALOS_DATA']) / 'datasets' / 'mab-scaleup-synthetic'
batches = pd.read_csv(dataset_dir / 'batches.csv')
agitation_means, airflow_means = [], []
for batch_id in batches['batch_id']:
    sensors = pd.read_csv(dataset_dir / 'sensors' / f'{batch_id}_sensors.csv')
    agitation_means.append(sensors['agitation_rpm'].mean())
    airflow_means.append(sensors['airflow_L_per_min'].mean())
batches['agitation_rpm'] = agitation_means
batches['airflow_L_per_min'] = airflow_means

report = leave_one_scale_out_report(
    batches, 'titer_g_per_L', ['ph_setpoint', 'dissolved_oxygen_pct', 'temperature_C']
)
print(report['by_direction']['extrapolate_up'])
print(report['overall'])
"
```

Or, equivalently, run the already-committed real-data suites with the same
env var set: `KALOS_DATA=/path/to/kalos-data .venv/bin/python -m pytest
tests/test_scale_integration.py tests/test_scale_v1_real_data.py`.

The exact permutation p-value is computed by enumerating every permutation
of the 5 held-out actual values against the 5 fixed predictions and taking
the fraction whose `|Spearman rho|` is at least the observed `0.7` - the
same method `tests/test_scale_v1_real_data.py` already uses and asserts a
qualitative (not exact-digit) bound on.
