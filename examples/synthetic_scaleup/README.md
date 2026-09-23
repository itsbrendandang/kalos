# Synthetic scale-up dataset

This is a SYNTHETIC dataset for the Scale-Up Readout demo.
It simulates a bioprocess scale-up effect, not real process data.
Every value comes from a fixed seed, not a measurement.
There is no client or proprietary data in this directory.

## What it is

90 rows across 9 scales (10 runs per scale): `1, 3, 10, 30, 100, 300, 1000, 3000, 5000` liters.
The generator plants a recipe-by-scale interaction: the optimal `ph_setpoint` shifts
from 6.6 at the smallest scale to 7.4 at the largest.
That shift is wide enough to flip which end of the sampled pH range wins between the
smallest and largest scale, so the dataset has a genuine, learnable rank crossing.

This dataset is promoted from `fabricate_harder_synthetic` in
`tests/test_scale_v1_harder_synthetic.py`.
That test module's docstring has the full backstory on why the dataset was built and how
it was used in the scale-up model's v1 promotion decision.

## Columns

- `scale_L`: bioreactor scale, in liters.
- `agitation_rpm`: impeller agitation speed, in rpm.
- `airflow_L_per_min`: sparge airflow, in liters per minute.
- `ph_setpoint`: process input, the pH setpoint for the run.
- `temperature_C`: process input, temperature in Celsius (a nuisance feature, no planted
  interaction with scale).
- `titer_g_per_L`: the target, simulated titer in grams per liter.

`scale_L`, `agitation_rpm`, and `airflow_L_per_min` match the default column names in
`kalos.scale.transfer.ScaleFeatureConfig`, so this fixture plugs into the scale-up
transfer model with no renaming.

## Seed and regeneration

Seed: `20260825`, set in `make_synthetic.py`.
The generator is deterministic: the same seed always produces the same rows.

To regenerate the CSV:

```
.venv/bin/python examples/synthetic_scaleup/make_synthetic.py
```

This overwrites `synthetic_scaleup.csv` with a byte-for-byte identical file, since the
generator is seeded.

## Loading it

```python
from examples.synthetic_scaleup.load import load_frame, load_features

df = load_frame()               # the raw fixture
X, y, feature_names = load_features()  # process + physics-proxy features, ready to fit
```
