# Scale-Up Readout

A one-page answer to one question: what will this process do at the next scale, and should you believe the number?

Upload a multi-scale run sheet and a target scale.
Kalos backtests itself on your own scale ladder, compares against two naive baselines, and either issues a prediction with an interval or refuses and says why.
Refusing is a feature: the readout never claims more than your data supports.

Intended use: development decision support only.
Not a GMP or regulatory record.
Not validated under 21 CFR Part 11.

## Request

`POST /api/scale/readout` (requires the `write` scope), multipart form:

- `file`: the run sheet (CSV or XLSX), parsed by the same hardened uploader as `/api/run`.
  One row per run with the process parameters, `scale_L`, `agitation_rpm`, `airflow_L_per_min`, and the response column.
- `target`: JSON with `scale_L`, `agitation_rpm`, `airflow_L_per_min`, `target_column`, `process_params` (a dict whose keys are the process columns), and optional `physics_overrides` (any `ScaleFeatureConfig` field).

Responses:

- `200 text/html`: the readout page. Save it as PDF from the browser; page 1 holds the decision, and a provenance appendix follows.
- `422`: `{"failed_check": ..., "detail": ...}` when the sheet or target fails the gate or the target JSON is malformed.
- `400`: upload rejected (same rules as `/api/run`).
- `503` + `Retry-After`: another analysis is running (one analysis slot, shared with `/api/run`).

Nothing is stored on the server.
To regenerate a readout, upload the same sheet again; the page prints the SHA-256 of the raw upload so anyone holding the file can verify it.

## How the decision is made

1. **Gate.** At least 3 distinct scales, at least 3 runs per scale, at least 10 rows at the third-smallest scale and above, at most 20% non-finite rows, a target strictly larger than every trained scale, and a target at most 20x the largest trained scale.
   The first failing check is named in the 422.
2. **Ladder backtest.** For each scale whose smaller scales give at least 2 distinct scales and 4 rows, fit on the smaller scales only and predict that scale (`leave_one_scale_out_report(..., include_oof=True, held_out_scales=[k])`).
   The one-scale first rung is skipped and listed.
   Every rung uses the same explicit bounds as the final model, so the backtest evaluates the model that produces the number.
   Those bounds span the training scales and the target, so the same sheet can backtest differently for different targets: on the demo sheet the 10 L rung's MAE is 0.126 for a 7,500 L target and 0.603 for a 40,000 L target.
   For a fixed sheet and target the readout is deterministic.
3. **Decision.**
   The reference step ratio is the largest step ratio among rungs whose own MAE beats both naive baselines.
   - *Prediction issued*: the pooled ladder MAE beats both baselines, there are at least 10 ladder residuals, the target ratio is at most 2x the reference, and every process parameter is inside the trained range.
   - *Prediction issued with warnings*: as above, but a process parameter is out of range or the target ratio is 2x to 5x the reference.
   - *No prediction*: no rung beats both baselines, the pooled ladder does not, fewer than 10 residuals, or the target ratio exceeds 5x the reference.
4. **Interval.** Split-conformal on the pooled ladder residuals, alpha 0.1, labeled *approximate coverage*: exchangeability does not hold for an unseen larger scale, so it is not called calibrated.
   A number is never issued without an interval.

All thresholds are named constants in `kalos/scale/readout.py` and are printed on every readout.

## What it does not claim

It predicts a level at the target scale, not a ranking of recipes there.
Ranking at the target scale is unmeasured at current replication; see `docs/SCALE_EVIDENCE.md`.

## Demo

```bash
.venv/bin/python examples/synthetic_scaleup/make_synthetic.py   # regenerates the committed CSV
```

`examples/synthetic_scaleup/synthetic_scaleup.csv` is synthetic (simulated scale effects, 9 scales from 1 to 5000 L, 10 runs each).
With a 1.5x target (7500 L) it issues a prediction: the pooled ladder MAE is 0.263 vs 0.699 and 0.729 for the naive baselines over 70 residuals, and the reference step ratio is 3.33x.
