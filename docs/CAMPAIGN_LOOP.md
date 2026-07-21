# Campaign loop

The closed optimization loop behind the kalos-web `/decide` surface.

## Why

The engine proposes a batch of recipes, but until a scientist can run those recipes, record what they measured, and get the *next* batch that accounts for those results, kalos is a one-shot analysis viewer, not a product.
This is the loop that closes: propose -> run -> log outcome -> re-propose.

It deliberately reuses the existing analysis path (`_analyze` in `kalos/portal/analysis.py`) and adds no new engine capability.
A campaign is just a growing dataset that gets re-analyzed each round.

## Concept

A **campaign** is one optimization target plus a growing table of (recipe -> measured outcome) rows.

- **base rows**: the dataset the analysis fits on. Seeded from the last `/api/run` upload, then grows as logged runs are folded in.
- **pending runs**: recipes the scientist started from a proposed batch. Each is either *awaiting* a measured outcome (`result: null`) or *measured* (`result` filled), waiting to be folded into base on the next re-analyze.

One campaign at a time (single local user). It lives in `~/.kalos/campaign.json`, guarded by a lock and written atomically, mirroring the `_LATEST` state pattern in `kalos/portal/app.py`.

## State shape (`~/.kalos/campaign.json`)

```json
{
  "target": "lipase_titer",
  "features": ["pH", "Methanol"],
  "base_rows": [{"pH": 6.5, "Methanol": 2.0, "lipase_titer": 4.6}],
  "pending": [
    {
      "id": "b1e2...",
      "recipe": {"pH": 6.46, "Methanol": 2.29},
      "pred": 1.58, "std": 0.33, "mode": "explore", "reason": "diversifies the batch",
      "result": null,
      "created_at": 1690000000.0,
      "measured_at": null
    }
  ],
  "round": 2,
  "updated_at": 1690000000.0
}
```

`features` and `target` come from the analysis result (`proposal_features`, `target`), so the campaign never guesses column roles - it takes them from the engine.

## Seeding

The campaign base is seeded/replaced whenever `/api/run` succeeds.
`_run_uploaded_sync` (`kalos/portal/app.py`) already holds the parsed `df` and calls `_analyze` then `_save_latest`; it also seeds the campaign from `(df, result["target"], result["proposal_features"])`.
A fresh upload starts a fresh campaign (new base, empty pending, round 0).

## Endpoints (`/api/campaign` router, mounted like `/api/experiments`)

| Method | Path | Body | Does |
| --- | --- | --- | --- |
| GET | `/api/campaign` | - | Campaign summary for the `/decide` view (below), or `{"has_campaign": false}` before any upload. |
| POST | `/api/campaign/start` | `{"recipes": [{recipe, pred, std, mode, reason}]}` | Append each recipe as a pending awaiting run. Returns the appended runs with ids. |
| POST | `/api/campaign/result` | `{"id": "...", "value": 3.2}` | Set the measured outcome on a pending run. 400 on unknown id or non-finite value. |
| POST | `/api/campaign/reanalyze` | - | Fold every measured pending run into base rows, drop them from pending, increment `round`, rebuild the DataFrame, run `_analyze`, `_save_latest`. Returns `{analysis, campaign}`. Awaiting (unmeasured) runs stay pending. |

`GET /api/campaign` response:

```json
{
  "has_campaign": true,
  "target": "lipase_titer",
  "features": ["pH", "Methanol"],
  "n_base": 40,
  "best": 5.9,
  "round": 2,
  "pending": [{ "id": "...", "recipe": {...}, "pred": 1.58, "std": 0.33,
               "mode": "explore", "reason": "...", "result": null, "awaiting": true }],
  "n_awaiting": 3,
  "n_measured": 0
}
```

`best` is `max(target over base_rows)` - the best *measured* value so far, not a prediction.

## Re-analyze = the loop closing

`reanalyze` is the whole point:

1. Every pending run with a non-null `result` becomes a base row: `{**recipe, target: result}`.
2. Those runs leave `pending`; awaiting runs stay.
3. `round += 1`.
4. Build `df = DataFrame(base_rows)`, call `_analyze(df, target, profile=BIOPROCESS_PROFILE)`, then `_save_latest(result, "campaign round N")`.
5. Return the fresh analysis (same shape `/api/latest` returns) plus the updated campaign.

The next `/api/latest` and the next `GET /api/campaign` both reflect the folded-in data, so `/decide` shows a batch that accounts for the results just logged.
This adds no engine capability - it is the same `_analyze` the upload path runs, on a dataset that grew by one round.

## Honesty constraints (do not regress)

- `best` is measured, never predicted.
- A run is only foldable once it has a real measured `result`; an awaiting run never silently becomes a data point.
- Re-analyze routes through the same leakage-controlled `_analyze`, so grouped-CV reliability, conformal bands, and the "not modeled" callouts stay first-class each round.

## Frontend contract (kalos-web)

`lib/api.ts` gains a campaign client: `getCampaign()`, `startRuns(recipes)`, `logResult(id, value)`, `reanalyze()`.
`/decide` uses it to turn the prototype's local Queue/Start state into the real loop:

1. **Start** the queued proposals -> `POST /api/campaign/start`.
2. A **campaign panel** lists the real pending runs (`GET /api/campaign`) with a measured-value input per run -> `POST /api/campaign/result`.
3. Once runs are measured, **Re-propose with N results** -> `POST /api/campaign/reanalyze`, then refresh `getLatest()` (new batch) and `getCampaign()`.
4. Best-so-far and round come from `GET /api/campaign`.
