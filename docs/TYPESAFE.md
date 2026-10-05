# TypeSafe decisions

[TypeSafe](https://typesafe.ai)'s System One model (Jev) returns typed judgments with
probabilities instead of generated text. Kalos uses it where ordinary code needs semantic
understanding - reading what a client's column *means* - and nowhere else:

| Question per column | TypeSafe primitive | How code uses the answer |
| --- | --- | --- |
| What role does it play: target, feature, group, metadata, free text? | **Choice** | acted on only above `KALOS_TYPESAFE_MIN_CONFIDENCE` (default 0.6); below it the offline guess is kept |
| Does it identify a client, a person, or a sample? | **Noul** | dropped as identity at p >= 0.5 - a privacy gate, not a preference |
| Which known canonical name is it (or its own)? | **Choice** over names code builds | select, never generate; exact aliases are resolved in code and never asked |

Code keeps the rules: units are an exact registry lookup, at most one target survives (a second
"outcome" becomes metadata, never an input), name collisions are resolved deterministically, and
every decision's probabilities are written into the plan's `note` for audit. The model only ever
sees the identity-screened payload from `normalize/payload.py`. Any failure (no key, network,
malformed answer) falls back to the offline plan - TypeSafe can improve the decision, never block
an upload.

```bash
python -m pip install -e ".[typesafe]"     # typesafe-sdk
export KALOS_LLM_PROVIDER=typesafe TYPESAFE_API_KEY=...
python -c "import pandas as pd; from kalos.normalize import propose_plan; print(propose_plan(pd.read_csv('sheet.csv')).to_json())"
curl -F file=@sheet.csv -F roles=auto http://127.0.0.1:8050/api/run   # TypeSafe-decided roles
```

`GET /api/providers` reports whether the key is present. The Claude Code plugin
(`typesafe@typesafe-ai`) carries TypeSafe's own design guidance for extending this tier.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `KALOS_LLM_PROVIDER` | `anthropic` | `typesafe` selects this tier (`ollama`, `none` are the others) |
| `TYPESAFE_API_KEY` | unset | required for the live tier; unset means the offline plan runs |
| `KALOS_TYPESAFE_MODEL` | `jev-latest` | the System One model alias |
| `KALOS_TYPESAFE_MIN_CONFIDENCE` | `0.6` | below this Choice confidence a role or name falls back to the offline guess |

A confident "target" on a column that is not numeric is kept as metadata, never
the target: the optimizer can only maximize a number. Code lives in
`src/kalos/normalize/typesafe_tier.py`; tests in `tests/test_typesafe_tier.py`.
