# Kalos documentation

Long-form write-ups that sit behind the [root README](../README.md). The README
says what Kalos does and how to run it; these are the contracts, results, and
decisions in depth.

**Using and extending the engine**

| Doc | What it covers | Read it when |
| --- | --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The module map of `src/kalos/`, the experiment store, the campaign loop in brief, other domains, device notes | finding your way around the code |
| [API.md](API.md) | `POST /api/run`: upload guarantees, provenance, validation, `roles=auto`, the XGBoost baseline | calling the engine or reading its response |
| [TYPESAFE.md](TYPESAFE.md) | TypeSafe column decisions: the questions asked, the policy code applies, configuration | turning on or extending the TypeSafe tier |

**Contracts, results, and operations**

| Doc | What it covers | Read it when |
| --- | --- | --- |
| [CAMPAIGN_LOOP.md](CAMPAIGN_LOOP.md) | The closed loop behind `/api/campaign*`: state shape, seeding, re-analysis, in-flight runs, honesty constraints | changing the propose -> run -> log -> re-propose flow or the kalos-web `/decide` surface |
| [M2_INTEGRATION.md](M2_INTEGRATION.md) | The experiment store, the `BackendAdapter` seam, the Singleton runner, and the `/api/experiments` contract | integrating with the Voyager platform or the runner |
| [BENCHMARK.md](BENCHMARK.md) | Does BO beat LHS/random at a fixed budget? Synthetic surfaces with known optima, the noise sweep, and the real-data pool retrospective | making any claim about optimizer performance |
| [HARDENING.md](HARDENING.md) | The in-house production track: auth tokens, tenancy, CORS, configuration reference, status log | touching auth, tenancy, or deployment security |
| [DESIGN.md](DESIGN.md) | The engine portal's visual system (kalos blue on Satoshi), tokens, and anti-slop rules | changing `src/kalos/portal/index.html` |
| [ROADMAP.md](ROADMAP.md) | The prioritized backlog: known bugs, structural refactors, product gaps, ops, CI | choosing what to work on next |

Elsewhere in the repo:

- [deploy/README.md](../deploy/README.md) - every container file (production and dev), its build context, and the checks that guard it.
- [deploy/RUNBOOK.md](../deploy/RUNBOOK.md) - deploying, upgrading, rolling back, and the backup/restore drill.
- [experiments/README.md](../experiments/README.md) - off-path research prototypes and their results.
- [CHANGELOG.md](../CHANGELOG.md) - every change, newest first, with the numbers behind it.
- [.env.example](../.env.example) - every environment variable, its default, and what happens when it is unset.

## Where the pipeline's decisions are documented

| Decision | Code | Doc |
| --- | --- | --- |
| Is this upload safe to parse? | `src/kalos/portal/uploads.py` | [API.md](API.md) |
| Is the data physically plausible? | `src/kalos/validation/` | [API.md](API.md), "Data validation gate" |
| What does each column mean? | `src/kalos/normalize/` (offline, LLM, TypeSafe tiers), `src/kalos/domains/` | [TYPESAFE.md](TYPESAFE.md), [ARCHITECTURE.md](ARCHITECTURE.md) "Domains" |
| How reliable is the model? | `src/kalos/core/evaluation.py`, `src/kalos/core/gates.py` | [API.md](API.md) (incl. the XGBoost baseline), [BENCHMARK.md](BENCHMARK.md) |
| What should run next? | `src/kalos/core/optimize.py`, `src/kalos/core/feasibility.py` | [CAMPAIGN_LOOP.md](CAMPAIGN_LOOP.md), [BENCHMARK.md](BENCHMARK.md) |
