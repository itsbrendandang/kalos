# experiments/

Off-path research prototypes. Nothing in this directory is imported by the
`kalos` package, shipped in its wheel, linted or tested by CI, or covered by its
guarantees. A result here is evidence for (or against) a change to kalos, not
the change itself.

| Prototype | Question | Status | Start with |
| --- | --- | --- | --- |
| [missingness_indicator/](missingness_indicator/) | Does a binary "was missing" column recover signal that zero-fill loses? | Synthetic-only; does not justify a production change | [RESULTS.md](missingness_indicator/RESULTS.md) |
| [leadgene_pipeline/](leadgene_pipeline/) | Rank clone wells by predicted titer from DO/pH time courses (a vendored, config-driven port of `Leadgene_Clone_Picker`) | Directional on real data (`USABLE`, n=10); a stand-in acquisition, not kalos BO | [README.md](leadgene_pipeline/README.md), then [TODO.md](leadgene_pipeline/TODO.md) |

## Running them

Each prototype keeps its own dependencies. From the repo root:

```bash
# missingness_indicator: needs only the torch-free kalos core + scikit-learn
python experiments/missingness_indicator/run.py

# leadgene_pipeline: its own requirements, run from its own directory
KALOS_LEADGENE=1 bash scripts/setup-dev.sh        # adds requirements.txt to .venv
cd experiments/leadgene_pipeline && python -m pytest -q
```

## Rules for adding one

- One directory per question, with a results write-up that states its status
  (synthetic-only, real-data, validated) in its first lines.
- Seed every random draw so the result reproduces from committed code.
- Promote into `kalos/` only through a normal change with tests, citing the
  write-up - never by importing from here.
