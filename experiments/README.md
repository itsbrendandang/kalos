# experiments/

This directory holds off-path research prototypes.
It is intentionally NOT part of the `kalos` package (see `pyproject.toml`, which only packages `kalos*`) and carries none of the package's guarantees.
Nothing here is validated on real data or wired into the production engine; treat every result as a synthetic characterization of an idea, not a production recommendation.

## Contents

- `missingness_indicator/` - synthetic characterization of a missingness-indicator feature.
- `leadgene_pipeline/` - a self-contained titer-prediction/well-ranking pipeline (config-driven ingest/train/predict, a five-model cross-validated blend, a bioprocess feature extractor, and an honesty/verdict layer). Standalone: it keeps its own `pipeline/` package, `config/`, and `tests/`, and is run in place (`cd leadgene_pipeline && python -m pipeline --help`). See its `README.md`. Ported from the `ml-leadgene-pipeline` tool; not yet validated inside kalos.
