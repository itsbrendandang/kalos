"""End-to-end smoke test on synthetic data: train (folder of CSVs) -> artifact
-> predict titer -> output, plus determinism. No real data required."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline import cli

FEATURES = [f"DO_f{i}" for i in range(8)] + [f"pH_f{i}" for i in range(4)]


def _make_wells(rng, prefix, n, with_titer):
    rows = []
    for i in range(n):
        feats = rng.normal(size=len(FEATURES))
        row = {"clone": f"{prefix}_c{i}", "well_id": f"{prefix}_w{i}"}
        row.update(dict(zip(FEATURES, feats)))
        if with_titer:  # a real (noisy) titer signal so CV has something to find
            row["titer"] = float(50 + feats[0] * 8 + feats[3] * 4 + rng.normal(scale=2))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_dataset(tmp: Path) -> Path:
    rng = np.random.default_rng(0)
    train_dir = tmp / "training_data"
    train_dir.mkdir()
    # two training files with DIFFERENT feature columns -> exercises the union
    _make_wells(rng, "a", 8, with_titer=True).to_csv(train_dir / "sourceA.csv", index=False)
    fileB = _make_wells(rng, "b", 6, with_titer=True).drop(columns=["pH_f3"])
    fileB.to_csv(train_dir / "sourceB.csv", index=False)

    predict_dir = tmp / "prediction_data"
    predict_dir.mkdir()
    _make_wells(rng, "p", 10, with_titer=False).to_csv(predict_dir / "predict.csv", index=False)

    cfg = {
        "seed": 20260712,
        "artifact_path": str(tmp / "artifacts" / "model.joblib"),
        "data": {"train_dir": str(train_dir),
                 "predict_csv": str(predict_dir / "predict.csv"),
                 "output_csv": str(tmp / "outputs" / "predictions.csv")},
        "columns": {"target": "titer", "id_col": "well_id", "group_col": "clone"},
        "features": {"exclude": []},
        "models": ["point_gb", "bootstrap_ensemble", "gaussian_process",
                   "bayesian_ridge", "copula_augmented"],
        "reference_model": "point_gb",
        "outputs": {"analysis": True, "visualize": True, "n_top": 3, "top_features": 5},
        "preprocess": {"max_numeric_features": 8},
        "cv": {"n_boot": 200},
    }
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_train_predict_end_to_end(tmp_path):
    import joblib
    config = _write_dataset(tmp_path)

    cli.main(["train", "--config", str(config)])
    artifact = tmp_path / "artifacts" / "model.joblib"
    assert artifact.exists()
    # reference block persisted (drives the figures)
    bundle = joblib.load(artifact)
    assert bundle["version"] == 2
    ref = bundle["reference"]
    assert ref["model_name"] == "point_gb"
    assert "cv_spearman" in ref and ref["importance"] and "titer_mean" in ref

    cli.main(["predict", "--config", str(config)])
    outdir = tmp_path / "outputs"
    out = pd.read_csv(outdir / "predictions.csv")

    # one row per prediction well, expected columns present
    assert len(out) == 10
    for col in ("well_id", "predicted_titer", "blend_rank", "confidence_tier"):
        assert col in out.columns
    for m in ("point_gb", "gaussian_process"):
        assert f"{m}_pred" in out.columns
    assert out["predicted_titer"].between(0.0, 200.0).all()

    # new outputs: manifest, analysis, the two figures, and parity CSVs
    assert (outdir / "predictions_manifest.json").exists()
    assert (outdir / "predictions_analysis.md").exists()
    assert (outdir / "predictions_figures" / "fig1_ranking.png").exists()
    assert (outdir / "predictions_figures" / "fig2_drivers.png").exists()
    ranking = pd.read_csv(outdir / "predictions_clone_ranking.csv")
    assert list(ranking.columns) == ["well_id", "pred_titer", "rank"] and len(ranking) == 10
    drivers = pd.read_csv(outdir / "predictions_drivers.csv")
    assert list(drivers.columns) == ["feature", "importance"] and drivers["importance"].notna().all()


def test_include_sources_filters_training_pool(tmp_path):
    """data.include_sources restricts TrainingPool.from_dir to rows whose
    source_col matches -- used to keep a reference model's training data in
    the same measurement/duration class as the cohort it ranks."""
    from pipeline.data import TrainingPool

    train_dir = tmp_path / "training_data"
    train_dir.mkdir()
    pd.DataFrame({
        "well_id": ["a1", "a2", "b1"],
        "source": ["A", "A", "B"],
        "titer": [10.0, 12.0, 1000.0],
        "f1": [1.0, 2.0, 3.0],
    }).to_csv(train_dir / "pooled.csv", index=False)

    base_cols = {"target": "titer", "id_col": "well_id", "source_col": "source"}
    filtered = TrainingPool.from_dir(train_dir, {"columns": base_cols, "data": {"include_sources": ["A"]}})
    assert filtered.n_rows == 2
    assert set(filtered.df["source"]) == {"A"}

    unfiltered = TrainingPool.from_dir(train_dir, {"columns": base_cols, "data": {}})
    assert unfiltered.n_rows == 3


def test_load_predict_cohort_rejects_duplicate_well_id(tmp_path):
    """A duplicate row key silently misaligns per-well predictions and the propose
    diversity batch, so the cohort loader must reject it up front."""
    import pytest

    from pipeline.data import load_predict_cohort

    csv = tmp_path / "predict.csv"
    pd.DataFrame({"well_id": ["a", "b", "a"], "f1": [1.0, 2.0, 3.0]}).to_csv(csv, index=False)
    cfg = {"columns": {"id_col": "well_id"}, "data": {"predict_csv": str(csv)}}
    with pytest.raises(ValueError, match="duplicate well_id"):
        load_predict_cohort(cfg)


def test_predict_is_deterministic(tmp_path):
    config = _write_dataset(tmp_path)
    cli.main(["train", "--config", str(config)])
    cli.main(["predict", "--config", str(config)])
    first = (tmp_path / "outputs" / "predictions.csv").read_text()
    cli.main(["predict", "--config", str(config)])
    second = (tmp_path / "outputs" / "predictions.csv").read_text()
    assert first == second
