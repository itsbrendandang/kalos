"""Terminal interface: `train` fits + persists a model artifact; `predict`
loads it and scores a cohort CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import artifact
from .config import load_config
from .pipeline import PredictPipeline, TrainPipeline


def _apply_overrides(cfg: dict, args) -> None:
    if getattr(args, "train_dir", None):
        cfg["data"]["train_dir"] = args.train_dir
    if getattr(args, "predict_csv", None):
        cfg["data"]["predict_csv"] = args.predict_csv
    if getattr(args, "output", None):
        cfg["data"]["output_csv"] = args.output
    if getattr(args, "artifact", None):
        cfg["artifact_path"] = args.artifact


def _artifact_path(cfg: dict) -> str:
    path = cfg.get("artifact_path")
    if not path:
        raise SystemExit("artifact_path is not set in config (or pass --artifact)")
    return path


def cmd_ingest(args) -> None:
    from . import ingest

    raw_train_dir = Path(args.raw_train_dir)
    if raw_train_dir.exists():
        df = ingest.write_combined_training_table(raw_train_dir, args.train_out)
        print(f"{args.train_out}: {len(df)} rows x {df.shape[1]} cols "
              f"(sources: {', '.join(sorted(df['source_dataset'].unique()))})")
    else:
        print(f"skip: {raw_train_dir} not found")

    raw_predict_dir = Path(args.raw_predict_dir)
    if raw_predict_dir.exists() and any(raw_predict_dir.glob("*.csv")):
        df = ingest.write_predict_table(raw_predict_dir, args.predict_out)
        print(f"{args.predict_out}: {len(df)} rows x {df.shape[1]} cols")
    else:
        print(f"skip: no CSVs found in {raw_predict_dir}")


def cmd_train(args) -> None:
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)

    pipeline = TrainPipeline(cfg)
    methods, feature_cols, reference = pipeline.run()
    out = pipeline.save(_artifact_path(cfg), methods, feature_cols, reference)

    print(f"Trained on {methods[0].n_train} rows, {methods[0].n_unique_groups} groups, "
          f"{len(feature_cols)} candidate features.")
    print("\nMethod comparison (CV Spearman, bootstrap 95% CI, verdict):")
    for m in methods:
        print(f"  {m.name:22s} n={m.n_train:3d} feats={m.n_features:2d} "
              f"rho={m.cv_spearman:+.3f} CI=[{m.ci_low:+.3f}, {m.ci_high:+.3f}] -> {m.verdict}")
    print(f"\nReference model '{reference['model_name']}': CV Spearman "
          f"{reference['cv_spearman']:+.3f} (drives the ranking / driver figures).")
    print(f"Wrote artifact: {out}")


def cmd_predict(args) -> None:
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)

    bundle = artifact.load(_artifact_path(cfg))
    report = PredictPipeline(bundle, cfg).run()

    output_csv = Path(cfg["data"].get("output_csv", "outputs/predictions.csv"))
    outdir, stem = output_csv.parent, output_csv.stem
    # All sidecar artifacts follow the output basename so different cohorts never
    # clobber each other's manifest / analysis / figures.
    written = report.save(output_csv, outdir / f"{stem}_manifest.json")

    print("Blend weights ({}):".format(report.note))
    for name, w in sorted(report.weights.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} weight={w:.3f}")

    print("\nTop predictions:")
    top = report.predictions_table().head(10)
    show = [c for c in ("well_id", "predicted_titer", "blend_rank", "confidence_tier") if c in top.columns]
    print(top[show].to_string(index=False))

    outputs_cfg = cfg.get("outputs", {})
    if outputs_cfg.get("analysis", True):
        from . import analysis
        ref = report.reference or {}
        if ref.get("cv_spearman") is not None:
            print(f"\nReference model '{ref['model_name']}': CV Spearman {ref['cv_spearman']:+.3f}")
        nov = analysis.novelty(report)
        if nov:
            print(f"WARNING: {len(nov)} feature(s) off-distribution vs training (|z|>3): "
                  f"{', '.join(nov)}")
        written["analysis"] = analysis.write_analysis(report, outdir / f"{stem}_analysis.md")
        if outputs_cfg.get("visualize", True):
            for p in analysis.write_visualizations(report, outdir / f"{stem}_figures", cfg):
                written[f"figure:{p.name}"] = p

    print("\nWrote:")
    for label, path in written.items():
        print(f"  {label}: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline",
                                     description="Train on a folder of CSVs / predict titer for a cohort.")
    sub = parser.add_subparsers(dest="command", required=True)

    i = sub.add_parser("ingest", help="engineer features from raw CSVs into training_data/ + prediction_data/")
    i.add_argument("--raw-train-dir", default="raw_training_data")
    i.add_argument("--raw-predict-dir", default="raw_prediction_data")
    i.add_argument("--train-out", default="training_data/combined_ml_training_table.csv")
    i.add_argument("--predict-out", default="prediction_data/predict.csv")
    i.set_defaults(func=cmd_ingest)

    t = sub.add_parser("train", help="fit models on the training folder and save an artifact")
    t.add_argument("--config", required=True, help="path to the YAML config")
    t.add_argument("--train-dir", help="override data.train_dir")
    t.add_argument("--artifact", help="override artifact_path")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="load an artifact and score the prediction CSV")
    p.add_argument("--config", required=True, help="path to the YAML config")
    p.add_argument("--predict-csv", help="override data.predict_csv")
    p.add_argument("--artifact", help="override artifact_path")
    p.add_argument("--output", help="override data.output_csv")
    p.set_defaults(func=cmd_predict)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
