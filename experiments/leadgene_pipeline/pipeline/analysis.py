"""Diagnostics + visualizations for a prediction run.

Compute (numeric) is separated from render (matplotlib): `diagnose`,
`within_plate_contrast`, and `novelty` produce plain data; `write_analysis`
(markdown) and `write_visualizations` (PNGs/CSVs) consume it. The two figures
mirror Leadgene_Clone_Picker's rank_external_cohort.py / driver_and_contrast_figure.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .preprocess import cohort_novelty

# Leadgene slide palette.
ORANGE, GRAY, BLUE, LIGHTBLUE = "#d95f02", "#999999", "#1f5fa8", "#4292c6"


# --------------------------------------------------------------------------- #
# Compute
# --------------------------------------------------------------------------- #
def _label(feat: str) -> str:
    return feat.replace("_", " ")


def reference_ranking(report) -> pd.DataFrame:
    """Cohort ranked by the reference model's predicted titer (drives fig1)."""
    ref = report.reference_result()
    if ref is not None:
        df = pd.DataFrame({"well_id": ref.client_well_ids, "pred_titer": ref.client_mean})
    else:  # no reference persisted -> fall back to the blended prediction
        t = report.predictions_table()
        df = t[["well_id", "predicted_titer"]].rename(columns={"predicted_titer": "pred_titer"})
    df = df.sort_values("pred_titer", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def within_plate_contrast(cohort: pd.DataFrame, feats: list[str], top_wells: set) -> list[float]:
    """Top-N minus rest, z-scored within the cohort plate (units of within-plate SD)."""
    out = []
    for feat in feats:
        if feat not in cohort.columns:
            out.append(0.0)
            continue
        vals = pd.to_numeric(cohort[feat], errors="coerce")
        sd = vals.std(ddof=0) or 1.0
        z = (vals - vals.mean()) / sd
        top_z = z[cohort["well_id"].isin(top_wells)].mean()
        rest_z = z[~cohort["well_id"].isin(top_wells)].mean()
        out.append(float(top_z - rest_z) if pd.notna(top_z) and pd.notna(rest_z) else 0.0)
    return out


def novelty(report) -> dict:
    """Features where the cohort sits off the reference training distribution."""
    ref = report.reference
    if not ref or report.cohort is None or "train_features" not in ref:
        return {}
    return cohort_novelty(ref["train_features"], report.cohort, ref.get("feature_cols", []))


def diagnose(report) -> dict:
    table = report.predictions_table()
    titer = table["predicted_titer"].to_numpy(dtype=float)
    r0 = report.results[0]
    surviving = {n: w for n, w in report.weights.items() if w > 0}
    top = float(np.max(titer))
    ref = report.reference or {}
    return {
        "n_wells": int(len(titer)),
        "n_train": r0.n_train,
        "n_unique_groups": r0.n_unique_groups,
        "n_features_selected": r0.n_features,
        "n_models": len(report.results),
        "n_surviving": len(surviving),
        "all_equal_weight": len(set(round(w, 6) for w in report.weights.values())) == 1,
        "titer_min": float(np.min(titer)),
        "titer_max": top,
        "n_distinct": int(np.unique(np.round(titer, 6)).size),
        "n_tied_at_top": int(np.sum(np.isclose(titer, top))),
        "tier_counts": table["confidence_tier"].value_counts().to_dict(),
        "reference_model": ref.get("model_name"),
        "reference_cv_spearman": ref.get("cv_spearman"),
        "reference_ci": [ref.get("ci_low"), ref.get("ci_high")],
        "reference_verdict": ref.get("verdict"),
        "novelty": novelty(report),
        "methods": {r.name: {"cv_spearman": r.cv_spearman, "ci": [r.ci_low, r.ci_high],
                             "verdict": r.verdict, "collapsed": r.collapsed_on_client,
                             "weight": report.weights.get(r.name, 0.0)}
                    for r in report.results},
    }


def _degenerate(d: dict) -> bool:
    return (d["all_equal_weight"] or d["n_surviving"] <= 1
            or d["n_distinct"] < 0.5 * d["n_wells"] or d["n_tied_at_top"] > 1)


# --------------------------------------------------------------------------- #
# Render: markdown
# --------------------------------------------------------------------------- #
def write_analysis(report, path: str | Path) -> Path:
    d = diagnose(report)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    L: list[str] = ["# Prediction analysis\n"]
    L.append(f"Predicted titer for **{d['n_wells']}** wells with a model trained on "
             f"**{d['n_train']}** rows across **{d['n_unique_groups']}** groups, "
             f"using **{d['n_features_selected']}** auto-selected features.\n")

    if d["reference_model"]:
        rho, (lo, hi) = d["reference_cv_spearman"], d["reference_ci"]
        L.append(f"**Reference model (`{d['reference_model']}`, drives the ranking figure): "
                 f"CV Spearman {rho:+.3f}** (95% CI [{lo:+.3f}, {hi:+.3f}]) — {d['reference_verdict']}.\n")

    L.append("## Per-model trust (train-time CV) and blend weight\n")
    L.append("| model | CV Spearman | 95% CI | verdict | collapsed on cohort | blend weight |")
    L.append("|---|---|---|---|---|---|")
    for name, m in d["methods"].items():
        L.append(f"| {name} | {m['cv_spearman']:+.3f} | [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}] | "
                 f"{m['verdict']} | {'yes' if m['collapsed'] else 'no'} | {m['weight']:.3f} |")
    L.append("")

    L.append("## Predicted-titer spread (blend)\n")
    L.append(f"- Predicted titer range: **{d['titer_min']:.3f} – {d['titer_max']:.3f}**")
    L.append(f"- Distinct predicted values: **{d['n_distinct']}** across {d['n_wells']} wells")
    L.append(f"- Wells tied at the top: **{d['n_tied_at_top']}**")
    L.append(f"- Confidence tiers: {', '.join(f'{k}: {v}' for k, v in d['tier_counts'].items())}\n")

    if d["novelty"]:
        L.append("## Off-distribution warning\n")
        L.append(f"{len(d['novelty'])} feature(s) where the cohort median is >3 training-SDs from "
                 "the reference mean (the model is extrapolating on these axes):")
        L.append(", ".join(f"`{k}` (z={v})" for k, v in d["novelty"].items()) + "\n")

    L.append("## Interpretation\n")
    ref_lo = d["reference_ci"][0]
    reference_validated = ref_lo is not None and ref_lo > 0
    if _degenerate(d):
        L.append("**This ranking is not trustworthy as a differentiator.** " + " ".join(filter(None, [
            (f"No model validated at train time, so the blend fell back to an equal-weight "
             f"consensus of all {d['n_models']} models." if d["all_equal_weight"] else
             (f"Only {d['n_surviving']} of {d['n_models']} models survived the blend." if d["n_surviving"] <= 1 else "")),
            (f"The models resolve the {d['n_wells']} wells into only {d['n_distinct']} distinct "
             "predicted titers — the off-distribution signature of reverting toward a constant."
             if d["n_distinct"] < 0.5 * d["n_wells"] else ""),
        ])))
        L.append("")
        L.append("Root cause is data, not code: small training set (feature cap "
                 f"{d['n_features_selected']}), no cross-validated ranking clears 0, and/or the "
                 "cohort is out-of-distribution vs the training wells. See `TODO.md`.")
    elif not reference_validated:
        L.append("**The blend differentiates the cohort, but the reference model's ranking "
                 "signal is NOT VALIDATED** (its bootstrap CI on CV Spearman includes 0), so "
                 "treat the ranking as directional only, not a validated result. "
                 "`predicted_titer` is the blended estimate (titer units); `confidence_tier` is "
                 "cross-model agreement, not statistical validation.")
    else:
        L.append("The reference model shows a validated ranking signal; `predicted_titer` is the "
                 "blended estimate (titer units), wells are ranked by predicted titer, and "
                 "`confidence_tier` is cross-model agreement.")
    L.append("\n> Duration caveat: a reference model should be trained on the same "
             "measurement/duration class as the cohort it ranks (MP ~4d vs SF ~10–14d). "
             "Pooling raw titer across durations conflates duration with productivity — see `TODO.md`.")
    path.write_text("\n".join(L))
    return path


# --------------------------------------------------------------------------- #
# Render: figures + parity CSVs
# --------------------------------------------------------------------------- #
def write_visualizations(report, figdir: str | Path, cfg: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    outdir = figdir.parent
    stem = figdir.name.replace("_figures", "")
    n_top = int(cfg.get("outputs", {}).get("n_top", 5))
    top_features = int(cfg.get("outputs", {}).get("top_features", 7))
    ref = report.reference or {}
    written: list[Path] = []

    # ---- fig1: ranking by reference-model predicted titer ----
    ranked = reference_ranking(report)
    ranked.to_csv(outdir / f"{stem}_clone_ranking.csv", index=False)
    written.append(outdir / f"{stem}_clone_ranking.csv")

    top_wells = set(ranked.head(n_top)["well_id"])
    cv = ref.get("cv_spearman")
    title = (f"Top {n_top} clones to scale (CV Spearman {cv:.2f})"
             if cv is not None else f"Top {n_top} clones to scale")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [ORANGE if w in top_wells else GRAY for w in ranked["well_id"]]
    ax.barh(ranked["well_id"][::-1], ranked["pred_titer"][::-1], color=colors[::-1])
    if ref.get("titer_mean") is not None:
        ax.axvline(ref["titer_mean"], ls="--", color=BLUE, lw=1, label="reference mean")
        ax.legend()
    ax.set_xlabel("Predicted titer (reference signal, relative units)")
    ax.set_ylabel("Clone / plate well")
    ax.set_title(title)
    fig.tight_layout()
    p = figdir / "fig1_ranking.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # ---- fig2: drivers (permutation importance) + within-plate contrast ----
    imp = pd.DataFrame(ref.get("importance") or [], columns=["feature", "importance"])
    imp.to_csv(outdir / f"{stem}_drivers.csv", index=False)
    written.append(outdir / f"{stem}_drivers.csv")

    if not imp.empty:
        top = imp.head(top_features)
        feats = top["feature"].tolist()
        labels = [_label(f) for f in feats]
        contrast = (within_plate_contrast(report.cohort, feats, top_wells)
                    if report.cohort is not None else [0.0] * len(feats))

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        axes[0].barh(labels[::-1], top["importance"][::-1], color=BLUE)
        axes[0].set_xlabel("Importance in the reference model\n(drop in titer-ranking skill when shuffled, in-sample)")
        axes[0].set_title("(a) What the model weights")

        cc = [ORANGE if v >= 0 else LIGHTBLUE for v in contrast]
        axes[1].barh(labels[::-1], contrast[::-1], color=cc[::-1])
        axes[1].axvline(0, color="black", lw=0.8)
        axes[1].set_xlabel("Top-N picks minus rest of plate (within-plate SD)")
        axes[1].set_title("(b) Why these picks stand out on THIS plate")

        fig.suptitle("What the model weights — and how the top picks differ",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        p = figdir / "fig2_drivers.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    return written
