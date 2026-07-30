#!/usr/bin/env python3
"""Run the engine on real media data (media composition + pH -> lipase titer).

Honest test of the BoTorch engine on actual Anagram runs: leakage-controlled
grouped-CV of the surrogate, a proposed next batch, and the multi-objective
titer-vs-purity Pareto front. The data is proprietary and read from
KALOS_MEDIA_DATA (a merged *_protein_expression_and_media_composition.tsv);
nothing is vendored.

Two things keep the headline number honest, both learned the hard way on this
dataset: inputs that were not measured on every run are dropped rather than
zero-filled (zero-filling invented a dose axis and roughly doubled the score),
and the CV groups on campaign x medium, not medium alone. Expect a modest
Spearman with a wide CI - the signal here is weak and does not transfer to a
held-out campaign.

Run:  KALOS_MEDIA_DATA=/path/to/combined.tsv python examples/run_on_media_data.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kalos import (  # noqa: E402
    MultiObjectiveSurrogate,
    Surrogate,
    check_gates,
    propose,
    propose_multiobjective,
)
from kalos.core.evaluation import grouped_cv_report  # noqa: E402

FEATURES = [
    "pH After Adjustment", "Iron_II_sulfate_heptahydrate", "Copper_II_sulfate_pentahydrate",
    "Boric_acid", "Zinc_sulfate_heptahydrate", "Sodium_molybdate_dihydrate",
    "Manganese_II_chloride_tetrahydrate", "Cobalt_II_chloride_hexahydrate", "Sulfuric_acid_95_98",
    "Ammonium_sulfate", "Calcium_chloride_dihydrate", "Citric_acid_monohydrate",
    "Magnesium_sulfate_heptahydrate", "Potassium_phosphate_monobasic", "Biotin",
    "Glycerol", "Methanol", "Potassium_hydroxide", "Ammonia_hydroxide", "PPG-2000",
]
TARGET = "Lipase_g.L"
PURITY = "% Purity"
# A `Medium` label means a DIFFERENT recipe in each campaign, so grouping on it
# alone lets the same medium sit in both train and val with different chemistry.
# Group on the campaign too.
GROUP_COLS = ["Experiment", "Medium"]


def load(path: Path):
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)
    if "Sample Name" in df:  # drop complex-medium controls, as the analysis does
        df = df[~df["Sample Name"].astype(str).str.startswith("BMMY")]
    feats = [c for c in FEATURES if c in df.columns]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].reset_index(drop=True)
    # A column that was not measured on every run cannot be used as a feature.
    # Imputing 0.0 turns "not recorded" into "zero concentration", inventing a
    # fake low-dose arm out of the rows where the assay simply was not run.
    # On the Anagram data that one substitution carried most of the headline
    # score: zero-filling `Ammonia_hydroxide` (25 of 82 rows missing) took the
    # grouped-CV Spearman from 0.46 to 0.76, and on complete cases only the same
    # axis gives 0.36. The hazard is easiest to see in `Potassium_hydroxide`,
    # where `!= 0` is exactly `Experiment == S26APR07` -- though that column is
    # inert to the score; it is the imputed dose axis, not the batch label, that
    # does the damage.
    unmeasured = {c: int(df[c].isna().sum()) for c in feats if df[c].isna().any()}
    feats = [c for c in feats if c not in unmeasured]
    feats = [c for c in feats if df[c].std() > 1e-9]  # only varying inputs
    return df, feats, unmeasured


def main() -> int:
    path = Path(os.environ.get("KALOS_MEDIA_DATA", "")).expanduser()
    if not path.exists():
        print("set KALOS_MEDIA_DATA to a merged media+titer TSV.")
        return 1
    df, feats, unmeasured = load(path)
    if not feats:
        print(f"no usable inputs in {path.name}: every declared feature was either constant "
              f"or partly unmeasured ({len(unmeasured)} dropped for missingness). Nothing to model.")
        return 1
    X = df[feats].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    bounds = np.vstack([X.min(0), X.max(0)])
    gcols = [c for c in GROUP_COLS if c in df.columns]
    groups = df[gcols].astype(str).agg(" | ".join, axis=1).tolist() if gcols else None
    if gcols != GROUP_COLS:  # degrading the key silently is how leakage gets back in
        missing = [c for c in GROUP_COLS if c not in df.columns]
        print(f"  WARNING: grouping on {gcols or 'nothing'} - {missing} absent from this file.\n"
              f"           The CV below is weaker than it looks; treat it as an upper bound.")

    print("=" * 72)
    print(f"KALOS on real media data  ({path.name})")
    print(f"  {len(df)} runs x {len(feats)} varying inputs -> {TARGET}")
    print(f"  titer range {y.min():.4f}..{y.max():.4f}  | non-producers {int((y == 0).sum())} "
          f"({100 * (y == 0).mean():.0f}%)  | groups: {' x '.join(gcols) or 'none'}")
    if unmeasured:
        dropped = ", ".join(f"{c} ({n} missing)" for c, n in unmeasured.items())
        print(f"  dropped {len(unmeasured)} partly-unmeasured input(s): {dropped}")
    print("=" * 72)

    # 1) honest leakage-controlled evaluation
    rep = grouped_cv_report(X, y, groups=groups, n_splits=5)
    rho, (lo, hi) = rep["spearman"], rep["ci95"]
    print(f"\nGrouped-CV surrogate Spearman (titer): {rho:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"  {rep['n_oof']} held-out points across {rep['n_groups']} groups, "
          f"{rep['n_folds']} folds")
    if lo <= 0.0 <= hi:
        print("  CI crosses zero: this data does not yet support a ranking claim.")

    # 2) promotion gate (fail-closed: feasibility/calibration not yet wired -> blocked)
    res = check_gates({"surrogate_spearman": rho})
    print(f"Promotion gate: {res.summary}")

    # 3) single-objective: propose the next titer-maximizing batch.
    #    Print every dimension - a partial recipe reads like a complete one.
    s = Surrogate().fit(X, y, bounds=bounds)
    batch = propose(s, bounds, q=5)
    w = max(9, *(len(f) for f in feats))
    print("\nNext batch (titer), all proposed dimensions:")
    print("  " + "  ".join(f"{f[:w]:>{w}}" for f in feats))
    for row in batch:
        print("  " + "  ".join(f"{v:>{w}.2f}" for v in row))

    # 4) multi-objective: titer AND purity together
    if PURITY in df.columns:
        p = pd.to_numeric(df[PURITY], errors="coerce")
        m = p.notna().to_numpy()
        if m.sum() > len(feats) + 2:
            Y = np.column_stack([y[m], p[m].to_numpy(float)])
            mo = MultiObjectiveSurrogate().fit(X[m], Y, bounds=bounds)
            # `pareto()` is descriptive: it filters the OBSERVED points, so on its
            # own it would make the GP fit above dead compute. Report it as the
            # observed front, then actually use the fitted model to propose points
            # that would expand it.
            _, pf = mo.pareto()
            order = np.argsort(pf[:, 0])
            print(f"\nMulti-objective (titer, purity) on {int(m.sum())} runs")
            print(f"  observed Pareto front ({len(pf)} of {int(m.sum())} runs non-dominated):")
            for i in order:
                print(f"    titer {pf[i, 0]:.4f}   purity {pf[i, 1]:5.1f}%")
            nd = np.unique(np.round(p[m].to_numpy(float), 6))
            if len(nd) <= 5:
                vals = ", ".join(f"{v:g}" for v in nd)
                print(f"  WARNING: purity takes only {len(nd)} distinct values ({vals}) across these "
                      f"runs.\n           Treat this front as a data-quality flag, not an optimum.")
            mob = propose_multiobjective(mo, bounds, q=3)
            print(f"  model-proposed batch to expand the front: {mob.shape[0]} points x "
                  f"{mob.shape[1]} inputs, all in-bounds="
                  f"{bool(((mob >= bounds[0]) & (mob <= bounds[1])).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
