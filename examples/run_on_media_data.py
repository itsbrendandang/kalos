#!/usr/bin/env python3
"""Run the engine on real media data (media composition + pH -> lipase titer).

Honest test of the BoTorch engine on actual Anagram runs: leakage-controlled
grouped-CV of the surrogate, a proposed next batch, and the multi-objective
titer-vs-purity Pareto front. The data is proprietary and read from
VOYAGER_MEDIA_DATA (a merged *_protein_expression_and_media_composition.tsv);
nothing is vendored.

Run:  VOYAGER_MEDIA_DATA=/path/to/combined.tsv python examples/run_on_media_data.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from voyager import (  # noqa: E402
    MultiObjectiveSurrogate,
    Surrogate,
    check_gates,
    grouped_cv_spearman,
    propose,
    propose_multiobjective,
)

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
GROUP = "Medium"


def load(path: Path):
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)
    if "Sample Name" in df:  # drop complex-medium controls, as the analysis does
        df = df[~df["Sample Name"].astype(str).str.startswith("BMMY")]
    feats = [c for c in FEATURES if c in df.columns]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna()].reset_index(drop=True)
    feats = [c for c in feats if df[c].std() > 1e-9]  # only varying inputs
    return df, feats


def main() -> int:
    path = Path(os.environ.get("VOYAGER_MEDIA_DATA", "")).expanduser()
    if not path.exists():
        print("set VOYAGER_MEDIA_DATA to a merged media+titer TSV.")
        return 1
    df, feats = load(path)
    X = df[feats].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    bounds = np.vstack([X.min(0), X.max(0)])
    groups = df[GROUP].tolist() if GROUP in df.columns else None

    print("=" * 72)
    print(f"VOYAGER ENGINE on real media data  ({path.name})")
    print(f"  {len(df)} runs x {len(feats)} varying inputs -> {TARGET}")
    print(f"  titer range {y.min():.4f}..{y.max():.4f}  | non-producers {int((y == 0).sum())} "
          f"({100 * (y == 0).mean():.0f}%)  | groups: {GROUP}")
    print("=" * 72)

    # 1) honest leakage-controlled evaluation
    rho = grouped_cv_spearman(X, y, groups=groups, n_splits=5)
    print(f"\nGrouped-CV surrogate Spearman (titer): {rho:.2f}")

    # 2) promotion gate (fail-closed: feasibility/calibration not yet wired -> blocked)
    res = check_gates({"surrogate_spearman": rho})
    print(f"Promotion gate: {res.summary}")

    # 3) single-objective: propose the next titer-maximizing batch
    s = Surrogate().fit(X, y, bounds=bounds)
    batch = propose(s, bounds, q=5)
    show = [f for f in ("Methanol", "Glycerol", "pH After Adjustment") if f in feats]
    cols = [feats.index(f) for f in show]
    print(f"\nNext batch (titer): top inputs {show}")
    for row in np.round(batch[:, cols], 2):
        print("  " + "  ".join(f"{v:6.2f}" for v in row))

    # 4) multi-objective: titer AND purity together
    if PURITY in df.columns:
        p = pd.to_numeric(df[PURITY], errors="coerce")
        m = p.notna().to_numpy()
        if m.sum() > len(feats) + 2:
            Y = np.column_stack([y[m], p[m].to_numpy(float)])
            mo = MultiObjectiveSurrogate().fit(X[m], Y, bounds=bounds)
            _, pf = mo.pareto()
            order = np.argsort(pf[:, 0])
            print(f"\nMulti-objective (titer, purity) on {int(m.sum())} runs — observed Pareto front:")
            for i in order:
                print(f"  titer {pf[i, 0]:.4f}   purity {pf[i, 1]:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
