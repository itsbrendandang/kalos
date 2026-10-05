#!/usr/bin/env python3
"""Demo: embed product proteins with ESM-2 and build a surrogate feature table.

Run:  python examples/run_protein_embed.py   (downloads a small ESM-2 model once)
Needs the protein extra:  pip install -e ".[protein]"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # run without installing

import numpy as np  # noqa: E402

from kalos.features import ESM2Embedder, build_feature_table  # noqa: E402

# illustrative product protein fragments (public sequences)
PRODUCTS = {
    "HSA": "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAF",
    "Herceptin": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVAR",
    "Lipase": "MKLLSLTGVAGVLATCVAATPLVKRSPNSTPDAVQTSADFAQGNYAEMV",
}


def main() -> int:
    emb = ESM2Embedder()
    print(f"ESM-2 {emb.model_name} on {emb.device} (dim {emb.dim})")
    table = build_feature_table(PRODUCTS, emb)
    print("\nprotein feature table (first 5 of", emb.dim, "dims):")
    print(table.iloc[:, :6].round(3).to_string(index=False))

    # pairwise cosine: distinct proteins, shared protein-space structure
    vecs = {n: emb.embed(s) for n, s in PRODUCTS.items()}
    names = list(vecs)
    print("\npairwise cosine similarity:")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            va, vb = vecs[a], vecs[b]
            cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
            print(f"  {a:10s} vs {b:10s}  {cos:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
