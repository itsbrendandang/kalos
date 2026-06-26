"""Protein-sequence embeddings as surrogate features.

Embed the expressed product's amino-acid sequence so one surrogate can transfer
across products/strains. Two backends behind one interface:

  - `ESM2Embedder` — the real deep-learning model (ESM-2 via Hugging Face
    transformers, the same family NVIDIA BioNeMo serves). Runs on Apple MPS or
    CPU. Mean-pools the residue embeddings to one vector per sequence.
  - `KmerEmbedder` — a dependency-free hashed k-mer stand-in that runs with no
    model download, for tests and offline use.

Trust boundary: an embedder only ever sees a PUBLIC protein sequence, never
proprietary process data.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Protocol

import numpy as np
import pandas as pd

_NON_AA = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def clean_sequence(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise TypeError("protein sequence must be a string of amino-acid letters")
    seq = _NON_AA.sub("", sequence.strip().upper())
    if not seq:
        raise ValueError("sequence has no valid amino-acid residues")
    return seq


class ProteinEmbedder(Protocol):
    dim: int

    def embed(self, sequence: str) -> np.ndarray: ...


class KmerEmbedder:
    """Deterministic hashed k-mer counts (L2-normalized). No model, no download."""

    def __init__(self, k: int = 3, dim: int = 64) -> None:
        if k < 1 or dim < 1:
            raise ValueError("k and dim must be >= 1")
        self.k = k
        self.dim = dim

    def _bucket(self, kmer: str) -> int:
        return int.from_bytes(hashlib.blake2b(kmer.encode(), digest_size=8).digest(), "big") % self.dim

    def embed(self, sequence: str) -> np.ndarray:
        seq = clean_sequence(sequence)
        vec = np.zeros(self.dim, dtype=float)
        for i in range(len(seq) - self.k + 1):
            vec[self._bucket(seq[i : i + self.k])] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class ESM2Embedder:
    """ESM-2 protein language model (Hugging Face transformers) on MPS/CPU.

    Default model is the small `esm2_t6_8M_UR50D` (8M params, 320-dim, ~30 MB) so
    it runs fast locally; swap `model` for a larger ESM-2 for stronger features.
    """

    def __init__(self, model: str = "facebook/esm2_t6_8M_UR50D", device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device).eval()
        self.dim = int(self.model.config.hidden_size)

    def embed(self, sequence: str) -> np.ndarray:
        torch = self._torch
        seq = clean_sequence(sequence)
        with torch.no_grad():
            enc = self.tokenizer(seq, return_tensors="pt").to(self.device)
            out = self.model(**enc).last_hidden_state[0]  # (L, H) incl. BOS/EOS
            emb = out[1:-1].mean(dim=0)  # mean-pool real residues
        return emb.float().cpu().numpy()


def build_feature_table(
    products: Dict[str, str], embedder: ProteinEmbedder, prefix: str = "prot_emb_"
) -> pd.DataFrame:
    """One embedding row per product. Join onto run data on `product` to add the
    protein embedding as extra surrogate features."""
    rows: List[dict] = []
    for name, seq in products.items():
        e = embedder.embed(seq)
        rows.append({"product": name, **{f"{prefix}{i}": float(e[i]) for i in range(len(e))}})
    return pd.DataFrame(rows)


__all__ = ["ProteinEmbedder", "KmerEmbedder", "ESM2Embedder", "build_feature_table", "clean_sequence"]
