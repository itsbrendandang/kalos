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


# Immutable commit pin for the default ESM-2 checkpoint.
#
# `from_pretrained("facebook/esm2_t6_8M_UR50D")` with no `revision` resolves to
# whatever that repo's `main` branch points at TODAY. A branch name is mutable:
# if the upstream repo is ever re-uploaded or its weights re-quantized, every
# embedding this class produces changes, silently, with no diff in our own code.
# That breaks the platform's reproducibility contract - a result must be
# reproducible from committed code and config - and it breaks it in the worst
# way, by changing model features under a fixed random seed while every test
# still passes.
#
# So the default is pinned to an explicit commit SHA. Verified against
# https://huggingface.co/api/models/facebook/esm2_t6_8M_UR50D on 2026-08-08:
# sha c731040fcd8d73dceaa04b0a8e6329b345b0f5df, upstream lastModified
# 2023-03-21. Bumping this pin is a deliberate, reviewable act: change the SHA,
# re-run the embedding tests, and note the change, because it invalidates every
# previously computed embedding.
DEFAULT_ESM2_MODEL = "facebook/esm2_t6_8M_UR50D"
DEFAULT_ESM2_REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"


class ESM2Embedder:
    """ESM-2 protein language model (Hugging Face transformers) on MPS/CPU.

    Default model is the small `esm2_t6_8M_UR50D` (8M params, 320-dim, ~30 MB) so
    it runs fast locally; swap `model` for a larger ESM-2 for stronger features.

    `revision` pins the checkpoint to an immutable commit (see
    `DEFAULT_ESM2_REVISION` above for why this is not optional). Pass
    `revision=None` to deliberately track the upstream branch instead, and
    accept that embeddings then depend on when you ran the code rather than on
    what you committed. When swapping `model` for a larger ESM-2, pass that
    model's own commit SHA as `revision` - carrying the default pin over to a
    different repo would raise, since the SHA does not exist there.
    """

    def __init__(
        self,
        model: str = DEFAULT_ESM2_MODEL,
        device: str | None = None,
        *,
        revision: str | None = DEFAULT_ESM2_REVISION,
    ) -> None:
        # Argument validation first, before the heavy optional imports: a caller
        # who mispaired model and revision should get a fast, clear error rather
        # than one that only surfaces on machines with the `protein` extra
        # installed, or a 404 raised deep inside transformers.
        if revision == DEFAULT_ESM2_REVISION and model != DEFAULT_ESM2_MODEL:
            # A commit SHA is repo-specific, so carrying the default pin over to
            # a different checkpoint can never resolve.
            raise ValueError(
                f"revision {DEFAULT_ESM2_REVISION!r} is the pin for "
                f"{DEFAULT_ESM2_MODEL!r} and does not exist in {model!r}; "
                "pass that model's own commit SHA, or revision=None to "
                "track its branch."
            )

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model_name = model
        # Recorded so a run's provenance can state exactly which weights produced
        # its features, not merely which model name was requested.
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModel.from_pretrained(model, revision=revision).to(device).eval()
        self.dim = int(self.model.config.hidden_size)

    def embed(self, sequence: str) -> np.ndarray:
        torch = self._torch
        seq = clean_sequence(sequence)
        with torch.no_grad():
            # truncate to ESM-2's context (BOS + 1022 residues + EOS = 1024) so a
            # long sequence or an absolute-position variant degrades, not crashes.
            enc = self.tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
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


__all__ = [
    "DEFAULT_ESM2_MODEL",
    "DEFAULT_ESM2_REVISION",
    "ESM2Embedder",
    "KmerEmbedder",
    "ProteinEmbedder",
    "build_feature_table",
    "clean_sequence",
]
