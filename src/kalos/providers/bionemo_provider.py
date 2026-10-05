"""SLOT for a future hosted NVIDIA BioNeMo endpoint - not a live integration.

Be honest about what this is: there is no BioNeMo client anywhere in this
codebase today. `kalos/features/protein.py` already runs real ESM-2 protein
embeddings via Hugging Face `transformers` (`ESM2Embedder`, pulling the
public, non-gated `facebook/esm2_t6_8M_UR50D` checkpoint) - that path needs
no token at all. This provider exists so a future hosted BioNeMo endpoint, or
gated Hugging Face model pulls, have a named, documented credential slot to
light up. It implements no HTTP client, no request builder, no retry logic -
only a name and a credential check.
"""
from __future__ import annotations

import os

from kalos.providers.base import ProviderStatus

_NVIDIA_API_KEY_ENV_VAR = "NVIDIA_API_KEY"
_HF_TOKEN_ENV_VAR = "HF_TOKEN"

_CAPABILITY = (
    "Real ESM-2 protein-sequence embeddings from a hosted NVIDIA BioNeMo "
    "endpoint (NVIDIA_API_KEY), plus pulling gated Hugging Face checkpoints "
    "(HF_TOKEN) - this is a SLOT for a future integration, not a working "
    "BioNeMo client today."
)
_FALLBACK = (
    "kalos.features.protein already runs real ESM-2 embeddings via the "
    "public, non-gated facebook/esm2_t6_8M_UR50D Hugging Face checkpoint "
    "(ESM2Embedder), which needs no token; the lean k-mer embedder "
    "(KmerEmbedder) covers the fully offline case."
)


class BioNemoProvider:
    """Credential check for a future hosted BioNeMo endpoint. `NVIDIA_API_KEY`
    is required; `HF_TOKEN` is optional (only needed for gated HF model
    pulls, and is not required for the existing public ESM-2 checkpoint)."""

    name = "bionemo"

    def available(self) -> bool:
        return bool(os.environ.get(_NVIDIA_API_KEY_ENV_VAR))

    def status(self) -> ProviderStatus:
        ok = self.available()
        return ProviderStatus(
            name=self.name,
            available=ok,
            required_env=(_NVIDIA_API_KEY_ENV_VAR,),
            missing_env=() if ok else (_NVIDIA_API_KEY_ENV_VAR,),
            capability=_CAPABILITY,
            fallback=_FALLBACK,
        )


__all__ = ["BioNemoProvider"]
