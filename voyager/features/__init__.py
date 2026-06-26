"""Feature sources. KmerEmbedder needs no extra deps; ESM2Embedder needs transformers."""
from .protein import KmerEmbedder, ESM2Embedder, build_feature_table, clean_sequence

__all__ = ["KmerEmbedder", "ESM2Embedder", "build_feature_table", "clean_sequence"]
