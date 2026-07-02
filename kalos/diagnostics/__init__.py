"""Diagnostic figures for the small-data BO regime (Python engine side only).

Opt-in — importing this needs matplotlib (`pip install -e ".[viz]"`); the core
kalos package does not. See `figures.py`.
"""
from .figures import (
    parity,
    cv_forest,
    parallel_coordinates,
    pca_scatter,
    partial_dependence,
)

__all__ = ["parity", "cv_forest", "parallel_coordinates", "pca_scatter", "partial_dependence"]
