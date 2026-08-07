"""Domain layer: declared column roles, domain profiles, and mixed design
spaces that map a tabular problem onto the domain-neutral engine.

Torch-free: importing this package never loads torch/botorch/gpytorch.
"""
from __future__ import annotations

from .bioprocess import BIOPROCESS_PROFILE
from .generic import GENERIC_PROFILE
from .profile import (
    ColumnRoles,
    DesignSpace,
    Dimension,
    DimensionKind,
    DomainProfile,
    build_design_space,
)

__all__ = [
    "ColumnRoles",
    "DesignSpace",
    "Dimension",
    "DimensionKind",
    "DomainProfile",
    "build_design_space",
    "BIOPROCESS_PROFILE",
    "GENERIC_PROFILE",
]
