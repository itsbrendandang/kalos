"""Torch-free primitives shared across the go-forward engine and the torch-free
reference repo. Importing this must never load torch.

Thin re-export facade: nothing here is defined, everything is a straight
re-export of an existing torch-free module under `kalos.core` / `kalos.data`,
so `from kalos.core.drivers import ...`-style imports keep working unchanged.
`kalos.core.evaluation`, `kalos.core.surrogate`, and `kalos.core.optimize` pull
in torch/botorch/gpytorch and are deliberately NOT re-exported here - they are
engine-only, not part of the torch-free kit.
"""
from __future__ import annotations

from ..core.conformal import conformal_interval, q_from_residuals, split_conformal
from ..core.drivers import bootstrap_spearman, rank_drivers, spearman_driver_matrix
from ..core.gates import GatesConfig, PromotionResult, check_gates
from ..core.splits import assert_no_group_leakage, make_splits, row_hash_groups
from ..data.anonymizer import (
    DROP_EXACT,
    DROP_SUBSTR,
    HASH_EXACT,
    HASH_SUBSTR,
    Anonymizer,
    _hash,
    default_salt,
)

__all__ = [
    # kalos.core.splits
    "row_hash_groups",
    "make_splits",
    "assert_no_group_leakage",
    # kalos.core.drivers
    "spearman_driver_matrix",
    "bootstrap_spearman",
    "rank_drivers",
    # kalos.core.conformal
    "q_from_residuals",
    "split_conformal",
    "conformal_interval",
    # kalos.core.gates
    "GatesConfig",
    "PromotionResult",
    "check_gates",
    # kalos.data.anonymizer
    "Anonymizer",
    "_hash",
    "default_salt",
    "DROP_EXACT",
    "DROP_SUBSTR",
    "HASH_EXACT",
    "HASH_SUBSTR",
]
