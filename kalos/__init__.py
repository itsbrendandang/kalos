"""Kalos platform: BoTorch Bayesian optimization for bioprocess development, with
NVIDIA BioNeMo / ESM-2 protein features and a barcode-organized data registry.
"""
from .core.surrogate import Surrogate
from .core.optimize import propose
from .core.multiobjective import MultiObjectiveSurrogate, propose_multiobjective
from .core.gates import GatesConfig, PromotionResult, check_gates
from .core.evaluation import grouped_cv_spearman
from .core.splits import row_hash_groups, make_splits, assert_no_group_leakage
from .core.drivers import bootstrap_spearman, rank_drivers
from .core.conformal import split_conformal
from .data.barcode_registry import BarcodeRegistry
from .data.anonymizer import Anonymizer

__all__ = [
    "Surrogate", "propose", "MultiObjectiveSurrogate", "propose_multiobjective",
    "GatesConfig", "PromotionResult", "check_gates", "grouped_cv_spearman",
    "row_hash_groups", "make_splits", "assert_no_group_leakage",
    "bootstrap_spearman", "rank_drivers", "split_conformal",
    "BarcodeRegistry", "Anonymizer",
]
__version__ = "0.1.0"
