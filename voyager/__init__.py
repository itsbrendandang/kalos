"""Voyager platform: BoTorch Bayesian optimization for bioprocess development,
with NVIDIA BioNeMo / ESM-2 protein features.
"""
from .core.surrogate import Surrogate
from .core.optimize import propose
from .core.multiobjective import MultiObjectiveSurrogate, propose_multiobjective
from .core.gates import GatesConfig, PromotionResult, check_gates
from .core.evaluation import grouped_cv_spearman

__all__ = [
    "Surrogate",
    "propose",
    "MultiObjectiveSurrogate",
    "propose_multiobjective",
    "GatesConfig",
    "PromotionResult",
    "check_gates",
    "grouped_cv_spearman",
]
__version__ = "0.1.0"
