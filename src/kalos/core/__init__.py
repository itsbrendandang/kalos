from .gates import GatesConfig, PromotionResult, check_gates
from .splits import row_hash_groups, make_splits, assert_no_group_leakage
from .drivers import spearman_driver_matrix, bootstrap_spearman, rank_drivers
from .conformal import split_conformal, conformal_interval

# `kalos.core` is torch-free except for three modules with top-level
# torch/botorch/gpytorch imports (~220 MB of RSS): surrogate.py, optimize.py,
# and multiobjective.py. Loaded lazily via `__getattr__` (PEP 562) below, so
# importing any sibling submodule of `kalos.core` (which always runs this
# `__init__.py` first) does not pay the torch tax before one of these is
# actually touched. Callers elsewhere in the codebase (e.g. the portal and
# the `--watch` poller) preserve the same contract by importing
# `multiobjective` lazily too, deferring the import until a fit actually
# runs rather than at module load time. See `kalos/__init__.py` for the
# matching top-level lazy exports.
_LAZY = {
    "Surrogate": (".surrogate", "Surrogate"),
    "propose": (".optimize", "propose"),
    "MultiObjectiveSurrogate": (".multiobjective", "MultiObjectiveSurrogate"),
    "propose_multiobjective": (".multiobjective", "propose_multiobjective"),
    "grouped_cv_spearman": (".evaluation", "grouped_cv_spearman"),
    "grouped_cv_report": (".evaluation", "grouped_cv_report"),
    "grouped_folds": (".evaluation", "grouped_folds"),
}

__all__ = [
    "Surrogate", "propose", "MultiObjectiveSurrogate", "propose_multiobjective",
    "GatesConfig", "PromotionResult", "check_gates",
    "grouped_cv_spearman", "grouped_cv_report", "grouped_folds",
    "row_hash_groups", "make_splits", "assert_no_group_leakage",
    "spearman_driver_matrix", "bootstrap_spearman", "rank_drivers",
    "split_conformal", "conformal_interval",
]


def __getattr__(name: str):
    try:
        module_path, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_path, __name__), attr)
    globals()[name] = value  # cache: repeat access hits the module dict, not __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
