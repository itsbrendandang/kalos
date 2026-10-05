"""Promotion gates: a model is only fit to serve if it clears the floors.

Carried over from the lean engine (and the multi-agent review that hardened it):
fail-CLOSED. A missing or NaN required metric BLOCKS promotion rather than
silently passing — the gate must not vanish on exactly the data it exists to
guard (e.g. an unmeasurable feasibility AUC on single-class grouped folds).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypeGuard


@dataclass
class GatesConfig:
    min_spearman: float = 0.20
    min_feasibility_auc: float = 0.65
    max_ece: float = 0.15
    max_brier: float = 0.25
    enabled: bool = True


@dataclass
class PromotionResult:
    passed: bool
    failures: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    summary: str = ""


def _finite(x: Any) -> TypeGuard[float]:
    # bool is an int subclass; a stray True/False is not a real metric value.
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_gates(stats: Dict[str, Any], config: GatesConfig | None = None) -> PromotionResult:
    cfg = config or GatesConfig()
    if not cfg.enabled:
        return PromotionResult(True, [], {}, "promotion gates disabled")

    f: List[str] = []
    m: Dict[str, float] = {}

    def floor(key: str, lo: float) -> None:
        v = stats.get(key)
        if _finite(v):
            m[key] = float(v)
            if v < lo:
                f.append(f"{key}={v:.3f} < min {lo}")
        else:
            f.append(f"{key} missing/NaN (unmeasured)")  # fail closed

    def ceiling(key: str, hi: float) -> None:
        v = stats.get(key)
        if _finite(v):
            m[key] = float(v)
            if v > hi:
                f.append(f"{key}={v:.3f} > max {hi}")
        else:
            f.append(f"{key} missing/NaN (unmeasured)")  # fail closed

    floor("surrogate_spearman", cfg.min_spearman)
    floor("feasibility_auc", cfg.min_feasibility_auc)
    ceiling("ece", cfg.max_ece)
    ceiling("brier", cfg.max_brier)

    passed = not f
    summary = "ALL GATES PASSED" if passed else "GATES FAILED: " + "; ".join(f)
    return PromotionResult(passed, f, m, summary)


__all__ = ["GatesConfig", "PromotionResult", "check_gates"]
