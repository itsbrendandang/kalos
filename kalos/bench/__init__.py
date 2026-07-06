"""Closed-loop benchmark: does the kalos BO loop actually beat space-filling?

The one honest question a bioprocess-optimization product must answer before it
is sold: given a fixed experiment budget, does the Bayesian-optimization loop
reach a good recipe in FEWER experiments than just running a space-filling
design (Latin Hypercube) or random sampling?

This package answers it on synthetic surfaces with KNOWN optima (so true simple
regret is measurable and the whole thing is reproducible from committed code),
while sweeping observation noise - the same measurement noise that governs
whether the model helps at all on real, noisy assay data.
"""
from .objectives import Objective, ackley, gaussian_bump  # noqa: F401
from .benchmark import run_benchmark, run_one, summarize  # noqa: F401

__all__ = [
    "Objective",
    "ackley",
    "gaussian_bump",
    "run_benchmark",
    "run_one",
    "summarize",
]
