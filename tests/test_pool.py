"""kalos/bench pool-based harness: feature selection excludes outputs/ids, and on
a CLEAN synthetic pool the model-guided strategy must beat random (the fairness
control that lets the real-data result be read as a statement about the data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kalos.bench.pool import pool_from_frame, run_pool


def test_pool_from_frame_keeps_inputs_drops_outputs_and_ids():
    df = pd.DataFrame({
        "Well": ["A1", "A2", "A3", "A4", "A5", "A6"],
        "Medium": ["M1", "M2", "M3", "M1", "M2", "M3"],
        "Conc. (ng/ul)": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # other output -> drop
        "% Purity": [10, 20, 30, 40, 50, 60],  # other output -> drop
        "Glycerol": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # process input -> keep
        "Const": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # constant -> drop
        "Lipase_g.L": [0.0, 0.02, 0.05, 0.01, 0.09, 0.03],  # target
    })
    X, y, feats = pool_from_frame(df, target="Lipase_g.L")
    assert feats == ["Glycerol"]
    assert X.shape == (6, 1)
    assert len(y) == 6


def test_pool_bo_beats_random_on_a_clean_pool():
    # Same harness as the real-data benchmark, but a smooth low-noise pool: the
    # model-guided pick must end at least as good as random on average. This is the
    # control that shows a real-data BO loss is about the data, not the harness.
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(80, 6))
    center = np.full(6, 0.55)
    y = 10.0 * np.exp(-3.0 * ((X - center) ** 2).sum(1)) + rng.normal(0, 0.05, 80)
    res = run_pool(X, y, n_init=8, budget=22, seeds=range(5))
    assert res["bo"]["mean"][-1] > res["random"]["mean"][-1]
