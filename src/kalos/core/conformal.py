"""Split-conformal prediction intervals — distribution-free coverage.

Ported from the lean engine. Calibrate on a held-out set's absolute residuals to
get a half-width q such that [mu - q, mu + q] covers at least (1 - alpha) of
outcomes, with NO assumption about the error distribution (valid even when titer
residuals are skewed or heteroscedastic). Pairs naturally with the BoTorch
surrogate: pass `surrogate.posterior(X)[0]` (the mean) as the prediction.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def q_from_residuals(residuals, alpha: float = 0.1) -> float:
    """Conformal half-width q from precomputed absolute residuals.

    Same finite-sample quantile as `split_conformal`, but for when you already
    hold the held-out residuals (e.g. pooled out-of-fold residuals from grouped
    CV) instead of a calibration predict-fn. Coverage is exact for iid
    split-conformal; with grouped OOF residuals treat it as approximate.
    """
    res = np.abs(np.asarray(residuals, float).reshape(-1))
    n = len(res)
    if n == 0:
        raise ValueError("empty residual set")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(res, level))


def split_conformal(predict_fn: Callable[[np.ndarray], np.ndarray], X_cal, y_cal, alpha: float = 0.1) -> float:
    """Half-width q for >= 1-alpha coverage, from absolute residuals on a
    calibration set (held out from training to guarantee valid coverage)."""
    mu = np.asarray(predict_fn(X_cal), float).reshape(-1)
    return q_from_residuals(np.asarray(y_cal, float).reshape(-1) - mu, alpha)


def conformal_interval(mean: np.ndarray, q: float) -> np.ndarray:
    """Return an (n, 2) array of [lower, upper] = mean +/- q."""
    m = np.asarray(mean, float).reshape(-1)
    return np.column_stack([m - q, m + q])


__all__ = ["q_from_residuals", "split_conformal", "conformal_interval"]
