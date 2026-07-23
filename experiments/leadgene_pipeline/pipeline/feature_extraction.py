"""Timecourse -> feature extraction (pure, config-free).

Ported verbatim from Leadgene_Clone_Picker's
clone_select/ingest/tabularize_wrapper.py:extract_features. Turns one
(time, signal) trace into ~50 scalar features so a DO/pH/VCD timecourse becomes a
fixed-length feature row. `channel_features` prefixes each key with the channel
name (`DO_r2_full`, `pH_mad`, ...) to match the naming the rest of the pipeline
keys on. Kept dependency-light (numpy/scipy/pandas only) and side-effect-free so
it is trivially unit-testable and reusable.

`phase_and_rate_features` adds a second, small set of features that
`extract_features` does NOT already cover (phase-thirds mean/slope,
rate-of-change dispersion/oscillation, threshold-crossing time) so callers that
want both (e.g. `pipeline.ingest`) get no duplicate columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, linregress, skew


def extract_features(t: np.ndarray, y: np.ndarray, rolling_window: int = 5,
                     prominence_factor: float = 0.5,
                     min_peak_distance_minutes: float = 5.0) -> dict:
    out: dict[str, float] = {}
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    out["n_points"] = int(len(y))
    out["n_missing"] = int(mask.size - len(y))
    if len(y) < 3:
        return out

    out["start_val"], out["end_val"] = float(y[0]), float(y[-1])
    out["delta_end_start"] = float(y[-1] - y[0])
    out["min"], out["max"] = float(np.min(y)), float(np.max(y))
    out["range"] = out["max"] - out["min"]
    out["mean"], out["median"] = float(np.mean(y)), float(np.median(y))
    out["std"] = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    out["cv"] = out["std"] / out["mean"] if out["mean"] else np.nan
    out["skew"] = float(skew(y)) if len(y) > 2 else np.nan
    out["kurtosis"] = float(kurtosis(y)) if len(y) > 3 else np.nan
    q1, q3 = np.percentile(y, [25, 75])
    out["iqr"] = float(q3 - q1)
    out["mad"] = float(np.median(np.abs(y - np.median(y))))
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    out["auc"] = float(trapz(y, t))

    lr = linregress(t, y)
    out["slope_full"], out["intercept"], out["r2_full"] = float(lr.slope), float(lr.intercept), float(lr.rvalue ** 2)
    n = len(y)
    n20 = max(2, int(0.2 * n))
    out["slope_early"] = float(linregress(t[:n20], y[:n20]).slope)
    out["slope_late"] = float(linregress(t[-n20:], y[-n20:]).slope)
    out["slope_mid"] = float(linregress(t[n20:-n20], y[n20:-n20]).slope) if n > 2 * n20 else np.nan
    out["slope_change_late_early"] = out["slope_late"] - out["slope_early"]

    dy = np.gradient(y, t)
    out["max_dy"], out["min_dy"], out["mean_abs_dy"] = float(np.max(dy)), float(np.min(dy)), float(np.mean(np.abs(dy)))
    ddy = np.gradient(dy, t)
    out["max_ddy"], out["min_ddy"], out["mean_abs_ddy"] = float(np.max(ddy)), float(np.min(ddy)), float(np.mean(np.abs(ddy)))

    out["t_at_max"], out["t_at_min"] = float(t[np.argmax(y)]), float(t[np.argmin(y)])
    idx_max = int(np.argmax(y))
    if idx_max < len(y) - 1:
        half = y[idx_max] * 0.5
        below = np.where(y[idx_max:] <= half)[0]
        out["t_to_half_peak_recovery"] = float(t[idx_max:][below[0]] - out["t_at_max"]) if len(below) else np.nan
    else:
        out["t_to_half_peak_recovery"] = np.nan

    out["stability_frac_within_mad"] = float(np.mean(np.abs(y - out["median"]) <= out["mad"])) if out["mad"] > 0 else 1.0
    if len(y) > rolling_window:
        rs = pd.Series(y).rolling(rolling_window)
        out["mean_rolling_std"] = float(rs.std().mean())
        out["max_dev_from_rolling_mean"] = float(np.max(np.abs(y - rs.mean().to_numpy())))
        idx = rs.mean().diff().abs().idxmax()
        out["t_max_rolling_mean_change"] = float(t[idx]) if pd.notna(idx) else np.nan
    else:
        out["mean_rolling_std"] = out["max_dev_from_rolling_mean"] = out["t_max_rolling_mean_change"] = np.nan
    out["autocorr_lag1"] = float(pd.Series(y).autocorr(lag=1)) if len(y) > 1 else np.nan

    out["z_mean_abs_dy"] = float(np.mean(np.abs(np.gradient((y - out["mean"]) / out["std"], t)))) if out["std"] > 0 else 0.0
    out["mm_mean_abs_dy"] = float(np.mean(np.abs(np.gradient((y - out["min"]) / out["range"], t)))) if out["range"] > 0 else 0.0

    prom = max(prominence_factor * (out["std"] if out["std"] > 0 else out["range"] * 0.1), 1e-9)
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    dist = max(int(np.ceil(min_peak_distance_minutes / dt)), 1)
    peaks, pp = find_peaks(y, prominence=prom, distance=dist)
    troughs, tp = find_peaks(-y, prominence=prom, distance=dist)
    out["n_peaks"], out["n_troughs"] = int(len(peaks)), int(len(troughs))
    out["peak_prom_mean"] = float(np.mean(pp["prominences"])) if len(peaks) else np.nan
    out["peak_prom_max"] = float(np.max(pp["prominences"])) if len(peaks) else np.nan
    out["t_first_peak"] = float(t[peaks[0]]) if len(peaks) else np.nan
    out["t_last_peak"] = float(t[peaks[-1]]) if len(peaks) else np.nan
    out["trough_prom_mean"] = float(np.mean(tp["prominences"])) if len(troughs) else np.nan
    out["trough_prom_max"] = float(np.max(tp["prominences"])) if len(troughs) else np.nan
    out["t_first_trough"] = float(t[troughs[0]]) if len(troughs) else np.nan
    out["t_last_trough"] = float(t[troughs[-1]]) if len(troughs) else np.nan
    return out


def channel_features(t, y, channel: str) -> dict:
    """extract_features on one trace, keys prefixed with the channel name
    (`{channel}_{feature}`). Non-finite / too-short traces yield an empty-ish row."""
    t = pd.to_numeric(pd.Series(t), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    return {f"{channel}_{k}": v for k, v in extract_features(t, y).items()}


def phase_and_rate_features(t: np.ndarray, y: np.ndarray, channel: str,
                            threshold: float, threshold_label: str) -> dict:
    """Features NOT already produced by `extract_features`: phase-thirds
    mean/slope (early/mid/late, split by index like `np.array_split`), the
    dispersion/oscillation-count of the point-to-point rate of change, and the
    first time the signal drops below `threshold`. (`extract_features` already
    covers global stats, AUC, overall/20%-early-late slopes, and the
    max/min/mean of the derivative, so those are not recomputed here.)"""
    t = pd.to_numeric(pd.Series(t), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    out: dict[str, float] = {}
    if len(y) < 3:
        return out

    for name, idx in zip(("early", "mid", "late"), np.array_split(np.arange(len(y)), 3)):
        if len(idx) == 0:
            out[f"{channel}_phase_{name}_mean"] = np.nan
            out[f"{channel}_phase_{name}_slope"] = np.nan
            continue
        ti, yi = t[idx], y[idx]
        out[f"{channel}_phase_{name}_mean"] = float(np.mean(yi))
        out[f"{channel}_phase_{name}_slope"] = (
            float(np.polyfit(ti, yi, 1)[0]) if len(idx) > 1 and np.ptp(ti) > 0 else np.nan
        )

    rate = np.diff(y) / np.diff(t)
    out[f"{channel}_rate_std"] = float(np.std(rate, ddof=0)) if len(rate) else np.nan
    out[f"{channel}_oscillation_count"] = int(np.sum(np.diff(np.sign(rate)) != 0)) if len(rate) > 1 else 0

    below = np.where(y < threshold)[0]
    out[f"{channel}_time_below_{threshold_label}"] = float(t[below[0]]) if len(below) else np.nan
    return out
