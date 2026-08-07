"""Raw-CSV -> featurized-CSV ingestion.

Reads the raw, single-purpose CSVs a user drops into `raw_training_data/` and
produces the combined, engineered `training_data/` table the rest of the
pipeline already consumes. `raw_prediction_data/` mirrors the DO/pH/plate-map
shape (no titer) and produces `prediction_data/predict.csv`.

`raw_training_data/` layout: each 24-well plate is its own subfolder (any
name) containing that plate's `raw_passage24w_do.csv`, `raw_passage24w_ph.csv`,
`raw_passage24w_titer.csv`, `raw_passage24w_vcd.csv`, and `raw_plate_map.csv` --
wells are only unique *within* a plate, so each plate needs its own map from
well -> pool_name. `raw_fedbatch_titer.csv` (one run, no plate) stays directly
in `raw_training_data/`.

Kept separate from `pipeline.feature_extraction` (pure timecourse -> features):
this module owns the *joins* (plate map -> pool -> titer) and the
*bioprocess-specific* engineering (fed-batch growth/IVCD/qP, seed context,
duration normalization) that only make sense once you know which raw file is
which. Steps below follow the ingestion spec 1:1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .feature_extraction import channel_features, phase_and_rate_features

DO_THRESHOLD, DO_THRESHOLD_LABEL = 20.0, "20"
PH_THRESHOLD, PH_THRESHOLD_LABEL = 6.8, "6_8"

# Every plate subfolder under raw_training_data/ must contain these.
PLATE_FILES = ("raw_passage24w_do.csv", "raw_passage24w_ph.csv",
              "raw_passage24w_titer.csv", "raw_passage24w_vcd.csv", "raw_plate_map.csv")
FEDBATCH_FILE = "raw_fedbatch_titer.csv"

# Step 5 column ordering: put these first (if present) so all four modalities
# are visible without scrolling; everything else follows in original order.
FRONT_COLS = ["source_dataset", "plate_id", "run_id", "culture_duration_days",
              "titer_final_mg_L", "titer_per_day_mg_L"]
SUMMARY_COLS = [
    "DO_mean", "DO_min", "DO_max", "DO_slope_full",
    "pH_mean", "pH_min", "pH_max", "pH_slope_full",
    "seed_vcd_total_1e4cells_mL_day0", "seed_vcd_total_1e4cells_mL_day4",
    "seed_viability_pct_day0", "seed_viability_pct_day4",
    "peak_vcd_1e6_cells_per_mL", "day_of_peak_vcd",
    "viability_first", "viability_last", "viability_decline_rate_pct_per_day",
]


# ---------------------------------------------------------------------------
# Step 1: per-well DO/pH feature engineering
# ---------------------------------------------------------------------------

def _well_features(t: np.ndarray, y: np.ndarray, channel: str,
                   threshold: float, threshold_label: str) -> dict:
    """One well's full feature row for one channel: the ~50-feature extractor
    plus the phase/rate/threshold features it doesn't already cover."""
    row = channel_features(t, y, channel)
    row.update(phase_and_rate_features(t, y, channel, threshold, threshold_label))
    return row


def _channel_table(long_df: pd.DataFrame, value_col: str, channel: str,
                   threshold: float, threshold_label: str) -> pd.DataFrame:
    """Long-format (well, running_time_min, value) -> one row per well."""
    rows = []
    for well, g in long_df.sort_values("running_time_min").groupby("well"):
        t = g["running_time_min"].to_numpy(dtype=float)
        y = g[value_col].to_numpy(dtype=float)
        rows.append({"well": well, **_well_features(t, y, channel, threshold, threshold_label)})
    return pd.DataFrame(rows)


def build_passage_features(do_path: Path, ph_path: Path) -> pd.DataFrame:
    """Merge per-well DO + pH feature tables on `well` (one row per well)."""
    do = _channel_table(pd.read_csv(do_path), "DO_pct", "DO", DO_THRESHOLD, DO_THRESHOLD_LABEL)
    ph = _channel_table(pd.read_csv(ph_path), "pH", "pH", PH_THRESHOLD, PH_THRESHOLD_LABEL)
    return do.merge(ph, on="well", how="outer")


def attach_titer_and_plate_map(features_df: pd.DataFrame, plate_map_path: Path,
                               titer_path: Path) -> pd.DataFrame:
    """well -> pool_name (plate map) -> titer/culture_days (titer sheet, left join).
    Pools in the titer sheet that never appear in the plate map are naturally
    dropped since we start from the measured wells, not the titer sheet. A
    measured well with no plate-map entry, or a pool absent from the titer
    sheet, has no usable target -> dropped (it can't be a training row)."""
    plate_map = pd.read_csv(plate_map_path)
    titer = pd.read_csv(titer_path).rename(columns={
        "titer_f_quant_mg_per_L": "titer_final_mg_L",
        "culture_days": "culture_duration_days",
    })
    df = features_df.merge(plate_map, on="well", how="left")
    df = df.merge(titer[["pool_name", "titer_final_mg_L", "culture_duration_days"]],
                  on="pool_name", how="left")
    return df.dropna(subset=["titer_final_mg_L"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: fed-batch run-level feature engineering
# ---------------------------------------------------------------------------

def build_fedbatch_row(titer_path: Path) -> dict:
    df = pd.read_csv(titer_path).sort_values("day").reset_index(drop=True)
    day = df["day"].to_numpy(dtype=float)
    vcd = df["vcd_1e6_cells_per_mL"].to_numpy(dtype=float)
    viab = df["viability_pct"].to_numpy(dtype=float)
    titer = df["titer_mg_per_L"].to_numpy(dtype=float)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    out: dict = {"run_id": df["run_id"].iloc[0]}
    out["culture_duration_days"] = float(np.max(day))

    idx_peak = int(np.argmax(vcd))
    out["peak_vcd_1e6_cells_per_mL"] = float(vcd[idx_peak])
    out["day_of_peak_vcd"] = float(day[idx_peak])

    growth = day <= out["day_of_peak_vcd"]
    out["specific_growth_rate_mu_per_day"] = (
        float(np.polyfit(day[growth], np.log(vcd[growth]), 1)[0]) if growth.sum() >= 2 else np.nan
    )

    out["viability_first"] = float(viab[0])
    out["viability_at_peak_vcd"] = float(viab[idx_peak])
    out["viability_last"] = float(viab[-1])

    decline = day >= out["day_of_peak_vcd"]
    out["viability_decline_rate_pct_per_day"] = (
        float(np.polyfit(day[decline], viab[decline], 1)[0]) if decline.sum() >= 2 else np.nan
    )

    below90, below80 = np.where(viab < 90)[0], np.where(viab < 80)[0]
    out["day_viability_below_90"] = float(day[below90[0]]) if len(below90) else np.nan
    out["day_viability_below_80"] = float(day[below80[0]]) if len(below80) else np.nan

    viable_cell_density = vcd * viab / 100.0
    ivcd_cum = np.zeros(len(day))
    for i in range(1, len(day)):
        ivcd_cum[i] = ivcd_cum[i - 1] + trapz(viable_cell_density[i - 1:i + 1], day[i - 1:i + 1])
    out["ivcd_total"] = float(ivcd_cum[-1])

    measured = np.isfinite(titer)
    if measured.sum() >= 1:
        m_day, m_val = day[measured], titer[measured]
        out["titer_first_measured_day"] = float(m_day[0])
        out["titer_first_measured_mg_L"] = float(m_val[0])
        out["titer_final_day"] = float(m_day[-1])
        out["titer_final_mg_L"] = float(m_val[-1])
        out["titer_accumulation_rate_mg_L_per_day"] = (
            float(np.polyfit(m_day, m_val, 1)[0]) if measured.sum() >= 2 else np.nan
        )
        ivcd_first = float(np.interp(out["titer_first_measured_day"], day, ivcd_cum))
        ivcd_final = float(np.interp(out["titer_final_day"], day, ivcd_cum))
        denom = ivcd_final - ivcd_first
        out["specific_productivity_qp"] = (
            (out["titer_final_mg_L"] - out["titer_first_measured_mg_L"]) / denom if denom else np.nan
        )
    else:
        for k in ("titer_first_measured_day", "titer_first_measured_mg_L", "titer_final_day",
                  "titer_final_mg_L", "titer_accumulation_rate_mg_L_per_day", "specific_productivity_qp"):
            out[k] = np.nan
    return out


# ---------------------------------------------------------------------------
# Step 3: duration normalization (both datasets)
# ---------------------------------------------------------------------------

def normalize_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["titer_per_day_mg_L"] = df["titer_final_mg_L"] / df["culture_duration_days"]
    return df


# ---------------------------------------------------------------------------
# Step 4: seed-culture context, broadcast onto every 24-well row
# ---------------------------------------------------------------------------

def attach_seed_context(mp_df: pd.DataFrame, vcd_path: Path) -> pd.DataFrame:
    vcd = pd.read_csv(vcd_path)
    day0 = vcd.loc[vcd["day"] == 0].iloc[0]
    day4 = vcd.loc[vcd["day"] == 4].iloc[0]

    seed = {
        "seed_vcd_total_1e4cells_mL_day0": float(day0["total_1e4_cells_per_mL"]),
        "seed_vcd_total_1e4cells_mL_day4": float(day4["total_1e4_cells_per_mL"]),
        "seed_viability_pct_day0": float(day0["viability_pct"]),
        "seed_viability_pct_day4": float(day4["viability_pct"]),
    }
    seed["seed_viability_decline_pct_per_day"] = (
        seed["seed_viability_pct_day4"] - seed["seed_viability_pct_day0"]) / 4.0
    seed["seed_vcd_growth_rate_per_day"] = (
        seed["seed_vcd_total_1e4cells_mL_day4"] - seed["seed_vcd_total_1e4cells_mL_day0"]) / 4.0

    df = mp_df.copy()
    for k, v in seed.items():
        df[k] = v
    return df


# ---------------------------------------------------------------------------
# Step 5: concatenate, order, sort
# ---------------------------------------------------------------------------

def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = [c for c in FRONT_COLS if c in df.columns]
    summary = [c for c in SUMMARY_COLS if c in df.columns and c not in front]
    rest = [c for c in df.columns if c not in front and c not in summary]
    return df[front + summary + rest]


def discover_plate_dirs(raw_dir: Path) -> list[Path]:
    """Every immediate subfolder of raw_dir that contains a full set of
    24-well plate files. Order is deterministic (sorted by name)."""
    raw_dir = Path(raw_dir)
    return sorted(p for p in raw_dir.iterdir()
                  if p.is_dir() and all((p / f).exists() for f in PLATE_FILES))


def discover_fedbatch_dirs(raw_dir: Path) -> list[Path]:
    """Every immediate subfolder of raw_dir that holds a fed-batch run
    (`raw_fedbatch_titer.csv`). Supports any number of runs, each its own folder."""
    raw_dir = Path(raw_dir)
    return sorted(p for p in raw_dir.iterdir() if p.is_dir() and (p / FEDBATCH_FILE).exists())


def build_plate_table(plate_dir: Path) -> pd.DataFrame:
    """Steps 1 + 4 for one 24-well plate folder, using *that plate's own*
    well -> pool_name map. `plate_id` (the folder name) namespaces `run_id`
    so the same pool_name reused across different plates never collides."""
    mp = build_passage_features(plate_dir / "raw_passage24w_do.csv", plate_dir / "raw_passage24w_ph.csv")
    mp = attach_titer_and_plate_map(mp, plate_dir / "raw_plate_map.csv", plate_dir / "raw_passage24w_titer.csv")
    mp = attach_seed_context(mp, plate_dir / "raw_passage24w_vcd.csv")
    mp["source_dataset"] = "MP_24w_Passage"
    mp["plate_id"] = plate_dir.name
    mp["run_id"] = mp["plate_id"] + "::" + mp["pool_name"].astype(str)
    return mp


def build_combined_training_table(raw_dir: Path) -> pd.DataFrame:
    """Discovers every dataset subfolder under raw_dir -- a 24-well plate
    folder (own do/ph/titer/vcd/plate map) or a fed-batch run folder (own
    `raw_fedbatch_titer.csv`) -- processes each independently, and pools the
    results. Adding a new plate or run is just adding a new folder."""
    raw_dir = Path(raw_dir)

    plate_dirs = discover_plate_dirs(raw_dir)
    fedbatch_dirs = discover_fedbatch_dirs(raw_dir)
    if not plate_dirs and not fedbatch_dirs:
        raise ValueError(
            f"no dataset folders found under {raw_dir} -- each 24-well plate needs its "
            f"own subfolder containing {', '.join(PLATE_FILES)}, and each fed-batch run "
            f"needs its own subfolder containing {FEDBATCH_FILE}")

    parts = [build_plate_table(d) for d in plate_dirs]
    for d in fedbatch_dirs:
        row = build_fedbatch_row(d / FEDBATCH_FILE)
        row["run_id"] = f"{d.name}::{row['run_id']}"  # namespaced, same as plates
        row["source_dataset"] = "SF_FedBatch"
        parts.append(pd.DataFrame([row]))

    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined = normalize_duration(combined)
    combined = _order_columns(combined)
    return combined.sort_values("titer_per_day_mg_L", ascending=False).reset_index(drop=True)


def write_combined_training_table(raw_dir: Path, out_path: Path) -> pd.DataFrame:
    df = build_combined_training_table(raw_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Prediction-side ingest: same DO/pH engineering, no titer. raw_prediction_data/
# is flat (one cohort) -- the DO/pH CSVs are found by their value column
# (DO_pct / pH), not by a fixed filename, since cohort exports may be named
# anything (e.g. `scc_24w_do_passage.csv`). An optional raw_plate_map.csv gives
# friendly well labels.
# ---------------------------------------------------------------------------

def _find_channel_csv(raw_dir: Path, value_col: str) -> Path:
    for p in sorted(raw_dir.glob("*.csv")):
        if value_col in pd.read_csv(p, nrows=0).columns:
            return p
    raise FileNotFoundError(f"no CSV with a '{value_col}' column found in {raw_dir}")


def build_predict_table(raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    do_path = _find_channel_csv(raw_dir, "DO_pct")
    ph_path = _find_channel_csv(raw_dir, "pH")
    df = build_passage_features(do_path, ph_path)

    plate_map_path = raw_dir / "raw_plate_map.csv"
    if plate_map_path.exists():
        df = df.merge(pd.read_csv(plate_map_path), on="well", how="left")
        df["well_id"] = df["pool_name"].fillna(df["well"])
    else:
        df["well_id"] = df["well"]
    df["source_dataset"] = "cohort"
    return df


def write_predict_table(raw_dir: Path, out_path: Path) -> pd.DataFrame:
    df = build_predict_table(raw_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
