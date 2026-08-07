"""Synthetic raw CSVs -> pipeline.ingest -> combined training / predict tables.
No real instrument data required; schemas match the raw-data ingestion spec."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ingest

WELLS = {"B2": "pool1", "B3": "pool2", "B4": "pool3"}
T = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]

# well -> (DO trace, pH trace); B2 crosses both thresholds, B3/B4 never do.
DO_TRACES = {
    "B2": np.linspace(90, 10, len(T)),
    "B3": 50 + 8 * np.sin(np.linspace(0, 6 * np.pi, len(T))),
    "B4": np.linspace(95, 85, len(T)),
}
PH_TRACES = {
    "B2": np.linspace(7.4, 6.5, len(T)),
    "B3": np.full(len(T), 7.0),
    "B4": np.full(len(T), 7.2),
}


def _write_channel(path: Path, traces: dict, value_col: str) -> None:
    rows = []
    for well, y in traces.items():
        for t, v in zip(T, y):
            rows.append({"timestamp": t, "running_time_min": t, "well": well, value_col: v})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_plate(plate_dir: Path, wells: dict, do_traces: dict, ph_traces: dict) -> None:
    """One 24-well plate folder: its own DO/pH/titer/VCD + plate map."""
    plate_dir.mkdir(parents=True, exist_ok=True)
    _write_channel(plate_dir / "raw_passage24w_do.csv", do_traces, "DO_pct")
    _write_channel(plate_dir / "raw_passage24w_ph.csv", ph_traces, "pH")

    pd.DataFrame([{"well": w, "pool_name": p} for w, p in wells.items()]
                ).to_csv(plate_dir / "raw_plate_map.csv", index=False)

    # an extra pool with no DO/pH/plate-map data -> must be dropped from the combined table
    pd.DataFrame([
        {"pool_name": p, "titer_f_quant_mg_per_L": t, "culture_days": 4}
        for p, t in zip(list(wells.values()) + ["unmapped_pool"], [44.0, 52.0, 66.8, 30.0][:len(wells) + 1])
    ]).to_csv(plate_dir / "raw_passage24w_titer.csv", index=False)

    pd.DataFrame([
        {"day": 0, "date_time": "d0", "name": "seed", "total_1e4_cells_per_mL": 50,
         "live_1e4_cells_per_mL": 48, "dead_1e4_cells_per_mL": 2, "viability_pct": 96.0},
        {"day": 4, "date_time": "d4", "name": "seed", "total_1e4_cells_per_mL": 180,
         "live_1e4_cells_per_mL": 170, "dead_1e4_cells_per_mL": 10, "viability_pct": 94.4},
    ]).to_csv(plate_dir / "raw_passage24w_vcd.csv", index=False)


def _write_fedbatch(fedbatch_dir: Path, run_id: str = "SF_run1") -> None:
    """One fed-batch run folder: its own raw_fedbatch_titer.csv."""
    fedbatch_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "run_id": [run_id] * 7,
        "day": [0, 4, 6, 8, 10, 12, 14],
        "vcd_1e6_cells_per_mL": [0.5, 3.0, 5.5, 7.0, 6.5, 5.8, 5.0],
        "viability_pct": [99, 98, 95, 90, 85, 78, 70],
        "titer_mg_per_L": [np.nan, np.nan, np.nan, np.nan, 1200.0, 1500.0, 1680.0],
    }).to_csv(fedbatch_dir / "raw_fedbatch_titer.csv", index=False)


def _write_raw_training(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_plate(raw_dir / "plateA", WELLS, DO_TRACES, PH_TRACES)
    _write_fedbatch(raw_dir / "sf_fedbatch")


def test_build_combined_training_table(tmp_path):
    raw_dir = tmp_path / "raw_training_data"
    _write_raw_training(raw_dir)

    df = ingest.build_combined_training_table(raw_dir)

    # 3 MP wells + 1 SF row; unmapped_pool (no DO/pH/plate-map) correctly dropped
    assert len(df) == 4
    assert set(df["source_dataset"]) == {"MP_24w_Passage", "SF_FedBatch"}
    mp_ids = sorted(df.loc[df["source_dataset"] == "MP_24w_Passage", "run_id"])
    assert mp_ids == ["plateA::pool1", "plateA::pool2", "plateA::pool3"]
    assert (df.loc[df["source_dataset"] == "MP_24w_Passage", "plate_id"] == "plateA").all()

    # no merge-collision suffixes -> phase/rate features didn't clash with extract_features
    assert not any(c.endswith(("_x", "_y")) for c in df.columns)
    for col in ("DO_slope_full", "DO_phase_early_mean", "DO_phase_late_slope",
                "DO_rate_std", "DO_oscillation_count", "DO_time_below_20",
                "pH_time_below_6_8"):
        assert col in df.columns, col

    # threshold crossing: B2/pool1 crosses both thresholds, B4/pool3 crosses neither
    row_pool1 = df.loc[df["run_id"] == "plateA::pool1"].iloc[0]
    row_pool3 = df.loc[df["run_id"] == "plateA::pool3"].iloc[0]
    assert pd.notna(row_pool1["DO_time_below_20"]) and pd.notna(row_pool1["pH_time_below_6_8"])
    assert pd.isna(row_pool3["DO_time_below_20"]) and pd.isna(row_pool3["pH_time_below_6_8"])

    # Step 3: duration normalization
    assert (df.loc[df["source_dataset"] == "MP_24w_Passage", "culture_duration_days"] == 4).all()
    sf = df.loc[df["source_dataset"] == "SF_FedBatch"].iloc[0]
    assert sf["culture_duration_days"] == 14
    assert sf["titer_final_mg_L"] == 1680.0
    np.testing.assert_allclose(sf["titer_per_day_mg_L"], 1680.0 / 14.0)
    np.testing.assert_allclose(row_pool1["titer_per_day_mg_L"], 44.0 / 4.0)

    # Step 2: fed-batch run-level features
    assert sf["peak_vcd_1e6_cells_per_mL"] == 7.0 and sf["day_of_peak_vcd"] == 8.0
    assert sf["day_viability_below_90"] == 10.0 and sf["day_viability_below_80"] == 12.0
    assert sf["specific_growth_rate_mu_per_day"] > 0
    assert sf["viability_decline_rate_pct_per_day"] < 0
    assert pd.notna(sf["ivcd_total"]) and sf["ivcd_total"] > 0
    assert pd.notna(sf["specific_productivity_qp"])

    # Step 4: seed context broadcast identically onto MP rows, NaN on the SF row
    mp = df.loc[df["source_dataset"] == "MP_24w_Passage"]
    assert (mp["seed_vcd_total_1e4cells_mL_day0"] == 50.0).all()
    assert (mp["seed_vcd_total_1e4cells_mL_day4"] == 180.0).all()
    np.testing.assert_allclose(mp["seed_viability_decline_pct_per_day"].iloc[0], (94.4 - 96.0) / 4.0)
    np.testing.assert_allclose(mp["seed_vcd_growth_rate_per_day"].iloc[0], (180.0 - 50.0) / 4.0)
    assert pd.isna(sf["seed_vcd_total_1e4cells_mL_day0"])

    # Step 5: front-loaded columns + descending sort by titer_per_day_mg_L
    assert list(df.columns[:6]) == ["source_dataset", "plate_id", "run_id", "culture_duration_days",
                                    "titer_final_mg_L", "titer_per_day_mg_L"]
    assert df["titer_per_day_mg_L"].is_monotonic_decreasing
    assert df.iloc[0]["run_id"] == "sf_fedbatch::SF_run1"  # 120 mg/L/day dominates the ~11-17 MP rows


def test_write_combined_training_table_roundtrip(tmp_path):
    raw_dir = tmp_path / "raw_training_data"
    _write_raw_training(raw_dir)
    out_path = tmp_path / "training_data" / "combined_ml_training_table.csv"

    df = ingest.write_combined_training_table(raw_dir, out_path)
    assert out_path.exists()
    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == len(df) == 4


def test_multiple_plates_same_pool_name_no_collision(tmp_path):
    """Two plates that happen to reuse the same well labels AND the same
    pool_name must not collide -- plate_id namespaces run_id, and each
    plate's own plate map is used for its own wells."""
    raw_dir = tmp_path / "raw_training_data"
    raw_dir.mkdir(parents=True)
    same_wells = {"B2": "pool1", "B3": "pool2"}
    plateA_do = {w: DO_TRACES[w] for w in same_wells}
    plateA_ph = {w: PH_TRACES[w] for w in same_wells}
    _write_plate(raw_dir / "plateA", same_wells, plateA_do, plateA_ph)
    # plateB reuses the same well labels + pool names, but with plateB/B4's trace under "B2"
    _write_plate(raw_dir / "plateB", same_wells,
                {"B2": DO_TRACES["B4"], "B3": DO_TRACES["B3"]},
                {"B2": PH_TRACES["B4"], "B3": PH_TRACES["B3"]})
    _write_fedbatch(raw_dir / "sf_fedbatch")

    df = ingest.build_combined_training_table(raw_dir)
    mp = df.loc[df["source_dataset"] == "MP_24w_Passage"]
    assert len(mp) == 4  # 2 wells x 2 plates, no dedup / collision
    assert set(mp["run_id"]) == {"plateA::pool1", "plateA::pool2", "plateB::pool1", "plateB::pool2"}
    assert set(mp["plate_id"]) == {"plateA", "plateB"}
    # same pool_name in each plate, but each plate's own DO trace is preserved distinctly
    a_pool1 = mp.loc[mp["run_id"] == "plateA::pool1", "DO_mean"].iloc[0]
    b_pool1 = mp.loc[mp["run_id"] == "plateB::pool1", "DO_mean"].iloc[0]
    assert a_pool1 != b_pool1


def test_multiple_fedbatch_runs(tmp_path):
    """Two fed-batch run folders (e.g. a future second SF run) both contribute
    their own SF_FedBatch row, namespaced by folder even if their internal
    run_id happens to collide."""
    raw_dir = tmp_path / "raw_training_data"
    raw_dir.mkdir(parents=True)
    _write_plate(raw_dir / "plateA", WELLS, DO_TRACES, PH_TRACES)
    _write_fedbatch(raw_dir / "sf_run_1", run_id="SF_run1")
    _write_fedbatch(raw_dir / "sf_run_2", run_id="SF_run1")  # same internal run_id on purpose

    df = ingest.build_combined_training_table(raw_dir)
    sf = df.loc[df["source_dataset"] == "SF_FedBatch"]
    assert len(sf) == 2
    assert set(sf["run_id"]) == {"sf_run_1::SF_run1", "sf_run_2::SF_run1"}


def test_build_predict_table(tmp_path):
    raw_dir = tmp_path / "raw_prediction_data"
    raw_dir.mkdir(parents=True)
    # arbitrary filenames -- discovery is by column content (DO_pct / pH), not filename
    _write_channel(raw_dir / "scc_24w_do_passage.csv", {"C2": DO_TRACES["B2"], "C3": DO_TRACES["B4"]}, "DO_pct")
    _write_channel(raw_dir / "scc_24w_ph_passage.csv", {"C2": PH_TRACES["B2"], "C3": PH_TRACES["B4"]}, "pH")
    pd.DataFrame([{"well": "C2", "pool_name": "cohortA"}, {"well": "C3", "pool_name": "cohortB"}]
                ).to_csv(raw_dir / "raw_plate_map.csv", index=False)

    df = ingest.build_predict_table(raw_dir)
    assert len(df) == 2
    assert set(df["well_id"]) == {"cohortA", "cohortB"}
    assert "titer_final_mg_L" not in df.columns
