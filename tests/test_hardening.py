"""Wave A1 hardening: upload guards, error hygiene, provenance, privacy,
reproducibility, and bounds-sanity for the client-facing /api/run path."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.core.optimize import propose  # noqa: E402
from kalos.core.surrogate import Surrogate, sanitize_bounds  # noqa: E402
from kalos.portal import app as portal  # noqa: E402
from kalos.portal.app import MAX_COLUMNS, MAX_UPLOAD_BYTES, app  # noqa: E402

client = TestClient(app)


def _good_sheet(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    methanol = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    titer = 1.5 * methanol - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "medium": rng.choice(["A", "B", "C"], n),
        "Methanol": methanol.round(3),
        "pH": ph.round(2),
        "lipase_titer": titer.round(3),
    })


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _post(content: bytes, filename: str = "runs.csv", **data):
    return client.post(
        "/api/run",
        files={"file": (filename, content, "application/octet-stream")},
        data=data,
    )


# --- upload guards ----------------------------------------------------------- #

def test_oversized_upload_rejected_400():
    big = b"a,b,c\n" + b"1,2,3\n" * (MAX_UPLOAD_BYTES // 6 + 10)
    assert len(big) > MAX_UPLOAD_BYTES
    resp = _post(big)
    assert resp.status_code == 400
    assert resp.json()["error"] == "The uploaded file is too large."


def test_wrong_magic_bytes_rejected_400():
    # A PNG header is neither a zip (xlsx) nor tabular text -> generic parse error.
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 512
    resp = _post(png, filename="image.csv")
    assert resp.status_code == 400
    # binary blob yields no numeric columns -> the generic parse message
    assert resp.json()["error"].startswith("Could not parse")


def test_too_many_columns_rejected_400():
    cols = MAX_COLUMNS + 5
    header = ",".join(f"c{i}" for i in range(cols))
    row = ",".join("1" for _ in range(cols))
    content = (header + "\n" + row + "\n").encode("utf-8")
    resp = _post(content)
    assert resp.status_code == 400
    assert resp.json()["error"] == "The uploaded file has too many columns."


def test_corrupt_zip_xlsx_rejected_400_not_500():
    # A zip magic header on a non-xlsx / corrupt zip must be a client 400, not a 500.
    corrupt = b"PK\x03\x04" + b"\x00" * 300
    resp = _post(corrupt, filename="book.xlsx")
    assert resp.status_code == 400
    assert resp.json()["error"].startswith("Could not parse")
    assert "Traceback" not in resp.text and "/Users/" not in resp.text


def test_xlsx_sniffed_by_magic_not_extension():
    # An xlsx uploaded with a .csv name must still be read as Excel (magic bytes).
    df = _good_sheet()
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    resp = _post(buf.getvalue(), filename="runs.csv")
    assert resp.status_code == 200
    assert resp.json()["target"] == "lipase_titer"


# --- error hygiene ----------------------------------------------------------- #

def test_malformed_bytes_generic_400_no_leak():
    # Random binary that decodes to junk -> generic message, no internals leaked.
    blob = bytes(range(256)) * 4
    resp = _post(blob, filename="junk.csv")
    assert resp.status_code == 400
    body = resp.text
    # no stack trace, no file-system path, no exception class name in the body
    assert "Traceback" not in body
    assert "/Users/" not in body and "kalos/portal" not in body
    assert "Error" not in resp.json()["error"] or resp.json()["error"].startswith("Could not")
    assert "line " not in body.lower()


def test_short_sheet_error_does_not_echo_details():
    # A real ValueError from _analyze ("need at least 6 rows ...") must be masked.
    tiny = pd.DataFrame({"Methanol": [1.0, 2.0], "lipase_titer": [3.0, 4.0]})
    resp = _post(_csv_bytes(tiny))
    assert resp.status_code == 400
    assert resp.json()["error"] == (
        "Could not parse the uploaded file. Check it is a CSV or Excel run-sheet."
    )
    assert "6 rows" not in resp.text  # the raw ValueError text is not echoed


# --- provenance -------------------------------------------------------------- #

def _messy_sheet() -> pd.DataFrame:
    n = 30
    rng = np.random.default_rng(3)
    methanol = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    titer = 1.5 * methanol - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({
        "Sample Name": [f"S{i}" for i in range(n)],       # id -> dropped_id
        "medium": rng.choice(["A", "B"], n),               # group col (categorical)
        "Methanol": methanol.round(3),                     # kept feature
        "Temp": [f"{t:.1f} C" for t in rng.uniform(30, 37, n)],  # units-in-cells
        "Load_pct": [f"{p:.0f} %" for p in rng.uniform(50, 90, n)],  # %-strings
        "pH": ph.round(2),                                 # kept feature
        "Const": [7.0] * n,                                # constant -> dropped
        "lipase_titer": titer.round(3),                    # target
    })
    # a duplicate column name (same label appears twice)
    df["Methanol_dup"] = methanol.round(3)
    df = df.rename(columns={"Methanol_dup": "Methanol"})
    return df


def test_provenance_reports_every_column_status():
    out = portal._analyze(_messy_sheet())
    prov = out["provenance"]
    assert "provenance" in out and len(prov) >= 8

    statuses = {row["name"]: row["status"] for row in prov}
    assert statuses["lipase_titer"] == "target"
    assert statuses["pH"] == "kept_feature"
    assert statuses["Methanol"] == "kept_feature"
    assert statuses["Sample Name"] == "dropped_id"
    assert statuses["Const"] == "dropped_constant"
    # the duplicate column is de-duplicated (X, X.1) and both stay in the report
    assert "Methanol.1" in statuses
    assert statuses["Methanol.1"] == "kept_feature"

    # units-in-cells columns were coerced; the non-numeric cell count is surfaced
    temp = next(r for r in prov if r["name"] == "Temp")
    assert temp["coerced_cells"] >= 1  # "34.6 C" does not parse as a bare number
    load = next(r for r in prov if r["name"] == "Load_pct")
    assert load["coerced_cells"] >= 1


# --- privacy / anonymize ----------------------------------------------------- #

def test_anonymize_masks_only_identifier_columns():
    plain = portal._analyze(_good_sheet())
    anon = portal._analyze(_good_sheet(), anonymize=True)
    # process features and target keep their real names (owner UI needs them)
    assert anon["target"] == "lipase_titer"
    assert "Methanol" in anon["features"]
    # the id-type group column is pseudonymized
    assert plain["group_col"] == "medium"
    assert anon["group_col"] != "medium" and anon["group_col"].startswith("col_")
    # provenance keeps feature/target names but masks the id column
    anon_prov = {r["status"]: r["name"] for r in anon["provenance"]}
    assert anon_prov["target"] == "lipase_titer"


# --- reproducibility + audit ------------------------------------------------- #

def test_same_upload_yields_identical_proposals():
    a = portal._analyze(_good_sheet(seed=7))
    b = portal._analyze(_good_sheet(seed=7))
    assert a["proposals"] == b["proposals"]
    assert a["best"] == b["best"]


def test_audit_fields_present():
    out = portal._analyze(_good_sheet())
    assert isinstance(out["seed"], int)
    assert isinstance(out["timestamp"], int) and out["timestamp"] > 0
    assert isinstance(out["engine_version"], str) and out["engine_version"]


def test_response_contract_fields_preserved():
    out = portal._analyze(_good_sheet())
    required = {
        "target", "targets", "features", "best", "n", "d", "group_col",
        "cv_spearman", "cv_ci95", "cv_n_groups", "drivers", "proposal_features",
        "proposals", "oof", "conformal_q", "reliability",
    }
    assert required.issubset(out.keys())
    # new fields are additive
    assert {"provenance", "seed", "timestamp", "engine_version"}.issubset(out.keys())


# --- bounds-sanity ----------------------------------------------------------- #

def test_propose_clamps_into_observed_box():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (16, 2))
    y = X[:, 0] - 0.5 * X[:, 1]
    bounds = np.vstack([X.min(0), X.max(0)])
    s = Surrogate().fit(X, y, bounds=bounds)
    nxt = propose(s, bounds, q=5)
    assert (nxt >= bounds[0] - 1e-9).all()
    assert (nxt <= bounds[1] + 1e-9).all()


def test_degenerate_feature_never_proposes_out_of_range():
    # Regression for the historical Culture_Volume ~= 33,000,000 blow-up: a
    # near-constant feature must not let a proposal escape the observed range.
    rng = np.random.default_rng(1)
    n = 20
    methanol = rng.uniform(0, 4, n)
    culture_volume = np.full(n, 1000.0)  # constant column
    X = np.column_stack([methanol, culture_volume])
    y = 1.5 * methanol + rng.normal(0, 0.1, n)
    bounds = np.vstack([X.min(0), X.max(0)])
    s = Surrogate().fit(X, y, bounds=bounds)
    nxt = propose(s, bounds, q=5)
    # the constant feature (col 1) must stay pinned at ~1000, never explode
    assert np.all(np.abs(nxt[:, 1] - 1000.0) <= 1e-3)
    assert np.all(nxt[:, 0] >= methanol.min() - 1e-6)
    assert np.all(nxt[:, 0] <= methanol.max() + 1e-6)


def test_sanitize_bounds_handles_nonfinite_and_degenerate():
    lower, upper = sanitize_bounds(np.array([[np.nan, 5.0], [np.inf, 5.0]]))
    assert np.isfinite(lower).all() and np.isfinite(upper).all()
    assert (upper >= lower).all()
    # the constant dimension (5, 5) is widened to a non-zero, valid interval
    assert upper[1] > lower[1]
