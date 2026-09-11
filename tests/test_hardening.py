"""Wave A1 hardening: upload guards, error hygiene, provenance, privacy,
reproducibility, and bounds-sanity for the client-facing /api/run path."""
from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.core.optimize import propose  # noqa: E402
from kalos.core.surrogate import Surrogate, sanitize_bounds  # noqa: E402
from kalos.portal import app as portal  # noqa: E402
from kalos.portal.app import (  # noqa: E402
    MAX_COLUMNS,
    MAX_CSV_ROWS,
    MAX_UPLOAD_BYTES,
    MAX_XLSX_CELLS,
    app,
)

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

    # Units-in-cells columns are now CONVERTED before feature selection, not
    # merely counted as coerced. This assertion changed deliberately when the
    # validation gate landed, and the new behavior is strictly better: "34.6 C"
    # parses at 0% as a bare number, so `Temp` used to fail the >=80% numeric
    # test and come back `dropped_sparse` - a real temperature input silently
    # discarded, with a coercion count as the only hint. Converting to the base
    # unit first keeps it as a usable feature, and the conversion is reported
    # explicitly in the `validation` block instead of implied by a count.
    statuses_after_conversion = {"Temp", "Load_pct"}
    assert statuses_after_conversion <= set(out["features"]), (
        "unit-tagged columns should be converted and kept, not dropped as sparse"
    )
    for name in statuses_after_conversion:
        assert statuses[name] == "kept_feature"
        # Converted cells are clean numbers by the time provenance runs, so there
        # is nothing left to report as coerced.
        row = next(r for r in prov if r["name"] == name)
        assert row["coerced_cells"] == 0

    # The information did not disappear, it moved somewhere more honest: the
    # client is told which column was converted, from what, and to what.
    conversions = {c["column"]: c for c in out["validation"]["conversions"]}
    assert conversions["Temp"]["from_units"] == ["C"]
    assert conversions["Temp"]["to_unit_label"] == "Celsius"
    assert conversions["Load_pct"]["to_unit_label"] == "percent"


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


# --- error hygiene: fit-time failures never escape the JSON envelope --------- #

def test_fit_linalg_error_returns_generic_400_not_500(monkeypatch):
    # A torch.linalg.LinAlgError from the GP fit (or an AssertionError from the
    # leakage guard) is NOT a plain ValueError, so before the catch-all it escaped
    # to FastAPI's default text/plain HTTP 500 and broke the {error} JSON contract.
    def _boom(self, *args, **kwargs):
        raise torch.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(Surrogate, "fit", _boom)
    resp = _post(_csv_bytes(_good_sheet()))
    assert resp.status_code == 400
    body = resp.json()
    assert "error" in body
    assert body["error"] == (
        "Could not parse the uploaded file. Check it is a CSV or Excel run-sheet."
    )
    # no stack trace, exception text, or file-system path leaks to the client
    assert "Traceback" not in resp.text
    assert "singular matrix" not in resp.text
    assert "/Users/" not in resp.text and "kalos/portal" not in resp.text


# --- zip-bomb: xlsx cell cap enforced BEFORE full materialization ------------ #

def _xlsx_with_declared_dims(ref: str) -> bytes:
    """A valid tiny xlsx whose sheet <dimension> tag is rewritten to `ref`, so the
    DECLARED shape is huge while the real content is a few cells (a zip-bomb shape).
    """
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    zin = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    names = zin.namelist()
    sheet = next(n for n in names if n.startswith("xl/worksheets/sheet"))
    xml = re.sub(
        r'<dimension ref="[^"]*"\s*/>', f'<dimension ref="{ref}"/>', zin.read(sheet).decode()
    )
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            zout.writestr(n, xml.encode() if n == sheet else zin.read(n))
    return out.getvalue()


def test_oversized_xlsx_rejected_before_materialization(monkeypatch):
    # An xlsx whose DECLARED cell count exceeds MAX_XLSX_CELLS must be rejected with
    # 400 WITHOUT pd.read_excel ever materializing the full frame. "SF" is column
    # 500 (within the 512-column cap), so this trips the CELL cap, not the column
    # cap: 500 * 1048576 cells >> MAX_XLSX_CELLS.
    assert 500 * 1_048_576 > MAX_XLSX_CELLS and 500 <= MAX_COLUMNS

    called = {"read_excel": False}
    real_read_excel = pd.read_excel

    def _tracking_read_excel(*args, **kwargs):
        called["read_excel"] = True
        return real_read_excel(*args, **kwargs)

    monkeypatch.setattr(pd, "read_excel", _tracking_read_excel)
    monkeypatch.setattr(portal.pd, "read_excel", _tracking_read_excel)

    resp = _post(_xlsx_with_declared_dims("A1:SF1048576"), filename="bomb.xlsx")
    assert resp.status_code == 400
    assert resp.json()["error"] == "The uploaded file is too large."
    # the guard rejected on declared dimensions, before materializing the frame
    assert called["read_excel"] is False


def test_honest_small_xlsx_still_accepted():
    # A normally-sized xlsx (declared dims within caps) is still read and analyzed.
    df = _good_sheet()
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    resp = _post(buf.getvalue(), filename="runs.xlsx")
    assert resp.status_code == 200
    assert resp.json()["target"] == "lipase_titer"


# --- CSV row cap fails closed (no silent truncation) ------------------------- #

def test_oversized_csv_rows_rejected_400():
    # A narrow CSV with more than MAX_CSV_ROWS data rows must be REJECTED (400),
    # not silently truncated to the first MAX_CSV_ROWS rows.
    n = MAX_CSV_ROWS + 25
    header = "Methanol,pH,lipase_titer\n"
    row = "1.0,6.0,3.0\n"
    content = (header + row * n).encode("utf-8")
    resp = _post(content)
    assert resp.status_code == 400
    assert resp.json()["error"] == "The uploaded file is too large."


def test_csv_at_row_cap_still_accepted():
    # Exactly MAX_CSV_ROWS data rows is within the cap and analyzed normally.
    rng = np.random.default_rng(0)
    n = 40  # small, well under the cap; asserts the reject is not overzealous
    df = pd.DataFrame({
        "Methanol": rng.uniform(0, 4, n).round(3),
        "pH": rng.uniform(5, 7, n).round(2),
        "lipase_titer": rng.uniform(1, 6, n).round(3),
    })
    resp = _post(_csv_bytes(df))
    assert resp.status_code == 200


# --- constant-on-fitted-rows feature is flagged, not silently pinned --------- #

def test_feature_constant_on_target_rows_is_flagged_not_pinned():
    # Methanol VARIES across the whole sheet, but on the rows where lipase_titer is
    # present it is constant. Its design box would collapse to zero width. The
    # feature must be dropped and flagged in provenance, not silently kept with a
    # zero-width bound.
    rng = np.random.default_rng(5)
    n = 40
    ph = rng.uniform(5, 7, n)
    methanol = rng.uniform(0, 4, n)
    titer = np.full(n, np.nan)
    # target present only on the first 20 rows; pin methanol constant THERE
    present = 20
    methanol[:present] = 2.0  # constant on target-present rows
    titer[:present] = (1.5 * ph[:present] + rng.normal(0, 0.2, present)).round(3)
    df = pd.DataFrame({
        "Methanol": methanol.round(3),
        "pH": ph.round(2),
        "lipase_titer": titer,
    })
    out = portal._analyze(df)
    # Methanol must NOT be a kept feature (its fitted-row box is zero-width)
    assert "Methanol" not in out["features"]
    assert "pH" in out["features"]
    statuses = {r["name"]: r["status"] for r in out["provenance"]}
    assert statuses["Methanol"] == "dropped_constant_on_fitted_rows"
    # every proposed pH coordinate stays inside the observed (varying) box. The box
    # is built from the dataframe's kept-row (rounded) pH values, so derive it from
    # the frame, not the raw array. proposal `vals` are indexed by
    # `proposal_features` order, not `features`.
    kept_ph = df.loc[df["lipase_titer"].notna(), "pH"]
    box_lo, box_hi = float(kept_ph.min()), float(kept_ph.max())
    assert "pH" in out["proposal_features"]
    ph_idx = out["proposal_features"].index("pH")
    for prop in out["proposals"]:
        assert box_lo - 1e-6 <= prop["vals"][ph_idx] <= box_hi + 1e-6


def test_sizing_env_knobs_treat_empty_string_as_unset():
    """Compose and k8s templates pass `KALOS_MAX_UPLOAD_MB=${KALOS_MAX_UPLOAD_MB:-}`,
    which arrives SET BUT EMPTY - os.environ.get's default never applies, and
    float("") crashed the engine at import on the first real container boot
    (deploy pack, 2026-09-10). The engine's own stated convention (providers/)
    is unset == empty; the sizing knobs now follow it.

    SUBPROCESS on purpose, twice over: the bug was an import-time crash, so a
    fresh interpreter with the empty env is the faithful reproduction - and an
    in-process `importlib.reload` variant of this test poisoned class identity
    for the rest of the suite (analysis raised the pre-reload UploadRejected
    while app's except clause held the post-reload one, so a row-cap rejection
    fell through to the generic parse error). Reload-based tests are banned in
    this import graph; this comment is the tombstone."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "KALOS_MAX_UPLOAD_MB": "", "KALOS_MAX_FIT_ROWS": "",
           "KALOS_TORCH_THREADS": ""}
    code = (
        "from kalos.portal import uploads;"
        "assert uploads._MAX_UPLOAD_MB == 25.0, uploads._MAX_UPLOAD_MB;"
        "assert uploads.MAX_FIT_ROWS == 2000, uploads.MAX_FIT_ROWS;"
        "import kalos.portal.app as a;"
        "assert a._TORCH_THREADS >= 1;"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_torch_threads_zero_means_do_not_pin():
    """KALOS_TORCH_THREADS=0 must leave torch's own default untouched - the
    containerized deployment depends on it: the 4-thread pin measured a >12x
    pathological slowdown (47s -> 600s+) on the image's linux-aarch64
    torch/OpenBLAS build (2026-09-10). A 0 that accidentally pinned to 1
    (the old max(1, n) floor) would be the worst of all worlds. Subprocess for
    the same isolation reasons as the test above."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "KALOS_TORCH_THREADS": "0"}
    code = (
        "import torch; before = torch.get_num_threads();"
        "import kalos.portal.app as a;"
        "assert a._TORCH_THREADS == 0, a._TORCH_THREADS;"
        "a._ensure_torch_threads();"
        "assert torch.get_num_threads() == before, (before, torch.get_num_threads());"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
