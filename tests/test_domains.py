"""The domain layer + the mixed (continuous + categorical) engine path.

Covers the torch-free contract of `kalos.domains`, the `ColumnRoles`/`DesignSpace`
mapping, the mixed GP fit + `optimize_acqf_mixed` proposal, and the declared-roles
analyze path on a non-bio run sheet. The continuous bioprocess path parity is
covered by `tests/test_portal.py`.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from kalos.domains import (
    BIOPROCESS_PROFILE,
    GENERIC_PROFILE,
    ColumnRoles,
    DesignSpace,
    Dimension,
    build_design_space,
)


def test_domains_import_does_not_load_torch():
    # The domain layer must be importable from the torch-free kit without pulling
    # in the torch/botorch/gpytorch stack. Checked in a fresh subprocess.
    out = subprocess.run(
        [sys.executable, "-c", "import sys; import kalos.domains; print('torch' in sys.modules)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "False", "importing kalos.domains must never load torch"


def test_column_roles_from_dict_parses_and_validates():
    roles = ColumnRoles.from_dict(
        {"target": "adhesion", "features": ["temp", "resin"], "categoricals": "resin", "ids": ["run"]}
    )
    assert roles.target == "adhesion"
    assert roles.features == ("temp", "resin")
    assert roles.categoricals == ("resin",)  # a bare string is accepted as a 1-list
    assert roles.ids == ("run",)
    with pytest.raises(ValueError):
        ColumnRoles.from_dict({"features": ["a"]})  # missing target


def test_design_space_encode_decode_roundtrip_and_bounds():
    df = pd.DataFrame({"temp": [20.0, 40.0, 60.0], "resin": ["epoxy", "acrylic", "epoxy"]})
    ds = build_design_space(df, ["temp", "resin"], categoricals=("resin",))
    assert ds.cat_dims == [1]
    assert ds.cont_indices == [0]
    assert ds.cat_cardinalities == [2]
    # levels are sorted for determinism
    assert ds.dims[1].levels == ("acrylic", "epoxy")
    b = ds.bounds()
    assert b.shape == (2, 2)
    assert list(b[0]) == [20.0, 0.0] and list(b[1]) == [60.0, 1.0]
    X = ds.encode_frame(df)
    assert X.shape == (3, 2)
    assert list(X[:, 1]) == [1.0, 0.0, 1.0]  # epoxy=1, acrylic=0
    # a proposed row decodes categorical codes back to labels
    assert ds.decode_row([55.0, 0.0]) == [55.0, "acrylic"]
    assert ds.decode_row([55.0, 1.4]) == [55.0, "epoxy"]  # rounded + clamped


def test_design_space_no_categoricals_is_all_continuous():
    ds = DesignSpace((Dimension("a", "continuous", 0.0, 1.0), Dimension("b", "continuous", 2.0, 5.0)))
    assert ds.cat_dims == [] and not ds.has_categoricals
    assert ds.cat_cardinalities == []


def _mixed_sheet(n=60, seed=0):
    rng = np.random.default_rng(seed)
    temp = rng.uniform(20, 80, n)
    time = rng.uniform(1, 10, n)
    resin = rng.choice(["epoxy", "acrylic", "urethane"], n)
    # peak near temp=60, higher time, and the "urethane" level adds a clear bonus
    adhesion = -0.01 * (temp - 60) ** 2 + 0.5 * time + np.where(resin == "urethane", 3.0, 0.0)
    adhesion = adhesion + rng.normal(0, 0.2, n)
    return pd.DataFrame({
        "run_id": range(n),
        "cure_temp": temp.round(2),
        "cure_time": time.round(2),
        "resin_type": resin,
        "adhesion": adhesion.round(3),
    })


def test_mixed_surrogate_fit_and_propose_picks_favorable_level():
    pytest.importorskip("botorch")
    import torch

    from kalos.core.optimize import propose
    from kalos.core.surrogate import Surrogate

    torch.manual_seed(0)
    np.random.seed(0)
    df = _mixed_sheet()
    ds = build_design_space(
        df, ["cure_temp", "cure_time", "resin_type"], categoricals=("resin_type",)
    )
    X = ds.encode_frame(df)
    y = df["adhesion"].to_numpy(float)
    s = Surrogate().fit(X, y, bounds=ds.bounds(), cat_dims=ds.cat_dims)
    assert s._cat_dims == [2]
    batch = propose(s, ds.bounds(), q=3, cat_dims=ds.cat_dims, cat_cardinalities=ds.cat_cardinalities)
    assert batch.shape == (3, 3)
    # categorical codes are snapped to integers within range
    codes = batch[:, 2]
    assert np.all(codes == np.rint(codes))
    assert np.all((codes >= 0) & (codes <= 2))
    # the favorable level dominates the proposed batch
    decoded = [ds.decode_row(r)[2] for r in batch]
    assert decoded.count("urethane") >= 2


def test_analyze_declared_roles_mixed_end_to_end():
    from kalos.portal.analysis import _analyze

    df = _mixed_sheet()
    roles = ColumnRoles(
        target="adhesion",
        features=("cure_temp", "cure_time", "resin_type"),
        ids=("run_id",),
        categoricals=("resin_type",),
    )
    out = _analyze(df, roles=roles, profile=GENERIC_PROFILE)
    assert out["target"] == "adhesion"
    assert out["categorical_features"] == ["resin_type"]
    assert set(out["features"]) == {"cure_temp", "cure_time", "resin_type"}
    # drivers are continuous-only (no nominal category masquerading as a driver)
    assert all(d["name"] in {"cure_temp", "cure_time"} for d in out["drivers"])
    # each proposal exposes a decoded recipe with a real resin label
    rec = out["proposals"][0]["recipe"]
    assert set(rec) == {"cure_temp", "cure_time", "resin_type"}
    assert rec["resin_type"] in {"epoxy", "acrylic", "urethane"}
    # a mixed problem this small uses the exact enumerating optimizer
    assert out["proposal_optimizer"] == "mixed_exact"
    # provenance is honest: declared roles (incl. the ignored id) are marked declared,
    # and the declared id reads as an explicit id-drop, not a data-quality "sparse".
    prov = {p["name"]: p for p in out["provenance"]}
    assert prov["adhesion"]["source"] == "declared"
    assert prov["resin_type"]["source"] == "declared"
    assert prov["run_id"]["source"] == "declared"
    assert prov["run_id"]["status"] == "dropped_id"


def test_declared_role_typo_raises_not_silently_dropped():
    # A declared feature/categorical/group name that does not match a header must
    # fail loudly - silently dropping it would tell the client it was honored.
    from kalos.portal.analysis import _analyze

    df = _mixed_sheet(n=20)
    roles = ColumnRoles(target="adhesion", features=("cure_temp", "cure_tmp"))  # typo
    with pytest.raises(ValueError, match="not found in the sheet"):
        _analyze(df, roles=roles, profile=GENERIC_PROFILE)


def test_declared_non_numeric_feature_reported_honestly():
    # A column declared as a continuous feature but holding text (the caller forgot
    # to mark it categorical) is reported as dropped_non_numeric, not dropped_sparse.
    from kalos.portal.analysis import _analyze

    df = _mixed_sheet(n=30)
    roles = ColumnRoles(
        target="adhesion", features=("cure_temp", "cure_time", "resin_type")
    )  # resin_type is text but NOT declared categorical
    out = _analyze(df, roles=roles, profile=GENERIC_PROFILE)
    prov = {p["name"]: p for p in out["provenance"]}
    assert prov["resin_type"]["status"] == "dropped_non_numeric"
    assert prov["resin_type"]["source"] == "declared"


def test_analyze_bioprocess_inference_unchanged_by_default():
    # With no declared roles the default (bioprocess) profile picks the titer
    # target and excludes other measured outputs - identical to the legacy path.
    from kalos.portal.analysis import _analyze

    rng = np.random.default_rng(0)
    n = 40
    methanol = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    titer = 1.5 * methanol - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({
        "medium": rng.choice(["A", "B", "C"], n),
        "Methanol": methanol.round(3),
        "pH": ph.round(2),
        "biomass_od600": rng.uniform(0, 1, n).round(3),
        "lipase_titer": titer.round(3),
    })
    out = _analyze(df, profile=BIOPROCESS_PROFILE)
    assert out["target"] == "lipase_titer"
    assert out["categorical_features"] == []
    assert out["proposal_optimizer"] == "continuous"
    assert "biomass_od600" not in out["features"]
    assert "Methanol" in out["features"]
