"""Wave A1.2 production hardening: the GP-training-row cap, the env-configurable
anonymization salt, and the distinct FitError -> 400 mapping.

These cover the three load-bearing behaviors added in A1.2 that the existing
suite did not exercise. The concurrency offload (run_in_threadpool) and the
torch thread cap are configuration-level and are not unit-tested here; they are
covered by the fact that the happy-path /api/run tests still pass through the
threaded handler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.core import analysis as core_analysis  # noqa: E402
from kalos.core.surrogate import FitError  # noqa: E402
from kalos.data.anonymizer import _hash, default_salt  # noqa: E402
from kalos.portal import app as portal  # noqa: E402
from kalos.portal.app import (  # noqa: E402
    _ERR_FIT,
    _ERR_PARSE,
    _ERR_TOO_MANY_FIT_ROWS,
    app,
)

client = TestClient(app)


def _sheet(n: int = 40, seed: int = 0) -> pd.DataFrame:
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


def _post(df: pd.DataFrame, **data):
    return client.post(
        "/api/run",
        files={"file": ("runs.csv", df.to_csv(index=False).encode("utf-8"), "application/octet-stream")},
        data=data,
    )


def test_row_cap_rejects_before_fitting(monkeypatch):
    # The O(n^2) exact GP must be guarded separately from the raw-upload cap. With
    # a low cap, a sheet above it is rejected (not silently subsampled, not OOM'd).
    # `_analyze` lives in `kalos.core.analysis` and reads the cap from its own
    # module globals, so patch it THERE (kalos.portal.analysis only re-exports it).
    monkeypatch.setattr(core_analysis, "MAX_FIT_ROWS", 20)
    r = _post(_sheet(40))
    assert r.status_code == 400
    assert r.json()["error"] == _ERR_TOO_MANY_FIT_ROWS


def test_salt_reads_from_env(monkeypatch):
    # The salt must be operator-controlled, not a fixed hardcoded pseudonym.
    monkeypatch.setenv("KALOS_ANON_SALT", "prod-secret-xyz")
    assert default_salt() == "prod-secret-xyz"
    # A different salt must change the hash of the same identifier, so a barcode is
    # not a globally-stable, dictionary-attackable pseudonym across deployments.
    assert _hash("CHO-K1", "prod-secret-xyz") != _hash("CHO-K1", "other-secret")
    monkeypatch.delenv("KALOS_ANON_SALT", raising=False)
    assert default_salt() == "kalos"  # documented dev fallback


def test_fit_failure_returns_distinct_honest_400(monkeypatch):
    # A numerically-hard-but-valid file (FitError) must not masquerade as a parse
    # failure: it gets its own honest message, still in the {error} JSON envelope.
    def _boom(*args, **kwargs):
        raise FitError("ill-conditioned")

    monkeypatch.setattr(portal, "_analyze", _boom)
    r = _post(_sheet(40))
    assert r.status_code == 400
    assert r.json()["error"] == _ERR_FIT
    assert _ERR_FIT != _ERR_PARSE
