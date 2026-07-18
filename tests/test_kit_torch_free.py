"""Guard: `kalos.kit` must stay importable without ever loading torch.

`kalos.kit` is the torch-free facade a sibling repo (voyager-brain-rebuild,
deliberately torch-free) is meant to import instead of keeping its own copies
of these primitives. torch IS installed in this venv (it is a normal test
dependency via the `ml` extra), so the check below runs in a fresh
subprocess and asserts torch is not LOADED as a side effect of the import -
that is the actual contract, and it must hold regardless of what other tests
in this suite have already imported into the current process.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np


def test_kalos_kit_import_does_not_load_torch():
    out = subprocess.run(
        [sys.executable, "-c", "import sys; import kalos.kit; print('torch' in sys.modules)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "False", "importing kalos.kit must never load torch"


def test_kalos_kit_reexports_are_callable():
    import kalos.kit as kit

    # splits: a tiny group-aware split on 6 rows / 3 groups.
    X = np.arange(6).reshape(-1, 1)
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1, 2, 2])
    splits = kit.make_splits(X, y, groups, n_splits=2)
    kit.assert_no_group_leakage(splits, groups)
    assert splits, "expected at least one split for 3 groups"

    # drivers: signed Spearman rho of a feature perfectly correlated with y.
    Z = np.array([1.0, 2.0, 3.0, 4.0])
    signal = np.array([10.0, 20.0, 30.0, 40.0])
    summary = kit.spearman_driver_matrix(Z, signal)
    assert summary["rho"][0] == 1.0

    # conformal: half-width from a small residual set.
    q = kit.q_from_residuals([0.1, 0.2, 0.3, 0.4])
    assert q > 0

    # anonymizer: identity keys get dropped, none of them survive.
    anon = kit.Anonymizer()
    clean = anon.anonymize_meta({"client_id": "acme", "temperature": 37.0})
    assert "client_id" not in clean
    assert clean["temperature"] == 37.0

    # gates: fails closed on missing metrics rather than passing silently.
    result = kit.check_gates({})
    assert result.passed is False
