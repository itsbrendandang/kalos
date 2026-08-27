"""Portal /api/run analysis path: leakage-controlled grouped CV + honest CI."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from kalos.portal.app import _analyze


def _sheet(n=40, seed=0):
    rng = np.random.default_rng(seed)
    methanol = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    other_output = rng.uniform(0, 1, n)   # a DIFFERENT measured output (leakage bait)
    titer = 1.5 * methanol - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "medium": rng.choice(["A", "B", "C"], n),   # group column
        "Methanol": methanol.round(3),
        "pH": ph.round(2),
        "biomass_od600": other_output.round(3),      # output -> must NOT become a feature
        "lipase_titer": titer.round(3),              # the target
    })


def test_analyze_excludes_outputs_and_reports_honest_cv():
    out = _analyze(_sheet())
    assert out["target"] == "lipase_titer"
    # other measured outputs must never become input features (anti-leakage)
    assert "biomass_od600" not in out["features"]
    assert "lipase_titer" not in out["features"]
    assert "Methanol" in out["features"]
    # the honest grouped-CV report is surfaced with a confidence band + group count
    assert out["cv_spearman"] is not None
    assert out["cv_ci95"] is not None and len(out["cv_ci95"]) == 2
    assert isinstance(out["cv_n_groups"], int) and out["cv_n_groups"] >= 2
    assert len(out["oof"]) > 0
    assert len(out["proposals"]) >= 1
    # honest uncertainty band + reliability verdict (only what this path can assess)
    assert isinstance(out["conformal_q"], float) and out["conformal_q"] >= 0
    rel = out["reliability"]
    assert rel["spearman_floor"] == 0.20
    assert rel["clears_floor"] == (out["cv_spearman"] >= 0.20)
    # An EXACT set on purpose: the reliability block is the client-facing trust
    # verdict, so a field appearing here should be a deliberate act. The producer_*
    # group was added when the pooled Spearman was shown to be satisfiable by
    # separating zeros from non-zeros - a feasibility classifier rather than a
    # ranking of recipes - so the producer-only ranking is reported beside it.
    # Note `clears_floor` still keys off the POOLED score alone: tightening the
    # gate changes which uploads the API accepts and is a product decision, not a
    # side effect of adding a measurement.
    #
    # `calibration` joined it for the same reason as the producer_* group: the
    # pooled Spearman says the model RANKS held-out runs correctly and says
    # nothing about whether its error bars are the right size, and the scientist
    # acts on the +/-. The held-out sd it needs was already computed inside the
    # CV and discarded, so this was reported as unmodeled for want of a number
    # the engine had already paid for. It is reported beside the verdict, not
    # folded into it.
    assert set(rel) == {
        "spearman",
        "ci95",
        "spearman_floor",
        "clears_floor",
        "ci_excludes_zero",
        "unmodeled",
        "producer_spearman",
        "producer_clears_floor",
        "n_producers",
        "producer_threshold",
        "calibration",
    }
    assert "scale-up transfer" in rel["unmodeled"]


def test_latest_reflects_last_upload(tmp_path, monkeypatch):
    from kalos.portal import app as portal
    from kalos.portal.auth import READ, Principal

    princ = Principal(subject="anon", tenant="default", scopes=frozenset({READ}))

    monkeypatch.setattr(portal, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(portal, "_LATEST", {})

    # before any upload the Overview must know there is no data (not fake it)
    assert portal.latest(princ) == {"has_data": False}

    result = portal._analyze(_sheet())
    portal._save_latest(result, "runs.csv")

    got = portal.latest(princ)
    assert got["has_data"] is True
    assert got["dataset"] == "runs.csv"
    assert got["target"] == "lipase_titer"
    assert got["cv_spearman"] is not None
    assert len(got["proposals"]) >= 1
    assert "updated" in got

    # survives a restart (reloads from disk when the in-memory copy is gone)
    monkeypatch.setattr(portal, "_LATEST", {})
    reloaded = portal.latest(princ)
    assert reloaded["has_data"] is True and reloaded["dataset"] == "runs.csv"


def test_importing_the_analysis_module_does_not_pull_torch():
    """The idle `--watch` poller and the portal both import
    `kalos.portal.analysis` at boot and may never run an analysis, so the
    torch/botorch/gpytorch stack is deferred to `_analyze`. That intent lived
    only in a comment, and a single top-level
    `from kalos.core.evaluation import producer_only_spearman` - added for one
    call site inside `_analyze` - quietly defeated it: `evaluation` imports
    `surrogate`, so importing this module cost 1.2s and loaded the whole stack.

    Asserted in a SUBPROCESS because the test session has torch loaded already.
    """
    import subprocess
    import sys

    code = (
        "import sys; import kalos.portal.analysis; "
        "print('torch' in sys.modules or 'botorch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False", (
        "kalos.portal.analysis pulled torch at import time; move the offending "
        "import into the deferred block inside _analyze"
    )
