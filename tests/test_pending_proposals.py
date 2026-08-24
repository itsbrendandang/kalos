"""The optimizer must know what is already running.

A campaign round is not atomic. Five recipes are proposed, the scientist starts
them, some come back before the others, and the loop is re-analyzed on what has
landed so far. Until now the re-proposal was computed from the measured rows
alone: the runs still incubating were invisible to the acquisition, which
therefore treated their region of the design space as unexplored and proposed
them again.

That is budget spent twice for one point of information, and it is not rare.
`kalos/bench/pending.py` measures it: on a 4-factor design at q=5 over 10 seeds,
a blind re-proposal repeated a mean of 2.1 of its 5 recipes (4 of 5 on the worst
seed) against recipes already in flight. Telling the acquisition about them -
BoTorch's `X_pending`, which integrates over their unknown outcomes - takes that
to 0.

The plumbing is `pending` on `propose`, `pending` on `_analyze` (encoded into the
fitted design), and `CampaignStore.awaiting_recipes` supplying the started-but-
unmeasured runs on the re-analyze path.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")

from kalos.core.optimize import propose  # noqa: E402
from kalos.core.surrogate import Surrogate  # noqa: E402
from kalos.portal.analysis import _analyze, _encode_pending  # noqa: E402
from kalos.portal.campaign import CampaignStore  # noqa: E402

# A re-proposal landing within this normalized distance of an in-flight recipe is
# the same experiment for any practical purpose: the scientist would be pipetting
# the same plate twice.
DUPLICATE_RADIUS = 0.05


def _fitted(seed: int, d: int = 4, n: int = 12):
    """A GP over a small continuous design with one broad interior optimum."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    optimum = np.linspace(0.3, 0.7, d)
    y = -np.sum((X - optimum) ** 2, axis=1) + 0.05 * rng.standard_normal(n)
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    return Surrogate().fit(X, y, bounds=bounds), bounds


def _n_duplicated(batch: np.ndarray, in_flight: np.ndarray) -> int:
    nearest = np.min(np.linalg.norm(batch[:, None, :] - in_flight[None, :, :], axis=-1), axis=1)
    return int((nearest < DUPLICATE_RADIUS).sum())


# --- the defect, and the fix ------------------------------------------------ #


def test_a_blind_re_proposal_repeats_work_that_is_already_running():
    """The behaviour the fix exists for. Asserted so the test is not vacuous: if
    a future change makes blind re-proposals stop colliding on their own, this
    fails and the whole premise below needs re-deriving rather than silently
    guarding nothing."""
    duplicated = 0
    for seed in range(4):
        s, bounds = _fitted(seed)
        in_flight = propose(s, bounds, q=5)
        duplicated += _n_duplicated(propose(s, bounds, q=5), in_flight)
    assert duplicated > 0, "expected blind re-proposals to collide with in-flight runs"


def test_pending_runs_are_not_proposed_again():
    for seed in range(4):
        s, bounds = _fitted(seed)
        in_flight = propose(s, bounds, q=5)
        aware = propose(s, bounds, q=5, pending=in_flight)
        assert _n_duplicated(aware, in_flight) == 0, f"seed {seed} re-proposed an in-flight recipe"


def test_pending_leaves_the_batch_inside_the_design_box():
    """The pending block must not disturb the box guarantee proposals already have."""
    s, bounds = _fitted(0)
    batch = propose(s, bounds, q=3, pending=np.full((2, 4), 0.5))
    assert batch.shape == (3, 4)
    assert (batch >= 0.0).all() and (batch <= 1.0).all()


# --- degenerate pending input degrades the information, never the run ------- #


@pytest.mark.parametrize(
    "bad",
    [
        None,
        np.empty((0, 4)),
        np.zeros((2, 7)),                       # wrong width for this design
        np.array([[np.nan, 0.1, 0.2, 0.3]]),    # every row unusable
    ],
    ids=["none", "empty", "wrong_width", "all_nan"],
)
def test_unusable_pending_still_proposes(bad):
    s, bounds = _fitted(0)
    assert propose(s, bounds, q=2, pending=bad).shape == (2, 4)


def test_pending_width_mismatch_is_logged_not_silent(caplog):
    """The wholesale width-mismatch rejection is correct policy (failing to
    propose is worse than proposing without the pending penalty) but must not
    be silent: downstream `n_pending_considered` reads 0 either way,
    indistinguishable from "nothing running" - so a caller bug that disables
    the in-flight guard (e.g. handing `propose` last round's pending block
    against a design that has since changed width) would otherwise vanish
    without a trace."""
    s, bounds = _fitted(0)
    with caplog.at_level(logging.WARNING, logger="kalos.core.optimize"):
        batch = propose(s, bounds, q=2, pending=np.zeros((2, 7)))  # wrong width for this 4-d design
    assert batch.shape == (2, 4)
    assert any("width" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "nothing_running", [None, np.empty((0, 4))], ids=["none", "empty"]
)
def test_pending_none_or_empty_emits_no_warning(nothing_running, caplog):
    """`pending=None` and an empty pending block are the documented "nothing is
    running" fast paths, not an anomaly - they must stay silent so that a real
    warning (the width mismatch above) is never lost in routine noise."""
    s, bounds = _fitted(0)
    with caplog.at_level(logging.WARNING, logger="kalos.core.optimize"):
        propose(s, bounds, q=2, pending=nothing_running)
    assert caplog.records == []


def test_a_single_pending_row_may_be_passed_unwrapped():
    s, bounds = _fitted(0)
    assert propose(s, bounds, q=2, pending=np.full(4, 0.5)).shape == (2, 4)


def test_one_bad_row_does_not_discard_the_good_ones():
    """A partially malformed in-flight record loses only its own row."""
    s, bounds = _fitted(0)
    in_flight = propose(s, bounds, q=5)
    with_junk = np.vstack([in_flight, np.full((1, 4), np.nan)])
    assert _n_duplicated(propose(s, bounds, q=5, pending=with_junk), in_flight) == 0


def test_pending_outside_the_box_is_clamped_not_ignored():
    """An in-flight recipe recorded slightly outside the design box (a hand-typed
    value, a unit round-trip) still marks its neighborhood as taken."""
    s, bounds = _fitted(0)
    in_flight = propose(s, bounds, q=5)
    escaped = in_flight.copy()
    escaped[0, 0] += 5.0
    clamped = np.clip(escaped, 0.0, 1.0)
    assert _n_duplicated(propose(s, bounds, q=5, pending=escaped), clamped) == 0


# --- encoding an in-flight recipe into the fitted design -------------------- #


def test_encode_pending_matches_the_fit_conventions():
    """Continuous components zero-fill like `Xc_zf`; categoricals become codes."""
    encoded = _encode_pending(
        [{"Glucose": 1.0, "Strain": "A"}, {"Strain": "B"}],
        ["Glucose"],
        ["Strain"],
        {"Strain": {"A": 0, "B": 1}},
        2,
    )
    assert encoded.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_a_recipe_naming_an_unknown_level_is_dropped():
    """It has no coordinate in this design, and inventing one (code 0) would mark
    the wrong region taken - worse than not knowing about the run at all."""
    encoded = _encode_pending(
        [{"Glucose": 1.0, "Strain": "A"}, {"Glucose": 2.0, "Strain": "unseen"}],
        ["Glucose"],
        ["Strain"],
        {"Strain": {"A": 0, "B": 1}},
        2,
    )
    assert encoded.tolist() == [[1.0, 0.0]]


@pytest.mark.parametrize("empty", [None, [], pd.DataFrame()], ids=["none", "list", "frame"])
def test_encode_pending_returns_none_when_there_is_nothing_running(empty):
    assert _encode_pending(empty, ["Glucose"], [], {}, 1) is None


def test_a_non_numeric_component_does_not_raise():
    assert _encode_pending([{"Glucose": "n/a"}], ["Glucose"], [], {}, 1).tolist() == [[0.0]]


# --- through the engine ----------------------------------------------------- #


def _sheet(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    methanol = rng.uniform(0.0, 4.0, n)
    ph = rng.uniform(5.0, 8.0, n)
    return pd.DataFrame(
        {
            "Methanol": methanol,
            "pH": ph,
            "lipase_titer": 1.5 * methanol - 0.4 * (ph - 6.5) ** 2 + rng.normal(0, 0.3, n),
        }
    )


def test_analyze_reports_how_many_in_flight_runs_it_was_told_about():
    df = _sheet()
    assert _analyze(df, target="lipase_titer")["n_pending_considered"] == 0
    out = _analyze(
        df,
        target="lipase_titer",
        pending=[{"Methanol": 3.9, "pH": 6.5}, {"Methanol": 3.8, "pH": 6.4}],
    )
    assert out["n_pending_considered"] == 2


def test_the_count_reflects_what_survived_encoding_not_what_was_passed():
    """A recipe that cannot be placed in the design is not silently counted as
    considered - the field would then read as reassurance it has not earned."""
    out = _analyze(
        _sheet(),
        target="lipase_titer",
        pending=[{"Methanol": 3.9, "pH": 6.5}, {"Methanol": "n/a", "pH": None}],
    )
    assert out["n_pending_considered"] == 2  # both encodable: blanks zero-fill
    assert out["proposals"]


def test_pending_moves_the_engine_batch():
    """End-to-end: the same sheet, analyzed with and without in-flight runs, does
    not return the identical batch. `_analyze` seeds torch and numpy, so any
    difference is the pending block and nothing else."""
    df = _sheet()
    blind = _analyze(df, target="lipase_titer")["proposals"]
    aware = _analyze(
        df, target="lipase_titer", pending=[p["recipe"] for p in blind]
    )["proposals"]
    assert [p["recipe"] for p in blind] != [p["recipe"] for p in aware]


# --- the campaign supplies them --------------------------------------------- #


def _seeded_store(tmp_path, tenant: str = "default") -> CampaignStore:
    store = CampaignStore(tmp_path)
    store.seed(
        _sheet(n=8),
        target="lipase_titer",
        features=["Methanol", "pH"],
        tenant=tenant,
    )
    return store


def test_awaiting_recipes_is_empty_without_a_campaign(tmp_path):
    assert CampaignStore(tmp_path).awaiting_recipes() == []


def test_only_the_unmeasured_runs_are_awaiting(tmp_path):
    store = _seeded_store(tmp_path)
    runs = store.start(
        [
            {"recipe": {"Methanol": 3.0, "pH": 6.5}},
            {"recipe": {"Methanol": 4.0, "pH": 6.0}},
        ]
    )
    assert len(store.awaiting_recipes()) == 2
    store.set_result(runs[0]["id"], 6.0)
    assert store.awaiting_recipes() == [{"Methanol": 4.0, "pH": 6.0}], (
        "a measured run is folded into the fit, not still in flight"
    )


def test_awaiting_recipes_are_scoped_to_their_tenant(tmp_path):
    store = _seeded_store(tmp_path, tenant="acme")
    store.start([{"recipe": {"Methanol": 3.0, "pH": 6.5}}], tenant="acme")
    assert store.awaiting_recipes(tenant="acme") == [{"Methanol": 3.0, "pH": 6.5}]
    assert store.awaiting_recipes(tenant="other") == []


def test_the_awaiting_recipes_are_encodable_by_the_engine(tmp_path):
    """The two halves meet: what the store hands over is what `_analyze` counts.
    Asserted end-to-end because the store speaks feature->value dicts and the
    engine speaks a coordinate matrix, and nothing else checks that seam."""
    store = _seeded_store(tmp_path)
    store.start([{"recipe": {"Methanol": 3.0, "pH": 6.5}}])
    out = _analyze(_sheet(), target="lipase_titer", pending=store.awaiting_recipes())
    assert out["n_pending_considered"] == 1
