"""`propose` reproducibility is the caller's job today - `seed` is an opt-in escape hatch.

`kalos.core.optimize.propose` and `kalos.core.multiobjective.propose_multiobjective`
are stochastic: multi-start acquisition optimization for both, plus the qLogNEHVI
Sobol sampler for the multi-objective path. Reproducibility has always come from
the CALLER seeding the global torch RNG before calling - the portal's
`_seed_everything` and the bench harness both do this today. A library consumer
that imports `kalos.core.optimize` directly and never seeds the global RNG gets
silently non-reproducible proposals under that contract, with no way to opt out.

`seed` closes that gap without changing the contract for existing callers: `None`
(the default) seeds nothing, so behavior is byte-identical to before this
parameter existed. A given `seed` is applied inside `torch.random.fork_rng()`, so
a seeded call is reproducible on its own AND does not clobber the caller's global
RNG state as a side effect - a library function silently resetting its caller's
RNG state on every call would be a new bug, not a fix for the one this closes.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("torch")
pytest.importorskip("botorch")

from kalos.core.multiobjective import MultiObjectiveSurrogate, propose_multiobjective  # noqa: E402
from kalos.core.optimize import propose  # noqa: E402
from kalos.core.surrogate import Surrogate  # noqa: E402


def _fitted_single(seed: int = 0, d: int = 2, n: int = 12):
    """A small GP over a continuous design with one broad interior optimum."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    optimum = np.linspace(0.3, 0.7, d)
    y = -np.sum((X - optimum) ** 2, axis=1) + 0.05 * rng.standard_normal(n)
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    return Surrogate().fit(X, y, bounds=bounds), bounds


def _fitted_multi(seed: int = 0, d: int = 2, n: int = 12):
    """A small two-objective GP pair with a real tension between the objectives."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    Y = np.stack([-((X[:, 0] - 0.7) ** 2), -((X[:, 0] - 0.2) ** 2)], axis=-1)
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    return MultiObjectiveSurrogate().fit(X, Y, bounds=bounds), bounds


# --- single-objective propose() ---------------------------------------------- #


def test_seed_gives_identical_batches_with_no_prior_global_seeding():
    """A library caller who never seeds torch's global RNG still gets a
    reproducible batch by passing `seed` explicitly - the gap this fix closes."""
    s, bounds = _fitted_single()
    a = propose(s, bounds, q=3, seed=123)
    b = propose(s, bounds, q=3, seed=123)
    assert np.array_equal(a, b)


def test_seed_omitted_matches_todays_caller_seeds_globally_contract():
    """The historical contract - the caller reseeds torch globally before each
    call - is unchanged. `seed=None` (the default) still leaves reproducibility
    entirely up to the caller's own global seeding, exactly as before this
    parameter existed."""
    s, bounds = _fitted_single()
    torch.manual_seed(7)
    a = propose(s, bounds, q=3)
    torch.manual_seed(7)
    b = propose(s, bounds, q=3)
    assert np.array_equal(a, b)


def test_seeded_call_does_not_disturb_the_callers_global_rng():
    """A seeded `propose` call must not leak its internal seeding into the
    caller's own subsequent random draws. Draw a torch random before and after
    a seeded call under a fixed global seed: the sequence must be unaffected,
    proving `fork_rng` actually isolated the call rather than merely
    reproducing results by accident."""
    s, bounds = _fitted_single()
    torch.manual_seed(99)
    before = torch.rand(5)

    torch.manual_seed(99)
    propose(s, bounds, q=3, seed=555)  # seeded call - must not touch global state
    after = torch.rand(5)

    assert torch.equal(before, after)


# --- multi-objective propose_multiobjective() -------------------------------- #


def test_multiobjective_seed_gives_close_batches_with_no_prior_global_seeding():
    """Same reproducibility contract as `propose`, on the qLogNEHVI path. Exact
    equality is not asserted here: the multi-objective optimizer carries extra
    internal state (box-decomposition bookkeeping feeding the sampler) that can
    make bit-identical output flaky across calls even under the same seed, so
    `np.allclose` is the honest bar for "reproducible" on this path."""
    s, bounds = _fitted_multi()
    a = propose_multiobjective(s, bounds, q=2, seed=321)
    b = propose_multiobjective(s, bounds, q=2, seed=321)
    assert np.allclose(a, b)


def test_multiobjective_seed_omitted_matches_todays_contract():
    """`seed=None` on the multi-objective path is likewise unchanged from
    before: reproducibility remains the caller's responsibility."""
    s, bounds = _fitted_multi()
    torch.manual_seed(11)
    a = propose_multiobjective(s, bounds, q=2)
    torch.manual_seed(11)
    b = propose_multiobjective(s, bounds, q=2)
    assert np.allclose(a, b)


def test_multiobjective_seeded_call_does_not_disturb_the_callers_global_rng():
    """Same isolation guarantee as the single-objective path, including the
    SobolQMCNormalSampler's own `seed=seed` - none of it should leak out."""
    s, bounds = _fitted_multi()
    torch.manual_seed(88)
    before = torch.rand(5)

    torch.manual_seed(88)
    propose_multiobjective(s, bounds, q=2, seed=444)
    after = torch.rand(5)

    assert torch.equal(before, after)
