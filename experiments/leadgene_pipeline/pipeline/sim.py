"""Mechanistic CHO fed-batch bioprocess simulator (clean-room).

A minimal, standard structured model of a fed-batch mammalian cell culture that
maps a few CONTROLLABLE process parameters to an endpoint titer. It gives us two
things: realistic synthetic training data for the pipeline, and a known
ground-truth optimum for closed-loop optimization benchmarks (see
`examples/closed_loop_benchmark.py`).

Model structure (textbook bioprocess kinetics, not any vendor's parameterization):
- Monod (1949) substrate-limited specific growth on glucose.
- Lactate inhibition of growth and lactate-induced death (overflow metabolism).
- Luedeking & Piret (1959) product formation with a non-growth-associated
  (growth-decoupled) term - CHO titer typically rises as growth slows.

States (y): VCD [1e6 cells/mL], Glc [mM], Lac [mM], Titer [mg/L]; time in hours.

The kinetic constants below are our own, chosen for plausible CHO fed-batch
dynamics (14-day culture, peak VCD ~40e6 cells/mL for an intensified fed-batch,
titer O(1000) mg/L), and
tuned so endpoint titer has an INTERIOR optimum in the design space: too little
glucose feed starves growth, too much drives lactate overflow that inhibits and
kills, and growth/product formation peak near pH 7.0. This is a benchmark tool,
NOT a validated predictor of any real process - it must never be presented as one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import qmc

# --- fixed kinetics (our chosen CHO-plausible constants) --------------------- #
MU_G_MAX = 0.045      # max specific growth rate [1/h] at optimal pH
MU_D_MAX = 0.012      # max specific death rate [1/h]
K_GLC = 0.8           # glucose Monod half-saturation [mM]
KI_LAC = 40.0         # lactate half-inhibition of growth [mM]
KD_LAC = 45.0         # lactate half-saturation of death [mM]
Q_GLC = 0.14          # max specific glucose uptake [mM per 1e6 cells/mL per h]
Y_LAC = 0.9           # lactate produced per glucose consumed [mol/mol] (overflow)
Q_LAC_C = 0.10        # max specific lactate re-uptake [mM per 1e6 cells/mL per h]
K_LAC_GATE = 5.0      # glucose below this switches cells onto lactate (diauxie) [mM]
Q_P_MAX = 0.9         # max specific productivity [mg/L per 1e6 cells/mL per h]
PH_OPT = 7.0          # growth-optimal pH
PH_WIDTH = 0.25       # Gaussian pH tolerance
CULTURE_HOURS = 336.0  # 14-day fed-batch
FEED_START_H = 72.0   # bolus/continuous feed begins on day 3


@dataclass(frozen=True)
class ProcessParams:
    """The controllable design variables that define one experiment/run."""
    glc_feed_rate: float   # continuous glucose feed [mM/h] over the feed window
    glc_0: float           # initial glucose [mM]
    vcd_0: float           # seeding viable cell density [1e6 cells/mL]
    ph_setpoint: float     # controlled culture pH


# Design space (lo, hi) for each controllable parameter.
DESIGN_SPACE: dict[str, tuple[float, float]] = {
    "glc_feed_rate": (0.0, 2.5),
    "glc_0": (20.0, 60.0),
    "vcd_0": (0.2, 1.0),
    "ph_setpoint": (6.6, 7.4),
}


def _mu_g_max_at_ph(ph: float) -> float:
    """Growth rate ceiling as a Gaussian penalty around the optimal pH."""
    return MU_G_MAX * float(np.exp(-(((ph - PH_OPT) / PH_WIDTH) ** 2)))


def _rhs(t: float, y: np.ndarray, p: ProcessParams, mu_g_max: float) -> list[float]:
    vcd, glc, lac, _titer = y
    glc = max(glc, 0.0)
    lac = max(lac, 0.0)
    vcd = max(vcd, 0.0)

    mu_g = mu_g_max * (glc / (K_GLC + glc)) * (KI_LAC / (KI_LAC + lac))
    mu_d = MU_D_MAX * (lac / (KD_LAC + lac))
    uptake = Q_GLC * (glc / (K_GLC + glc)) * vcd      # glucose consumption rate
    # Diauxic lactate re-uptake: cells consume lactate once glucose runs low, which
    # both keeps lactate in a realistic range and makes over-feeding (glucose never
    # scarce -> lactate never cleared -> inhibition/death) genuinely costly.
    lac_uptake = Q_LAC_C * (lac / (10.0 + lac)) * (K_LAC_GATE / (K_LAC_GATE + glc)) * vcd
    feed = p.glc_feed_rate if t >= FEED_START_H else 0.0
    growth_ratio = mu_g / mu_g_max if mu_g_max > 0 else 0.0

    d_vcd = (mu_g - mu_d) * vcd
    d_glc = -uptake + feed
    d_lac = Y_LAC * uptake - lac_uptake
    d_titer = Q_P_MAX * (1.0 - growth_ratio) * vcd    # growth-decoupled production
    return [d_vcd, d_glc, d_lac, d_titer]


def simulate(p: ProcessParams, *, hours: float = CULTURE_HOURS,
             n_points: int = 60) -> pd.DataFrame:
    """Integrate the culture and return the time-course of all states."""
    mu_g_max = _mu_g_max_at_ph(p.ph_setpoint)
    t_eval = np.linspace(0.0, hours, n_points)
    sol = solve_ivp(_rhs, (0.0, hours), [p.vcd_0, p.glc_0, 0.0, 0.0],
                    args=(p, mu_g_max), t_eval=t_eval, method="LSODA",
                    rtol=1e-6, atol=1e-8)
    vcd, glc, lac, titer = sol.y
    return pd.DataFrame({"time_h": sol.t, "VCD": np.clip(vcd, 0, None),
                         "Glc": np.clip(glc, 0, None), "Lac": np.clip(lac, 0, None),
                         "titer": np.clip(titer, 0, None)})


def endpoint_titer(p: ProcessParams, *, noise_cv: float = 0.0,
                   rng: np.random.Generator | None = None) -> float:
    """Final-day titer [mg/L]. `noise_cv` adds multiplicative assay noise so the
    benchmark can exercise the pipeline's noise-vs-signal honesty (real assays are
    noisy; 0.0 is the deterministic ground truth)."""
    titer = float(simulate(p).iloc[-1]["titer"])
    if noise_cv > 0:
        rng = rng or np.random.default_rng()
        titer *= float(np.exp(rng.normal(0.0, noise_cv)))
    return titer


def sample_doe(n: int, *, seed: int = 0) -> list[ProcessParams]:
    """Space-filling Latin-hypercube sample over the design space (a proper DoE,
    not ad-hoc random draws)."""
    keys = list(DESIGN_SPACE)
    lo = np.array([DESIGN_SPACE[k][0] for k in keys])
    hi = np.array([DESIGN_SPACE[k][1] for k in keys])
    unit = qmc.LatinHypercube(d=len(keys), seed=seed).random(n)
    scaled = qmc.scale(unit, lo, hi)
    return [ProcessParams(**dict(zip(keys, row))) for row in scaled]


def to_featurized_rows(params: list[ProcessParams], titers: list[float],
                       *, prefix: str = "run") -> pd.DataFrame:
    """Assemble simulated runs into the pipeline's featurized-CSV shape: one row
    per run, the controllable parameters as features, plus `well_id` and `titer`.
    Each run is its own group (no replicate structure) unless noise is layered on
    separately."""
    rows = []
    for i, (p, t) in enumerate(zip(params, titers)):
        rows.append({"well_id": f"{prefix}_{i:03d}", "clone": f"{prefix}_{i:03d}",
                     **asdict(p), "titer": round(float(t), 4)})
    return pd.DataFrame(rows)


def true_optimum(*, n_search: int = 4000, seed: int = 0) -> tuple[ProcessParams, float]:
    """Approximate the design-space optimum by a dense space-filling search
    (deterministic ground truth for regret curves in the closed-loop benchmark)."""
    cand = sample_doe(n_search, seed=seed)
    titers = [endpoint_titer(p) for p in cand]
    best = int(np.argmax(titers))
    return cand[best], float(titers[best])
