"""Physics-informed scale-up features.

ScaleBridge's standing design (owner-approved, 2026-07-13) is physics priors
first, client data second: the scale-up SIGNAL a model gets to use should come
from a documented correlation, not a fitted slope with no mechanism behind it.
Every function below computes one feature from columns a real batch sheet
plausibly has (`scale_L`, `agitation_rpm`, `airflow_L_per_min`), and every
constant it uses is a FITTABLE PARAMETER with a literature-typical default and
range stated in its dataclass docstring - never a precise "universal" number
invented for this module.

GEOMETRY ASSUMPTION, stated once because every downstream feature depends on
it: a bioreactor vessel is modeled as a right circular cylinder at a fixed
liquid-height-to-tank-diameter ratio, with the working volume equal to the
liquid volume (headspace, dished/domed bottoms, and baffle volume are
ignored). This is the standard simplification used for order-of-magnitude
mixing/mass-transfer estimates in bioprocess scale-up texts (e.g. Doran,
"Bioprocess Engineering Principles", 2nd ed., Ch. 8-9; Nienow, A.W., "Reactor
Engineering in Large Scale Animal Cell Culture", Cytotechnology 50 (2006)
9-33) and is exactly what it claims to be: a PROXY good enough to rank scales
against each other, not a vessel-specific mechanical design calculation.

UNITS. All public functions take `scale_L` in liters (the unit every batch
sheet in this dataset uses) and internally convert to SI (m3, m, m/s, W/m3,
Pa) for the physics, then convert results back to the unit a process engineer
would recognize (W/m3, m/s, 1/s, mmHg) - stated per function.

MISSING DATA. No function here imputes. `agitation_rpm` or `airflow_L_per_min`
missing (NaN) for a row propagates to NaN in every feature that depends on it
(ordinary IEEE-754 NaN propagation through the arithmetic) - the caller
decides whether to drop, impute, or accept a partial feature set, not this
module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

# Gravitational acceleration, m/s^2. Not a fittable parameter - a physical
# constant, not a process assumption.
_G_M_PER_S2 = 9.80665
# 1 mmHg in Pa (exact, by definition of the conventional mmHg). Used only to
# report the hydrostatic-pressure proxy in the same unit pCO2 is reported in.
_MMHG_PER_PA = 1.0 / 133.322


@dataclass(frozen=True)
class GeometryAssumptions:
    """Fittable geometric ratios for the cylindrical-vessel proxy.

    `aspect_ratio_h_over_t`: liquid height / tank diameter (H/T). Typical
    stirred bioreactors run H/T ~= 1.0-1.5 (Doran, Ch. 9); default 1.0 is the
    common "square" tank assumption at small-to-pilot scale.

    `impeller_to_tank_diameter_ratio`: impeller diameter / tank diameter
    (D/T). Typical range 0.3-0.4 for Rushton and pitched-blade turbines used
    in stirred-tank cell culture (Nienow 2006); default 1/3 is the textbook
    "D = T/3" rule of thumb.

    `liquid_density_kg_per_m3`: bulk density of the culture broth. Mammalian
    cell culture media is aqueous and close to water; typical range
    1000-1050 kg/m3. Default 1000.0 (pure-water approximation) is
    deliberately conservative rather than tuned to any specific medium.
    """

    aspect_ratio_h_over_t: float = 1.0
    impeller_to_tank_diameter_ratio: float = 1.0 / 3.0
    liquid_density_kg_per_m3: float = 1000.0


@dataclass(frozen=True)
class PowerNumberAssumption:
    """Fittable impeller power number (Np) for the P/V proxy.

    Np relates rotational speed to power draw via P = Np * rho * N^3 * D^5
    (the standard turbulent-regime mixing power correlation; Rushton, Costich
    & Everett, "Power characteristics of mixing impellers", Chem. Eng. Prog.
    46 (1950); reproduced in Doran Ch. 8). Np is impeller-geometry-specific
    and turbulent-regime (Reynolds number > ~10^4), not a universal constant:
    typical values range from ~1.5 (pitched-blade turbine, axial flow) to
    ~5-6 (Rushton disc turbine, radial flow). Default 5.0 (Rushton-type) is
    the most common cell-culture impeller in this range.
    """

    power_number: float = 5.0


@dataclass(frozen=True)
class VantRietParams:
    """Fittable coefficients for the van't Riet-form kLa correlation.

    kLa = a * (P/V)^alpha * vs^beta

    This is the canonical empirical form for volumetric mass-transfer
    coefficient in stirred, sparged vessels (van't Riet, K., "Review of
    measuring methods and results in nonviscous gas-liquid mass transfer in
    stirred vessels", Ind. Eng. Chem. Process Des. Dev. 18(3) (1979)
    357-364), with P/V in W/m3 and vs (superficial gas velocity) in m/s.

    The literature reports two regimes and neither should be trusted as a
    universal constant here - they are the reason `a`/`alpha`/`beta` are
    exposed rather than baked in:
      - coalescing (electrolyte-free aqueous) systems: a ~= 0.026,
        alpha ~= 0.4, beta ~= 0.5 (van't Riet 1979).
      - non-coalescing (surfactant-containing, e.g. media with Pluronic/
        Poloxamer) systems: a is roughly an order of magnitude smaller and
        alpha runs higher, beta lower - reported ranges in the bioprocess
        literature are roughly a ~= 0.0002-0.003, alpha ~= 0.5-0.9,
        beta ~= 0.2-0.6 (see Doran Ch. 9 for a compiled range; the exact
        values are system-specific and this module does not assert one).

    Defaults below use the coalescing form as the documented starting point;
    a caller with fitted values for their own medium should pass them.
    """

    a: float = 0.026
    alpha: float = 0.4
    beta: float = 0.5


DEFAULT_GEOMETRY = GeometryAssumptions()
DEFAULT_POWER_NUMBER = PowerNumberAssumption()
DEFAULT_VANT_RIET = VantRietParams()


def _liters_to_m3(scale_L: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(scale_L, dtype=float) / 1000.0


def tank_geometry_proxy(
    scale_L: ArrayLike, geometry: GeometryAssumptions = DEFAULT_GEOMETRY
) -> pd.DataFrame:
    """Cylindrical-vessel geometry implied by `scale_L` under `geometry`.

    Solves V = (pi/4) * T^2 * H for tank diameter T with H = r*T (r the
    assumed aspect ratio), i.e. T = (4V / (pi*r))^(1/3), then derives liquid
    height, cross-sectional area, and impeller diameter from T. See the
    module docstring for the cylindrical-vessel / no-headspace assumption
    this rests on.

    Returns a DataFrame (index-aligned with `scale_L`) with columns:
      - `tank_diameter_m`
      - `liquid_height_m`
      - `cross_section_area_m2` (pi/4 * T^2)
      - `impeller_diameter_m`

    `scale_L` <= 0 or NaN propagates to NaN in every output column (no
    imputation, no clamping).
    """
    scale = np.asarray(scale_L, dtype=float)
    v_m3 = _liters_to_m3(scale)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_m3 = np.where(v_m3 > 0, v_m3, np.nan)
        r = geometry.aspect_ratio_h_over_t
        tank_diameter_m = (4.0 * v_m3 / (np.pi * r)) ** (1.0 / 3.0)
        liquid_height_m = r * tank_diameter_m
        cross_section_area_m2 = (np.pi / 4.0) * tank_diameter_m**2
        impeller_diameter_m = geometry.impeller_to_tank_diameter_ratio * tank_diameter_m
    return pd.DataFrame(
        {
            "tank_diameter_m": tank_diameter_m,
            "liquid_height_m": liquid_height_m,
            "cross_section_area_m2": cross_section_area_m2,
            "impeller_diameter_m": impeller_diameter_m,
        }
    )


def log_volume_ratio(scale_L: ArrayLike, reference_scale_L: float = 1.0) -> NDArray[np.float64]:
    """log10(scale_L / reference_scale_L): the single most standard scale-up
    feature (every decade of scale-up is one unit of this feature).

    `reference_scale_L` fixes the zero point (default 1 L, an arbitrary but
    conventional bench-scale reference); it does not change the feature's
    SHAPE, only its offset, so it matters only if the raw value (not just
    relative differences) is interpreted directly. `scale_L` <= 0 or NaN, or
    a non-positive `reference_scale_L`, propagates to NaN.
    """
    scale = np.asarray(scale_L, dtype=float)
    if not np.isfinite(reference_scale_L) or reference_scale_L <= 0:
        return np.full(scale.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = scale / float(reference_scale_L)
        ratio = np.where(ratio > 0, ratio, np.nan)
        return np.log10(ratio)


def specific_power_proxy(
    scale_L: ArrayLike,
    agitation_rpm: ArrayLike,
    geometry: GeometryAssumptions = DEFAULT_GEOMETRY,
    power: PowerNumberAssumption = DEFAULT_POWER_NUMBER,
) -> NDArray[np.float64]:
    """Estimated power input per unit volume (P/V, W/m3) from agitation rpm.

    P = Np * rho * N^3 * D^5 (turbulent-regime mixing power correlation;
    Rushton, Costich & Everett 1950 - see `PowerNumberAssumption`), with N in
    rev/s and D the impeller diameter implied by `tank_geometry_proxy` under
    `geometry`. P/V = P divided by the liquid volume in m3.

    This is a PROXY: it assumes single-impeller turbulent-regime mixing at
    the geometric ratios in `geometry`, ignores multi-impeller stacks, baffle
    configuration, and non-Newtonian effects. It exists to rank scales/
    agitation settings against each other, not to size an impeller motor.

    Missing/non-positive `scale_L` or `agitation_rpm` propagates to NaN.
    """
    scale = np.asarray(scale_L, dtype=float)
    rpm = np.asarray(agitation_rpm, dtype=float)
    geom = tank_geometry_proxy(scale, geometry)
    n_rev_per_s = rpm / 60.0
    v_m3 = _liters_to_m3(scale)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_m3 = np.where(v_m3 > 0, v_m3, np.nan)
        p_watts = (
            power.power_number
            * geometry.liquid_density_kg_per_m3
            * n_rev_per_s**3
            * geom["impeller_diameter_m"].to_numpy() ** 5
        )
        return p_watts / v_m3


def superficial_gas_velocity_proxy(
    scale_L: ArrayLike,
    airflow_L_per_min: ArrayLike,
    geometry: GeometryAssumptions = DEFAULT_GEOMETRY,
) -> NDArray[np.float64]:
    """Estimated superficial gas velocity (vs, m/s) from sparge airflow.

    vs = Q_gas / A, the standard definition used throughout the mass-transfer
    literature (e.g. van't Riet 1979): volumetric gas flow rate divided by
    the vessel cross-sectional area implied by `tank_geometry_proxy` under
    `geometry`. `airflow_L_per_min` is converted L/min -> m3/s.

    This is a PROXY: it assumes the sparged gas crosses the full liquid
    cross-section uniformly (no bubble-column-specific hold-up correction).
    Missing/non-positive `scale_L` or `airflow_L_per_min` propagates to NaN.
    """
    scale = np.asarray(scale_L, dtype=float)
    airflow = np.asarray(airflow_L_per_min, dtype=float)
    geom = tank_geometry_proxy(scale, geometry)
    q_m3_per_s = airflow / 1000.0 / 60.0
    with np.errstate(invalid="ignore", divide="ignore"):
        area = geom["cross_section_area_m2"].to_numpy()
        area = np.where(area > 0, area, np.nan)
        return q_m3_per_s / area


def kla_proxy(
    power_per_volume_w_per_m3: ArrayLike,
    superficial_velocity_m_per_s: ArrayLike,
    params: VantRietParams = DEFAULT_VANT_RIET,
) -> NDArray[np.float64]:
    """Estimated volumetric mass-transfer coefficient (kLa, 1/s), van't Riet
    form: kLa = a * (P/V)^alpha * vs^beta (van't Riet 1979 - see
    `VantRietParams` for the coalescing/non-coalescing coefficient ranges and
    why `a`/`alpha`/`beta` are exposed rather than fixed).

    Takes the already-computed P/V and vs proxies rather than raw columns, so
    it composes directly with `specific_power_proxy` /
    `superficial_gas_velocity_proxy` (or with client-measured P/V and vs,
    when a client has instrumented values instead of the geometric proxy).

    Negative P/V or vs is physically invalid (both are magnitudes) and
    propagates to NaN rather than a complex/undefined fractional power; NaN
    inputs propagate to NaN as everywhere else in this module.
    """
    p_v = np.asarray(power_per_volume_w_per_m3, dtype=float)
    vs = np.asarray(superficial_velocity_m_per_s, dtype=float)
    with np.errstate(invalid="ignore"):
        p_v = np.where(p_v >= 0, p_v, np.nan)
        vs = np.where(vs >= 0, vs, np.nan)
        return params.a * p_v**params.alpha * vs**params.beta


def hydrostatic_pressure_proxy(
    scale_L: ArrayLike, geometry: GeometryAssumptions = DEFAULT_GEOMETRY
) -> NDArray[np.float64]:
    """Estimated hydrostatic pressure at the vessel base (mmHg), from liquid
    height under `geometry`: dP = rho * g * h (standard fluid statics).

    This is the mechanistic PROXY for the scale-dependent pCO2 accumulation
    documented in the dataset (`DATA.md`: "pCO2 accumulation at >500 L") -
    a taller liquid column at large scale both raises static pressure on
    rising gas bubbles (shrinking them, per Boyle's law, and so slowing their
    rise and CO2-stripping efficiency) and increases the hydrostatic head the
    sparge gas must be delivered against. This feature does not model CO2
    solubility/stripping kinetics directly (that would need gas hold-up time
    and mass-transfer driving force, not just static pressure); it is a
    single-number driver a GP can use to pick up the scale-dependent pCO2
    effect the way van't Riet's kLa proxy picks up the scale-dependent
    mixing effect.

    Reported in mmHg (the same unit `pCO2_peak_mmHg` uses in this dataset)
    via the exact mmHg/Pa conversion, so it is directly comparable in
    magnitude to a measured pCO2 value. Missing/non-positive `scale_L`
    propagates to NaN.
    """
    geom = tank_geometry_proxy(scale_L, geometry)
    h_m = geom["liquid_height_m"].to_numpy()
    p_pa = geometry.liquid_density_kg_per_m3 * _G_M_PER_S2 * h_m
    return p_pa * _MMHG_PER_PA


PHYSICS_FEATURE_COLUMNS: tuple[str, ...] = (
    "log_volume_ratio",
    "specific_power_w_per_m3",
    "superficial_gas_velocity_m_per_s",
    "kla_proxy_per_s",
    "hydrostatic_pressure_mmHg",
)

# The subset of PHYSICS_FEATURE_COLUMNS computable from `scale_L` alone - no
# `agitation_rpm`/`airflow_L_per_min` column needed. This is the scale-only
# fallback feature set (see `kalos.scale.transfer.ScaleFeatureConfig.feature_set`)
# for a run sheet that does not record agitation/airflow: `log_volume_ratio`
# depends only on `scale_L`, and `hydrostatic_pressure_mmHg` depends only on
# `scale_L` and geometry - neither reads `agitation_rpm`/`airflow_L_per_min`.
# Derived BY NAME from `PHYSICS_FEATURE_COLUMNS` (filtered, order preserved)
# rather than listed independently, so the two constants cannot drift apart.
_SCALE_ONLY_NAMES = frozenset({"log_volume_ratio", "hydrostatic_pressure_mmHg"})
SCALE_ONLY_FEATURE_COLUMNS: tuple[str, ...] = tuple(c for c in PHYSICS_FEATURE_COLUMNS if c in _SCALE_ONLY_NAMES)
assert set(SCALE_ONLY_FEATURE_COLUMNS) == _SCALE_ONLY_NAMES, (
    "SCALE_ONLY_FEATURE_COLUMNS drifted from PHYSICS_FEATURE_COLUMNS - a name in "
    "_SCALE_ONLY_NAMES no longer matches a column in PHYSICS_FEATURE_COLUMNS"
)


def compute_scale_features(
    df: pd.DataFrame,
    *,
    scale_column: str = "scale_L",
    agitation_column: str = "agitation_rpm",
    airflow_column: str = "airflow_L_per_min",
    reference_scale_L: float = 1.0,
    geometry: GeometryAssumptions = DEFAULT_GEOMETRY,
    power: PowerNumberAssumption = DEFAULT_POWER_NUMBER,
    kla_params: VantRietParams = DEFAULT_VANT_RIET,
) -> pd.DataFrame:
    """All five physics scale features in one call, index-aligned with `df`.

    Requires `scale_column` (raises `KeyError` if absent - a scale-up feature
    set with no scale is a contradiction). `agitation_column` and
    `airflow_column` are OPTIONAL: if either is missing from `df`, the
    features that depend on it (`specific_power_w_per_m3` and/or
    `kla_proxy_per_s` for agitation; `superficial_gas_velocity_m_per_s`
    and/or `kla_proxy_per_s` for airflow) are all-NaN columns rather than a
    raised error, so a caller with a partial batch sheet still gets the
    features it can support. This is the one place in the module that
    substitutes an all-NaN column for an ABSENT column; a present column with
    NaN values still propagates NaN per-row as everywhere else.

    Returns a DataFrame with columns `PHYSICS_FEATURE_COLUMNS`, in that
    order.
    """
    if scale_column not in df.columns:
        raise KeyError(f"'{scale_column}' is required to compute scale features")
    scale = df[scale_column]

    if agitation_column in df.columns:
        p_v = specific_power_proxy(scale, df[agitation_column], geometry, power)
    else:
        p_v = np.full(len(df), np.nan)

    if airflow_column in df.columns:
        vs = superficial_gas_velocity_proxy(scale, df[airflow_column], geometry)
    else:
        vs = np.full(len(df), np.nan)

    return pd.DataFrame(
        {
            "log_volume_ratio": log_volume_ratio(scale, reference_scale_L),
            "specific_power_w_per_m3": p_v,
            "superficial_gas_velocity_m_per_s": vs,
            "kla_proxy_per_s": kla_proxy(p_v, vs, kla_params),
            "hydrostatic_pressure_mmHg": hydrostatic_pressure_proxy(scale, geometry),
        },
        index=df.index,
    )


__all__ = [
    "GeometryAssumptions",
    "PowerNumberAssumption",
    "VantRietParams",
    "DEFAULT_GEOMETRY",
    "DEFAULT_POWER_NUMBER",
    "DEFAULT_VANT_RIET",
    "PHYSICS_FEATURE_COLUMNS",
    "SCALE_ONLY_FEATURE_COLUMNS",
    "tank_geometry_proxy",
    "log_volume_ratio",
    "specific_power_proxy",
    "superficial_gas_velocity_proxy",
    "kla_proxy",
    "hydrostatic_pressure_proxy",
    "compute_scale_features",
]
