"""ScaleBridge v0: physics-informed scale-up prediction for kalos.

Standalone library layer (no portal/CLI integration - see the package's
tests for usage). Three modules:

  - `kalos.scale.features` - physics-informed scale features computed from
    columns a batch sheet actually has (`scale_L`, `agitation_rpm`,
    `airflow_L_per_min`): log volume ratio, a specific-power (P/V) proxy, a
    superficial gas velocity proxy, a van't-Riet-form kLa proxy, and a
    hydrostatic-pressure proxy for the scale-dependent pCO2 driver. Every
    constant is a fittable parameter with a literature-typical default -
    see each function's docstring for the citation.
  - `kalos.scale.transfer` - `ScaleUpTransferModel`: the v0 transfer model,
    a `kalos.core.surrogate.Surrogate` GP over process + physics-scale
    features (feature-engineering over the proven surrogate, not a new
    model class).
  - `kalos.scale.evaluation` - `leave_one_scale_out_report`: honest
    cross-scale evaluation (per-scale + extrapolation-direction breakdown)
    against naive baselines, reusing `kalos.core.evaluation.logo_report`.
"""
from __future__ import annotations

from .evaluation import leave_one_scale_out_report
from .features import (
    DEFAULT_GEOMETRY,
    DEFAULT_POWER_NUMBER,
    DEFAULT_VANT_RIET,
    PHYSICS_FEATURE_COLUMNS,
    GeometryAssumptions,
    PowerNumberAssumption,
    VantRietParams,
    compute_scale_features,
    hydrostatic_pressure_proxy,
    kla_proxy,
    log_volume_ratio,
    specific_power_proxy,
    superficial_gas_velocity_proxy,
    tank_geometry_proxy,
)
from .transfer import DEFAULT_SCALE_FEATURE_CONFIG, ScaleFeatureConfig, ScaleUpTransferModel, build_feature_matrix

__all__ = [
    "GeometryAssumptions",
    "PowerNumberAssumption",
    "VantRietParams",
    "DEFAULT_GEOMETRY",
    "DEFAULT_POWER_NUMBER",
    "DEFAULT_VANT_RIET",
    "PHYSICS_FEATURE_COLUMNS",
    "tank_geometry_proxy",
    "log_volume_ratio",
    "specific_power_proxy",
    "superficial_gas_velocity_proxy",
    "kla_proxy",
    "hydrostatic_pressure_proxy",
    "compute_scale_features",
    "ScaleFeatureConfig",
    "DEFAULT_SCALE_FEATURE_CONFIG",
    "build_feature_matrix",
    "ScaleUpTransferModel",
    "leave_one_scale_out_report",
]
