"""Unit tests for `kalos.scale.features` - physics-informed scale features.

All numeric expectations here are derived independently from the documented
formulas (van't Riet kLa, the P = Np*rho*N^3*D^5 mixing power correlation,
vs = Q/A, dP = rho*g*h), not by calling the function under test to produce
its own "expected" value. Every test runs on tiny fabricated frames - no
dependency on the real dataset.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from kalos.scale.features import (
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

# One reference case, worked out independently in the docstring/PR notes:
# scale_L=1000 (V=1 m3), default geometry (H/T=1, D/T=1/3, rho=1000 kg/m3).
_SCALE_L = 1000.0
_V_M3 = 1.0
_T_M = (4 * _V_M3 / math.pi) ** (1 / 3)
_H_M = _T_M
_AREA_M2 = (math.pi / 4) * _T_M**2
_D_IMPELLER_M = _T_M / 3.0


def test_tank_geometry_proxy_matches_hand_solved_cylinder():
    geom = tank_geometry_proxy(pd.Series([_SCALE_L]))
    assert geom["tank_diameter_m"].iloc[0] == pytest.approx(_T_M, rel=1e-9)
    assert geom["liquid_height_m"].iloc[0] == pytest.approx(_H_M, rel=1e-9)
    assert geom["cross_section_area_m2"].iloc[0] == pytest.approx(_AREA_M2, rel=1e-9)
    assert geom["impeller_diameter_m"].iloc[0] == pytest.approx(_D_IMPELLER_M, rel=1e-9)


def test_tank_geometry_proxy_nonpositive_and_nan_scale_is_nan():
    geom = tank_geometry_proxy(np.array([0.0, -5.0, np.nan]))
    assert geom.isna().all(axis=None)


def test_log_volume_ratio_known_values():
    out = log_volume_ratio(np.array([1.0, 10.0, 1000.0]), reference_scale_L=1.0)
    np.testing.assert_allclose(out, [0.0, 1.0, 3.0])


def test_log_volume_ratio_custom_reference_shifts_offset_not_shape():
    base = log_volume_ratio(np.array([10.0, 100.0]), reference_scale_L=1.0)
    shifted = log_volume_ratio(np.array([10.0, 100.0]), reference_scale_L=10.0)
    np.testing.assert_allclose(shifted, base - 1.0)


def test_log_volume_ratio_nonpositive_scale_or_reference_is_nan():
    assert np.isnan(log_volume_ratio(np.array([-1.0]))[0])
    assert np.isnan(log_volume_ratio(np.array([0.0]))[0])
    assert np.isnan(log_volume_ratio(np.array([10.0]), reference_scale_L=0.0)[0])


def test_specific_power_proxy_matches_hand_solved_value():
    rpm = 120.0
    n_rev_s = rpm / 60.0
    expected_p_watts = 5.0 * 1000.0 * n_rev_s**3 * _D_IMPELLER_M**5
    expected_pv = expected_p_watts / _V_M3

    out = specific_power_proxy(np.array([_SCALE_L]), np.array([rpm]))
    assert out[0] == pytest.approx(expected_pv, rel=1e-9)


def test_specific_power_proxy_custom_power_number_and_density_scale_linearly():
    rpm = np.array([120.0])
    scale = np.array([_SCALE_L])
    base = specific_power_proxy(scale, rpm)
    doubled_np = specific_power_proxy(scale, rpm, power=PowerNumberAssumption(power_number=10.0))
    doubled_rho = specific_power_proxy(
        scale, rpm, geometry=GeometryAssumptions(liquid_density_kg_per_m3=2000.0)
    )
    assert doubled_np[0] == pytest.approx(2 * base[0], rel=1e-9)
    assert doubled_rho[0] == pytest.approx(2 * base[0], rel=1e-9)


def test_specific_power_proxy_missing_inputs_propagate_nan():
    out = specific_power_proxy(np.array([_SCALE_L, np.nan, -1.0]), np.array([100.0, 100.0, 100.0]))
    assert np.isfinite(out[0])
    assert np.isnan(out[1])
    assert np.isnan(out[2])


def test_superficial_gas_velocity_proxy_matches_hand_solved_value():
    airflow = 500.0
    q_m3_s = airflow / 1000.0 / 60.0
    expected_vs = q_m3_s / _AREA_M2

    out = superficial_gas_velocity_proxy(np.array([_SCALE_L]), np.array([airflow]))
    assert out[0] == pytest.approx(expected_vs, rel=1e-9)


def test_superficial_gas_velocity_proxy_missing_airflow_is_nan():
    out = superficial_gas_velocity_proxy(np.array([_SCALE_L]), np.array([np.nan]))
    assert np.isnan(out[0])


def test_kla_proxy_matches_van_t_riet_form():
    p_v = np.array([246.2])
    vs = np.array([0.009])
    params = VantRietParams(a=0.026, alpha=0.4, beta=0.5)
    expected = 0.026 * (246.2**0.4) * (0.009**0.5)
    out = kla_proxy(p_v, vs, params)
    assert out[0] == pytest.approx(expected, rel=1e-9)


def test_kla_proxy_negative_inputs_are_nan_not_complex():
    out = kla_proxy(np.array([-1.0, 5.0]), np.array([0.01, -0.5]))
    assert np.isnan(out).all()


def test_kla_proxy_exponents_are_exposed_and_change_the_result():
    p_v = np.array([100.0])
    vs = np.array([0.01])
    coalescing = kla_proxy(p_v, vs, VantRietParams(a=0.026, alpha=0.4, beta=0.5))
    non_coalescing = kla_proxy(p_v, vs, VantRietParams(a=0.001, alpha=0.7, beta=0.3))
    assert coalescing[0] != pytest.approx(non_coalescing[0])


def test_hydrostatic_pressure_proxy_matches_hand_solved_value():
    rho = 1000.0
    g = 9.80665
    expected_pa = rho * g * _H_M
    expected_mmhg = expected_pa / 133.322

    out = hydrostatic_pressure_proxy(np.array([_SCALE_L]))
    assert out[0] == pytest.approx(expected_mmhg, rel=1e-6)


def test_hydrostatic_pressure_proxy_increases_with_scale():
    out = hydrostatic_pressure_proxy(np.array([1.0, 10.0, 1000.0, 2000.0]))
    assert np.all(np.diff(out) > 0)


def test_compute_scale_features_returns_all_five_columns_in_order():
    df = pd.DataFrame(
        {
            "scale_L": [1.0, 1000.0],
            "agitation_rpm": [200.0, 120.0],
            "airflow_L_per_min": [0.5, 500.0],
        }
    )
    out = compute_scale_features(df)
    assert list(out.columns) == [
        "log_volume_ratio",
        "specific_power_w_per_m3",
        "superficial_gas_velocity_m_per_s",
        "kla_proxy_per_s",
        "hydrostatic_pressure_mmHg",
    ]
    assert len(out) == 2
    assert out.notna().all(axis=None)


def test_compute_scale_features_requires_scale_column():
    df = pd.DataFrame({"agitation_rpm": [100.0]})
    with pytest.raises(KeyError):
        compute_scale_features(df)


def test_compute_scale_features_missing_sensor_columns_are_all_nan_not_raised():
    df = pd.DataFrame({"scale_L": [10.0, 100.0]})
    out = compute_scale_features(df)
    assert out["log_volume_ratio"].notna().all()
    assert out["hydrostatic_pressure_mmHg"].notna().all()
    assert out["specific_power_w_per_m3"].isna().all()
    assert out["superficial_gas_velocity_m_per_s"].isna().all()
    assert out["kla_proxy_per_s"].isna().all()


def test_compute_scale_features_never_imputes_row_level_nan():
    df = pd.DataFrame(
        {
            "scale_L": [10.0, 100.0],
            "agitation_rpm": [150.0, np.nan],
            "airflow_L_per_min": [1.0, 2.0],
        }
    )
    out = compute_scale_features(df)
    assert out["specific_power_w_per_m3"].iloc[0] == pytest.approx(out["specific_power_w_per_m3"].iloc[0])
    assert np.isnan(out["specific_power_w_per_m3"].iloc[1])
    assert np.isnan(out["kla_proxy_per_s"].iloc[1])
