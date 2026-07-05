"""
Bit-identity: numba ETS kernel == pure-numpy reference recursion.

The 4.6.6 performance change swaps the interpreted per-timestep loop in
``ets_recursion`` for a numba kernel (``ets_recursion_nb``).  Because the
kernel is compiled at ``fastmath=False`` it must reproduce the blessed
pure-numpy reference (``_ets_recursion_reference``) to the last bit in
fp64 — otherwise the maximum-likelihood optimiser would see a different
objective and the whole fit could drift.

These tests assert EXACT equality (atol = 0, rtol = 0) of fitted values,
residuals, and the full state history, across every ETS(error, trend,
season) family and a parameter / initial-state grid that spans what the
fitter actually traverses (including the multiplicative small-denominator
guards).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pystatistics.timeseries._ets_models import (
    ETSSpec,
    _ets_recursion_reference,
    ets_recursion,
    parse_ets_spec,
)

# --------------------------------------------------------------------------
# Grid construction
# --------------------------------------------------------------------------

_ERRORS = ["A", "M"]
_TRENDS = ["N", "A", "Ad"]
_SEASONS = ["N", "A", "M"]
_PERIOD = 4


def _model_string(error: str, trend: str, season: str) -> str:
    return f"{error}{trend}{season}"


def _build_params(spec: ETSSpec, alpha: float, beta: float,
                  gamma: float, phi: float) -> np.ndarray:
    """Assemble the smoothing vector in canonical order for ``spec``."""
    vals = [alpha]
    if spec.trend in ("A", "Ad"):
        vals.append(beta)
    if spec.season in ("A", "M"):
        vals.append(gamma)
    if spec.damped:
        vals.append(phi)
    return np.asarray(vals, dtype=np.float64)


def _build_init_states(spec: ETSSpec, level: float, trend: float,
                       season: np.ndarray) -> np.ndarray:
    """Assemble the initial-state vector ``[l0, b0?, s_{1-m}..s_0?]``."""
    vals = [level]
    if spec.trend in ("A", "Ad"):
        vals.append(trend)
    if spec.season in ("A", "M"):
        vals.extend(season.tolist())
    return np.asarray(vals, dtype=np.float64)


# A strictly positive series so multiplicative error/season stay well
# defined; deterministic (no RNG) so the grid is reproducible.
_T = np.arange(60, dtype=np.float64)
_SERIES = {
    "trend_season": 100.0 + 1.5 * _T
    + 5.0 * np.sin(2.0 * np.pi * _T / _PERIOD) + 20.0,
    "near_flat": 50.0 + 0.01 * _T + 1e-3 * np.cos(_T),
    "steep": 10.0 + 3.0 * _T,
}

# Parameter / initial-state grid.  Values are chosen to exercise both the
# usual region and its edges; the recursion itself is unconstrained, so any
# finite inputs are a valid parity check.
_PARAM_GRID = [
    # (alpha, beta, gamma, phi, l0, b0, season_pattern)
    (0.3, 0.1, 0.2, 0.9, 100.0, 1.0, np.array([1.1, 0.9, 1.05, 0.95])),
    (0.05, 0.02, 0.05, 0.85, 120.0, 0.5, np.array([2.0, -1.0, -0.5, -0.5])),
    (0.8, 0.3, 0.15, 0.98, 90.0, 2.0, np.array([1.2, 0.8, 1.0, 1.0])),
    (0.5, 0.4, 0.3, 0.8, 105.0, -1.0, np.array([0.5, 1.5, 1.0, 1.0])),
]


def _all_specs() -> list[ETSSpec]:
    specs = []
    for error, trend, season in itertools.product(_ERRORS, _TRENDS, _SEASONS):
        period = _PERIOD if season in ("A", "M") else 1
        specs.append(parse_ets_spec(_model_string(error, trend, season),
                                    period=period))
    return specs


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

@pytest.mark.parametrize("spec", _all_specs(), ids=lambda s: s.name)
@pytest.mark.parametrize("series_key", list(_SERIES))
def test_kernel_matches_reference_bit_for_bit(spec, series_key):
    """JIT kernel output == numpy reference output, exactly, for every model
    family across the parameter grid and several series."""
    y = _SERIES[series_key]
    for (alpha, beta, gamma, phi, l0, b0, season) in _PARAM_GRID:
        params = _build_params(spec, alpha, beta, gamma, phi)
        init = _build_init_states(spec, l0, b0, season)

        f_jit, r_jit, s_jit = ets_recursion(y, spec, params, init)
        f_ref, r_ref, s_ref = _ets_recursion_reference(y, spec, params, init)

        assert_array_equal(
            f_jit, f_ref,
            err_msg=f"fitted mismatch: {spec.name} / {series_key} "
                    f"alpha={alpha}",
        )
        assert_array_equal(
            r_jit, r_ref,
            err_msg=f"residuals mismatch: {spec.name} / {series_key} "
                    f"alpha={alpha}",
        )
        assert_array_equal(
            s_jit, s_ref,
            err_msg=f"states mismatch: {spec.name} / {series_key} "
                    f"alpha={alpha}",
        )


def test_kernel_output_shapes():
    """Kernel returns the reference's shapes for a representative seasonal
    model (states has n+1 rows and 1 + trend + period columns)."""
    spec = parse_ets_spec("AAA", period=_PERIOD)
    y = _SERIES["trend_season"]
    params = _build_params(spec, 0.3, 0.1, 0.2, 0.9)
    init = _build_init_states(spec, 100.0, 1.0, np.array([1.0, -1.0, 0.5, -0.5]))
    fitted, resid, states = ets_recursion(y, spec, params, init)
    assert fitted.shape == (len(y),)
    assert resid.shape == (len(y),)
    assert states.shape == (len(y) + 1, 1 + 1 + _PERIOD)


def test_multiplicative_season_small_denominator_guard():
    """A multiplicative-season / additive-error model with a seasonal state
    driven near zero exercises the 1e-15 denominator guard; the kernel must
    still match the reference bit-for-bit there."""
    spec = parse_ets_spec("ANM", period=_PERIOD)
    y = _SERIES["trend_season"]
    # Seasonal pattern containing a value that can be driven toward the guard.
    params = _build_params(spec, 0.6, 0.0, 0.6, 0.0)
    init = _build_init_states(spec, 100.0, 0.0,
                              np.array([1e-16, 2.0, 1.0, 1.0]))
    f_jit, r_jit, s_jit = ets_recursion(y, spec, params, init)
    f_ref, r_ref, s_ref = _ets_recursion_reference(y, spec, params, init)
    assert_array_equal(f_jit, f_ref)
    assert_array_equal(r_jit, r_ref)
    assert_array_equal(s_jit, s_ref)
