"""Bit-identity: compiled Cython Kalman kernels == pure-numpy reference.

The compiled kernels in ``_arima_kalman_kernel`` (Cython) must reproduce the
pure-numpy reference in ``_arima_kalman_ref`` to the LAST BIT — exactly, not
within a tolerance. This is the parity contract the wheel build must preserve
on every platform (compiled with ``-ffp-contract=off``, never ``-ffast-math``).

Both implementations are scalar recursions in the same operation order (all
matrix products are explicit loops, never BLAS), so identical last bits are
achievable and are the required guarantee. If this test ever fails on a given
wheel, that wheel was built with an FP flag that reassociates arithmetic
(FMA contraction or fast-math) — fix the build, do not relax the tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pystatistics.timeseries import _arima_kalman_kernel as cy
from pystatistics.timeseries import _arima_kalman_ref as ref
from pystatistics.timeseries._arima_kalman import _build_state_space

# (ar, ma) specs covering pure-AR, pure-MA, mixed, seasonal-sized r, and a
# non-stationary AR that exercises the ok=False doubling-blowup fallback.
_SPECS = [
    (np.array([0.5]), np.array([])),
    (np.array([]), np.array([0.3])),
    (np.array([0.6, -0.2]), np.array([0.4])),
    (np.array([0.4]), np.array([0.3, -0.1, 0.05])),
    (np.array([0.5, -0.3, 0.2, -0.1]), np.array([0.6, -0.4])),
    (np.array([]), np.concatenate([[-0.4], np.zeros(10), [-0.5], [0.2]])),
    (np.array([0.2] + [0.0] * 10 + [-0.3]), np.array([-0.6])),
    (np.array([1.5]), np.array([])),  # non-stationary -> ok=False
]


def _phi_rvec(ar: np.ndarray, ma: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    _, r_vec, r = _build_state_space(ar, ma)
    phi = np.zeros(r, dtype=np.float64)
    if len(ar) > 0:
        phi[: len(ar)] = ar
    return phi, r_vec, r


@pytest.mark.parametrize("ar,ma", _SPECS)
def test_stationary_init_matches_reference_bit_for_bit(ar, ma):
    phi, r_vec, _ = _phi_rvec(ar, ma)
    p_cy, ok_cy = cy.stationary_init(phi.copy(), r_vec.copy())
    p_ref, ok_ref = ref.stationary_init(phi.copy(), r_vec.copy())
    assert ok_cy == ok_ref
    assert_array_equal(np.asarray(p_cy), np.asarray(p_ref))


@pytest.mark.parametrize("ar,ma", _SPECS)
@pytest.mark.parametrize("series_seed", [0, 1, 2, 3])
def test_kalman_loop_matches_reference_bit_for_bit(ar, ma, series_seed):
    phi, r_vec, r = _phi_rvec(ar, ma)
    _, ok_ref = ref.stationary_init(phi.copy(), r_vec.copy())
    p_ref, _ = ref.stationary_init(phi.copy(), r_vec.copy())
    p0 = np.asarray(p_ref) if ok_ref else 1e6 * np.eye(r)

    rng = np.random.default_rng(series_seed)  # NON-DETERMINISTIC: fixed seed
    z = rng.standard_normal(120)

    a_cy = np.zeros(r)
    p_cy = p0.copy()
    innov_cy, f_cy, ok_cy = cy.kalman_loop(z.copy(), phi.copy(), r_vec.copy(), a_cy, p_cy)

    a_rf = np.zeros(r)
    p_rf = p0.copy()
    innov_rf, f_rf, ok_rf = ref.kalman_loop(z.copy(), phi.copy(), r_vec.copy(), a_rf, p_rf)

    assert ok_cy == ok_rf
    assert_array_equal(np.asarray(innov_cy), np.asarray(innov_rf))
    assert_array_equal(np.asarray(f_cy), np.asarray(f_rf))
    # In-place mutation of state mean and covariance must match too.
    assert_array_equal(np.asarray(a_cy), np.asarray(a_rf))
    assert_array_equal(np.asarray(p_cy), np.asarray(p_rf))


@pytest.mark.parametrize("ar,ma", _SPECS[:-1])  # skip non-stationary ok=False spec
def test_workspace_matches_single_call_kernels_bit_for_bit(ar, ma):
    """The fused ArmaKalmanWorkspace must reproduce the single-call kernels'
    innov/F exactly, so the production path is bit-identical to the reference."""
    phi, r_vec, r = _phi_rvec(ar, ma)
    rng = np.random.default_rng(7)  # NON-DETERMINISTIC: fixed seed
    z = rng.standard_normal(140)

    # single-call reference path (init -> diffuse fallback -> loop)
    p0, ok_init = cy.stationary_init(phi.copy(), r_vec.copy())
    P = np.asarray(p0) if ok_init else 1e6 * np.eye(r)
    a = np.zeros(r)
    innov_ref, F_ref, ok_ref = cy.kalman_loop(z.copy(), phi.copy(), r_vec.copy(), a, P.copy())

    # workspace fused path
    ws = cy.ArmaKalmanWorkspace(r, z.size)
    ok, sse, sum_log_F = ws.loglik_parts(z.copy(), phi.copy(), r_vec.copy(), 1e6)

    assert ok == ok_ref
    if ok_ref:
        with np.errstate(divide="ignore", invalid="ignore"):
            sse_ref = float(np.sum(innov_ref * innov_ref / F_ref))
            slF_ref = float(np.sum(np.log(F_ref)))
        assert sse == sse_ref       # bit-for-bit
        assert sum_log_F == slF_ref


def test_workspace_rejects_wrong_dimensions():
    """Capacity guard: a workspace sized (r, n) must fail loud, not corrupt
    memory, if handed mismatched arrays."""
    ws = cy.ArmaKalmanWorkspace(3, 50)
    phi = np.zeros(5)
    r_vec = np.zeros(5)
    r_vec[0] = 1.0
    with pytest.raises((ValueError, IndexError)):
        ws.loglik_parts(np.zeros(50), phi, r_vec, 1e6)


def test_kernel_mutates_state_in_place():
    """Forecasting relies on a/P holding the post-filter predictive state."""
    phi, r_vec, r = _phi_rvec(np.array([0.6, -0.2]), np.array([0.4]))
    p_ref, _ = ref.stationary_init(phi.copy(), r_vec.copy())
    a = np.zeros(r)
    P = np.asarray(p_ref).copy()
    a_before = a.copy()
    cy.kalman_loop(np.ones(50), phi.copy(), r_vec.copy(), a, P)
    assert not np.array_equal(a, a_before)  # a was updated in place
