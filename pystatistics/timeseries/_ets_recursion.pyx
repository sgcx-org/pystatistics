# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""
Compiled ETS state-space forward recursion (Cython port of the former Numba
``@njit`` kernel).

The ETS one-step recursion (prediction, one-step error, six-way level/trend/
season update) is inherently sequential, so numpy vectorisation cannot help;
the ML optimiser calls it once per objective evaluation. This is a faithful,
branch-for-branch scalar translation of the pure-numpy reference
(``_ets_models._ets_recursion_reference``) and reproduces it BIT-FOR-BIT under
``-ffp-contract=off`` (no FMA contraction, no reassociation). The reference
stays the blessed definition; parity is pinned by
``tests/timeseries/test_ets_kernel_parity.py``.

Flag encoding (matching the reference): ``mult_error`` (bool), ``has_trend``
(bool), ``season_code`` (0=none, 1=additive, 2=multiplicative). Damping is
carried entirely by ``phi_val`` (1.0 when undamped).
"""

import numpy as np
from libc.math cimport fabs


def ets_recursion_nb(double[::1] y, double alpha, double beta, double gamma,
                     double phi_val, bint mult_error, bint has_trend,
                     int season_code, int m, double l0, double b0,
                     double[::1] s0):
    """Run the ETS forward recursion. Returns (fitted, residuals, states)."""
    cdef Py_ssize_t n = y.shape[0]
    cdef Py_ssize_t j, t, s_idx, col

    # Local seasonal ring buffer (copied so the caller's array is untouched).
    cdef double[::1] s = np.empty(m, dtype=np.float64)
    for j in range(m):
        s[j] = s0[j]

    cdef double l_prev = l0
    cdef double b_prev = b0

    cdef Py_ssize_t n_cols = 1
    if has_trend:
        n_cols += 1
    if season_code != 0:
        n_cols += m

    cdef double[:, ::1] states = np.empty((n + 1, n_cols), dtype=np.float64)
    cdef double[::1] fitted = np.empty(n, dtype=np.float64)
    cdef double[::1] residuals = np.empty(n, dtype=np.float64)

    cdef double mu, e, s_old, l_old, b_old, mu_val, base, denom_s, denom_l

    with nogil:
        # Store initial state row (row 0).
        states[0, 0] = l_prev
        col = 1
        if has_trend:
            states[0, 1] = b_prev
            col = 2
        if season_code != 0:
            for j in range(m):
                states[0, col + j] = s[j]

        for t in range(n):
            s_idx = t % m

            # --- one-step-ahead prediction (mu_t) ---
            if season_code == 0:
                mu = l_prev if not has_trend else l_prev + phi_val * b_prev
            elif season_code == 1:
                if not has_trend:
                    mu = l_prev + s[s_idx]
                else:
                    mu = l_prev + phi_val * b_prev + s[s_idx]
            else:  # season_code == 2 (multiplicative)
                if not has_trend:
                    mu = l_prev * s[s_idx]
                else:
                    mu = (l_prev + phi_val * b_prev) * s[s_idx]

            fitted[t] = mu

            # --- error ---
            if mult_error:
                e = (y[t] - mu) if fabs(mu) < 1e-15 else (y[t] / mu) - 1.0
                residuals[t] = e
            else:
                e = y[t] - mu
                residuals[t] = e

            # --- state update ---
            s_old = s[s_idx]
            l_old = l_prev
            b_old = b_prev

            if season_code == 0 and not has_trend:
                # ETS(.,N,N)
                l_prev = l_old * (1.0 + alpha * e) if mult_error else l_old + alpha * e

            elif season_code == 0 and has_trend:
                # ETS(.,A,N) or ETS(.,Ad,N)
                if mult_error:
                    l_prev = (l_old + phi_val * b_old) * (1.0 + alpha * e)
                    b_prev = phi_val * b_old + beta * (l_old + phi_val * b_old) * e
                else:
                    l_prev = l_old + phi_val * b_old + alpha * e
                    b_prev = phi_val * b_old + beta * e

            elif season_code == 1 and not has_trend:
                # ETS(.,N,A)
                if mult_error:
                    l_prev = l_old + alpha * (l_old + s_old) * e
                    s[s_idx] = s_old + gamma * (l_old + s_old) * e
                else:
                    l_prev = l_old + alpha * e
                    s[s_idx] = s_old + gamma * e

            elif season_code == 1 and has_trend:
                # ETS(.,A,A) or ETS(.,Ad,A)
                if mult_error:
                    mu_val = l_old + phi_val * b_old + s_old
                    l_prev = l_old + phi_val * b_old + alpha * mu_val * e
                    b_prev = phi_val * b_old + beta * mu_val * e
                    s[s_idx] = s_old + gamma * mu_val * e
                else:
                    l_prev = l_old + phi_val * b_old + alpha * e
                    b_prev = phi_val * b_old + beta * e
                    s[s_idx] = s_old + gamma * e

            elif season_code == 2 and not has_trend:
                # ETS(.,N,M)
                if mult_error:
                    l_prev = l_old * (1.0 + alpha * e)
                    s[s_idx] = s_old * (1.0 + gamma * e)
                else:
                    denom_s = s_old if fabs(s_old) > 1e-15 else 1e-15
                    denom_l = l_old if fabs(l_old) > 1e-15 else 1e-15
                    l_prev = l_old + alpha * e / denom_s
                    s[s_idx] = s_old + gamma * e / denom_l

            else:  # season_code == 2 and has_trend
                # ETS(.,A,M) or ETS(.,Ad,M)
                base = l_old + phi_val * b_old
                if mult_error:
                    l_prev = base * (1.0 + alpha * e)
                    b_prev = phi_val * b_old + beta * base * e
                    s[s_idx] = s_old * (1.0 + gamma * e)
                else:
                    denom_s = s_old if fabs(s_old) > 1e-15 else 1e-15
                    denom_l = base if fabs(base) > 1e-15 else 1e-15
                    l_prev = base + alpha * e / denom_s
                    b_prev = phi_val * b_old + beta * e / denom_s
                    s[s_idx] = s_old + gamma * e / denom_l

            # --- store state row (t + 1) ---
            states[t + 1, 0] = l_prev
            col = 1
            if has_trend:
                states[t + 1, 1] = b_prev
                col = 2
            if season_code != 0:
                for j in range(m):
                    states[t + 1, col + j] = s[j]

    return np.asarray(fitted), np.asarray(residuals), np.asarray(states)
