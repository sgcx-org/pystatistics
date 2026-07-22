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
the ML optimiser calls it once per objective evaluation. Faithful,
branch-for-branch scalar translation of the pure-numpy reference
(``_ets_models._ets_recursion_reference``); reproduces it BIT-FOR-BIT under
``-ffp-contract=off``. Parity pinned by ``test_ets_kernel_parity.py``.

Structure:
  * ``_ets_core`` — allocation-free ``noexcept nogil`` core with the arithmetic.
  * ``ets_recursion_nb`` — allocating Python entry (historical API).
  * ``EtsWorkspace`` — fit-scoped scratch reused across the optimizer's many
    objective evaluations (removes per-eval allocation; the objective needs
    only fitted/residuals, so it never wants the state history).

Flag encoding: ``mult_error`` (bool), ``has_trend`` (bool), ``season_code``
(0=none, 1=additive, 2=multiplicative). Damping is carried by ``phi_val``.
"""

import numpy as np
from libc.math cimport fabs


cdef void _ets_core(const double* yp, Py_ssize_t n, double alpha, double beta,
                    double gamma, double phi_val, bint mult_error,
                    bint has_trend, int season_code, int m, double l0,
                    double b0, const double* s0p, double* sp, double* fp,
                    double* rp, double* stp, bint want_states,
                    Py_ssize_t n_cols) noexcept nogil:
    """Run the ETS recursion into caller-provided buffers. sp (len m) is the
    seasonal ring buffer (seeded from s0p); fp/rp (len n) receive fitted/
    residuals; stp (len (n+1)*n_cols) the state history when want_states."""
    cdef Py_ssize_t j, t, s_idx, col
    cdef double l_prev = l0, b_prev = b0
    cdef double mu, e, s_old, l_old, b_old, mu_val, base, denom_s, denom_l

    for j in range(m):
        sp[j] = s0p[j]

    # Store initial state row (row 0).
    if want_states:
        stp[0] = l_prev
        col = 1
        if has_trend:
            stp[1] = b_prev
            col = 2
        if season_code != 0:
            for j in range(m):
                stp[col + j] = sp[j]

    for t in range(n):
        s_idx = t % m

        # --- one-step-ahead prediction (mu_t) ---
        if season_code == 0:
            mu = l_prev if not has_trend else l_prev + phi_val * b_prev
        elif season_code == 1:
            if not has_trend:
                mu = l_prev + sp[s_idx]
            else:
                mu = l_prev + phi_val * b_prev + sp[s_idx]
        else:  # season_code == 2 (multiplicative)
            if not has_trend:
                mu = l_prev * sp[s_idx]
            else:
                mu = (l_prev + phi_val * b_prev) * sp[s_idx]

        fp[t] = mu

        # --- error ---
        if mult_error:
            e = (yp[t] - mu) if fabs(mu) < 1e-15 else (yp[t] / mu) - 1.0
            rp[t] = e
        else:
            e = yp[t] - mu
            rp[t] = e

        # --- state update ---
        s_old = sp[s_idx]
        l_old = l_prev
        b_old = b_prev

        if season_code == 0 and not has_trend:
            l_prev = l_old * (1.0 + alpha * e) if mult_error else l_old + alpha * e

        elif season_code == 0 and has_trend:
            if mult_error:
                l_prev = (l_old + phi_val * b_old) * (1.0 + alpha * e)
                b_prev = phi_val * b_old + beta * (l_old + phi_val * b_old) * e
            else:
                l_prev = l_old + phi_val * b_old + alpha * e
                b_prev = phi_val * b_old + beta * e

        elif season_code == 1 and not has_trend:
            if mult_error:
                l_prev = l_old + alpha * (l_old + s_old) * e
                sp[s_idx] = s_old + gamma * (l_old + s_old) * e
            else:
                l_prev = l_old + alpha * e
                sp[s_idx] = s_old + gamma * e

        elif season_code == 1 and has_trend:
            if mult_error:
                mu_val = l_old + phi_val * b_old + s_old
                l_prev = l_old + phi_val * b_old + alpha * mu_val * e
                b_prev = phi_val * b_old + beta * mu_val * e
                sp[s_idx] = s_old + gamma * mu_val * e
            else:
                l_prev = l_old + phi_val * b_old + alpha * e
                b_prev = phi_val * b_old + beta * e
                sp[s_idx] = s_old + gamma * e

        elif season_code == 2 and not has_trend:
            if mult_error:
                l_prev = l_old * (1.0 + alpha * e)
                sp[s_idx] = s_old * (1.0 + gamma * e)
            else:
                denom_s = s_old if fabs(s_old) > 1e-15 else 1e-15
                denom_l = l_old if fabs(l_old) > 1e-15 else 1e-15
                l_prev = l_old + alpha * e / denom_s
                sp[s_idx] = s_old + gamma * e / denom_l

        else:  # season_code == 2 and has_trend
            base = l_old + phi_val * b_old
            if mult_error:
                l_prev = base * (1.0 + alpha * e)
                b_prev = phi_val * b_old + beta * base * e
                sp[s_idx] = s_old * (1.0 + gamma * e)
            else:
                denom_s = s_old if fabs(s_old) > 1e-15 else 1e-15
                denom_l = base if fabs(base) > 1e-15 else 1e-15
                l_prev = base + alpha * e / denom_s
                b_prev = phi_val * b_old + beta * e / denom_s
                sp[s_idx] = s_old + gamma * e / denom_l

        # --- store state row (t + 1) ---
        if want_states:
            stp[(t + 1) * n_cols] = l_prev
            col = 1
            if has_trend:
                stp[(t + 1) * n_cols + 1] = b_prev
                col = 2
            if season_code != 0:
                for j in range(m):
                    stp[(t + 1) * n_cols + col + j] = sp[j]


def ets_recursion_nb(double[::1] y, double alpha, double beta, double gamma,
                     double phi_val, bint mult_error, bint has_trend,
                     int season_code, int m, double l0, double b0,
                     double[::1] s0, bint want_states=True):
    """Run the ETS forward recursion. Returns (fitted, residuals, states).

    ``want_states`` (default True): return the full state history. False on the
    ML objective's hot path (only fitted/residuals used) skips the
    ``(n+1, n_cols)`` allocation and its per-step writes; fitted/residuals are
    bit-identical either way and states comes back empty.
    """
    cdef Py_ssize_t n = y.shape[0]
    cdef Py_ssize_t n_cols = 1
    if has_trend:
        n_cols += 1
    if season_code != 0:
        n_cols += m

    cdef double[::1] s = np.empty(m, dtype=np.float64)
    cdef double[:, ::1] states = np.empty(((n + 1) if want_states else 0,
                                           n_cols if want_states else 0),
                                          dtype=np.float64)
    cdef double[::1] fitted = np.empty(n, dtype=np.float64)
    cdef double[::1] residuals = np.empty(n, dtype=np.float64)
    cdef double* stp = &states[0, 0] if want_states else NULL

    _ets_core(&y[0], n, alpha, beta, gamma, phi_val, mult_error, has_trend,
              season_code, m, l0, b0, &s0[0], &s[0], &fitted[0], &residuals[0],
              stp, want_states, n_cols)
    return np.asarray(fitted), np.asarray(residuals), np.asarray(states)


cdef class EtsWorkspace:
    """Fit-scoped scratch for the ETS ML objective: reuse fitted/residuals/s
    buffers across every likelihood evaluation of a single fit, avoiding
    per-eval allocation. Sized for a fixed series length ``n`` and seasonal
    period ``m``.

    Mutable, single-owner scratch — one per independent fit, never shared across
    concurrent evaluations (buffers are overwritten each call). The objective
    needs only fitted/residuals, so this never computes the state history.
    """
    cdef Py_ssize_t n, m
    cdef double[::1] _s, _fitted, _resid

    def __cinit__(self, Py_ssize_t n, Py_ssize_t m):
        self.n = n
        self.m = m
        self._s = np.empty(m, dtype=np.float64)
        self._fitted = np.empty(n, dtype=np.float64)
        self._resid = np.empty(n, dtype=np.float64)

    def recurse(self, double[::1] y, double alpha, double beta, double gamma,
                double phi_val, bint mult_error, bint has_trend,
                int season_code, int m, double l0, double b0, double[::1] s0):
        """Run the recursion reusing this workspace's buffers.
        Returns (fitted, residuals) views (valid until the next call)."""
        if y.shape[0] != self.n or m != self.m or s0.shape[0] != m:
            raise ValueError(
                f"workspace sized (n={self.n}, m={self.m}) but got "
                f"y={y.shape[0]}, m={m}, s0={s0.shape[0]}")
        _ets_core(&y[0], self.n, alpha, beta, gamma, phi_val, mult_error,
                  has_trend, season_code, m, l0, b0, &s0[0], &self._s[0],
                  &self._fitted[0], &self._resid[0], NULL, False, 0)
        return np.asarray(self._fitted), np.asarray(self._resid)
