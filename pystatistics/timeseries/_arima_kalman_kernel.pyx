# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""
Compiled Kalman kernels for ARMA exact-ML (Cython port of the former Numba
``@njit`` kernels in ``_arima_kalman``).

Structure:
  * ``_stationary_core`` / ``_kalman_core`` — ``noexcept nogil`` raw-pointer
    cores holding all the arithmetic. Scratch is passed in, so a caller can
    reuse buffers across many likelihood evaluations (see ``ArmaKalmanWorkspace``).
  * ``stationary_init`` / ``kalman_loop`` — thin Python entry points that
    allocate scratch and call the cores. These preserve the historical API the
    parity tests pin.
  * ``ArmaKalmanWorkspace`` — fused, buffer-reusing path (init+filter in one
    ``nogil`` call). Bit-identical to the single-call kernels. Driven once per
    fit by the ML optimizer loop (wired through ``_arima_kalman._new_workspace``
    and the ``_workspace=`` objective path); the single-call entry points remain
    for one-shot / low-level use.

Both cores are faithful scalar translations of the pure-numpy reference in
``_arima_kalman_ref`` and are checked bit-for-bit against it. All matrix
products are explicit scalar loops (never BLAS ``@``), so results are fixed and
platform-reproducible; combined with ``-ffp-contract=off`` at compile time (no
FMA contraction) this yields identical last bits to the reference on every
wheel we ship. The raw-pointer refactor changes NO arithmetic or accumulation
order relative to the reference.

Contract (Rule 2): callers pass C-contiguous float64 arrays with matching
shapes (phi, R_vec length r; a length r; P shape (r, r); z length n). The
Python wrappers in ``_arima_kalman`` guarantee this.
"""

import numpy as np
from libc.math cimport isfinite, fabs
from libc.string cimport memcpy


# ---------------------------------------------------------------------------
# Cores: all arithmetic lives here. noexcept nogil, raw pointers, no allocation.
# ---------------------------------------------------------------------------

cdef bint _stationary_core(const double* phi, Py_ssize_t r, const double* R_vec,
                           double* Pout, double* S, double* A, double* AS,
                           double* U, double* Anew) noexcept nogil:
    """Stationary covariance by doubling. Writes result into Pout, returns ok.

    On convergence Pout = 0.5(S+S'); on failure Pout = current S (matches the
    reference's abort value). S/A/AS/U/Anew are scratch of length r*r.
    """
    cdef Py_ssize_t i, j, k, it
    cdef Py_ssize_t rr = r * r
    cdef double acc, s_scale, u_max, av
    cdef double rtol = 1e-13
    cdef Py_ssize_t max_doublings = 60

    # A = T (companion form: AR coefs in column 0, ones on superdiagonal).
    for i in range(rr):
        A[i] = 0.0
    for i in range(r):
        A[i * r] = phi[i]
    for i in range(r - 1):
        A[i * r + (i + 1)] = 1.0

    # S = R R'
    for i in range(r):
        for j in range(r):
            S[i * r + j] = R_vec[i] * R_vec[j]

    for it in range(max_doublings):
        # AS = A @ S  (ikj; per-output k-accumulation order preserved)
        for i in range(rr):
            AS[i] = 0.0
        for i in range(r):
            for k in range(r):
                acc = A[i * r + k]
                for j in range(r):
                    AS[i * r + j] += acc * S[k * r + j]

        # U = AS @ A'  =>  U[i,j] = sum_k AS[i,k] * A[j,k]
        u_max = 0.0
        for i in range(r):
            for j in range(r):
                acc = 0.0
                for k in range(r):
                    acc += AS[i * r + k] * A[j * r + k]
                if not isfinite(acc):
                    memcpy(Pout, S, rr * sizeof(double))
                    return False
                U[i * r + j] = acc
                av = fabs(acc)
                if av > u_max:
                    u_max = av

        # S = S + U ; track scale
        s_scale = 0.0
        for i in range(rr):
            S[i] = S[i] + U[i]
            av = fabs(S[i])
            if av > s_scale:
                s_scale = av
        if not isfinite(s_scale):
            memcpy(Pout, S, rr * sizeof(double))
            return False

        # Converged: symmetrize into Pout.
        if u_max <= rtol * (1.0 if s_scale < 1.0 else s_scale):
            for i in range(r):
                for j in range(r):
                    Pout[i * r + j] = 0.5 * (S[i * r + j] + S[j * r + i])
            return True

        # A = A @ A  (ikj into Anew; old A read throughout)
        for i in range(rr):
            Anew[i] = 0.0
        for i in range(r):
            for k in range(r):
                acc = A[i * r + k]
                for j in range(r):
                    Anew[i * r + j] += acc * A[k * r + j]
        for i in range(rr):
            if not isfinite(Anew[i]):
                memcpy(Pout, S, rr * sizeof(double))
                return False
        memcpy(A, Anew, rr * sizeof(double))

    memcpy(Pout, S, rr * sizeof(double))
    return False


cdef bint _kalman_core(const double* z, Py_ssize_t n,
                       const double* phi, Py_ssize_t r, const double* R_vec,
                       double* a, double* P,
                       double* innov, double* F,
                       double* a_filt, double* K, double* P_filt,
                       double* TPf_row0) noexcept nogil:
    """Kalman forward pass. Mutates a and P in place; writes innov, F.
    Returns ok (False on non-finite value or F_t <= 0)."""
    cdef Py_ssize_t t, i, j
    cdef double v, f, inv_f, s, a_i0, a_ij1, val, p0j_ki

    for t in range(n):
        v = z[t] - a[0]
        f = P[0]
        if not isfinite(f) or f <= 0.0:
            return False
        innov[t] = v
        F[t] = f

        inv_f = 1.0 / f
        for i in range(r):
            K[i] = P[i * r] * inv_f

        for i in range(r):
            a_filt[i] = a[i] + K[i] * v
            p0j_ki = K[i]
            for j in range(r):
                P_filt[i * r + j] = P[i * r + j] - p0j_ki * P[j]

        for i in range(r):
            s = phi[i] * a_filt[0]
            if i + 1 < r:
                s += a_filt[i + 1]
            a[i] = s

        for j in range(r):
            TPf_row0[j] = phi[0] * P_filt[j]
            if 1 < r:
                TPf_row0[j] += P_filt[r + j]

        for i in range(r):
            if i == 0:
                a_i0 = TPf_row0[0]
            else:
                a_i0 = phi[i] * P_filt[0]
                if i + 1 < r:
                    a_i0 += P_filt[(i + 1) * r]
            for j in range(r):
                val = a_i0 * phi[j]
                if j + 1 < r:
                    if i == 0:
                        val += TPf_row0[j + 1]
                    else:
                        a_ij1 = phi[i] * P_filt[j + 1]
                        if i + 1 < r:
                            a_ij1 += P_filt[(i + 1) * r + (j + 1)]
                        val += a_ij1
                val += R_vec[i] * R_vec[j]
                if not isfinite(val):
                    return False
                P[i * r + j] = val

    return True


# ---------------------------------------------------------------------------
# Python entry points (historical API pinned by the parity tests).
# ---------------------------------------------------------------------------

def stationary_init(double[::1] phi, double[::1] R_vec):
    """Stationary initial covariance P0. Returns (P, ok)."""
    cdef Py_ssize_t r = phi.shape[0]
    cdef double[:, ::1] Pout = np.empty((r, r), dtype=np.float64)
    cdef double[:, ::1] S = np.empty((r, r), dtype=np.float64)
    cdef double[:, ::1] A = np.empty((r, r), dtype=np.float64)
    cdef double[:, ::1] AS = np.empty((r, r), dtype=np.float64)
    cdef double[:, ::1] U = np.empty((r, r), dtype=np.float64)
    cdef double[:, ::1] Anew = np.empty((r, r), dtype=np.float64)
    cdef bint ok = _stationary_core(&phi[0], r, &R_vec[0], &Pout[0, 0],
                                    &S[0, 0], &A[0, 0], &AS[0, 0],
                                    &U[0, 0], &Anew[0, 0])
    return np.asarray(Pout), ok


def kalman_loop(double[::1] z, double[::1] phi, double[::1] R_vec,
                double[::1] a, double[:, ::1] P):
    """Kalman forward pass. Mutates a and P in place; returns (innov, F, ok)."""
    cdef Py_ssize_t n = z.shape[0]
    cdef Py_ssize_t r = phi.shape[0]
    cdef double[::1] innov = np.empty(n, dtype=np.float64)
    cdef double[::1] F = np.empty(n, dtype=np.float64)
    cdef double[::1] a_filt = np.empty(r, dtype=np.float64)
    cdef double[::1] K = np.empty(r, dtype=np.float64)
    cdef double[:, ::1] P_filt = np.empty((r, r), dtype=np.float64)
    cdef double[::1] TPf_row0 = np.empty(r, dtype=np.float64)
    cdef bint ok = _kalman_core(&z[0], n, &phi[0], r, &R_vec[0],
                                &a[0], &P[0, 0], &innov[0], &F[0],
                                &a_filt[0], &K[0], &P_filt[0, 0], &TPf_row0[0])
    return np.asarray(innov), np.asarray(F), ok


# ---------------------------------------------------------------------------
# Production fused path: one workspace per fit, reused across all likelihood
# evaluations. Eliminates the per-eval allocation of every scratch matrix (the
# ~7 us/eval boundary cost the single-call entry points pay). The init+filter
# arithmetic is byte-for-byte the cores above; the sse / sum(log F) reductions
# stay in numpy so their pairwise-summation semantics are unchanged.
# ---------------------------------------------------------------------------

cdef class ArmaKalmanWorkspace:
    """Reusable Kalman workspace for a fixed (r, n) within one ARIMA fit.

    Construct once per (state-dim r, series-length n); call :meth:`loglik_parts`
    for each parameter vector the optimizer proposes. All scratch is allocated
    in ``__cinit__`` and reused, so a full optimization does O(1) allocation
    instead of O(evals).
    """
    cdef Py_ssize_t r, n
    cdef double[::1] _a, _P, _innov, _F, _a_filt, _K, _P_filt, _TPf
    cdef double[::1] _Pout, _S, _A, _AS, _U, _Anew

    def __cinit__(self, Py_ssize_t r, Py_ssize_t n):
        self.r = r
        self.n = n
        cdef Py_ssize_t rr = r * r
        self._a = np.empty(r, dtype=np.float64)
        self._P = np.empty(rr, dtype=np.float64)
        self._innov = np.empty(n, dtype=np.float64)
        self._F = np.empty(n, dtype=np.float64)
        self._a_filt = np.empty(r, dtype=np.float64)
        self._K = np.empty(r, dtype=np.float64)
        self._P_filt = np.empty(rr, dtype=np.float64)
        self._TPf = np.empty(r, dtype=np.float64)
        self._Pout = np.empty(rr, dtype=np.float64)
        self._S = np.empty(rr, dtype=np.float64)
        self._A = np.empty(rr, dtype=np.float64)
        self._AS = np.empty(rr, dtype=np.float64)
        self._U = np.empty(rr, dtype=np.float64)
        self._Anew = np.empty(rr, dtype=np.float64)

    def loglik_parts(self, double[::1] z, double[::1] phi, double[::1] R_vec,
                     double kappa):
        """Run stationary init + Kalman filter fused, reusing buffers.

        Returns ``(ok, sse, sum_log_F)`` where
        ``sse = sum(innov**2 / F)`` and ``sum_log_F = sum(log F)`` — the two
        reductions the concentrated log-likelihood needs. ``ok`` is False if the
        filter hit a non-finite value; the caller applies the usual penalty.
        On non-stationary init the diffuse fallback (kappa on the diagonal) is
        applied in-core, identically to the historical Python path.
        """
        cdef Py_ssize_t r = self.r, n = self.n, i
        cdef Py_ssize_t rr = r * r
        cdef double* Pp = &self._P[0]
        cdef bint init_ok, ok
        # Fail loud on any shape mismatch rather than silently computing with
        # the workspace's r/n and the caller's (different) arrays (Rule 1).
        if phi.shape[0] != r or R_vec.shape[0] != r or z.shape[0] != n:
            raise ValueError(
                f"workspace sized (r={r}, n={n}) but got phi={phi.shape[0]}, "
                f"R_vec={R_vec.shape[0]}, z={z.shape[0]}"
            )
        with nogil:
            init_ok = _stationary_core(&phi[0], r, &R_vec[0], &self._Pout[0],
                                       &self._S[0], &self._A[0], &self._AS[0],
                                       &self._U[0], &self._Anew[0])
            if init_ok:
                memcpy(Pp, &self._Pout[0], rr * sizeof(double))
            else:
                for i in range(rr):
                    Pp[i] = 0.0
                for i in range(r):
                    Pp[i * r + i] = kappa
            for i in range(r):
                self._a[i] = 0.0
            ok = _kalman_core(&z[0], n, &phi[0], r, &R_vec[0], &self._a[0], Pp,
                              &self._innov[0], &self._F[0], &self._a_filt[0],
                              &self._K[0], &self._P_filt[0], &self._TPf[0])
        if not ok:
            return False, 0.0, 0.0
        # Reductions stay in numpy: pairwise summation semantics unchanged.
        innov = np.asarray(self._innov)
        F = np.asarray(self._F)
        with np.errstate(divide="ignore", invalid="ignore"):
            sse = float(np.sum(innov * innov / F))
            sum_log_F = float(np.sum(np.log(F)))
        return True, sse, sum_log_F


