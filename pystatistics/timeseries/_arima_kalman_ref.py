"""Pure-numpy reference for the ARMA Kalman kernels — the bit-identity oracle.

These are line-for-line pure-numpy twins of the compiled kernels in
``_arima_kalman_kernel`` (Cython). They exist ONLY as a portable, compiler-
independent oracle for the parity tests: the compiled kernel must reproduce
these to the last bit on every platform we ship a wheel for.

They are deliberately written as scalar recursions in the *same operation
order* as the compiled kernel — NOT as vectorised numpy (``P @ T.T`` etc.),
because matmul reduction order differs from the triple-loop and would not be
bit-identical. Do not "optimise" this file; its only job is to be the exact
reference the kernel is checked against.

This module is imported by tests, never on the library's hot path.
"""

from __future__ import annotations

import numpy as np


def _matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Naive ijk scalar matrix product, k accumulated in index order.

    Written as explicit loops (not ``A @ B``) so the reduction order is
    fixed and reproducible, and the compiled Cython kernel can reproduce it
    to the last bit. BLAS ``@`` does NOT have a defined reduction order
    across implementations/threads, which is precisely the hidden non-
    determinism this port removes from ``stationary_init``.
    """
    n = A.shape[0]
    m = B.shape[1]
    k_ = A.shape[1]
    C = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            acc = 0.0
            for k in range(k_):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc
    return C


def stationary_init(phi: np.ndarray, R_vec: np.ndarray) -> tuple[np.ndarray, bool]:
    """Reference twin of the compiled ``stationary_init``.

    Solves ``P = T P T' + R R'`` by doubling. See the compiled kernel's
    docstring for the algorithm. Unlike the historical Numba kernel, the
    matrix products are explicit scalar loops (``_matmul``), so the result
    is bit-reproducible across platforms rather than BLAS-order-dependent.
    (This changes results vs. the old Numba path by ~1e-17 — far below the
    R-parity gate — while removing a hidden BLAS non-determinism source.)
    """
    r = phi.shape[0]

    T = np.zeros((r, r), dtype=np.float64)
    for i in range(r):
        T[i, 0] = phi[i]
    for i in range(r - 1):
        T[i, i + 1] = 1.0

    S = np.empty((r, r), dtype=np.float64)
    for i in range(r):
        for j in range(r):
            S[i, j] = R_vec[i] * R_vec[j]

    A = T.copy()
    rtol = 1e-13
    max_doublings = 60
    # Non-stationary parameters make the doubling blow up to +/-inf on
    # purpose; the isfinite guards below detect it and return ok=False. The
    # overflow is the detection mechanism, not an error, so silence it (the
    # compiled kernel produces the same inf silently).
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(max_doublings):
            U = _matmul(_matmul(A, S), A.T.copy())
            if not np.all(np.isfinite(U)):
                return S, False
            S = S + U
            s_scale = np.abs(S).max()
            if not np.isfinite(s_scale):
                return S, False
            if np.abs(U).max() <= rtol * max(1.0, s_scale):
                return 0.5 * (S + S.T), True
            A = _matmul(A, A)
            if not np.all(np.isfinite(A)):
                return S, False

    return S, False


def kalman_loop(
    z: np.ndarray,
    phi: np.ndarray,
    R_vec: np.ndarray,
    a: np.ndarray,
    P: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Reference twin of the compiled ``kalman_loop`` (forward pass).

    Mutates ``a`` and ``P`` in place exactly as the compiled kernel does, so
    callers that rely on the post-filter state (forecasting) see identical
    behaviour. Returns ``(innov, F, ok)``.
    """
    n = z.shape[0]
    r = phi.shape[0]

    innov = np.empty(n, dtype=np.float64)
    F = np.empty(n, dtype=np.float64)
    a_filt = np.empty(r, dtype=np.float64)
    K = np.empty(r, dtype=np.float64)
    P_filt = np.empty((r, r), dtype=np.float64)
    TPf_row0 = np.empty(r, dtype=np.float64)

    for t in range(n):
        v = z[t] - a[0]
        f = P[0, 0]
        if not np.isfinite(f) or f <= 0.0:
            return innov, F, False
        innov[t] = v
        F[t] = f

        inv_f = 1.0 / f
        for i in range(r):
            K[i] = P[i, 0] * inv_f

        for i in range(r):
            a_filt[i] = a[i] + K[i] * v
            p0j_ki = K[i]
            for j in range(r):
                P_filt[i, j] = P[i, j] - p0j_ki * P[0, j]

        for i in range(r):
            s = phi[i] * a_filt[0]
            if i + 1 < r:
                s += a_filt[i + 1]
            a[i] = s

        for j in range(r):
            TPf_row0[j] = phi[0] * P_filt[0, j]
            if 1 < r:
                TPf_row0[j] += P_filt[1, j]

        for i in range(r):
            if i == 0:
                a_i0 = TPf_row0[0]
            else:
                a_i0 = phi[i] * P_filt[0, 0]
                if i + 1 < r:
                    a_i0 += P_filt[i + 1, 0]
            for j in range(r):
                val = a_i0 * phi[j]
                if j + 1 < r:
                    if i == 0:
                        val += TPf_row0[j + 1]
                    else:
                        a_ij1 = phi[i] * P_filt[0, j + 1]
                        if i + 1 < r:
                            a_ij1 += P_filt[i + 1, j + 1]
                        val += a_ij1
                val += R_vec[i] * R_vec[j]
                if not np.isfinite(val):
                    return innov, F, False
                P[i, j] = val

    return innov, F, True
