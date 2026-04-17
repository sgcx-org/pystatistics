"""
Kalman-filter exact log-likelihood for ARMA(p, q) models.

Replaces the O(n^3) innovations-algorithm path in ``_arima_likelihood.py``
with the state-space / Kalman-filter approach used by R's
``stats::arima`` (Gardner, Harvey & Phillips 1980; Harvey 1989 ch. 3).
Per-iteration work is O(r^2) where r = max(p, q+1) is the state
dimension, so total cost is O(n * r^2). For the Box–Jenkins airline
model on AirPassengers (n=132, r=13) that is roughly 22k scalar ops
versus the innovations algorithm's 2.3M — two orders of magnitude fewer
FLOPS, and each step is a handful of small dense numpy ops that
vectorize cleanly.

State-space representation used here (Harvey 1989, "SSF") for an
ARMA(p, q) process on the differenced series:

    α_{t+1} = T α_t + R η_t,   η_t ~ N(0, σ²)
    y_t     = Z α_t            (measurement is exact — no obs noise)

with:

    r = max(p, q + 1)
    T  (r × r):  T[0, j] = φ_{j+1} (AR coefs padded with zeros)
                 T[i, i-1] = 1  for i ≥ 1  (shift up)
                 T[i, j]    = 0 otherwise
    R  (r × 1):  R[0]   = 1
                 R[i]   = θ_i  for 1 ≤ i ≤ q (padded with zeros to r-1)
    Z  (1 × r):  Z[0] = 1, rest 0

The initial state is stationary: α_0 = 0, P_0 solves the discrete
Lyapunov equation T P_0 T' - P_0 = -σ² R R' (via scipy).

During ML optimization the optimizer can briefly wander into parameter
regions where the AR polynomial is not strictly stationary (roots very
close to or inside the unit circle). If ``solve_discrete_lyapunov``
fails or returns a non-PSD matrix, we fall back to a diffuse init
(large κ on the diagonal), matching what R does under
``kappa = 1e6``.

References
----------
Gardner, G., Harvey, A. C., & Phillips, G. D. A. (1980).
    "An algorithm for exact maximum likelihood estimation of autoregressive–
    moving average models by means of Kalman filtering." J. R. Stat. Soc.
    Applied Statistics, 29(3): 311–322.
Harvey, A. C. (1989). Forecasting, Structural Time Series Models and the
    Kalman Filter. Cambridge University Press.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_lyapunov

__all__ = ["kalman_arma_loglik"]

# Non-stationary fallback: matches R's `kappa` in stats:::makeARIMA.
_DIFFUSE_KAPPA = 1.0e6


@njit(cache=True, fastmath=False)
def _sparse_T_times_M_times_TT(
    phi: np.ndarray,
    M: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute ``out = T M T'`` exploiting T's companion structure.

    Same structure used by ``_kalman_loop``: T[i, 0] = phi[i],
    T[i, i+1] = 1. Shared helper so the initial-covariance fixed-point
    iteration runs in O(r^2) per step instead of O(r^3) from a generic
    matmul. Writes result into ``out`` (must be pre-allocated).
    """
    r = phi.shape[0]
    # First compute A = T M. Top row: A[0, j] = phi[0]*M[0,j] + M[1,j] (if r>1)
    # Other rows: A[i, j] = phi[i]*M[0,j] + M[i+1,j] (if i+1<r)
    # Then out = A T': out[i, j] = A[i, 0]*phi[j] + A[i, j+1] (if j+1<r)
    for i in range(r):
        # A[i, 0]
        a_i0 = phi[i] * M[0, 0]
        if i + 1 < r:
            a_i0 += M[i + 1, 0]
        for j in range(r):
            # A[i, j+1] (if j+1 < r)
            if j + 1 < r:
                a_ij1 = phi[i] * M[0, j + 1]
                if i + 1 < r:
                    a_ij1 += M[i + 1, j + 1]
                out[i, j] = a_i0 * phi[j] + a_ij1
            else:
                out[i, j] = a_i0 * phi[j]


@njit(cache=True, fastmath=False)
def _stationary_init(phi: np.ndarray, R_vec: np.ndarray) -> tuple:
    """Stationary initial covariance P_0 for the ARMA state-space.

    Solves ``P = T P T' + R R'`` by fixed-point iteration. For stationary
    T (AR roots outside unit circle) this converges geometrically; for
    boundary or non-stationary T it fails to converge and the caller
    falls back to a diffuse init. Runs entirely in numba — the prior
    scipy-based ``solve_discrete_lyapunov`` cost ~150 µs per call on
    r=13 and was the remaining bottleneck after the Kalman loop itself
    was JIT'd.

    Returns
    -------
    P : shape (r, r)   stationary covariance (or partial result if
                       convergence not reached)
    ok : bool          True if the iteration converged within tolerance
                       and all entries are finite.
    """
    r = phi.shape[0]
    # RR' = outer(R_vec, R_vec)
    RR = np.empty((r, r), dtype=np.float64)
    for i in range(r):
        for j in range(r):
            RR[i, j] = R_vec[i] * R_vec[j]

    P = RR.copy()
    P_new = np.empty((r, r), dtype=np.float64)
    TMT = np.empty((r, r), dtype=np.float64)

    # Fixed-point iteration. Rate of convergence is spectral radius of
    # T (squared), so for stationary AR this is < 1 and converges
    # geometrically. 200 iterations is plenty for r <= ~30 and AR
    # eigenvalues bounded away from 1.
    max_iter = 200
    tol = 1e-12
    for _ in range(max_iter):
        _sparse_T_times_M_times_TT(phi, P, TMT)
        max_change = 0.0
        bad = False
        for i in range(r):
            for j in range(r):
                new_val = TMT[i, j] + RR[i, j]
                if not np.isfinite(new_val):
                    bad = True
                    break
                diff = new_val - P[i, j]
                if diff < 0.0:
                    diff = -diff
                if diff > max_change:
                    max_change = diff
                P_new[i, j] = new_val
            if bad:
                break
        if bad:
            return P, False
        # Copy P_new back into P.
        for i in range(r):
            for j in range(r):
                P[i, j] = P_new[i, j]
        if max_change < tol:
            return P, True

    # Did not converge; signal fallback.
    return P, False


@njit(cache=True, fastmath=False)
def _kalman_loop(
    z: np.ndarray,       # centered series, shape (n,)
    phi: np.ndarray,     # AR coefs padded to length r, shape (r,)
    R_vec: np.ndarray,   # state-noise loading, shape (r,)
    a: np.ndarray,       # initial state mean (zeros), shape (r,)
    P: np.ndarray,       # initial state covariance, shape (r, r)
) -> tuple:
    """Kalman filter forward pass (numba-JIT).

    The transition matrix T for ARMA state-space has a very specific
    companion-matrix structure:

        T = [[phi_1, 1, 0, ..., 0],
             [phi_2, 0, 1, ..., 0],
             ...
             [phi_r, 0, 0, ..., 0]]

    so ``T @ x`` is ``phi * x[0] + shift_up(x)`` and ``T @ M`` is
    ``outer(phi, M[0, :]) + shift_up_rows(M)``. We never materialize T;
    we just supply ``phi`` (AR coefficients padded with zeros to length
    r). This drops each state-propagation matmul from O(r^3) to O(r^2),
    which is where a typical SARIMA fit after seasonal expansion (r ~
    15) spends most of its time.

    Measurement Z = [1, 0, ..., 0] is implicit. Observation equation has
    no measurement noise — σ² enters through R η_t only.

    Returns
    -------
    innov : shape (n,)   innovations y_t - Z α̂_{t|t-1}
    F     : shape (n,)   innovation variances Z P_{t|t-1} Z' = P[0, 0]
    ok    : bool         False if any intermediate value was non-finite
                         or F_t went non-positive. Caller treats as a
                         penalty step.
    """
    n = z.shape[0]
    r = phi.shape[0]

    innov = np.empty(n, dtype=np.float64)
    F = np.empty(n, dtype=np.float64)
    a_filt = np.empty(r, dtype=np.float64)
    K = np.empty(r, dtype=np.float64)
    P_filt = np.empty((r, r), dtype=np.float64)
    TPf_row0 = np.empty(r, dtype=np.float64)   # top row of T @ P_filt
    # remaining rows of T @ P_filt are just P_filt[1:, :], accessed directly

    for t in range(n):
        # Innovation v = z[t] - α[0];  F = P[0, 0].
        v = z[t] - a[0]
        f = P[0, 0]
        if not np.isfinite(f) or f <= 0.0:
            return innov, F, False
        innov[t] = v
        F[t] = f

        # Kalman gain K = P[:, 0] / f
        inv_f = 1.0 / f
        for i in range(r):
            K[i] = P[i, 0] * inv_f

        # Filter update:
        #   a_filt = a + K * v
        #   P_filt = P - outer(K, P[0, :])
        for i in range(r):
            a_filt[i] = a[i] + K[i] * v
            p0j_ki = K[i]
            for j in range(r):
                P_filt[i, j] = P[i, j] - p0j_ki * P[0, j]

        # State mean propagation: a = T @ a_filt
        # Exploit T's structure:
        #   (T a_filt)[0]   = sum_k phi_k a_filt[k]
        #   (T a_filt)[i]   = a_filt[i + 1] for i >= 1 (shift up; last = 0
        #                     unless phi_r is part of AR, which it already
        #                     is in the [0] row via the full phi vector)
        #
        # Wait — the companion form here places phi in the FIRST COLUMN
        # of T, not the first row. Let me re-derive:
        #
        #   T[i, 0] = phi_{i+1}     (phi in column 0)
        #   T[i, i+1] = 1           (superdiagonal)
        #
        # So (T a_filt)[i] = phi_{i+1} * a_filt[0] + (a_filt[i+1] if i+1<r else 0).
        for i in range(r):
            s = phi[i] * a_filt[0]
            if i + 1 < r:
                s += a_filt[i + 1]
            a[i] = s

        # Covariance propagation: P_new = T @ P_filt @ T' + RR'
        # Using T[i, 0] = phi_{i+1}, T[i, i+1] = 1:
        #   (T @ P_filt)[i, j] = phi_{i+1} * P_filt[0, j] + P_filt[i+1, j] (if i+1<r)
        # Precompute the top row once, then for i>=1 we only need shifted
        # rows of P_filt — O(r^2) instead of O(r^3).
        for j in range(r):
            TPf_row0[j] = phi[0] * P_filt[0, j]
            if 1 < r:
                TPf_row0[j] += P_filt[1, j]

        # Now form P_new = (T P_filt) T' + R R'.
        #   (A T')[i, j] = A[i, 0] * phi[j] + (A[i, j+1] if j+1 < r else 0)
        # where A = T P_filt.
        for i in range(r):
            # A[i, 0]
            if i == 0:
                a_i0 = TPf_row0[0]
            else:
                a_i0 = phi[i] * P_filt[0, 0]
                if i + 1 < r:
                    a_i0 += P_filt[i + 1, 0]
            for j in range(r):
                val = a_i0 * phi[j]
                if j + 1 < r:
                    # A[i, j+1]
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


def _build_state_space(
    ar: NDArray,
    ma: NDArray,
) -> tuple[NDArray, NDArray, int]:
    """Build the ARMA state-space matrices T, R (and return state dim r).

    Parameters
    ----------
    ar : NDArray
        AR coefficients [φ_1, ..., φ_p]. Empty for pure MA.
    ma : NDArray
        MA coefficients [θ_1, ..., θ_q]. Empty for pure AR.

    Returns
    -------
    T : NDArray, shape (r, r)
    R : NDArray, shape (r,)   (kept as a 1-D vector; we never need R as a
        full matrix because the innovation variance is a scalar σ²)
    r : int
        State dimension.
    """
    p = len(ar)
    q = len(ma)
    r = max(p, q + 1) if (p + q) > 0 else 1

    T = np.zeros((r, r), dtype=np.float64)
    if p > 0:
        T[:p, 0] = ar
    # Shift: superdiagonal of ones (i.e. T[i, i+1] = 1 pattern when we
    # keep the "first column is AR, rest is a shift" convention).
    for i in range(r - 1):
        T[i, i + 1] = 1.0

    R_vec = np.zeros(r, dtype=np.float64)
    R_vec[0] = 1.0
    if q > 0:
        R_vec[1:1 + q] = ma

    return T, R_vec, r


def _initial_covariance(T: NDArray, R_vec: NDArray, sigma2: float) -> NDArray:
    """Stationary initial covariance P_0, falling back to diffuse init.

    For a stationary ARMA we solve the discrete Lyapunov equation

        T P_0 T' - P_0 = -σ² R R'

    If the AR polynomial has roots near or inside the unit circle the
    Lyapunov solve can fail or return a non-PSD matrix. In that case we
    fall back to a diffuse initialization (large diagonal), matching R's
    ``makeARIMA`` default kappa = 1e6. This is almost never the right
    answer for a converged fit but lets the optimizer explore
    non-stationary regions without crashing.
    """
    r = T.shape[0]
    Q = sigma2 * np.outer(R_vec, R_vec)
    try:
        P0 = solve_discrete_lyapunov(T, Q)
    except Exception:
        return _DIFFUSE_KAPPA * np.eye(r)

    # Sanity: must be finite and roughly PSD. A tiny negative eigenvalue
    # from floating-point noise is fine (we symmetrize below); anything
    # worse means the AR roots are not strictly outside the unit circle.
    if not np.all(np.isfinite(P0)):
        return _DIFFUSE_KAPPA * np.eye(r)

    # Symmetrize (Lyapunov solver can leave tiny asymmetry).
    P0 = 0.5 * (P0 + P0.T)

    # Cheap PSD check — the smallest eigenvalue of a 1-D ARMA cov is the
    # observational variance, which must be > 0. If it's significantly
    # negative, the fit point is not stationary and we fall back.
    min_eig = np.linalg.eigvalsh(P0).min() if r > 1 else P0[0, 0]
    if min_eig < -1e-8 * max(1.0, float(np.abs(P0).max())):
        return _DIFFUSE_KAPPA * np.eye(r)

    return P0


def kalman_arma_loglik(
    y: NDArray,
    ar: NDArray,
    ma: NDArray,
    mean: float,
) -> tuple[float, float]:
    """Exact Gaussian log-likelihood of ARMA(p, q) via the Kalman filter.

    Equivalent to the O(n^3) ``_innovations_algorithm`` + prediction-error
    computation in ``_arima_likelihood.exact_loglik``, but O(n * r^2) per
    fit where r = max(p, q+1). For typical SARIMA fits after seasonal
    expansion r ≤ ~25, so the savings vs. innovations (which is O(n^3))
    are ~100× on n=100 series and grow with n.

    Parameters
    ----------
    y : NDArray
        Differenced series (1-D). Mean will be subtracted internally.
    ar : NDArray
        AR coefficients (possibly empty).
    ma : NDArray
        MA coefficients (possibly empty).
    mean : float
        Estimated mean of the differenced series (0 if include_mean=False).

    Returns
    -------
    nll : float
        Negative log-likelihood. Returns a large finite penalty if
        numerical issues arise, so that the caller (scipy.optimize) just
        steps away from the bad region.
    sigma2 : float
        Profile ML estimate of the innovation variance at these (ar, ma).
        This is the sigma² that makes the partial derivative of the
        concentrated log-likelihood vanish; ``arima()`` uses it to report
        the final model variance.
    """
    z = y - mean
    n = z.size
    if n == 0:
        return 1e18, 1.0

    _, R_vec, r = _build_state_space(ar, ma)
    # phi = AR coefficients padded with zeros to length r (the column-0
    # entries of the implicit T). The Kalman loop never materializes T
    # — it uses phi directly to exploit the companion-matrix structure.
    phi = np.zeros(r, dtype=np.float64)
    p = len(ar)
    if p > 0:
        phi[:p] = ar

    # Stationary init via JIT fixed-point iteration. If it fails to
    # converge (AR polynomial has roots near/inside the unit circle),
    # fall back to the diffuse init R uses under `kappa = 1e6`.
    P, init_ok = _stationary_init(phi, R_vec)
    if not init_ok:
        P = _DIFFUSE_KAPPA * np.eye(r, dtype=np.float64)

    # Run the Kalman filter with sigma² = 1 (concentrated out). The
    # innovation variance F_t is linear in σ², so we compute F_t | σ²=1
    # and rescale at the end via the profile-ML estimator.
    a = np.zeros(r, dtype=np.float64)

    innov, F, ok = _kalman_loop(z, phi, R_vec, a, P)
    if not ok:
        return 1e18, 1.0

    # Concentrated ML for sigma²:
    #   sigma²_hat = (1/n) Σ innov_t² / F_t
    #   nll(sigma²_hat) = 0.5 n log(2π σ²_hat) + 0.5 Σ log F_t + 0.5 n
    # where F_t was computed under σ²=1 (so true F_t = σ² * F_t_unit).
    with np.errstate(divide='ignore', invalid='ignore'):
        sse = float(np.sum(innov * innov / F))
    if not np.isfinite(sse) or sse <= 0.0:
        return 1e18, 1.0

    sigma2_hat = sse / n
    nll = (
        0.5 * n * np.log(2.0 * np.pi * sigma2_hat)
        + 0.5 * float(np.sum(np.log(F)))
        + 0.5 * n
    )
    if not np.isfinite(nll):
        return 1e18, 1.0

    return nll, sigma2_hat
