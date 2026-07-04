"""
Kalman-filter exact log-likelihood and forecasting for ARMA(p, q).

The state-space / Kalman-filter approach used by R's ``stats::arima``
(Gardner, Harvey & Phillips 1980; Harvey 1989 ch. 3). Per-iteration
work is O(r^2) where r = max(p, q+1) is the state dimension, so total
cost is O(n * r^2) — for the Box–Jenkins airline model on AirPassengers
(n=132, r=13) roughly 22k scalar ops, each step a handful of small
dense numpy ops that vectorize cleanly.

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
Lyapunov equation T P_0 T' - P_0 = -σ² R R' (via a doubling iteration
in numba; see ``_stationary_init``).

During ML optimization the optimizer can briefly wander into parameter
regions where the AR polynomial is not strictly stationary (roots very
close to or inside the unit circle). If the doubling iteration blows
up or fails to decay, we fall back to a diffuse init (large κ on the
diagonal), matching what R does under ``kappa = 1e6``.

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

__all__ = ["kalman_arma_forecast", "kalman_arma_innovations", "kalman_arma_loglik"]

# Non-stationary fallback: matches R's `kappa` in stats:::makeARIMA.
_DIFFUSE_KAPPA = 1.0e6


@njit(cache=True, fastmath=False)
def _stationary_init(phi: np.ndarray, R_vec: np.ndarray) -> tuple:
    """Stationary initial covariance P_0 for the ARMA state-space.

    Solves ``P = T P T' + R R'`` (i.e. P = sum_k T^k RR' T'^k) by the
    doubling iteration

        S_0 = RR',  A_0 = T,
        S_{j+1} = S_j + A_j S_j A_j',  A_{j+1} = A_j A_j,

    so that S_j = sum_{k=0}^{2^j - 1} T^k RR' T'^k. Convergence is
    quadratic in the horizon: ~20 doublings cover a spectral radius of
    0.9999, where a linear fixed-point iteration needs ~10^5 steps.

    The previous implementation iterated linearly with max_iter=200 and
    an ABSOLUTE tol of 1e-12: a moderately persistent seasonal AR term
    (e.g. sar1 = -0.47 at lag 12 gives spectral radius 0.94) failed to
    converge in the budget and silently fell back to the diffuse init,
    which shifted the reported log-likelihood by ~80 units on airline-
    class models (RIGOR R18 follow-up). R never falls back for
    stationary models — it solves for Q0 exactly.

    Convergence is judged RELATIVE to the accumulated covariance scale.
    For non-stationary T the iterates blow up (non-finite, or no decay
    within the doubling budget) and the caller falls back to the
    diffuse init, as before.

    Returns
    -------
    P : shape (r, r)   stationary covariance (or partial result if
                       convergence not reached)
    ok : bool          True if the doubling converged and all entries
                       are finite.
    """
    r = phi.shape[0]

    # Materialize T (companion form: AR coefs in column 0, ones on the
    # superdiagonal) — the doubling squares A, which destroys sparsity,
    # so there is nothing to exploit beyond dense matmuls. r <= ~30, so
    # each O(r^3) product is trivial.
    T = np.zeros((r, r), dtype=np.float64)
    for i in range(r):
        T[i, 0] = phi[i]
    for i in range(r - 1):
        T[i, i + 1] = 1.0

    # S = RR' = outer(R_vec, R_vec)
    S = np.empty((r, r), dtype=np.float64)
    for i in range(r):
        for j in range(r):
            S[i, j] = R_vec[i] * R_vec[j]

    A = T.copy()
    rtol = 1e-13
    max_doublings = 60  # horizon 2^60; far beyond any stationary need
    for _ in range(max_doublings):
        U = A @ S @ A.T
        if not np.all(np.isfinite(U)):
            return S, False
        S = S + U
        # Relative convergence: the tail just added is negligible
        # against the accumulated covariance.
        s_scale = np.abs(S).max()
        if not np.isfinite(s_scale):
            return S, False
        if np.abs(U).max() <= rtol * max(1.0, s_scale):
            # Symmetrize: doubling preserves symmetry analytically, but
            # float matmuls drift by ~1e-12 relative on near-unit
            # systems; the filter expects an exactly symmetric P_0.
            return 0.5 * (S + S.T), True
        A = A @ A
        if not np.all(np.isfinite(A)):
            return S, False

    # No decay within the doubling budget: treat as non-stationary.
    return S, False


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


def kalman_arma_loglik(
    y: NDArray,
    ar: NDArray,
    ma: NDArray,
    mean: float,
) -> tuple[float, float]:
    """Exact Gaussian log-likelihood of ARMA(p, q) via the Kalman filter.

    O(n * r^2) per fit where r = max(p, q+1); for typical SARIMA fits
    after seasonal expansion r ≤ ~25.

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


def kalman_arma_forecast(
    z: NDArray,
    ar: NDArray,
    ma: NDArray,
    h: int,
) -> tuple[NDArray, NDArray]:
    """h-step forecasts of a zero-mean ARMA from the Kalman filter state.

    Runs the same filter as :func:`kalman_arma_loglik` over the sample
    and then iterates the state recursion for *h* steps. This is how R's
    ``predict.Arima`` (``KalmanForecast``) produces its forecasts: the
    forecast origin is the exact filtered state, not a CSS residual
    recursion. The distinction matters for models with a near-unit MA
    root, where zero-initialized CSS residuals still carry conditioning
    error at the end of the sample — on AirPassengers (2,1,1)(0,1,0)[12]
    (ma1 = -0.98) the CSS-seeded recursion was ~1.4 off R's forecasts;
    the Kalman state reproduces them (RIGOR R18 follow-up).

    Also returns the full h x h covariance matrix of the forecast
    ERRORS (under sigma2 = 1): with X_j the state prediction error at
    horizon j, Cov(X_1) = P_{n+1|n} and X_{j+1} = T X_j + R eta, so

        Cov(e_j, e_k) = [P_j (T')^{k-j}]_{00},   P_{j+1} = T P_j T' + RR'.

    The caller needs the cross-covariances, not just the diagonal, to
    aggregate forecast variances through un-differencing exactly the
    way R's integrated state space does.

    Parameters
    ----------
    z : NDArray
        Centered (mean-subtracted) differenced series.
    ar : NDArray
        Effective AR coefficients (multiplied-out for seasonal models).
    ma : NDArray
        Effective MA coefficients.
    h : int
        Forecast horizon (>= 1).

    Returns
    -------
    fc : NDArray, shape (h,)
        Point forecasts of z at times n+1 .. n+h.
    err_cov : NDArray, shape (h, h)
        Forecast-error covariance matrix under sigma2 = 1; multiply by
        the fitted innovation variance to get actual covariances.

    Raises
    ------
    ConvergenceError
        If the Kalman filter fails at the supplied parameters (only
        possible at numerically pathological parameter values).
    """
    from pystatistics.core.exceptions import ConvergenceError

    _, R_vec, r = _build_state_space(ar, ma)
    phi = np.zeros(r, dtype=np.float64)
    p = len(ar)
    if p > 0:
        phi[:p] = ar

    P, init_ok = _stationary_init(phi, R_vec)
    if not init_ok:
        P = _DIFFUSE_KAPPA * np.eye(r, dtype=np.float64)
    a = np.zeros(r, dtype=np.float64)

    # _kalman_loop mutates a and P in place; on return they hold the
    # one-step-ahead predictive state mean/covariance for time n+1.
    _, _, ok = _kalman_loop(
        np.ascontiguousarray(z, dtype=np.float64), phi, R_vec, a, P,
    )
    if not ok:
        raise ConvergenceError(
            "Kalman filter failed at the fitted parameters; cannot "
            "produce forecasts (non-finite state or non-positive "
            "innovation variance).",
            iterations=0,
            reason="kalman_eval_failed",
        )

    # T materialized once for the h-step state iteration.
    T = np.zeros((r, r), dtype=np.float64)
    for i in range(r):
        T[i, 0] = phi[i]
    for i in range(r - 1):
        T[i, i + 1] = 1.0
    RR = np.outer(R_vec, R_vec)

    fc = np.empty(h, dtype=np.float64)
    err_cov = np.zeros((h, h), dtype=np.float64)
    for k in range(h):
        fc[k] = a[0]
        err_cov[k, k] = P[0, 0]
        # Cross-covariances with later horizons: M = P_k (T')^s.
        M = P
        for s in range(k + 1, h):
            M = M @ T.T
            err_cov[k, s] = M[0, 0]
            err_cov[s, k] = M[0, 0]
        a = T @ a
        P = T @ P @ T.T + RR
    return fc, err_cov


def kalman_arma_innovations(
    z: NDArray,
    ar: NDArray,
    ma: NDArray,
) -> NDArray:
    """Standardized one-step innovations of a zero-mean ARMA via the
    Kalman filter.

    Returns v_t / sqrt(F_t), where v_t = z_t - E[z_t | z_1..z_{t-1}]
    and F_t is the innovation variance under sigma2 = 1 — exactly what
    R's ``stats::arima`` returns as ``residuals()`` for ML-family fits
    (``arima.c`` scales by ``sqrt(gain)``). The standardization gives
    the residuals CONSTANT variance sigma2 at every t (raw innovations
    are heteroscedastic early, where the state is still uncertain), so
    ``mean(residuals**2)`` equals the profile ML sigma2 identically and
    Ljung-Box/ACF/normality diagnostics see the homoscedastic white
    noise the model asserts. CSS residuals approximate these only up to
    a conditioning transient decaying like the largest MA root
    modulus^t — materially different near an MA unit root, and
    divergent beyond it.

    Parameters
    ----------
    z : NDArray
        Centered (mean-subtracted) differenced series.
    ar : NDArray
        Effective AR coefficients.
    ma : NDArray
        Effective MA coefficients.

    Returns
    -------
    NDArray
        Standardized innovations, same length as ``z``.

    Raises
    ------
    ConvergenceError
        If the filter fails at the supplied parameters.
    """
    from pystatistics.core.exceptions import ConvergenceError

    _, R_vec, r = _build_state_space(ar, ma)
    phi = np.zeros(r, dtype=np.float64)
    p = len(ar)
    if p > 0:
        phi[:p] = ar

    P, init_ok = _stationary_init(phi, R_vec)
    if not init_ok:
        P = _DIFFUSE_KAPPA * np.eye(r, dtype=np.float64)
    a = np.zeros(r, dtype=np.float64)

    innov, F, ok = _kalman_loop(
        np.ascontiguousarray(z, dtype=np.float64), phi, R_vec, a, P,
    )
    if not ok:
        raise ConvergenceError(
            "Kalman filter failed at the fitted parameters; "
            "innovations cannot be computed (non-finite state or "
            "non-positive innovation variance).",
            iterations=0,
            reason="kalman_eval_failed",
        )
    return innov / np.sqrt(F)
