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
from numpy.typing import NDArray

# Compiled Kalman kernels (Cython). These replace the former numba @njit
# kernels; the algorithm, operation order, and in-place mutation contract are
# unchanged. A pure-numpy twin lives in ``_arima_kalman_ref`` and is the
# bit-identity oracle the compiled kernels are tested against.
#
# ``_stationary_init`` solves the stationary initial covariance
# ``P = T P T' + R R'`` by a doubling iteration, judging convergence RELATIVE
# to the accumulated covariance scale. (An earlier ABSOLUTE-tol/linear version
# silently fell back to the diffuse init for persistent seasonal AR terms,
# shifting airline-class log-likelihoods by ~80 units — RIGOR R18 follow-up.)
# For non-stationary parameters the iterates blow up and it returns ok=False,
# and the caller falls back to the diffuse init R uses under ``kappa = 1e6``.
# The matrix products are explicit scalar loops (not BLAS), so P0 is now
# bit-reproducible across platforms rather than BLAS-reduction-order dependent.
#
# ``_kalman_loop`` is the forward filter, exploiting the companion-matrix
# structure of T so each state/covariance propagation is O(r^2) not O(r^3).
# It mutates ``a`` and ``P`` in place (callers rely on the post-filter state
# for forecasting) and returns (innov, F, ok).
from ._arima_kalman_kernel import ArmaKalmanWorkspace as _ArmaKalmanWorkspace  # noqa: E402
from ._arima_kalman_kernel import kalman_loop as _kalman_loop  # noqa: E402
from ._arima_kalman_kernel import stationary_init as _stationary_init  # noqa: E402

__all__ = ["kalman_arma_forecast", "kalman_arma_innovations", "kalman_arma_loglik"]

# Non-stationary fallback: matches R's `kappa` in stats:::makeARIMA.
_DIFFUSE_KAPPA = 1.0e6


def _new_workspace(p_eff: int, q_eff: int, n: int):
    """Create a fit-scoped Kalman workspace for an ARMA(p_eff, q_eff) fit.

    The workspace is **mutable, fit-scoped scratch storage**: it holds the
    reusable buffers for a single optimization over a fixed state dimension
    ``r = max(p_eff, q_eff+1)`` and series length ``n``. Reusing it across the
    optimizer's many likelihood evaluations avoids reallocating every scratch
    matrix per evaluation.

    Ownership rule: **one workspace per independent fit.** It must NOT be shared
    across concurrent objective evaluations (it is overwritten on every call);
    give each concurrently-running fit its own workspace.
    """
    r = max(p_eff, q_eff + 1) if (p_eff + q_eff) > 0 else 1
    return _ArmaKalmanWorkspace(r, n)


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


def _noise_loading(ar: NDArray, ma: NDArray) -> tuple[NDArray, int]:
    """State-noise loading R and state dim r, without materializing T.

    The Kalman kernels never use the dense transition matrix T (they exploit
    its companion structure via ``phi`` directly), so the loglik / forecast /
    innovation paths only need R and r. Building the r×r T on every likelihood
    evaluation was pure waste; this is the lean builder those paths use.
    """
    p = len(ar)
    q = len(ma)
    r = max(p, q + 1) if (p + q) > 0 else 1
    R_vec = np.zeros(r, dtype=np.float64)
    R_vec[0] = 1.0
    if q > 0:
        R_vec[1:1 + q] = ma
    return R_vec, r


def kalman_arma_loglik(
    y: NDArray,
    ar: NDArray,
    ma: NDArray,
    mean: float,
    *,
    _workspace=None,
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
    _workspace : ArmaKalmanWorkspace, optional
        Internal, fit-scoped scratch (see :func:`_new_workspace`). When
        supplied, the fused, buffer-reusing kernel path is used instead of
        allocating fresh scratch — bit-identical result, no per-call
        allocation. The workspace must be sized for this exact ``(r, n)`` and
        must not be shared across concurrent evaluations. Leave as ``None`` for
        one-shot / low-level calls (the historical allocating path).

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
    # The compiled kernel takes C-contiguous float64 (double[::1]); guarantee
    # it at the boundary rather than trusting the caller's array layout.
    z = np.ascontiguousarray(y - mean, dtype=np.float64)
    n = z.size
    if n == 0:
        return 1e18, 1.0

    R_vec, r = _noise_loading(ar, ma)
    # phi = AR coefficients padded with zeros to length r (the column-0
    # entries of the implicit T). The Kalman loop never materializes T
    # — it uses phi directly to exploit the companion-matrix structure.
    phi = np.zeros(r, dtype=np.float64)
    p = len(ar)
    if p > 0:
        phi[:p] = ar

    if _workspace is not None:
        # Fused path: init + diffuse-fallback + filter in one nogil call with
        # reused buffers. The sse / sum(log F) reductions are computed in numpy
        # inside the workspace, identically to the allocating path below, so
        # this returns the same (nll, sigma2) bit-for-bit.
        ok, sse, sum_log_F = _workspace.loglik_parts(z, phi, R_vec, _DIFFUSE_KAPPA)
        if not ok or not np.isfinite(sse) or sse <= 0.0:
            return 1e18, 1.0
        sigma2_hat = sse / n
        nll = (
            0.5 * n * np.log(2.0 * np.pi * sigma2_hat)
            + 0.5 * sum_log_F
            + 0.5 * n
        )
        if not np.isfinite(nll):
            return 1e18, 1.0
        return nll, sigma2_hat

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

    R_vec, r = _noise_loading(ar, ma)
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

    R_vec, r = _noise_loading(ar, ma)
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
