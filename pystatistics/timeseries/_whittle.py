"""Whittle (frequency-domain) approximate MLE for ARMA models.

The exact Gaussian likelihood for an ARMA(p, q) model needs a Kalman
filter forward-pass (``_arima_kalman``) whose cost scales as O(n) with
a large constant: at n ≈ 10 000 on a single series the Kalman path
runs ~30 ms per L-BFGS-B evaluation, and L-BFGS-B typically needs
50-200 evaluations, so a single fit crosses the second mark.

Whittle's approximation replaces the exact likelihood with a frequency-
domain criterion evaluated against the periodogram:

    L_W(phi, theta, sigma²) = -½ Σ_k [log f(ω_k; θ) + I(ω_k) / f(ω_k; θ)]

where the sum runs over the non-zero Fourier frequencies
``ω_k = 2πk/n`` for ``k = 1 … m = (n-1)//2``,
``I(ω) = |FFT(y)|² / n`` is the sample periodogram, and
``f(ω; θ) = (σ²/2π) · |MA(e^{iω})|² / |AR(e^{iω})|²`` is the ARMA
spectral density. The periodogram is one FFT (O(n log n)); the
spectrum is p + q complex exponentials per frequency. There is no
per-iteration state, so the whole thing is GPU-ideal — a batched
elementwise workload after one FFT.

This module exports the numpy CPU implementation. The GPU version
lives in ``backends/whittle_gpu.py`` and re-uses the same algebra on
device tensors via autograd.

Scope
-----
- Non-seasonal ARMA(p, q) on a stationary input series (the caller
  is responsible for differencing — see ``arima(..., method='whittle')``
  which applies the differencing before calling in).
- Concentrated out σ², so the optimizer works over (phi, theta) only.
- AR stationarity: not enforced by the parameterization; the fitter
  checks the AR polynomial roots at convergence and raises
  ``ConvergenceError`` if the optimum is non-stationary. For
  practical use on real (differenced) data this is almost never
  triggered; when it is, it signals the model order is wrong.

References
----------
- Whittle, P. (1953). Estimation and information in stationary time
  series. Arkiv för Matematik, 2(5), 423-434.
- Brockwell, P. J. & Davis, R. A. (1991). Time Series: Theory and
  Methods (2nd ed.), Ch. 10.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pystatistics.core.exceptions import ConvergenceError, ValidationError


def _periodogram(y: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Sample periodogram ``I(ω_k) = |FFT(y)|² / n`` at the ``(n-1)//2``
    non-zero, non-Nyquist Fourier frequencies.

    The DC (k=0) bin is skipped because it carries the mean-deviation,
    which we centred away. The Nyquist bin (for even n) is also
    skipped because ``|MA|²/|AR|²`` and the periodogram are both
    strictly real there, contributing a constant to the NLL.
    """
    n = y.shape[0]
    # rfft returns bins 0 .. n//2. We drop bin 0 and the Nyquist bin.
    fft_y = np.fft.rfft(y)
    spec = (fft_y.real ** 2 + fft_y.imag ** 2) / n
    m = (n - 1) // 2
    return spec[1 : 1 + m]


def _fourier_frequencies(n: int) -> NDArray[np.floating[Any]]:
    """Non-zero Fourier frequencies matching ``_periodogram``."""
    m = (n - 1) // 2
    return 2.0 * np.pi * np.arange(1, m + 1) / n


def _arma_log_g(
    phi: NDArray[np.floating[Any]],
    theta: NDArray[np.floating[Any]],
    freqs: NDArray[np.floating[Any]],
    *,
    cos_tab: NDArray[np.floating[Any]] | None = None,
    sin_tab: NDArray[np.floating[Any]] | None = None,
) -> NDArray[np.floating[Any]]:
    """Return ``log g(ω) = log(|MA(e^{iω})|² / |AR(e^{iω})|²)``.

    ``cos_tab`` and ``sin_tab`` are optional precomputed ``(m, k)``
    tables of ``cos(jω)`` / ``-sin(jω)`` for ``j = 1..k``; when
    supplied the per-call complex-exponential construction is skipped.
    The fitter in :func:`fit_whittle_arma` precomputes these tables
    once per fit and re-uses them across every L-BFGS-B evaluation,
    which shaves a large constant off long-series fits (at n=2·10⁵
    the cos/sin rebuild dominated total CPU time).
    """
    p = phi.shape[0]
    q = theta.shape[0]
    if cos_tab is None or sin_tab is None:
        k = max(p, q, 1)
        j = np.arange(1, k + 1, dtype=freqs.dtype)
        angle = -freqs[:, np.newaxis] * j[np.newaxis, :]
        cos_tab = np.cos(angle)
        sin_tab = np.sin(angle)

    if p > 0:
        cos_p = cos_tab[:, :p]
        sin_p = sin_tab[:, :p]
        ar_re = 1.0 - cos_p @ phi
        ar_im = -sin_p @ phi
        log_ar_mag2 = np.log(ar_re * ar_re + ar_im * ar_im)
    else:
        log_ar_mag2 = np.zeros_like(freqs)

    if q > 0:
        cos_q = cos_tab[:, :q]
        sin_q = sin_tab[:, :q]
        ma_re = 1.0 + cos_q @ theta
        ma_im = sin_q @ theta
        log_ma_mag2 = np.log(ma_re * ma_re + ma_im * ma_im)
    else:
        log_ma_mag2 = np.zeros_like(freqs)

    return log_ma_mag2 - log_ar_mag2


def _whittle_concentrated_nll(
    params: NDArray[np.floating[Any]],
    periodogram: NDArray[np.floating[Any]],
    freqs: NDArray[np.floating[Any]],
    p: int,
    q: int,
    cos_tab: NDArray[np.floating[Any]] | None = None,
    sin_tab: NDArray[np.floating[Any]] | None = None,
) -> float:
    """Concentrated (in σ²) Whittle negative log-likelihood.

    Setting the score for σ² to zero gives ``σ²_hat = mean(I/g)``
    under the convention ``f = σ² · g`` on the discrete Fourier grid
    (see the ``sigma2`` comment in :func:`fit_whittle_arma`).
    Substituting back drops the σ²-dependence of the objective,
    leaving a clean function of (phi, theta) only — which is what
    scipy optimizes.

    ``cos_tab`` and ``sin_tab`` are the precomputed cos/sin tables
    from :func:`fit_whittle_arma`; see :func:`_arma_log_g` for why.
    """
    phi = params[:p]
    theta = params[p : p + q]
    log_g = _arma_log_g(
        phi, theta, freqs, cos_tab=cos_tab, sin_tab=sin_tab,
    )

    # Safe-guard against numerical overflow at near-zero |AR|² (i.e.
    # AR roots near the unit circle). Clamp log_g at a finite floor;
    # the optimizer will naturally move away.
    log_g = np.clip(log_g, -50.0, 50.0)
    g = np.exp(log_g)

    # σ²_hat = 2π · mean(I / g). The 2π factor folds into the constant
    # of the NLL and we absorb it.
    mean_I_over_g = np.mean(periodogram / g)
    m = periodogram.shape[0]

    # Whittle NLL (up to additive constants):
    #   m · log(σ²_hat) + Σ log g
    return m * np.log(max(mean_I_over_g, 1e-20)) + log_g.sum()


def _check_ar_stationarity(phi: NDArray[np.floating[Any]]) -> bool:
    """Return True iff all AR polynomial roots lie outside the unit circle."""
    if phi.shape[0] == 0:
        return True
    # AR polynomial: 1 - φ_1 z - φ_2 z² - ... - φ_p z^p  (in terms of z).
    # Stationary ⇔ all roots have |z| > 1.
    poly = np.concatenate([[1.0], -phi])
    roots = np.roots(poly[::-1])  # numpy uses highest-degree-first
    return bool(np.all(np.abs(roots) > 1.0 + 1e-8))


def fit_whittle_arma(
    y: NDArray[np.floating[Any]],
    p: int,
    q: int,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
    start_params: NDArray[np.floating[Any]] | None = None,
) -> tuple[NDArray, NDArray, float, int, bool]:
    """Fit ARMA(p, q) via Whittle approximate MLE on a stationary series.

    Parameters
    ----------
    y : NDArray
        Stationary time series (the caller differences first).
    p, q : int
        AR and MA orders.
    tol : float
        Gradient tolerance for L-BFGS-B.
    max_iter : int
        Maximum optimizer iterations.
    start_params : NDArray or None
        Starting values for ``[phi_1..phi_p, theta_1..theta_q]``. If
        None, a small perturbation around zero is used.

    Returns
    -------
    phi, theta, sigma2, n_iter, converged

    Raises
    ------
    ValidationError
        On invalid order or insufficient data.
    ConvergenceError
        If the optimizer fails or lands at a non-stationary AR.
    """
    if p < 0 or q < 0:
        raise ValidationError(f"p, q must be >= 0, got p={p}, q={q}")
    n = y.shape[0]
    if n < 2 * (p + q + 1) + 4:
        raise ValidationError(
            f"Whittle ARMA: need n >= {2 * (p + q + 1) + 4} observations "
            f"for order (p={p}, q={q}); got n={n}."
        )

    # Centre y so the DC bin we drop really is the mean component.
    y_c = y - float(np.mean(y))

    I_omega = _periodogram(y_c)
    freqs = _fourier_frequencies(n)

    # Precompute cos/sin of jω once per fit; the optimizer reuses
    # them across every L-BFGS-B evaluation.
    k = max(p, q, 1)
    j = np.arange(1, k + 1, dtype=freqs.dtype)
    angle = -freqs[:, np.newaxis] * j[np.newaxis, :]
    cos_tab = np.cos(angle)
    sin_tab = np.sin(angle)

    n_params = p + q
    if start_params is None:
        # Small negative MA start mirrors the heuristic in the CSS-ML
        # path — real data tends to have mild positive serial
        # correlation at short lags, corresponding to negative θ. AR
        # starts at zero; the Whittle surface is smooth enough that
        # this works without Yule-Walker priming for typical orders.
        start_params = np.concatenate([
            np.zeros(p), np.full(q, -0.1),
        ]).astype(np.float64)

    if n_params == 0:
        # ARMA(0, 0): σ²_hat = var(y), zero params.
        sigma2 = float(np.dot(y_c, y_c) / n)
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            sigma2, 0, True,
        )

    # Numerical gradient via scipy's finite difference — good to ~1e-6
    # on the concentrated NLL. ftol tighter than that triggers L-BFGS-B
    # ABNORMAL (line-search stall on the noise floor), same mode as the
    # FP32 polr path. We don't build an analytical gradient on the CPU
    # side because (p+q) is tiny so finite differencing is cheap.
    result = minimize(
        _whittle_concentrated_nll,
        start_params,
        args=(I_omega, freqs, p, q, cos_tab, sin_tab),
        method="L-BFGS-B",
        options={
            "maxiter": max_iter,
            "gtol": max(tol, 1e-6),
            "ftol": max(tol, 1e-8),
        },
    )

    if not result.success:
        raise ConvergenceError(
            f"Whittle ARMA optimizer did not converge: {result.message}. "
            f"Completed {result.nit} iterations.",
            iterations=int(result.nit),
            reason=str(result.message),
        )

    phi = result.x[:p]
    theta = result.x[p : p + q]

    if not _check_ar_stationarity(phi):
        raise ConvergenceError(
            "Whittle ARMA converged to a non-stationary AR polynomial. "
            "This usually signals the wrong order or inadequate "
            "differencing. Try increasing ``d`` or reducing ``p``.",
            iterations=int(result.nit),
            reason="non-stationary AR",
        )

    # Recover σ² from the concentrated form.
    # Convention: the Whittle spectral density is f(ω) = σ² · g(ω)
    # (unnormalised, so the periodogram I(ω) has E[I] = σ² for white
    # noise directly — verified by the σ² ≈ var(y) limit at p=q=0).
    # The 2π that appears in the continuous-frequency spectral density
    # f(ω) = σ²/(2π) · g(ω) is absorbed into the periodogram scaling
    # on a discrete Fourier grid. See Brockwell & Davis, Ch. 10.
    log_g = _arma_log_g(
        phi, theta, freqs, cos_tab=cos_tab, sin_tab=sin_tab,
    )
    log_g = np.clip(log_g, -50.0, 50.0)
    g = np.exp(log_g)
    sigma2 = float(np.mean(I_omega / g))

    return phi, theta, sigma2, int(result.nit), True


def fit_arima_whittle(
    *,
    y_diff: NDArray[np.floating[Any]],
    n_obs: int,
    order: tuple[int, int, int],
    include_mean: bool,
    tol: float,
    max_iter: int,
    backend: str | None,
    use_fp64: bool,
    yule_walker_start: Callable[[NDArray, int], NDArray],
    css_residuals: Callable[[NDArray, NDArray, NDArray, float], NDArray],
    ARIMAResult: Any,
):
    """Assemble an :class:`ARIMAResult` for a Whittle-method ARIMA fit.

    Called by :func:`arima` when ``method='Whittle'``. Kept here (and
    not in ``_arima_fit.py``) so the Whittle implementation stays
    self-contained and the main arima-fit module stays under the
    500-LOC limit.

    The result's time-domain quantities (residuals, fitted values,
    log-likelihood, AIC/BIC) are reconstructed from the Whittle-fitted
    coefficients via the shared CSS residual evaluator. vcov is not
    computed — Whittle users who need coefficient SEs should fall back
    to method='ML' or 'CSS-ML'.
    """
    from scipy.optimize import minimize as _minimize

    p, d, q = order

    if backend is None:
        backend = "cpu"
    if backend not in ("cpu", "auto", "gpu"):
        raise ValidationError(
            f"backend: must be 'cpu', 'auto', or 'gpu', got {backend!r}"
        )

    # Whittle is centred internally. ``include_mean=True`` means "also
    # report the sample mean that we subtracted", not "estimate mean as
    # a free parameter".
    mu = float(np.mean(y_diff)) if include_mean else 0.0
    y_centered = y_diff - mu
    n_used = len(y_diff)

    gpu_like = None
    if backend != "cpu":
        from pystatistics.core.compute.device import select_device
        dev = select_device("gpu" if backend == "gpu" else "auto")
        if dev.is_gpu:
            from pystatistics.timeseries.backends.whittle_gpu import (
                WhittleGPULikelihood,
            )
            gpu_like = WhittleGPULikelihood(
                y_centered, p, q,
                device=dev.device_type, use_fp64=use_fp64,
            )
        elif backend == "gpu":
            raise RuntimeError(
                "backend='gpu' requested but no GPU is available. "
                "Install PyTorch with CUDA/MPS support or use "
                "backend='cpu'."
            )

    # FP32 noise floor: same convention as multinom / polr / gam.
    gpu_fp32_min_tol = 1e-5
    effective_tol = tol
    if gpu_like is not None and not use_fp64 and tol < gpu_fp32_min_tol:
        effective_tol = gpu_fp32_min_tol

    # Yule-Walker AR init (always stationary). Zero-start on AR is not
    # safe: the frequency-domain likelihood is symmetric under each AR
    # root's reciprocal, so a zero init can drift across the unit
    # circle into the non-stationary mirror basin.
    if p > 0:
        ar_start = np.clip(yule_walker_start(y_centered, p), -0.99, 0.99)
    else:
        ar_start = np.zeros(0, dtype=np.float64)
    start = np.concatenate([ar_start, np.full(q, -0.1)]).astype(np.float64)

    if p + q == 0:
        phi_hat = np.zeros(0, dtype=np.float64)
        theta_hat = np.zeros(0, dtype=np.float64)
        sigma2 = float(np.dot(y_centered, y_centered) / n_used)
        n_iter = 0
        converged = True
    elif gpu_like is not None:
        result = _minimize(
            gpu_like.fun, start, method="L-BFGS-B", jac=gpu_like.jac,
            options={"maxiter": max_iter, "gtol": effective_tol,
                     "ftol": 1e-12 if use_fp64 else effective_tol},
        )
        if not result.success:
            raise ConvergenceError(
                f"Whittle ARIMA (GPU) did not converge: {result.message}",
                iterations=int(result.nit),
                reason=str(result.message),
            )
        phi_hat = result.x[:p]
        theta_hat = result.x[p : p + q]
        if not _check_ar_stationarity(phi_hat):
            raise ConvergenceError(
                "Whittle ARIMA converged to a non-stationary AR "
                "polynomial. Check the model order and differencing.",
                iterations=int(result.nit),
                reason="non-stationary AR",
            )
        sigma2 = gpu_like.sigma2(result.x)
        n_iter = int(result.nit)
        converged = True
    else:
        phi_hat, theta_hat, sigma2, n_iter, converged = fit_whittle_arma(
            y_centered, p, q, tol=effective_tol, max_iter=max_iter,
            start_params=start,
        )

    residuals = css_residuals(y_diff, phi_hat, theta_hat, mu)
    fitted = y_diff - residuals
    loglik = -(
        0.5 * n_used * np.log(2.0 * np.pi)
        + 0.5 * n_used * np.log(max(sigma2, 1e-15))
        + 0.5 * n_used
    )
    n_coef = p + q + (1 if include_mean else 0)
    k = n_coef + 1
    aic = -2.0 * loglik + 2.0 * k
    bic = -2.0 * loglik + k * np.log(n_used)
    aicc = (
        aic + 2.0 * k * (k + 1.0) / (n_used - k - 1.0)
        if n_used - k - 1 > 0 else np.inf
    )
    vcov = np.full((n_coef, n_coef), np.nan)

    return ARIMAResult(
        order=order,
        seasonal_order=None,
        ar=phi_hat,
        ma=theta_hat,
        seasonal_ar=np.array([], dtype=np.float64),
        seasonal_ma=np.array([], dtype=np.float64),
        mean=(mu if include_mean else None),
        sigma2=sigma2,
        vcov=vcov,
        residuals=residuals,
        fitted_values=fitted,
        log_likelihood=loglik,
        aic=aic,
        aicc=aicc,
        bic=bic,
        n_obs=n_obs,
        n_used=n_used,
        method="Whittle" + ("-GPU" if gpu_like is not None else ""),
        converged=converged,
        n_iter=n_iter,
    )
