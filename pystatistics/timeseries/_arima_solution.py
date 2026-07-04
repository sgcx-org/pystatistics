"""
ARIMA result types.

Library-standard "Solution wraps Result[Params]" pattern for ARIMA fits:
- ARIMAParams: frozen dataclass holding the fitted-model data fields.
- ARIMASolution: wraps a :class:`Result` ``[ARIMAParams]`` envelope and
  exposes every datum via ``@property`` plus the shared metadata accessors
  (``.info``, ``.timing``, ``.backend_name``, ``.warnings``).

Kept in its own module (rather than ``_arima_fit.py``) so the main
arima-fit module stays under the 500-LOC limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result, SolutionReprMixin


@dataclass(frozen=True)
class ARIMAParams:
    """Immutable parameter payload for a fitted ARIMA model.

    Attributes
    ----------
    order : tuple[int, int, int]
        ``(p, d, q)`` — AR order, differencing order, MA order.
    seasonal_order : tuple[int, int, int, int] or None
        ``(P, D, Q, m)`` — seasonal orders and period, or ``None``.
    ar : NDArray
        AR coefficients (length p). For seasonal models, these are the
        non-seasonal AR coefficients only.
    ma : NDArray
        MA coefficients (length q). For seasonal models, these are the
        non-seasonal MA coefficients only.
    seasonal_ar : NDArray
        Seasonal AR coefficients (length P). Empty if non-seasonal.
    seasonal_ma : NDArray
        Seasonal MA coefficients (length Q). Empty if non-seasonal.
    mean : float or None
        Estimated mean of the differenced series (``None`` if
        ``include_mean=False`` or if the model has any differencing,
        matching R ``stats::arima``).
    sigma2 : float
        Estimated innovation variance.
    vcov : NDArray
        Variance-covariance matrix of the estimated coefficients
        (AR, MA, seasonal AR, seasonal MA, mean). Computed from the
        numerical Hessian of the negative log-likelihood.
    residuals : NDArray
        CSS (zero-conditioned) innovation residuals on the differenced
        scale. NOTE: for ML-family fits ``log_likelihood`` and
        ``sigma2`` come from the exact Kalman filter, so
        ``mean(residuals**2)`` may differ slightly from ``sigma2``
        (visibly so when an MA root is near the unit circle); R returns
        the Kalman innovations instead.
    fitted_values : NDArray
        One-step-ahead fitted values (length of the differenced series).
    log_likelihood : float
        Maximized log-likelihood value.
    aic : float
        Akaike information criterion: ``-2*loglik + 2*k``, where *k* is
        the number of FREE estimated parameters
        (``p + q + P + Q + mean-if-estimated + 1`` for sigma2 — equal to
        ``ARIMASolution.n_params``), matching R ``stats::arima``.
    aicc : float
        Corrected AIC: ``AIC + 2*k*(k+1)/(n-k-1)`` with the same free
        *k* and ``n = n_used``.
    bic : float
        Bayesian information criterion: ``-2*loglik + k*log(n)``.
    n_obs : int
        Length of the original (undifferenced) series.
    n_used : int
        Number of observations used in estimation (after differencing).
    method : str
        Estimation method: ``'CSS'``, ``'ML'``, or ``'CSS-ML'``.
    converged : bool
        Whether the optimizer converged.
    n_iter : int
        Number of optimizer iterations.
    """

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] | None
    ar: NDArray
    ma: NDArray
    seasonal_ar: NDArray
    seasonal_ma: NDArray
    mean: float | None
    sigma2: float
    vcov: NDArray
    residuals: NDArray
    fitted_values: NDArray
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    n_obs: int
    n_used: int
    method: str
    converged: bool
    n_iter: int


@dataclass
class ARIMASolution(SolutionReprMixin):
    """
    Result from fitting an ARIMA model.

    Wraps a :class:`Result` ``[ARIMAParams]`` envelope; every datum is
    exposed via a read-only ``@property`` so the public attribute surface
    is unchanged from the previous flat dataclass.

    Attributes
    ----------
    order : tuple[int, int, int]
        ``(p, d, q)`` — AR order, differencing order, MA order.
    seasonal_order : tuple[int, int, int, int] or None
        ``(P, D, Q, m)`` — seasonal orders and period, or ``None``.
    ar : NDArray
        AR coefficients (length p). For seasonal models, these are the
        non-seasonal AR coefficients only.
    ma : NDArray
        MA coefficients (length q). For seasonal models, these are the
        non-seasonal MA coefficients only.
    seasonal_ar : NDArray
        Seasonal AR coefficients (length P). Empty if non-seasonal.
    seasonal_ma : NDArray
        Seasonal MA coefficients (length Q). Empty if non-seasonal.
    mean : float or None
        Estimated mean of the differenced series (``None`` if
        ``include_mean=False``).
    sigma2 : float
        Estimated innovation variance.
    vcov : NDArray
        Variance-covariance matrix of the estimated coefficients.
    residuals : NDArray
        Innovation residuals (length of the differenced series).
    fitted_values : NDArray
        One-step-ahead fitted values (length of the differenced series).
    log_likelihood : float
        Maximized log-likelihood value.
    aic : float
        Akaike information criterion.
    aicc : float
        Corrected AIC.
    bic : float
        Bayesian information criterion.
    n_obs : int
        Length of the original (undifferenced) series.
    n_used : int
        Number of observations used in estimation (after differencing).
    method : str
        Estimation method: ``'CSS'``, ``'ML'``, or ``'CSS-ML'``.
    converged : bool
        Whether the optimizer converged.
    n_iter : int
        Number of optimizer iterations.
    """

    _result: Result[ARIMAParams]

    @property
    def order(self) -> tuple[int, int, int]:
        return self._result.params.order

    @property
    def seasonal_order(self) -> tuple[int, int, int, int] | None:
        return self._result.params.seasonal_order

    @property
    def ar(self) -> NDArray:
        return self._result.params.ar

    @property
    def ma(self) -> NDArray:
        return self._result.params.ma

    @property
    def seasonal_ar(self) -> NDArray:
        return self._result.params.seasonal_ar

    @property
    def seasonal_ma(self) -> NDArray:
        return self._result.params.seasonal_ma

    @property
    def mean(self) -> float | None:
        return self._result.params.mean

    @property
    def sigma2(self) -> float:
        return self._result.params.sigma2

    @property
    def vcov(self) -> NDArray:
        return self._result.params.vcov

    @property
    def residuals(self) -> NDArray:
        return self._result.params.residuals

    @property
    def fitted_values(self) -> NDArray:
        return self._result.params.fitted_values

    @property
    def log_likelihood(self) -> float:
        return self._result.params.log_likelihood

    @property
    def aic(self) -> float:
        return self._result.params.aic

    @property
    def aicc(self) -> float:
        return self._result.params.aicc

    @property
    def bic(self) -> float:
        return self._result.params.bic

    @property
    def n_obs(self) -> int:
        return self._result.params.n_obs

    @property
    def n_used(self) -> int:
        return self._result.params.n_used

    @property
    def method(self) -> str:
        return self._result.params.method

    @property
    def converged(self) -> bool:
        return self._result.params.converged

    @property
    def n_iter(self) -> int:
        return self._result.params.n_iter

    @property
    def info(self) -> dict:
        return self._result.info

    @property
    def timing(self) -> dict[str, float] | None:
        return self._result.timing

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    @property
    def n_params(self) -> int:
        """Total number of estimated parameters (AR + MA + seasonal + mean + sigma2)."""
        p = len(self.ar)
        q = len(self.ma)
        sp = len(self.seasonal_ar)
        sq = len(self.seasonal_ma)
        k = p + q + sp + sq + (1 if self.mean is not None else 0) + 1
        return k

    def summary(self) -> str:
        """
        R-style summary matching ``stats::arima()`` output.

        Returns
        -------
        str
            Multi-line summary string.
        """
        p, d, q = self.order
        if self.seasonal_order is not None:
            sp, sd, sq_s, m = self.seasonal_order
            header = f"ARIMA({p},{d},{q})({sp},{sd},{sq_s})[{m}]"
        else:
            header = f"ARIMA({p},{d},{q})"

        lines = [header, ""]
        names, coefs = self._collect_coef_names_values()

        if names:
            n_coef = len(coefs)
            se = np.sqrt(np.abs(np.diag(self.vcov[:n_coef, :n_coef])))
            cw = max(10, max(len(n) for n in names) + 2)

            lines.append("Coefficients:")
            lines.append("".join(f"{n:>{cw}}" for n in names))
            lines.append("".join(f"{c:>{cw}.4f}" for c in coefs))
            lines.append("s.e." + "".join(f"{s:>{cw}.4f}" for s in se))
            lines.append("")

        lines.append(
            f"sigma^2 = {self.sigma2:.4f}:  "
            f"log likelihood = {self.log_likelihood:.2f}"
        )
        lines.append(
            f"AIC={self.aic:.2f}   AICc={self.aicc:.2f}   BIC={self.bic:.2f}"
        )
        return "\n".join(lines)

    def _collect_coef_names_values(self) -> tuple[list[str], list[float]]:
        """Build parallel lists of coefficient names and values."""
        names: list[str] = []
        coefs: list[float] = []
        for i, v in enumerate(self.ar):
            names.append(f"ar{i + 1}"); coefs.append(float(v))
        for i, v in enumerate(self.ma):
            names.append(f"ma{i + 1}"); coefs.append(float(v))
        for i, v in enumerate(self.seasonal_ar):
            names.append(f"sar{i + 1}"); coefs.append(float(v))
        for i, v in enumerate(self.seasonal_ma):
            names.append(f"sma{i + 1}"); coefs.append(float(v))
        if self.mean is not None:
            names.append("intercept"); coefs.append(self.mean)
        return names, coefs

