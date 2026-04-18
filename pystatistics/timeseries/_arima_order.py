"""
Automatic ARIMA order selection.

Simplified version of R's ``forecast::auto.arima()`` implementing the
Hyndman--Khandakar (2008) stepwise algorithm as well as an exhaustive
grid search.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.timeseries._differencing import diff, ndiffs


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AutoARIMAResult:
    """Result from automatic ARIMA order selection.

    Attributes
    ----------
    best_model : ARIMAResult
        The fitted model with the best information criterion.
    best_order : tuple[int, int, int]
        The (p, d, q) order of the best model.
    best_seasonal : tuple[int, int, int, int] | None
        The (P, D, Q, m) seasonal order, or ``None``.
    best_aic : float
        Value of the chosen information criterion for the best model.
    models_fitted : int
        Total number of models successfully evaluated.
    search_results : list[tuple[tuple, float]]
        ``(order, ic_value)`` pairs for every model tried (including
        those that failed, recorded with ``inf``).

    Methods
    -------
    summary()
        Human-readable summary of the search.
    """

    best_model: object  # ARIMAResult (forward reference)
    best_order: tuple[int, int, int]
    best_seasonal: tuple[int, int, int, int] | None
    best_aic: float
    models_fitted: int
    search_results: list[tuple[tuple, float]]

    def summary(self) -> str:
        """Return a human-readable summary of the search.

        Returns
        -------
        str
            Multi-line summary.
        """
        lines = [
            "Auto ARIMA Search Results",
            "=" * 40,
            f"Best model: ARIMA{self.best_order}",
        ]
        if self.best_seasonal is not None:
            lines.append(f"Seasonal:   {self.best_seasonal}")
        lines.extend([
            f"Best IC:    {self.best_aic:.4f}",
            f"Models fitted: {self.models_fitted}",
            "",
            "Search history (order, IC):",
        ])
        for order, ic_val in self.search_results:
            tag = " *" if ic_val == self.best_aic else ""
            lines.append(f"  {order}  {ic_val:.4f}{tag}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ic(result: object, ic: str) -> float:
    """Extract the chosen information criterion from an ARIMAResult.

    Parameters
    ----------
    result : ARIMAResult
        Fitted model.
    ic : str
        One of ``'aic'``, ``'aicc'``, ``'bic'``.

    Returns
    -------
    float
        IC value.
    """
    if ic == "aic":
        return result.aic  # type: ignore[attr-defined]
    elif ic == "aicc":
        return result.aicc  # type: ignore[attr-defined]
    else:
        return result.bic  # type: ignore[attr-defined]


def _try_fit(
    y: NDArray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] | None,
    ic: str,
    tol: float,
    max_iter: int,
    method: str = "CSS-ML",
    backend: str | None = None,
) -> tuple[object | None, float]:
    """Attempt to fit an ARIMA model; return (result, ic_value).

    On failure returns ``(None, inf)``.

    Parameters
    ----------
    y : NDArray
        Time series.
    order : tuple
        (p, d, q).
    seasonal_order : tuple or None
        (P, D, Q, m) or ``None``.
    ic : str
        Information criterion name.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    tuple[ARIMAResult | None, float]
        Fitted model (or ``None``) and IC value (``inf`` on failure).
    """
    from pystatistics.timeseries._arima_fit import arima  # lazy import

    try:
        result = arima(
            y,
            order=order,
            seasonal=seasonal_order,
            method=method,
            tol=tol,
            max_iter=max_iter,
            backend=backend,
        )
        if not result.converged:
            return None, math.inf
        return result, _get_ic(result, ic)
    except (ConvergenceError, ValidationError, np.linalg.LinAlgError,
            ValueError, RuntimeError):
        return None, math.inf


def _determine_d(y: NDArray, max_d: int) -> int:
    """Determine the non-seasonal differencing order using ADF test.

    Parameters
    ----------
    y : NDArray
        Time series.
    max_d : int
        Maximum d to consider.

    Returns
    -------
    int
        Recommended d.
    """
    return ndiffs(y, test="adf", max_d=max_d)


def _determine_D(y: NDArray, period: int, max_D: int) -> int:
    """Determine the seasonal differencing order.

    Uses a simple heuristic: apply seasonal differencing once and check
    whether ``ndiffs`` of the seasonally differenced series is lower.
    If so, set D=1; otherwise D=0.

    Parameters
    ----------
    y : NDArray
        Time series.
    period : int
        Seasonal period.
    max_D : int
        Maximum D (0 or 1).

    Returns
    -------
    int
        Recommended D (0 or 1).
    """
    if max_D == 0 or period <= 1 or len(y) < 2 * period + 1:
        return 0

    # Seasonal strength heuristic:
    # If the variance of the seasonally differenced series is noticeably
    # smaller, set D=1.
    try:
        y_sdiff = diff(y, differences=1, lag=period)
        var_orig = np.var(y)
        var_sdiff = np.var(y_sdiff)
        if var_orig > 0 and var_sdiff / var_orig < 0.90:
            return 1
    except ValidationError:
        pass
    return 0


# ---------------------------------------------------------------------------
# Stepwise search (Hyndman-Khandakar 2008)
# ---------------------------------------------------------------------------

def _stepwise_search(
    y: NDArray,
    d: int,
    max_p: int,
    max_q: int,
    seasonal_order: tuple[int, int, int, int] | None,
    ic: str,
    tol: float,
    max_iter: int,
    method: str = "CSS-ML",
    backend: str | None = None,
) -> tuple[object, tuple[int, int, int], float, list[tuple[tuple, float]]]:
    """Stepwise ARIMA order selection.

    Parameters
    ----------
    y : NDArray
        Time series (original, un-differenced).
    d : int
        Differencing order.
    max_p, max_q : int
        Upper bounds on p and q.
    seasonal_order : tuple or None
        Fixed seasonal order.
    ic : str
        Information criterion.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    tuple
        ``(best_result, best_order, best_ic, search_results)``
    """
    search_results: list[tuple[tuple, float]] = []

    # Initial candidates: (0,d,0), (1,d,0), (0,d,1), (2,d,2)
    initial_orders = [
        (2, d, 2),
        (0, d, 0),
        (1, d, 0),
        (0, d, 1),
    ]
    # Filter to valid ranges
    initial_orders = [
        (p, dd, q) for p, dd, q in initial_orders
        if p <= max_p and q <= max_q
    ]

    best_result = None
    best_order: tuple[int, int, int] = (0, d, 0)
    best_ic = math.inf

    for order in initial_orders:
        result, ic_val = _try_fit(
            y, order, seasonal_order, ic, tol, max_iter, method, backend,
        )
        search_results.append((order, ic_val))
        if ic_val < best_ic:
            best_ic = ic_val
            best_order = order
            best_result = result

    if best_result is None:
        raise ConvergenceError(
            "auto_arima: all initial candidate models failed to converge",
            iterations=0,
            reason="all_failed",
        )

    # Stepwise neighbourhood search
    improved = True
    visited: set[tuple[int, int, int]] = {o for o in initial_orders}

    while improved:
        improved = False
        p, _d, q = best_order

        # Generate neighbours: p +/- 1, q +/- 1, and combinations
        neighbours = []
        for dp, dq in [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]:
            np_, nq = p + dp, q + dq
            if 0 <= np_ <= max_p and 0 <= nq <= max_q:
                candidate = (np_, _d, nq)
                if candidate not in visited:
                    neighbours.append(candidate)

        for order in neighbours:
            visited.add(order)
            result, ic_val = _try_fit(
                y, order, seasonal_order, ic, tol, max_iter, method, backend,
            )
            search_results.append((order, ic_val))
            if ic_val < best_ic:
                best_ic = ic_val
                best_order = order
                best_result = result
                improved = True

    return best_result, best_order, best_ic, search_results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def _grid_search(
    y: NDArray,
    d: int,
    max_p: int,
    max_q: int,
    seasonal_order: tuple[int, int, int, int] | None,
    ic: str,
    tol: float,
    max_iter: int,
    method: str = "CSS-ML",
    backend: str | None = None,
) -> tuple[object, tuple[int, int, int], float, list[tuple[tuple, float]]]:
    """Exhaustive grid search over all (p, d, q) combinations.

    Parameters
    ----------
    y : NDArray
        Time series.
    d : int
        Differencing order.
    max_p, max_q : int
        Upper bounds.
    seasonal_order : tuple or None
        Fixed seasonal order.
    ic : str
        Information criterion.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    tuple
        ``(best_result, best_order, best_ic, search_results)``
    """
    search_results: list[tuple[tuple, float]] = []
    best_result = None
    best_order: tuple[int, int, int] = (0, d, 0)
    best_ic = math.inf

    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        order = (p, d, q)
        result, ic_val = _try_fit(
            y, order, seasonal_order, ic, tol, max_iter, method, backend,
        )
        search_results.append((order, ic_val))
        if ic_val < best_ic:
            best_ic = ic_val
            best_order = order
            best_result = result

    if best_result is None:
        raise ConvergenceError(
            "auto_arima: no model converged during grid search",
            iterations=0,
            reason="all_failed",
        )

    return best_result, best_order, best_ic, search_results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_arima(
    y: ArrayLike,
    *,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
    max_P: int = 2,
    max_Q: int = 2,
    max_D: int = 1,
    period: int = 1,
    ic: str = "aicc",
    stepwise: bool = True,
    tol: float = 1e-8,
    max_iter: int = 1000,
    method: str = "CSS-ML",
    backend: str | None = None,
) -> AutoARIMAResult:
    """Automatic ARIMA model selection.

    Simplified version of R's ``forecast::auto.arima()``.

    For ``stepwise=True`` the Hyndman--Khandakar (2008) algorithm is
    used: start from a set of initial candidates and greedily explore
    neighbouring orders.

    For ``stepwise=False`` an exhaustive grid search over all
    ``p = 0 .. max_p``, ``q = 0 .. max_q`` combinations is performed
    (much slower but thorough).

    Parameters
    ----------
    y : ArrayLike
        Time series.
    max_p, max_q : int
        Maximum non-seasonal AR / MA orders.
    max_d : int
        Maximum non-seasonal differencing order.
    max_P, max_Q, max_D : int
        Maximum seasonal orders.
    period : int
        Seasonal period (1 = non-seasonal).
    ic : str
        Information criterion: ``'aic'``, ``'aicc'``, or ``'bic'``.
    stepwise : bool
        Use stepwise search (default) or grid search.
    tol : float
        Convergence tolerance passed to :func:`arima`.
    max_iter : int
        Maximum iterations passed to :func:`arima`.
    method : str
        Estimation method forwarded to every candidate fit. Default
        ``'CSS-ML'`` matches R. Use ``'Whittle'`` with ``backend='gpu'``
        to route each candidate through the frequency-domain GPU path.
    backend : str or None
        Backend forwarded to every candidate fit. Default ``None`` →
        CPU (R-reference path). Pass ``'gpu'`` or ``'auto'`` to
        opt into the GPU path; only meaningful when the candidate
        fits actually support it (e.g. ``method='Whittle'``).

    Returns
    -------
    AutoARIMAResult
        Best model and search history.

    Raises
    ------
    ValidationError
        If inputs are invalid.
    ConvergenceError
        If no model converges.
    """
    arr = np.asarray(y, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValidationError("y: must be non-empty")
    if arr.size < 10:
        raise ValidationError(
            f"y: series of length {arr.size} is too short for auto_arima "
            f"(requires at least 10 observations)"
        )

    valid_ic = ("aic", "aicc", "bic")
    if ic not in valid_ic:
        raise ValidationError(f"ic: must be one of {valid_ic}, got '{ic}'")

    if max_p < 0 or max_q < 0 or max_d < 0:
        raise ValidationError(
            f"max_p, max_q, max_d must be non-negative, "
            f"got max_p={max_p}, max_q={max_q}, max_d={max_d}"
        )
    if period < 1:
        raise ValidationError(f"period: must be >= 1, got {period}")

    # --- Determine differencing order ---
    d = _determine_d(arr, max_d)

    # --- Seasonal order ---
    seasonal_order: tuple[int, int, int, int] | None = None
    if period > 1:
        D = _determine_D(arr, period, max_D)
        # For the seasonal component, use simple fixed P=1, Q=1 when
        # the data is seasonal, capped by user limits.
        P = min(1, max_P)
        Q = min(1, max_Q)
        seasonal_order = (P, D, Q, period)

    # --- Search ---
    if stepwise:
        best_result, best_order, best_ic_val, search_results = _stepwise_search(
            arr, d, max_p, max_q, seasonal_order, ic, tol, max_iter,
            method, backend,
        )
    else:
        best_result, best_order, best_ic_val, search_results = _grid_search(
            arr, d, max_p, max_q, seasonal_order, ic, tol, max_iter,
            method, backend,
        )

    models_fitted = sum(1 for _, v in search_results if v < math.inf)

    return AutoARIMAResult(
        best_model=best_result,
        best_order=best_order,
        best_seasonal=seasonal_order,
        best_aic=best_ic_val,
        models_fitted=models_fitted,
        search_results=search_results,
    )
