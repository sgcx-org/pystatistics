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
from pystatistics.core.result import Result, SolutionReprMixin
from pystatistics.timeseries._differencing import diff, ndiffs


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AutoARIMAParams:
    """Immutable parameter payload for automatic ARIMA order selection.

    Attributes
    ----------
    best_model : ARIMASolution
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
        those that failed, recorded with ``inf``). For seasonal
        searches ``order`` is the pair ``((p, d, q), (P, D, Q, m))``.
    """

    best_model: object  # ARIMASolution (forward reference)
    best_order: tuple[int, int, int]
    best_seasonal: tuple[int, int, int, int] | None
    best_aic: float
    models_fitted: int
    search_results: list[tuple[tuple, float]]


@dataclass
class AutoARIMASolution(SolutionReprMixin):
    """Result from automatic ARIMA order selection.

    Wraps a :class:`Result` ``[AutoARIMAParams]`` envelope; every datum is
    exposed via a read-only ``@property`` so the public attribute surface is
    unchanged from the previous flat dataclass.

    Attributes
    ----------
    best_model : ARIMASolution
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
        those that failed, recorded with ``inf``). For seasonal
        searches ``order`` is the pair ``((p, d, q), (P, D, Q, m))``.

    Methods
    -------
    summary()
        Human-readable summary of the search.
    """

    _result: Result[AutoARIMAParams]

    @property
    def best_model(self) -> object:
        return self._result.params.best_model

    @property
    def best_order(self) -> tuple[int, int, int]:
        return self._result.params.best_order

    @property
    def best_seasonal(self) -> tuple[int, int, int, int] | None:
        return self._result.params.best_seasonal

    @property
    def best_aic(self) -> float:
        return self._result.params.best_aic

    @property
    def models_fitted(self) -> int:
        return self._result.params.models_fitted

    @property
    def search_results(self) -> list[tuple[tuple, float]]:
        return self._result.params.search_results

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
    """Extract the chosen information criterion from an ARIMASolution.

    Parameters
    ----------
    result : ARIMASolution
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
    tuple[ARIMASolution | None, float]
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
    seasonal_start: tuple[int, int, int, int] | None,
    ic: str,
    tol: float,
    max_iter: int,
    method: str = "CSS-ML",
    backend: str | None = None,
    max_P: int = 2,
    max_Q: int = 2,
) -> tuple[
    object,
    tuple[int, int, int],
    tuple[int, int, int, int] | None,
    float,
    list[tuple[tuple, float]],
]:
    """Stepwise ARIMA order selection (Hyndman-Khandakar 2008).

    For seasonal models the seasonal AR/MA orders (P, Q) are searched
    alongside (p, q) — matching R ``forecast::auto.arima`` — while the
    differencing orders d and D stay fixed. Previously (P, Q) was
    pinned at the starting value, which made models such as
    (2,1,1)(0,1,0)[12] unreachable (RIGOR R18).

    Parameters
    ----------
    y : NDArray
        Time series (original, un-differenced).
    d : int
        Differencing order.
    max_p, max_q : int
        Upper bounds on p and q.
    seasonal_start : tuple or None
        ``(P, D, Q, m)`` starting seasonal order (D and m are kept
        fixed; P and Q are searched), or ``None`` for non-seasonal.
    ic : str
        Information criterion.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    method, backend : str, str or None
        Forwarded to every candidate fit.
    max_P, max_Q : int
        Upper bounds on the seasonal AR / MA orders.

    Returns
    -------
    tuple
        ``(best_result, best_order, best_seasonal, best_ic,
        search_results)``. For seasonal searches each
        ``search_results`` entry is ``(((p,d,q), (P,D,Q,m)), ic)``;
        for non-seasonal searches it is ``((p,d,q), ic)``.
    """
    seasonal = seasonal_start is not None
    if seasonal:
        D, m = seasonal_start[1], seasonal_start[3]
    else:
        D, m = 0, 1

    def _orders(
        key: tuple[int, int, int, int],
    ) -> tuple[tuple[int, int, int], tuple[int, int, int, int] | None]:
        p, q, P, Q = key
        return (p, d, q), ((P, D, Q, m) if seasonal else None)

    # Initial candidates (Hyndman-Khandakar 2008), capped by the user
    # limits and de-duplicated: (2,d,2)(1,D,1), (0,d,0)(0,D,0),
    # (1,d,0)(1,D,0), (0,d,1)(0,D,1).
    raw_initial = [(2, 2, 1, 1), (0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1)]
    initial: list[tuple[int, int, int, int]] = []
    for p, q, P, Q in raw_initial:
        cand = (
            min(p, max_p), min(q, max_q),
            min(P, max_P) if seasonal else 0,
            min(Q, max_Q) if seasonal else 0,
        )
        if cand not in initial:
            initial.append(cand)

    search_results: list[tuple[tuple, float]] = []
    best_result: object | None = None
    best_key = initial[0]
    best_ic = math.inf
    visited: set[tuple[int, int, int, int]] = set()

    def _evaluate(key: tuple[int, int, int, int]) -> bool:
        """Fit one candidate; update the incumbent. Returns True on improvement."""
        nonlocal best_result, best_key, best_ic
        visited.add(key)
        order, s_order = _orders(key)
        result, ic_val = _try_fit(
            y, order, s_order, ic, tol, max_iter, method, backend,
        )
        search_results.append(
            ((order, s_order) if seasonal else order, ic_val)
        )
        if ic_val < best_ic:
            best_ic = ic_val
            best_key = key
            best_result = result
            return True
        return False

    for key in initial:
        _evaluate(key)

    if best_result is None:
        raise ConvergenceError(
            "auto_arima: all initial candidate models failed to converge",
            iterations=0,
            reason="all_failed",
        )

    # Neighbourhood moves: p/q +/- 1 alone and jointly; for seasonal
    # models also P/Q +/- 1 alone and jointly (Hyndman-Khandakar).
    pq_moves = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]
    PQ_moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]

    improved = True
    while improved:
        improved = False
        p, q, P, Q = best_key

        neighbours: list[tuple[int, int, int, int]] = []
        for dp, dq in pq_moves:
            cand = (p + dp, q + dq, P, Q)
            if (0 <= cand[0] <= max_p and 0 <= cand[1] <= max_q
                    and cand not in visited):
                neighbours.append(cand)
        if seasonal:
            for dP, dQ in PQ_moves:
                cand = (p, q, P + dP, Q + dQ)
                if (0 <= cand[2] <= max_P and 0 <= cand[3] <= max_Q
                        and cand not in visited):
                    neighbours.append(cand)

        for cand in neighbours:
            if _evaluate(cand):
                improved = True

    best_order, best_seasonal = _orders(best_key)
    return best_result, best_order, best_seasonal, best_ic, search_results


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def _grid_search(
    y: NDArray,
    d: int,
    max_p: int,
    max_q: int,
    seasonal_start: tuple[int, int, int, int] | None,
    ic: str,
    tol: float,
    max_iter: int,
    method: str = "CSS-ML",
    backend: str | None = None,
    max_P: int = 2,
    max_Q: int = 2,
) -> tuple[
    object,
    tuple[int, int, int],
    tuple[int, int, int, int] | None,
    float,
    list[tuple[tuple, float]],
]:
    """Exhaustive grid search over all (p, q) — and, for seasonal
    models, (P, Q) — combinations.

    Parameters
    ----------
    y : NDArray
        Time series.
    d : int
        Differencing order.
    max_p, max_q : int
        Upper bounds.
    seasonal_start : tuple or None
        ``(P, D, Q, m)`` — D and m are kept fixed, the P/Q grid is
        searched. ``None`` for non-seasonal.
    ic : str
        Information criterion.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    method, backend : str, str or None
        Forwarded to every candidate fit.
    max_P, max_Q : int
        Upper bounds on the seasonal AR / MA orders.

    Returns
    -------
    tuple
        ``(best_result, best_order, best_seasonal, best_ic,
        search_results)`` — same conventions as
        :func:`_stepwise_search`.
    """
    seasonal = seasonal_start is not None
    if seasonal:
        D, m = seasonal_start[1], seasonal_start[3]
        P_range: range | list[int] = range(max_P + 1)
        Q_range: range | list[int] = range(max_Q + 1)
    else:
        D, m = 0, 1
        P_range, Q_range = [0], [0]

    search_results: list[tuple[tuple, float]] = []
    best_result = None
    best_order: tuple[int, int, int] = (0, d, 0)
    best_seasonal: tuple[int, int, int, int] | None = (
        (0, D, 0, m) if seasonal else None
    )
    best_ic = math.inf

    for p, q, P, Q in itertools.product(
        range(max_p + 1), range(max_q + 1), P_range, Q_range,
    ):
        order = (p, d, q)
        s_order = (P, D, Q, m) if seasonal else None
        result, ic_val = _try_fit(
            y, order, s_order, ic, tol, max_iter, method, backend,
        )
        search_results.append(
            ((order, s_order) if seasonal else order, ic_val)
        )
        if ic_val < best_ic:
            best_ic = ic_val
            best_order = order
            best_seasonal = s_order
            best_result = result

    if best_result is None:
        raise ConvergenceError(
            "auto_arima: no model converged during grid search",
            iterations=0,
            reason="all_failed",
        )

    return best_result, best_order, best_seasonal, best_ic, search_results


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
) -> AutoARIMASolution:
    """Automatic ARIMA model selection.

    Simplified version of R's ``forecast::auto.arima()``.

    For ``stepwise=True`` the Hyndman--Khandakar (2008) algorithm is
    used: start from a set of initial candidates and greedily explore
    neighbouring orders. For seasonal models (``period > 1``) the
    seasonal AR/MA orders (P, Q) are searched alongside (p, q).

    For ``stepwise=False`` an exhaustive grid search over all
    ``p = 0 .. max_p``, ``q = 0 .. max_q`` — and, for seasonal models,
    ``P = 0 .. max_P``, ``Q = 0 .. max_Q`` — combinations is performed
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
    AutoARIMASolution
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
    if max_P < 0 or max_Q < 0 or max_D < 0:
        raise ValidationError(
            f"max_P, max_Q, max_D must be non-negative, "
            f"got max_P={max_P}, max_Q={max_Q}, max_D={max_D}"
        )
    if period < 1:
        raise ValidationError(f"period: must be >= 1, got {period}")

    # --- Determine differencing order ---
    d = _determine_d(arr, max_d)

    # --- Seasonal starting order (P and Q are then searched) ---
    seasonal_start: tuple[int, int, int, int] | None = None
    if period > 1:
        D = _determine_D(arr, period, max_D)
        seasonal_start = (min(1, max_P), D, min(1, max_Q), period)

    # --- Search ---
    if stepwise:
        best_result, best_order, best_seasonal, best_ic_val, search_results = (
            _stepwise_search(
                arr, d, max_p, max_q, seasonal_start, ic, tol, max_iter,
                method, backend, max_P, max_Q,
            )
        )
    else:
        best_result, best_order, best_seasonal, best_ic_val, search_results = (
            _grid_search(
                arr, d, max_p, max_q, seasonal_start, ic, tol, max_iter,
                method, backend, max_P, max_Q,
            )
        )

    models_fitted = sum(1 for _, v in search_results if v < math.inf)

    return AutoARIMASolution(
        _result=Result(
            params=AutoARIMAParams(
                best_model=best_result,
                best_order=best_order,
                best_seasonal=best_seasonal,
                best_aic=best_ic_val,
                models_fitted=models_fitted,
                search_results=search_results,
            ),
            info={
                "ic": ic,
                "stepwise": stepwise,
                "method": method,
                "models_fitted": models_fitted,
            },
            timing=None,
            backend_name="cpu",
            warnings=(),
        )
    )
