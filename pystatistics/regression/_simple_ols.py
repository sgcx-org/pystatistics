"""Lean univariate ordinary least squares — a 1-D front door for OLS.

``simple_ols(x, y)`` fits the with-intercept line ``y = intercept + slope·x``
to two equal-length 1-D vectors and returns just the handful of quantities a
univariate caller actually wants: slope, intercept, R², adjusted R², the
slope's standard error, and the sample size.

Why this exists — and why it is *not* ``fit()``:
    ``fit(X, y)`` is the full-inference workhorse. Each call assembles a
    :class:`~pystatistics.regression.design.Design`, resolves term names, and
    instantiates a backend so that lazy diagnostics (hat values, Cook's
    distance, ...) are available on demand. For a plain "fit a line to these
    two vectors" call inside a tight loop — e.g. PK terminal-slope (λz)
    estimation scoring many candidate windows per profile — that per-call
    ceremony is pure overhead, enough that domain code has historically
    reached past the engine for ``scipy.stats.linregress`` instead.
    ``simple_ols`` is the lean primitive that closes that gap: no Design, no
    backend, no lazy-inference object.

Deviation from convention (Rule 3 note): this module deliberately does NOT
return the usual ``Result[LinearParams]`` / ``LinearSolution`` wrapper. Staying
allocation-light in hot loops is the whole point, so the return type is a small
frozen :class:`SimpleOLSResult` dataclass rather than the heavy lazy-inference
object. Callers needing full inference (confidence intervals, influence
measures, multi-column designs) should use ``fit()``.

Validated against R ``lm(y ~ x)`` to ``rtol=1e-10`` for coefficients,
``summary()$r.squared`` / ``$adj.r.squared``, and the slope's ``Std. Error``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from pystatistics.core.exceptions import ValidationError


@dataclass(frozen=True)
class SimpleOLSResult:
    """Result of a with-intercept univariate OLS fit (``y = intercept + slope·x``).

    A deliberately lean, frozen container — not the library's usual
    ``Result[LinearParams]`` / ``LinearSolution`` — so ``simple_ols`` stays
    allocation-light when called thousands of times in a loop (see module
    docstring).

    Attributes:
        slope: Estimated slope ``Sxy / Sxx``.
        intercept: Estimated intercept ``ȳ − slope·x̄``.
        r_squared: Coefficient of determination ``Sxy² / (Sxx·Syy)``.
        adjusted_r_squared: ``1 − (1 − r²)·(n − 1)/(n − 2)`` (k = 2 params).
        slope_se: Standard error of the slope, ``sqrt((RSS/(n − 2)) / Sxx)``.
        n: Number of observations used.
    """

    slope: float
    intercept: float
    r_squared: float
    adjusted_r_squared: float
    slope_se: float
    n: int


def simple_ols(x: ArrayLike, y: ArrayLike) -> SimpleOLSResult:
    """Fit ``y = intercept + slope·x`` by ordinary least squares.

    A lean univariate front door for OLS: pass two equal-length 1-D vectors,
    get back slope, intercept, R², adjusted R², and the slope's standard error
    without constructing a Design or a full ``LinearSolution``.

    Args:
        x: Predictor, a 1-D array-like of length ``n`` (``n >= 3``).
        y: Response, a 1-D array-like of the same length as ``x``.

    Returns:
        A frozen :class:`SimpleOLSResult`.

    Raises:
        ValidationError: If ``x`` and ``y`` are not both 1-D; if their lengths
            differ; if ``n < 3`` (adjusted R² and ``slope_se`` need
            ``n − 2 >= 1``); if either contains a non-finite value (NaN/Inf are
            never silently dropped — the caller owns masking); if ``x`` has zero
            variance (``Sxx == 0``, slope undefined); or if ``y`` has zero
            variance (``Syy == 0``, R² undefined).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 1 or y.ndim != 1:
        raise ValidationError(
            f"x and y must both be 1-dimensional, got x.ndim={x.ndim}, "
            f"y.ndim={y.ndim}"
        )
    if x.shape[0] != y.shape[0]:
        raise ValidationError(
            f"x and y must have the same length, got {x.shape[0]} and "
            f"{y.shape[0]}"
        )

    n = x.shape[0]
    if n < 3:
        raise ValidationError(
            f"simple_ols needs at least 3 observations (n − 2 >= 1 for "
            f"adjusted R² and slope_se), got n={n}"
        )
    if not np.all(np.isfinite(x)):
        raise ValidationError(
            "x contains non-finite values (NaN/Inf); simple_ols does not "
            "silently drop them — mask before calling"
        )
    if not np.all(np.isfinite(y)):
        raise ValidationError(
            "y contains non-finite values (NaN/Inf); simple_ols does not "
            "silently drop them — mask before calling"
        )

    x_bar = x.mean()
    y_bar = y.mean()
    dx = x - x_bar
    dy = y - y_bar
    s_xx = float(dx @ dx)
    s_yy = float(dy @ dy)
    s_xy = float(dx @ dy)

    if s_xx == 0.0:
        raise ValidationError(
            "x has zero variance (all values identical); slope is undefined"
        )
    if s_yy == 0.0:
        raise ValidationError(
            "y has zero variance (constant response); R² is undefined"
        )

    slope = s_xy / s_xx
    intercept = y_bar - slope * x_bar

    # RSS from residuals directly (not Syy − Sxy²/Sxx) to avoid the
    # catastrophic cancellation of near-collinear fits where RSS is a tiny
    # difference of two large sums.
    residuals = y - (intercept + slope * x)
    rss = float(residuals @ residuals)

    r_squared = (s_xy * s_xy) / (s_xx * s_yy)
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - 2)
    slope_se = float(np.sqrt((rss / (n - 2)) / s_xx))

    return SimpleOLSResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
        slope_se=slope_se,
        n=n,
    )
