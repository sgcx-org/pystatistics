"""Cubic regression spline basis — mgcv-exact construction (Wood 2017 §5.3.1).

Implements the same cardinal natural-cubic-spline basis R's ``mgcv`` uses for
``s(x, bs="cr")``: knots at type-7 quantiles of the *unique* covariate values,
the banded ``D``/``B`` second-derivative relations, penalty ``S = D'B⁻¹D``, and
mgcv's ``scale.penalty`` normalisation. Verified against
``mgcv::smoothCon(s(x, bs="cr"), absorb.cons=FALSE)`` to ~1e-15 (basis matrix,
knots, penalty, and ``S.scale``).

Reference: Wood, S.N. (2017). Generalized Additive Models: An Introduction
with R (2nd ed.), §5.3.1. mgcv source: ``smooth.construct.cr.smooth.spec``,
``smoothCon`` (scale.penalty block).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.gam._basis_common import validate_k, validate_x


def cr_basis(
    x: NDArray[np.floating[Any]],
    k: int = 10,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], float]:
    """Construct the mgcv-exact cubic regression spline basis and penalty.

    Args:
        x: Predictor values, 1-D array of *n* observations (validated).
        k: Basis dimension — number of knots, exactly as mgcv's ``s(x, k=k)``.

    Returns:
        ``(X, S, s_scale)``: the ``(n, k)`` cardinal basis matrix, the
        ``(k, k)`` penalty already divided by mgcv's ``S.scale`` factor,
        and that ``s_scale`` factor itself (needed to convert smoothing
        parameters to/from function-space units).

    Raises:
        ValidationError: if inputs fail validation, or ``x`` has fewer than
            ``k`` unique values (mgcv fails on this too: the quantile knots
            would tie and the banded system would be singular).
    """
    x = validate_x(x)
    validate_k(k, n_unique=np.unique(x).shape[0])

    xk = place_knots_cr(x, k)
    X = _cardinal_basis(x, xk)
    S_raw = _penalty(xk)

    # mgcv smoothCon scale.penalty: S.scale = ||S||_1 / ||X||_inf^2,
    # S_used = S_raw / S.scale.  (norm(S) one-norm = max abs column sum;
    # norm(X, "I") = max abs row sum.)
    ma_xx = float(np.max(np.abs(X).sum(axis=1)) ** 2)
    s_scale = float(np.max(np.abs(S_raw).sum(axis=0)) / ma_xx)
    S = S_raw / s_scale

    return X, S, s_scale


def place_knots_cr(
    x: NDArray[np.floating[Any]], k: int
) -> NDArray[np.floating[Any]]:
    """mgcv ``place.knots``: type-7 quantiles of the unique covariate values.

    Args:
        x: Validated 1-D predictor.
        k: Number of knots (= basis dimension).

    Returns:
        Strictly increasing knot vector of length *k*.

    Raises:
        ValidationError: if the resulting knots are not strictly increasing
            (ties — cannot happen with >= k unique values, but guarded).
    """
    xu = np.unique(x)
    xk = np.quantile(xu, np.linspace(0.0, 1.0, k))
    if np.any(np.diff(xk) <= 0.0):
        raise ValidationError(
            f"cr knot placement produced tied knots (k={k}, "
            f"n_unique={xu.shape[0]}); reduce k or supply more distinct x"
        )
    return xk


def _band_matrices(
    xk: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """The banded ``D`` ((k-2, k)) and ``B`` ((k-2, k-2)) of Wood §5.3.1."""
    k = xk.shape[0]
    h = np.diff(xk)
    D = np.zeros((k - 2, k), dtype=np.float64)
    B = np.zeros((k - 2, k - 2), dtype=np.float64)
    for i in range(k - 2):
        D[i, i] = 1.0 / h[i]
        D[i, i + 1] = -1.0 / h[i] - 1.0 / h[i + 1]
        D[i, i + 2] = 1.0 / h[i + 1]
        B[i, i] = (h[i] + h[i + 1]) / 3.0
        if i + 1 < k - 2:
            B[i, i + 1] = h[i + 1] / 6.0
            B[i + 1, i] = h[i + 1] / 6.0
    return D, B


def _cardinal_basis(
    x: NDArray[np.floating[Any]],
    xk: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Evaluate the cardinal natural-cubic-spline basis at *x*.

    ``X[i, j] = b_j(x_i)`` where ``b_j`` is the natural cubic spline through
    ``b_j(xk_l) = delta_jl``. Points outside ``[xk_0, xk_-1]`` follow the
    natural linear extension.
    """
    k = xk.shape[0]
    h = np.diff(xk)
    D, B = _band_matrices(xk)
    B_inv = np.linalg.inv(B)
    # F maps knot values beta -> second derivatives at ALL knots (natural BC:
    # zero curvature at the boundary knots).
    F = np.vstack([np.zeros(k), B_inv @ D, np.zeros(k)])  # (k, k)

    n = x.shape[0]
    X = np.zeros((n, k), dtype=np.float64)

    inside = (x >= xk[0]) & (x <= xk[-1])
    xi = x[inside]
    j = np.clip(np.searchsorted(xk, xi, side="right") - 1, 0, k - 2)
    hj = h[j]
    lo = xk[j]
    hi = xk[j + 1]
    a_minus = (hi - xi) / hj
    a_plus = (xi - lo) / hj
    c_minus = ((hi - xi) ** 3 / hj - hj * (hi - xi)) / 6.0
    c_plus = ((xi - lo) ** 3 / hj - hj * (xi - lo)) / 6.0

    rows = np.where(inside)[0]
    X[rows, j] += a_minus
    X[rows, j + 1] += a_plus
    X[rows, :] += c_minus[:, None] * F[j, :] + c_plus[:, None] * F[j + 1, :]

    # Natural linear extension beyond the boundary knots. First derivative
    # at a boundary knot of the cardinal splines, from the piecewise form:
    #   f'(xk_0)  = (beta_1 - beta_0)/h_0 - h_0/6 * (2*delta_0 + delta_1)
    # with delta = F beta and delta_0 = 0 (natural), similarly at the right.
    left = x < xk[0]
    if np.any(left):
        d_left = np.zeros(k)
        d_left[0] = -1.0 / h[0]
        d_left[1] = 1.0 / h[0]
        d_left -= (h[0] / 6.0) * F[1, :]  # 2*F[0]=0 (natural)
        b_left = np.zeros(k)
        b_left[0] = 1.0
        X[left, :] = b_left[None, :] + np.outer(x[left] - xk[0], d_left)
    right = x > xk[-1]
    if np.any(right):
        d_right = np.zeros(k)
        d_right[-2] = -1.0 / h[-1]
        d_right[-1] = 1.0 / h[-1]
        d_right += (h[-1] / 6.0) * F[-2, :]
        b_right = np.zeros(k)
        b_right[-1] = 1.0
        X[right, :] = b_right[None, :] + np.outer(x[right] - xk[-1], d_right)

    return X


def _penalty(xk: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Exact ∫ f''(t)² dt penalty in knot-value coordinates: ``D'B⁻¹D``."""
    D, B = _band_matrices(xk)
    B_inv = np.linalg.inv(B)
    S = D.T @ B_inv @ D
    return 0.5 * (S + S.T)  # symmetrise round-off
