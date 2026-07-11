"""Cyclic cubic regression spline basis — mgcv-exact (Wood 2017 §5.3.1).

Implements the periodic cubic regression spline R's ``mgcv`` uses for
``s(x, bs="cc")``: knots at type-7 quantiles of the unique covariate values
(as for ``cr``), a *cyclic* second-derivative system so the fitted function and
its first two derivatives match at the endpoints, and mgcv's ``scale.penalty``
normalisation. The cyclic identification ``f(t_1) = f(t_k)`` leaves ``k - 1``
basis columns.

Verified against ``mgcv::smoothCon(s(x, bs="cc"), absorb.cons=FALSE)`` to ~1e-9
(basis matrix and penalty).

Reference: Wood, S.N. (2017). GAM: An Introduction with R (2nd ed.), §5.3.1
(cyclic cubic regression splines). mgcv: ``smooth.construct.cc.smooth.spec``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.gam._basis_common import validate_k, validate_x
from pystatistics.gam._basis_cr import place_knots_cr


def _cyclic_BD(h: NDArray[np.floating[Any]]) -> tuple[NDArray, NDArray]:
    """Cyclic ``(m, m)`` matrices B and D of Wood §5.3.1 (m = k-1).

    ``h`` are the ``k-1`` inter-knot gaps (the last wraps to the first). ``B δ =
    D β`` relates knot values ``β`` to second derivatives ``δ`` under periodicity.
    """
    m = h.shape[0]
    B = np.zeros((m, m))
    D = np.zeros((m, m))
    for i in range(m):
        hm = h[i - 1]        # h_{i-1}, wraps for i == 0
        hi = h[i]
        B[i, i] = (hm + hi) / 3.0
        B[i, (i + 1) % m] += hi / 6.0
        B[i, (i - 1) % m] += hm / 6.0
        D[i, i] = -(1.0 / hm + 1.0 / hi)
        D[i, (i + 1) % m] += 1.0 / hi
        D[i, (i - 1) % m] += 1.0 / hm
    return B, D


def cc_basis(
    x: NDArray[np.floating[Any]],
    k: int = 10,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], float]:
    """Construct the mgcv-exact cyclic cubic regression spline basis and penalty.

    Args:
        x: Predictor values, 1-D array of *n* observations (validated). Values
            are wrapped into the knot range (periodic domain).
        k: Basis dimension — number of knots, as mgcv's ``s(x, bs="cc", k=k)``.
            The returned basis has ``k - 1`` columns (cyclic identification).

    Returns:
        ``(X, S, s_scale)``: the ``(n, k-1)`` basis matrix, the ``(k-1, k-1)``
        penalty divided by mgcv's ``S.scale``, and that ``s_scale`` factor.

    Raises:
        ValidationError: invalid inputs, or fewer than ``k`` unique ``x``.
    """
    x = validate_x(x)
    validate_k(k, n_unique=np.unique(x).shape[0])

    xk = place_knots_cr(x, k)            # same quantile knots as cr
    m = k - 1                            # free parameters (β_k ≡ β_1)
    period = xk[-1] - xk[0]
    h = np.diff(xk)                      # length k-1

    B, D = _cyclic_BD(h)
    F = np.linalg.solve(B, D)            # β -> second derivatives δ (m x m)

    # Wrap x into [t_1, t_k) and evaluate the piecewise-cubic Hermite form.
    xw = xk[0] + np.mod(x - xk[0], period)
    n = x.shape[0]
    X = np.zeros((n, m), dtype=np.float64)
    j = np.clip(np.searchsorted(xk, xw, side="right") - 1, 0, k - 2)
    hj = h[j]
    a_minus = (xk[j + 1] - xw) / hj
    a_plus = (xw - xk[j]) / hj
    c_minus = ((xk[j + 1] - xw) ** 3 / hj - hj * (xk[j + 1] - xw)) / 6.0
    c_plus = ((xw - xk[j]) ** 3 / hj - hj * (xw - xk[j])) / 6.0

    jl = j % m                           # left knot's β index (cyclic)
    jr = (j + 1) % m                     # right knot's β index (cyclic)
    rows = np.arange(n)
    X[rows, jl] += a_minus
    X[rows, jr] += a_plus
    X += c_minus[:, None] * F[jl, :] + c_plus[:, None] * F[jr, :]

    S_raw = D.T @ np.linalg.solve(B, D)  # D' B^{-1} D
    S_raw = 0.5 * (S_raw + S_raw.T)

    ma_xx = float(np.max(np.abs(X).sum(axis=1)) ** 2)
    s_scale = float(np.max(np.abs(S_raw).sum(axis=0)) / ma_xx)
    S = S_raw / s_scale
    return X, S, s_scale
