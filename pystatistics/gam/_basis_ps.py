"""P-spline basis — mgcv-exact construction (Eilers & Marx 1996).

Implements the same evenly-spaced B-spline basis with a discrete difference
penalty that R's ``mgcv`` uses for ``s(x, bs="ps")`` with its default
``m = c(2, 2)``: a cubic (order ``m[1]+2``) B-spline basis on knots placed by
mgcv's ``smooth.construct.ps.smooth.spec`` rule, penalised by the ``m[2]``-th
order difference operator ``D'D`` and normalised by mgcv's ``S.scale``.

Verified against ``mgcv::smoothCon(s(x, bs="ps"), absorb.cons=FALSE)`` to ~1e-12
(basis matrix, penalty, and ``S.scale``).

Reference: Eilers, P.H.C. & Marx, B.D. (1996). Flexible smoothing with B-splines
and penalties. mgcv source: ``smooth.construct.ps.smooth.spec``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline

from pystatistics.gam._basis_common import validate_k, validate_x


def ps_basis(
    x: NDArray[np.floating[Any]],
    k: int = 10,
    degree: int = 3,
    pen_order: int = 2,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], float]:
    """Construct the mgcv-exact P-spline basis and difference penalty.

    Args:
        x: Predictor values, 1-D array of *n* observations (validated).
        k: Basis dimension, exactly as mgcv's ``s(x, bs="ps", k=k)``.
        degree: B-spline degree (mgcv's ``m[1]+1``; default 3 = cubic).
        pen_order: Order of the difference penalty (mgcv's ``m[2]``; default 2).

    Returns:
        ``(X, S, s_scale)``: the ``(n, k)`` B-spline basis matrix, the ``(k, k)``
        penalty divided by mgcv's ``S.scale``, and that ``s_scale`` factor.

    Raises:
        ValidationError: invalid inputs, or fewer than ``k`` unique ``x``.
    """
    x = validate_x(x)
    validate_k(k, n_unique=np.unique(x).shape[0])

    knots = _place_knots_ps(x, k, degree)
    X = BSpline.design_matrix(x, knots, degree, extrapolate=True).toarray()

    # m[2]-th order difference penalty: D is the pen_order-th difference of I_k.
    D = np.diff(np.eye(k), n=pen_order, axis=0)
    S_raw = D.T @ D

    # mgcv smoothCon scale.penalty: S.scale = ||S||_1 / ||X||_inf^2.
    ma_xx = float(np.max(np.abs(X).sum(axis=1)) ** 2)
    s_scale = float(np.max(np.abs(S_raw).sum(axis=0)) / ma_xx)
    S = S_raw / s_scale
    return X, 0.5 * (S + S.T), s_scale


def _place_knots_ps(
    x: NDArray[np.floating[Any]], k: int, degree: int
) -> NDArray[np.floating[Any]]:
    """mgcv ``ps`` knot placement: evenly spaced, with ``degree`` padding knots
    on each side of a slightly-expanded data range.

    Reproduces ``smooth.construct.ps.smooth.spec`` with ``m[1] = degree - 1``:
    ``nk = k - m[1]`` interior knots span ``[xl, xu]`` (the data range expanded by
    0.1%), and ``m[1]+1 = degree`` knots pad each end.
    """
    m1 = degree - 1
    xl, xu = float(np.min(x)), float(np.max(x))
    xr = xu - xl
    xl -= xr * 0.001
    xu += xr * 0.001
    nk = k - m1
    dx = (xu - xl) / (nk - 1)
    n_total = nk + 2 * (m1 + 1)
    return xl - dx * (m1 + 1) + dx * np.arange(n_total)
