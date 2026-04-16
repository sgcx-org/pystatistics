"""
Basis construction and penalty matrices for GAM smooth terms.

Provides cubic regression spline (cr) and thin plate regression spline (tp)
basis matrices with their associated second-derivative penalty matrices.

Algorithm references:
    - Wood, S.N. (2003). Thin plate regression splines.
      Journal of the Royal Statistical Society B, 65(1), 95-114.
    - Wood, S.N. (2017). Generalized Additive Models: An Introduction
      with R, 2nd ed.  CRC Press.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate, integrate

from pystatistics.core.exceptions import ValidationError

_MIN_K = 3
_MIN_UNIQUE_X = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cubic_regression_spline_basis(
    x: NDArray[np.floating[Any]],
    k: int = 10,
    knots: NDArray[np.floating[Any]] | None = None,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Construct a cubic regression spline basis matrix and penalty matrix.

    Cubic regression splines (cr) are the default basis in R's
    ``mgcv::gam()``.  They use a B-spline basis with knots placed at
    quantiles of the predictor variable.

    Algorithm:
        1. Place *k* knots at evenly-spaced quantiles of *x*.
        2. Build a cubic B-spline basis matrix **B** (n x k) via
           :func:`scipy.interpolate.BSpline`.
        3. Compute the second-derivative penalty **S** (k x k) via
           numerical integration:
           ``S_ij = integral B_i''(t) B_j''(t) dt``.

    Args:
        x: Predictor values, 1-D array of *n* observations.
        k: Number of basis functions (default 10, matching mgcv).
        knots: Optional interior knot locations.  If ``None`` they are
            placed at quantiles of *x*.

    Returns:
        ``(B, S)`` where **B** is (n, k) and **S** is (k, k).

    Raises:
        ValidationError: If inputs fail validation.
    """
    x = _validate_x(x, min_unique=_MIN_UNIQUE_X)
    _validate_k(k, n=x.shape[0])

    if knots is not None:
        knots = np.asarray(knots, dtype=np.float64).ravel()
        if knots.shape[0] < 1:
            raise ValidationError("knots array must be non-empty")
    else:
        knots = _place_knots(x, k)

    basis = _bspline_basis(x, knots, degree=3)
    penalty = _penalty_matrix_numerical(knots, degree=3)

    # Ensure dimensions match -- trim or pad if the B-spline
    # evaluation returned a slightly different column count.
    n_cols = basis.shape[1]
    if penalty.shape[0] != n_cols:
        # Recompute penalty at the correct size by using only the
        # first n_cols basis functions.
        penalty = penalty[:n_cols, :n_cols]

    return basis, penalty


def thin_plate_spline_basis(
    x: NDArray[np.floating[Any]],
    k: int = 10,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Construct a thin plate regression spline basis and penalty.

    Uses the truncated eigendecomposition approach of Wood (2003):
        1. Build the full TPS kernel matrix **E** (n x n) with
           ``E_ij = |x_i - x_j|^3`` (1-D thin plate kernel).
        2. Absorb the polynomial null space (constant + linear) to
           form the penalised part.
        3. Eigendecompose and retain the *k* eigenvectors with the
           largest eigenvalues as the basis.
        4. The penalty in this reduced basis is
           ``diag(selected eigenvalues)``.

    Args:
        x: Predictor values, 1-D array of *n* observations.
        k: Number of basis functions (default 10).

    Returns:
        ``(B, S)`` where **B** is (n, k) and **S** is (k, k).

    Raises:
        ValidationError: If inputs fail validation.
    """
    x = _validate_x(x, min_unique=_MIN_UNIQUE_X)
    _validate_k(k, n=x.shape[0])

    n = x.shape[0]

    # -- 1. Full TPS kernel matrix (1-D: E_ij = |x_i - x_j|^3) ----------
    diff = x[:, np.newaxis] - x[np.newaxis, :]       # (n, n)
    E = np.abs(diff) ** 3                              # (n, n)

    # -- 2. Polynomial null space (constant + linear) --------------------
    T = np.column_stack([np.ones(n), x])               # (n, 2)

    # QR of T to build projector away from null space
    Q, _ = np.linalg.qr(T, mode="reduced")             # Q: (n, 2)
    projector = np.eye(n) - Q @ Q.T                     # (n, n)

    # Penalised kernel in the complement of the null space
    E_proj = projector @ E @ projector                  # (n, n)

    # Symmetrise (numerical noise)
    E_proj = 0.5 * (E_proj + E_proj.T)

    # -- 3. Eigendecomposition -------------------------------------------
    eigenvalues, eigenvectors = np.linalg.eigh(E_proj)

    # eigh returns ascending order -- take the *k* largest
    idx = np.argsort(eigenvalues)[::-1][:k]
    selected_vals = eigenvalues[idx]
    selected_vecs = eigenvectors[:, idx]                # (n, k)

    # Replace any tiny / negative eigenvalues with a small positive value
    # (they correspond to the null space and numerical noise).
    floor = max(np.abs(selected_vals).max() * 1e-12, 1e-15)
    selected_vals = np.maximum(selected_vals, floor)

    # -- 4. Basis and penalty --------------------------------------------
    basis = selected_vecs                                # (n, k)
    penalty = np.diag(selected_vals)                     # (k, k)

    # Normalise columns so max(|col|) ~ 1
    col_scales = np.abs(basis).max(axis=0)
    col_scales = np.where(col_scales > 0, col_scales, 1.0)
    basis = basis / col_scales
    penalty = np.diag(selected_vals / (col_scales ** 2))

    return basis, penalty


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _place_knots(x: NDArray[np.floating[Any]], k: int) -> NDArray[np.floating[Any]]:
    """Place *k* interior knots at evenly-spaced quantiles of *x*.

    Boundary knots sit at ``min(x)`` and ``max(x)``.  The remaining
    ``k - 2`` interior knots are at quantiles
    ``1/(k-1), 2/(k-1), ..., (k-2)/(k-1)``.

    Args:
        x: 1-D predictor values (already validated).
        k: Desired number of interior knots.

    Returns:
        Sorted 1-D array of *k* knot positions.
    """
    n_interior = k - 2
    if n_interior <= 0:
        return np.array([x.min(), x.max()], dtype=np.float64)

    probs = np.linspace(0.0, 1.0, n_interior + 2)
    knots = np.quantile(x, probs)
    return np.sort(np.unique(knots)).astype(np.float64)


def _bspline_basis(
    x: NDArray[np.floating[Any]],
    knots: NDArray[np.floating[Any]],
    degree: int = 3,
) -> NDArray[np.floating[Any]]:
    """Evaluate B-spline basis functions at *x*.

    Constructs the full knot vector with clamped (repeated) boundary
    knots and evaluates each basis function column-by-column using
    :class:`scipy.interpolate.BSpline`.

    Args:
        x: Evaluation points (1-D).
        knots: Interior knot positions (sorted, 1-D).
        degree: Spline degree (3 for cubic).

    Returns:
        **B**: (n, n_basis) basis matrix where
        ``n_basis = len(knots) + degree - 1``.
    """
    knots = np.sort(knots)

    # Clamp: replicate boundary knots (degree + 1) times
    t_left = np.repeat(knots[0], degree)
    t_right = np.repeat(knots[-1], degree)
    t = np.concatenate([t_left, knots, t_right])

    n_basis = len(t) - degree - 1
    n = x.shape[0]

    basis = np.zeros((n, n_basis), dtype=np.float64)
    for i in range(n_basis):
        coeffs = np.zeros(n_basis, dtype=np.float64)
        coeffs[i] = 1.0
        spl = interpolate.BSpline(t, coeffs, degree, extrapolate=False)
        col = spl(x)
        # Replace NaN at boundaries with 0
        col = np.where(np.isfinite(col), col, 0.0)
        basis[:, i] = col

    return basis


def _penalty_matrix_numerical(
    knots: NDArray[np.floating[Any]],
    degree: int = 3,
) -> NDArray[np.floating[Any]]:
    """Compute the penalty matrix via numerical integration.

    ``S_ij = integral_{a}^{b} B_i''(t) B_j''(t) dt``

    Uses a fine grid of evaluation points and the trapezoidal rule
    for speed (compared to per-element ``scipy.integrate.quad``).

    Args:
        knots: Interior knot positions (sorted).
        degree: Spline degree.

    Returns:
        **S**: (n_basis, n_basis) symmetric positive semi-definite
        penalty matrix.
    """
    knots = np.sort(knots)

    # Full clamped knot vector
    t_left = np.repeat(knots[0], degree)
    t_right = np.repeat(knots[-1], degree)
    t = np.concatenate([t_left, knots, t_right])

    n_basis = len(t) - degree - 1

    # Fine evaluation grid
    a, b = float(knots[0]), float(knots[-1])
    n_grid = max(500, 50 * len(knots))
    grid = np.linspace(a, b, n_grid)

    # Evaluate second derivatives of each basis function on the grid
    deriv2 = np.zeros((n_grid, n_basis), dtype=np.float64)
    for i in range(n_basis):
        coeffs = np.zeros(n_basis, dtype=np.float64)
        coeffs[i] = 1.0
        spl = interpolate.BSpline(t, coeffs, degree, extrapolate=False)
        spl_d2 = spl.derivative(2)
        col = spl_d2(grid)
        col = np.where(np.isfinite(col), col, 0.0)
        deriv2[:, i] = col

    # Trapezoidal integration: S_ij = integral B_i'' B_j'' dt
    # ~ sum of (B_i'' * B_j'') * weights
    dx = np.diff(grid)
    weights = np.zeros(n_grid, dtype=np.float64)
    weights[0] = dx[0] / 2.0
    weights[-1] = dx[-1] / 2.0
    weights[1:-1] = (dx[:-1] + dx[1:]) / 2.0

    # S = D2^T @ diag(w) @ D2
    W_half = np.sqrt(weights)
    D2w = deriv2 * W_half[:, np.newaxis]
    S = D2w.T @ D2w

    # Symmetrise
    S = 0.5 * (S + S.T)

    return S


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_x(
    x: NDArray[np.floating[Any]],
    min_unique: int = _MIN_UNIQUE_X,
) -> NDArray[np.floating[Any]]:
    """Validate and flatten a 1-D predictor array.

    Args:
        x: Input array.
        min_unique: Minimum number of unique values required.

    Returns:
        Flattened float64 array.

    Raises:
        ValidationError: On any validation failure.
    """
    x = np.asarray(x, dtype=np.float64).ravel()

    if x.ndim != 1 or x.shape[0] == 0:
        raise ValidationError(
            f"x must be a non-empty 1-D array, got shape {x.shape}"
        )

    if np.any(~np.isfinite(x)):
        raise ValidationError("x contains non-finite values (NaN or Inf)")

    n_unique = len(np.unique(x))
    if n_unique < min_unique:
        raise ValidationError(
            f"x must have at least {min_unique} unique values, "
            f"got {n_unique}"
        )

    return x


def _validate_k(k: int, n: int) -> None:
    """Validate the number of basis functions.

    Args:
        k: Requested basis dimension.
        n: Number of observations.

    Raises:
        ValidationError: If k is invalid.
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValidationError(f"k must be an integer, got {type(k).__name__}")

    if k < _MIN_K:
        raise ValidationError(
            f"k must be >= {_MIN_K}, got k={k}"
        )

    if k > n:
        raise ValidationError(
            f"k={k} exceeds the number of observations n={n}"
        )
