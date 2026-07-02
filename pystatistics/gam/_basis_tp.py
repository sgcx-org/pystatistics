"""Thin plate regression spline basis — Wood (2003), mgcv-equivalent.

Implements the 1-D thin plate regression spline the way mgcv's
``s(x, bs="tp")`` does: eigen-truncation of the full TPS kernel
``E_ij = |x_i - x_j|^3 / 12`` (the d=1, m=2 Green's function, constant
included), the ``T'delta = 0`` side condition absorbed into the penalized
block, the polynomial null space ``{1, x - mean(x)}`` retained as unpenalized
trailing columns, and every column normalised to ``||col|| = sqrt(n)``.

The resulting (basis, penalty) pair is *function-space identical* to mgcv's
(verified: hat-matrix agreement ~4e-17 at fixed lambda; subspace angle
~2e-14). The coordinate system differs from mgcv's by an orthogonal
reparameterisation (LAPACK eigenvector conventions), so raw coefficients are
not comparable across engines — fits, EDFs and smoothing parameters (in
function space) are.

Knots: all unique covariate values, capped at ``max_knots`` (mgcv default
2000). Above the cap mgcv subsamples with its internal RNG; we take an
evenly-spaced subsample of the sorted unique values instead — deterministic
and documented, but a (bounded) divergence from mgcv above the cap.

Reference: Wood, S.N. (2003). Thin plate regression splines. JRSS-B 65(1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr

from pystatistics.gam._basis_common import validate_k, validate_x

_MAX_KNOTS_DEFAULT = 2000
_NULL_DIM = 2  # d=1, m=2: {1, x}


def tp_basis(
    x: NDArray[np.floating[Any]],
    k: int = 10,
    max_knots: int = _MAX_KNOTS_DEFAULT,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], float]:
    """Construct the thin plate regression spline basis and penalty.

    Args:
        x: Predictor values, 1-D array of *n* observations.
        k: Basis dimension INCLUDING the 2-dim polynomial null space,
            exactly as mgcv's ``s(x, bs="tp", k=k)``.
        max_knots: Eigen-decomposition size cap (mgcv ``max.knots``).

    Returns:
        ``(X, S, s_scale)``: the ``(n, k)`` basis (penalized columns first,
        then ``[1, x - mean]``, all normalised to ``sqrt(n)``), the ``(k, k)``
        penalty (zero on the null-space block) divided by mgcv's ``S.scale``
        rule, and the ``s_scale`` factor.

    Raises:
        ValidationError: on invalid input, or k > number of unique x values.
    """
    x = validate_x(x)
    xu = np.unique(x)
    validate_k(k, n_unique=xu.shape[0])

    shift = float(np.mean(x))
    xc = x - shift
    knots = xu - shift
    if knots.shape[0] > max_knots:
        # Deterministic even-index subsample of the sorted unique values.
        # (mgcv subsamples with its own RNG here; documented divergence.)
        idx = np.linspace(0, knots.shape[0] - 1, max_knots).round().astype(int)
        knots = knots[np.unique(idx)]

    m = knots.shape[0]
    n = x.shape[0]

    # TPS kernel on knots (d=1, m=2 Green's function, with its 1/12 constant
    # -- the constant fixes the lambda scale; verified against mgcv).
    E = np.abs(knots[:, None] - knots[None, :]) ** 3 / 12.0
    evals, evecs = np.linalg.eigh(E)
    order = np.argsort(np.abs(evals))[::-1][:k]
    d = evals[order]
    U = evecs[:, order]  # (m, k)

    # Absorb the natural-TPS side condition T'delta = 0 (T = [1, x] at knots):
    # delta = U delta_k, constraint (U'T)' delta_k = 0 -> delta_k = Z nu.
    T_knots = np.column_stack([np.ones(m), knots])
    Q_full, _ = qr(U.T @ T_knots, mode="full")  # (k, k)
    Z = Q_full[:, _NULL_DIM:]                    # (k, k-2)

    # Penalized design at the DATA points: E(x, knots) U Z; when knots == all
    # unique x and x has no duplicates this equals U diag(d) Z.
    E_data = np.abs(xc[:, None] - knots[None, :]) ** 3 / 12.0  # (n, m)
    X_pen = E_data @ (U @ Z)                                    # (n, k-2)
    S_pen = Z.T @ (d[:, None] * Z)                              # (k-2, k-2)
    S_pen = 0.5 * (S_pen + S_pen.T)

    # Null space columns at the data, then normalise everything to sqrt(n).
    X = np.hstack([X_pen, np.ones((n, 1)), xc[:, None]])
    col_norm = np.linalg.norm(X, axis=0) / np.sqrt(n)
    X = X / col_norm
    c_pen = col_norm[: k - _NULL_DIM]
    S_pen = S_pen / np.outer(c_pen, c_pen)

    S = np.zeros((k, k), dtype=np.float64)
    S[: k - _NULL_DIM, : k - _NULL_DIM] = S_pen

    # mgcv smoothCon scale.penalty rule (same as cr).
    ma_xx = float(np.max(np.abs(X).sum(axis=1)) ** 2)
    s_scale = float(np.max(np.abs(S).sum(axis=0)) / ma_xx)
    S = S / s_scale

    return X, S, s_scale
