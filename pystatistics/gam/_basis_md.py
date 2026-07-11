"""Isotropic multivariate thin-plate regression spline — mgcv ``s(x, z, ...)``.

Generalises the 1-D thin plate spline (:mod:`_basis_tp`) to ``d >= 2``
covariates that SHARE a scale (an isotropic smooth, penalising wiggliness
equally in every direction — the alternative to ``te()`` when the covariates
are on the same footing, e.g. spatial coordinates). Same construction as the
1-D case:

* thin-plate Green's function of the Euclidean knot distances for order
  ``m = 2`` in ``d`` dimensions (``r^(2m-d)`` when ``2m-d`` is odd,
  ``r^(2m-d) log r`` when even);
* the polynomial null space of total degree ``< m`` — for ``m = 2`` the
  ``d + 1`` functions ``{1, x_1, ..., x_d}`` — kept as unpenalised columns
  with the ``T' delta = 0`` side condition absorbed;
* eigen-truncation to ``k`` basis functions and column normalisation to
  ``sqrt(n)``.

The overall constant of the Green's function cancels out of both the
sqrt(n)-normalised basis and the ``S.scale``-normalised penalty, so the fit
is function-space identical to mgcv's ``s(x, z, bs="tp")`` (the eigenbasis
coordinate system differs by an orthogonal reparameterisation, exactly as in
the 1-D thin-plate case — compare via EDF/fitted values, not raw
coefficients).

Reference: Wood, S.N. (2003). Thin plate regression splines. JRSS-B 65(1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr

from pystatistics.core.exceptions import ValidationError

_MAX_KNOTS_DEFAULT = 2000
_M = 2  # penalty order (mgcv default for tprs)


def _green(r: NDArray[np.floating[Any]], d: int) -> NDArray[np.floating[Any]]:
    """Thin-plate Green's function of Euclidean distance ``r`` (order m=2)."""
    power = 2 * _M - d
    if power <= 0:
        raise ValidationError(
            f"thin-plate smooth of {d} variables needs order m > d/2; "
            f"m=2 supports up to 3 covariates (got d={d})"
        )
    if power % 2 == 0:
        # r^power * log r, with the r=0 limit (0) handled explicitly.
        out = np.zeros_like(r)
        nz = r > 0
        out[nz] = r[nz] ** power * np.log(r[nz])
        return out
    return r ** power


def md_tp_basis(
    coords: NDArray[np.floating[Any]],
    k: int = 10,
    max_knots: int = _MAX_KNOTS_DEFAULT,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], float]:
    """Isotropic multivariate thin-plate basis and penalty.

    Args:
        coords: ``(n, d)`` covariate matrix (``d >= 2`` columns).
        k: Basis dimension INCLUDING the ``d + 1`` polynomial null space,
            exactly as mgcv's ``s(..., bs="tp", k=k)``.
        max_knots: Eigen-decomposition size cap (mgcv ``max.knots``).

    Returns:
        ``(X, S, s_scale)``: the ``(n, k)`` basis (penalised columns first,
        then ``[1, x_1 - mean, ...]``), the ``(k, k)`` penalty (zero on the
        null-space block), and the ``S.scale`` factor.

    Raises:
        ValidationError: on invalid input or ``k`` too large for the data.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValidationError(
            "md_tp_basis needs an (n, d) matrix with d >= 2 columns"
        )
    if not np.all(np.isfinite(coords)):
        raise ValidationError("coords contains non-finite values")
    n, d = coords.shape
    null_dim = d + 1  # {1, x_1, ..., x_d} for m = 2

    shift = coords.mean(axis=0)
    xc = coords - shift
    knots = np.unique(xc, axis=0)
    if knots.shape[0] < k:
        raise ValidationError(
            f"k={k} exceeds the {knots.shape[0]} unique covariate points"
        )
    if knots.shape[0] > max_knots:
        idx = np.linspace(0, knots.shape[0] - 1, max_knots).round().astype(int)
        knots = knots[np.unique(idx)]

    m = knots.shape[0]
    # TPS kernel on the knots.
    dist = np.linalg.norm(knots[:, None, :] - knots[None, :, :], axis=2)
    E = _green(dist, d)
    evals, evecs = np.linalg.eigh(E)
    order = np.argsort(np.abs(evals))[::-1][:k]
    dvals = evals[order]
    U = evecs[:, order]  # (m, k)

    # Absorb the side condition T' delta = 0 with T = [1, knots].
    T_knots = np.column_stack([np.ones(m), knots])
    Q_full, _ = qr(U.T @ T_knots, mode="full")
    Z = Q_full[:, null_dim:]  # (k, k - null_dim)

    # Penalised design at the data points.
    dist_data = np.linalg.norm(
        xc[:, None, :] - knots[None, :, :], axis=2,
    )  # (n, m)
    E_data = _green(dist_data, d)
    X_pen = E_data @ (U @ Z)                    # (n, k - null_dim)
    S_pen = Z.T @ (dvals[:, None] * Z)          # (k - null_dim, k - null_dim)
    S_pen = 0.5 * (S_pen + S_pen.T)

    X = np.hstack([X_pen, np.ones((n, 1)), xc])  # null space [1, x_1..x_d]
    col_norm = np.linalg.norm(X, axis=0) / np.sqrt(n)
    X = X / col_norm
    c_pen = col_norm[: k - null_dim]
    S_pen = S_pen / np.outer(c_pen, c_pen)

    S = np.zeros((k, k), dtype=np.float64)
    S[: k - null_dim, : k - null_dim] = S_pen

    ma_xx = float(np.max(np.abs(X).sum(axis=1)) ** 2)
    s_scale = float(np.max(np.abs(S).sum(axis=0)) / ma_xx)
    S = S / s_scale
    return X, S, s_scale
