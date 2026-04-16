"""
Factor rotation methods.

Provides varimax (orthogonal) and promax (oblique) rotation,
matching R's ``stats::varimax()`` and ``stats::promax()``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ConvergenceError


def varimax(
    loadings: NDArray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[NDArray, NDArray]:
    """Varimax (orthogonal) rotation of a loadings matrix.

    Maximises the varimax criterion using Kaiser normalisation
    (normalise rows before rotation, denormalise after).

    Matches R's ``stats::varimax()``.

    Args:
        loadings: Unrotated loadings matrix, shape (p, m).
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on the rotation criterion change.

    Returns:
        Tuple of (rotated_loadings, rotation_matrix) where
        rotated_loadings has shape (p, m) and rotation_matrix has
        shape (m, m) and is orthogonal.

    Raises:
        ConvergenceError: If the algorithm does not converge.
    """
    p, m = loadings.shape

    # Kaiser normalisation: normalise each row to unit length
    row_norms = np.sqrt(np.sum(loadings ** 2, axis=1))
    # Protect against zero-norm rows
    row_norms = np.where(row_norms == 0, 1.0, row_norms)
    normalized = loadings / row_norms[:, np.newaxis]

    rotation_matrix = np.eye(m)
    d_old = 0.0

    for iteration in range(max_iter):
        rotated = normalized @ rotation_matrix
        # Compute the varimax criterion gradient components
        # For each pair of factors, perform a planar rotation
        # R's varimax uses the SVD-based algorithm from Kaiser (1958)
        B = rotated ** 2
        col_means = np.mean(B, axis=0)
        u = B - col_means[np.newaxis, :]
        # SVD of normalized' @ (rotated^3 - rotated @ diag(colMeans(rotated^2)))
        sv_target = normalized.T @ (rotated ** 3 - rotated @ np.diag(col_means))
        U, S, Vt = np.linalg.svd(sv_target)
        rotation_matrix = U @ Vt
        d_new = np.sum(S)

        if abs(d_new - d_old) < tol:
            rotated_loadings = normalized @ rotation_matrix
            # Undo Kaiser normalisation
            rotated_loadings = rotated_loadings * row_norms[:, np.newaxis]
            return rotated_loadings, rotation_matrix

        d_old = d_new

    raise ConvergenceError(
        f"Varimax rotation did not converge after {max_iter} iterations",
        iterations=max_iter,
        final_change=abs(d_new - d_old),
        reason="max_iterations",
        threshold=tol,
    )


def promax(
    loadings: NDArray,
    *,
    m: int = 4,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[NDArray, NDArray]:
    """Promax (oblique) rotation of a loadings matrix.

    Algorithm:
        1. Start with varimax rotation.
        2. Raise absolute elements to the *m*-th power (preserving signs)
           to create a target matrix.
        3. Find the rotation that best approximates the target via
           least-squares.

    Matches R's ``stats::promax()``.

    Args:
        loadings: Unrotated loadings matrix, shape (p, k).
        m: Power parameter for the target. Default 4 (R's default).
        max_iter: Maximum iterations for the initial varimax step.
        tol: Convergence tolerance for the initial varimax step.

    Returns:
        Tuple of (rotated_loadings, rotation_matrix).

    Raises:
        ConvergenceError: If the varimax step does not converge.
    """
    # Step 1: varimax rotation
    varimax_loadings, _ = varimax(loadings, max_iter=max_iter, tol=tol)

    # Step 2: create target by raising to the m-th power (R's promax convention)
    target = np.sign(varimax_loadings) * np.abs(varimax_loadings) ** m

    # Step 3: least-squares rotation  L @ Q ≈ target
    # Q = (L' L)^{-1} L' target, then normalise columns
    LtL = varimax_loadings.T @ varimax_loadings
    LtT = varimax_loadings.T @ target
    Q = np.linalg.solve(LtL, LtT)

    # Normalise columns of Q so that diag(Q'Q) = 1
    col_norms = np.sqrt(np.sum(Q ** 2, axis=0))
    col_norms = np.where(col_norms == 0, 1.0, col_norms)
    Q = Q / col_norms[np.newaxis, :]

    rotated_loadings = varimax_loadings @ Q

    return rotated_loadings, Q
