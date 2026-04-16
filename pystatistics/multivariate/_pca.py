"""
Principal Component Analysis via SVD.

Matches R's ``stats::prcomp()``, validated against R output.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.validation import check_2d, check_finite, check_array
from pystatistics.multivariate._common import PCAResult


def _fix_sign_convention(rotation: NDArray) -> NDArray:
    """Enforce R's sign convention for loadings.

    For each column (component) of the rotation matrix, ensure that the
    element with the largest absolute value is positive. This removes the
    sign ambiguity inherent in SVD and matches R's ``prcomp()`` convention.

    Args:
        rotation: Loadings matrix (p x k).

    Returns:
        Sign-corrected loadings matrix of same shape.
    """
    signs = np.sign(rotation[np.argmax(np.abs(rotation), axis=0), np.arange(rotation.shape[1])])
    # Replace any zeros with 1 (shouldn't happen with real data)
    signs = np.where(signs == 0, 1.0, signs)
    return rotation * signs[np.newaxis, :]


def pca(
    X: ArrayLike,
    *,
    center: bool = True,
    scale: bool = False,
    n_components: int | None = None,
    names: list[str] | None = None,
) -> PCAResult:
    """Principal Component Analysis via SVD.

    Matches R's ``stats::prcomp()``.

    Algorithm:
        1. Center *X* (subtract column means).
        2. Optionally scale (divide by column standard deviations).
        3. Thin SVD: X_centered = U @ diag(S) @ V'.
        4. sdev = S / sqrt(n - 1).
        5. rotation = V (right singular vectors = loadings).
        6. x = U @ diag(S) (scores = X_centered @ V).

    Args:
        X: Data matrix (n x p), n observations, p variables.
        center: Whether to center columns (subtract mean). Default True.
        scale: Whether to scale columns (divide by SD). Default False.
            Equivalent to R's ``prcomp(scale. = TRUE)``.
        n_components: Number of components to retain. Default: min(n, p).
        names: Variable names for the p columns.

    Returns:
        PCAResult with sdev, rotation (loadings), scores, etc.

    Raises:
        ValidationError: If inputs are invalid.

    Validates against: R ``stats::prcomp()``.
    """
    # ---- Input validation ----
    X_arr = check_array(X, "X")
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    check_2d(X_arr, "X")
    check_finite(X_arr, "X")

    n, p = X_arr.shape

    if n < 2:
        raise ValidationError("X: requires at least 2 observations, got 1")

    if names is not None:
        if len(names) != p:
            raise ValidationError(
                f"names: length {len(names)} does not match number of columns {p}"
            )
        var_names: tuple[str, ...] | None = tuple(names)
    else:
        var_names = None

    max_components = min(n, p)

    if n_components is not None:
        if n_components < 1:
            raise ValidationError(
                f"n_components: must be >= 1, got {n_components}"
            )
        if n_components > max_components:
            raise ValidationError(
                f"n_components: {n_components} exceeds max possible "
                f"min(n, p) = {max_components}"
            )
    else:
        n_components = max_components

    # ---- Centering ----
    if center:
        col_means = np.mean(X_arr, axis=0)
        X_centered = X_arr - col_means
    else:
        col_means = np.zeros(p)
        X_centered = X_arr.copy()

    # ---- Scaling ----
    if scale:
        col_sds = np.std(X_centered, axis=0, ddof=1)
        zero_sd = col_sds == 0
        if np.any(zero_sd):
            zero_cols = np.where(zero_sd)[0].tolist()
            raise ValidationError(
                f"X: columns {zero_cols} have zero variance; "
                f"cannot scale constant columns"
            )
        X_centered = X_centered / col_sds
        scale_values: NDArray | None = col_sds
    else:
        scale_values = None

    # ---- SVD ----
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # sdev = singular values / sqrt(n - 1)
    sdev = S / np.sqrt(n - 1)

    # rotation = V (right singular vectors, transposed from Vt)
    rotation = Vt.T  # shape (p, min(n, p))

    # Apply sign convention
    rotation = _fix_sign_convention(rotation)

    # scores = X_centered @ V = U @ diag(S)
    # Recompute scores using the sign-fixed rotation for consistency
    scores = X_centered @ rotation

    # ---- Truncate to n_components ----
    sdev = sdev[:n_components]
    rotation = rotation[:, :n_components]
    scores = scores[:, :n_components]

    return PCAResult(
        sdev=sdev,
        rotation=rotation,
        center=col_means,
        scale=scale_values,
        x=scores,
        n_obs=n,
        n_vars=p,
        var_names=var_names,
    )
