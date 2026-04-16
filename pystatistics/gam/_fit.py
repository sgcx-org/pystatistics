"""
Core GAM fitting engine: Penalized Iteratively Re-weighted Least Squares.

Provides the building blocks for fitting a Generalized Additive Model:

1. Model matrix construction (augmented design + block-diagonal penalty).
2. A single P-IRLS step.
3. Full P-IRLS iteration loop with fixed smoothing parameters.
4. Effective-degrees-of-freedom computation per smooth term.

References:
    Wood, S. N. (2017). Generalized Additive Models (2nd ed.), CRC Press.
    Wood, S. N. (2004). Stable and efficient multiple smoothing parameter
        estimation for generalized additive models. JASA 99(467), 673--686.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ConvergenceError, ValidationError

if TYPE_CHECKING:
    from pystatistics.regression.families import Family
    from pystatistics.gam._smooth import SmoothTerm


# ------------------------------------------------------------------
# Basis dispatch
# ------------------------------------------------------------------

def _compute_basis(
    x: NDArray,
    smooth: SmoothTerm,
) -> tuple[NDArray, NDArray]:
    """Compute the basis matrix and penalty for a single smooth term.

    Dispatches to the appropriate basis constructor based on
    ``smooth.bs`` and populates the ``SmoothTerm`` object.

    Args:
        x: Predictor values, shape ``(n,)``.
        smooth: The smooth term specification.

    Returns:
        ``(B, S)`` where ``B`` is ``(n, k)`` and ``S`` is ``(k, k)``.

    Raises:
        ValidationError: If the basis type is not recognised.
    """
    from pystatistics.gam._basis import (
        cubic_regression_spline_basis,
        thin_plate_spline_basis,
    )

    if smooth.bs == "cr":
        B, S = cubic_regression_spline_basis(x, k=smooth.k)
    elif smooth.bs == "tp":
        B, S = thin_plate_spline_basis(x, k=smooth.k)
    else:
        raise ValidationError(
            f"Unknown basis type {smooth.bs!r}; expected 'cr' or 'tp'"
        )
    smooth.basis_matrix = B
    smooth.penalty_matrix = S
    return B, S


# ------------------------------------------------------------------
# Model-matrix construction
# ------------------------------------------------------------------

def _build_model_matrix(
    X_parametric: NDArray | None,
    smooth_data: dict[str, NDArray],
    smooth_terms: list[SmoothTerm],
) -> tuple[NDArray, list[NDArray], list[tuple[int, int]]]:
    """Build the augmented design matrix and per-smooth penalty list.

    The augmented matrix is::

        X_aug = [X_parametric | B_1 | B_2 | ... | B_m]

    Each ``S_j`` in the returned list is padded to ``(p_total, p_total)``
    with zeros for columns that do not belong to term *j*.

    Args:
        X_parametric: ``(n, p_lin)`` parametric design matrix, or ``None``
            (intercept-only is handled by the caller).
        smooth_data: Mapping from variable names to ``(n,)`` arrays.
        smooth_terms: List of :class:`SmoothTerm` specifications.

    Returns:
        X_aug: ``(n, p_total)`` augmented design matrix.
        S_penalties: List of ``(p_total, p_total)`` penalty matrices
            (one per smooth, with lambda = 1).
        term_indices: ``(start, end)`` column slices for each smooth.

    Raises:
        ValidationError: If a required smooth variable is missing from
            *smooth_data*.
    """
    n = None
    blocks: list[NDArray] = []

    # Parametric columns first
    if X_parametric is not None:
        n = X_parametric.shape[0]
        blocks.append(np.asarray(X_parametric, dtype=np.float64))

    # Smooth basis blocks
    basis_list: list[NDArray] = []
    penalty_raw: list[NDArray] = []
    for st in smooth_terms:
        if st.var_name not in smooth_data:
            raise ValidationError(
                f"smooth_data is missing variable {st.var_name!r}"
            )
        x = np.asarray(smooth_data[st.var_name], dtype=np.float64).ravel()
        if n is None:
            n = x.shape[0]
        elif x.shape[0] != n:
            raise ValidationError(
                f"Length mismatch: expected {n} observations, "
                f"got {x.shape[0]} for variable {st.var_name!r}"
            )
        B, S = _compute_basis(x, st)
        basis_list.append(B)
        penalty_raw.append(S)
        blocks.append(B)

    if n is None:
        raise ValidationError("Cannot determine n_obs: no data provided")

    X_aug = np.hstack(blocks)
    p_total = X_aug.shape[1]
    p_lin = 0 if X_parametric is None else X_parametric.shape[1]

    # Build padded penalty matrices and track column ranges
    S_penalties: list[NDArray] = []
    term_indices: list[tuple[int, int]] = []
    col = p_lin
    for S_raw in penalty_raw:
        k = S_raw.shape[0]
        S_full = np.zeros((p_total, p_total), dtype=np.float64)
        S_full[col : col + k, col : col + k] = S_raw
        S_penalties.append(S_full)
        term_indices.append((col, col + k))
        col += k

    return X_aug, S_penalties, term_indices


# ------------------------------------------------------------------
# Single P-IRLS step
# ------------------------------------------------------------------

def _pirls_iteration(
    y: NDArray,
    X_aug: NDArray,
    S_penalties: list[NDArray],
    lambdas: NDArray,
    family: Family,
    mu: NDArray,
    parametric_cols: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Execute one Penalized IRLS step.

    Solves the penalised weighted least-squares problem::

        beta = (X'WX + sum_j lambda_j S_j)^{-1} X'Wz

    where ``W`` = diag of IRLS weights and ``z`` = working response.

    Args:
        y: Response vector ``(n,)``.
        X_aug: Augmented design matrix ``(n, p)``.
        S_penalties: Padded penalty matrices (one per smooth).
        lambdas: Smoothing parameters ``(n_smooths,)``.
        family: GLM family instance.
        mu: Current mean estimates ``(n,)``.
        parametric_cols: Number of leading parametric columns.

    Returns:
        ``(new_beta, new_mu, new_eta)`` after one update.
    """
    eta = family.link.link(mu)
    dmu_deta = family.link.mu_eta(eta)
    var_mu = family.variance(mu)

    # IRLS weights: (dmu/deta)^2 / V(mu)
    w = np.maximum(dmu_deta ** 2 / np.maximum(var_mu, 1e-20), 1e-20)
    W = np.sqrt(w)  # for numerical stability, use sqrt(W) approach

    # Working response
    z = eta + (y - mu) / np.maximum(dmu_deta, 1e-20)

    # Penalised normal equations: (X'WX + sum lam*S) beta = X'Wz
    XtW = X_aug.T * w[np.newaxis, :]          # (p, n)
    XtWX = XtW @ X_aug                         # (p, p)

    # Add penalties
    penalty = np.zeros_like(XtWX)
    for lam, S in zip(lambdas, S_penalties):
        penalty += lam * S

    A = XtWX + penalty
    b = XtW @ z

    # Solve via Cholesky for speed and numerical stability
    try:
        L = np.linalg.cholesky(A)
        beta = np.linalg.solve(L.T, np.linalg.solve(L, b))
    except np.linalg.LinAlgError:
        # Fall back to regularised solve
        A += 1e-8 * np.eye(A.shape[0])
        beta = np.linalg.solve(A, b)

    new_eta = X_aug @ beta
    new_mu = family.link.linkinv(new_eta)
    return beta, new_mu, new_eta


# ------------------------------------------------------------------
# Full P-IRLS loop
# ------------------------------------------------------------------

def _fit_gam_fixed_lambda(
    y: NDArray,
    X_aug: NDArray,
    S_penalties: list[NDArray],
    lambdas: NDArray,
    family: Family,
    parametric_cols: int,
    tol: float,
    max_iter: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray, float, int, bool]:
    """Fit a GAM with fixed smoothing parameters via P-IRLS.

    Iterates the penalised IRLS algorithm until the relative change in
    deviance falls below *tol* or *max_iter* iterations are reached.

    Args:
        y: Response ``(n,)``.
        X_aug: Augmented design ``(n, p)``.
        S_penalties: Padded penalties (one per smooth).
        lambdas: Smoothing parameters ``(n_smooths,)``.
        family: GLM family.
        parametric_cols: Number of parametric columns.
        tol: Convergence tolerance on relative deviance change.
        max_iter: Maximum iterations.

    Returns:
        ``(beta, mu, eta, W_final, deviance, n_iter, converged)``
        where ``W_final`` is the diagonal weight vector from the last
        iteration.
    """
    n = y.shape[0]
    wt = np.ones(n, dtype=np.float64)

    # Initialise mu
    mu = family.initialize(y)
    mu = np.asarray(mu, dtype=np.float64)

    dev_old = family.deviance(y, mu, wt)
    converged = False
    beta = np.zeros(X_aug.shape[1], dtype=np.float64)

    for it in range(1, max_iter + 1):
        beta, mu, eta = _pirls_iteration(
            y, X_aug, S_penalties, lambdas, family, mu, parametric_cols,
        )
        dev_new = family.deviance(y, mu, wt)

        # Convergence check
        if dev_old > 0:
            rel_change = abs(dev_new - dev_old) / (abs(dev_old) + 1e-20)
        else:
            rel_change = abs(dev_new - dev_old)

        if rel_change < tol and it > 1:
            converged = True
            dev_old = dev_new
            break
        dev_old = dev_new

    # Final IRLS weights for EDF computation
    eta_final = family.link.link(mu)
    dmu_deta = family.link.mu_eta(eta_final)
    var_mu = family.variance(mu)
    W_final = np.maximum(dmu_deta ** 2 / np.maximum(var_mu, 1e-20), 1e-20)

    return beta, mu, eta, W_final, dev_old, it, converged


# ------------------------------------------------------------------
# Effective degrees of freedom
# ------------------------------------------------------------------

def _compute_edf(
    X_aug: NDArray,
    W: NDArray,
    S_penalties: list[NDArray],
    lambdas: NDArray,
    term_indices: list[tuple[int, int]],
) -> NDArray:
    """Compute effective degrees of freedom for each smooth term.

    Uses the influence-matrix approach::

        F = (X'WX + sum lam*S)^{-1} X'WX

    Then ``edf_j = trace(F[j_start:j_end, j_start:j_end])``.

    Args:
        X_aug: Augmented design ``(n, p)``.
        W: IRLS weight vector ``(n,)`` (diagonal elements).
        S_penalties: Padded penalties.
        lambdas: Smoothing parameters.
        term_indices: ``(start, end)`` column slices per smooth.

    Returns:
        Array of effective degrees of freedom, one per smooth.
    """
    XtW = X_aug.T * W[np.newaxis, :]
    XtWX = XtW @ X_aug

    penalty = np.zeros_like(XtWX)
    for lam, S in zip(lambdas, S_penalties):
        penalty += lam * S

    A = XtWX + penalty

    # F = A^{-1} @ XtWX  =>  solve(A, XtWX)
    try:
        F = np.linalg.solve(A, XtWX)
    except np.linalg.LinAlgError:
        A += 1e-8 * np.eye(A.shape[0])
        F = np.linalg.solve(A, XtWX)

    edf = np.array(
        [np.trace(F[s:e, s:e]) for s, e in term_indices],
        dtype=np.float64,
    )
    return edf


def _compute_hat_matrix_trace(
    X_aug: NDArray,
    W: NDArray,
    S_penalties: list[NDArray],
    lambdas: NDArray,
) -> float:
    """Compute total trace of the hat matrix (total EDF).

    Args:
        X_aug: Augmented design ``(n, p)``.
        W: IRLS weight vector ``(n,)``.
        S_penalties: Padded penalties.
        lambdas: Smoothing parameters.

    Returns:
        Total effective degrees of freedom (scalar).
    """
    XtW = X_aug.T * W[np.newaxis, :]
    XtWX = XtW @ X_aug

    penalty = np.zeros_like(XtWX)
    for lam, S in zip(lambdas, S_penalties):
        penalty += lam * S

    A = XtWX + penalty
    try:
        F = np.linalg.solve(A, XtWX)
    except np.linalg.LinAlgError:
        A += 1e-8 * np.eye(A.shape[0])
        F = np.linalg.solve(A, XtWX)

    return float(np.trace(F))
