"""
Mathematical utilities for MVN MLE computation.

Core functions needed by the objective functions:
1. R-compatible pattern sorting (mysort)
2. Parameter reconstruction helpers
"""

import numpy as np
from typing import Tuple


def mysort_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort data by missingness patterns (direct port of R's mysort).

    This is CRITICAL for computational efficiency and R compatibility.
    The R algorithm sorts observations by missingness pattern to group
    identical patterns together for efficient likelihood computation.

    Parameters
    ----------
    data : np.ndarray, shape (n_observations, n_variables)
        Input data matrix with missing values as np.nan

    Returns
    -------
    sorted_data : np.ndarray
        Data matrix with rows reordered by missingness pattern
    freq : np.ndarray
        Number of observations in each missingness pattern block
    presence_absence : np.ndarray
        Binary matrix indicating observed variables for each pattern,
        shape (n_patterns, n_variables). 1 = observed, 0 = missing

    Notes
    -----
    This implements the exact algorithm from R's mvnmle package.
    """
    n_obs, n_vars = data.shape

    # Create binary representation (1=observed, 0=missing)
    # R: binrep <- ifelse(is.na(x), 0, 1)
    is_observed = (~np.isnan(data)).astype(int)

    # Convert to decimal representation for sorting
    # R: powers <- as.integer(2^((nvars-1):0))
    # R: decrep <- binrep %*% powers
    powers = 2 ** np.arange(n_vars - 1, -1, -1)
    pattern_codes = is_observed @ powers

    # Sort by pattern codes
    # R: sorted <- x[order(decrep), ]
    sort_indices = np.argsort(pattern_codes)
    sorted_data = data[sort_indices]
    sorted_patterns = is_observed[sort_indices]
    sorted_codes = pattern_codes[sort_indices]

    # Count frequency of each unique pattern
    # R: freq = as.vector(table(decrep))
    unique_codes, freq = np.unique(sorted_codes, return_counts=True)

    # Extract unique patterns
    presence_absence = []
    current_code = -1
    for i, code in enumerate(sorted_codes):
        if code != current_code:
            presence_absence.append(sorted_patterns[i])
            current_code = code

    presence_absence = np.array(presence_absence)

    return sorted_data, freq, presence_absence


def reconstruct_delta_matrix(theta: np.ndarray, n_vars: int) -> np.ndarray:
    """
    Reconstruct the upper triangular Delta matrix from parameter vector.

    The parameter vector is structured as:
    theta = [mu_1, ..., mu_p, log(delta_11), ..., log(delta_pp), delta_12, delta_13, delta_23, ...]

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector
    n_vars : int
        Number of variables (p)

    Returns
    -------
    np.ndarray, shape (n_vars, n_vars)
        Upper triangular Delta matrix

    Notes
    -----
    Critical for inverse Cholesky parameterization: Sigma = (Delta^-1)^T Delta^-1
    """
    Delta = np.zeros((n_vars, n_vars))

    # Extract diagonal elements (from log parameters to ensure positivity)
    log_diag = theta[n_vars:2*n_vars]
    Delta[np.diag_indices(n_vars)] = np.exp(log_diag)

    # Extract off-diagonal elements (upper triangle, column by column)
    idx = 2 * n_vars
    for j in range(n_vars):
        for i in range(j):
            Delta[i, j] = theta[idx]
            idx += 1

    return Delta


def extract_parameters(theta: np.ndarray, n_vars: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mean vector and covariance matrix from parameter vector.

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector
    n_vars : int
        Number of variables

    Returns
    -------
    mu : np.ndarray, shape (n_vars,)
        Mean vector
    sigma : np.ndarray, shape (n_vars, n_vars)
        Covariance matrix

    Notes
    -----
    Converts from inverse Cholesky parameterization back to mu, Sigma.
    """
    import warnings

    # Extract mean parameters
    mu = theta[:n_vars]

    # Reconstruct Delta matrix
    Delta = reconstruct_delta_matrix(theta, n_vars)

    # Convert to covariance matrix: Sigma = (Delta^-1)^T Delta^-1
    try:
        Delta_inv = np.linalg.inv(Delta)
        sigma = Delta_inv.T @ Delta_inv
    except np.linalg.LinAlgError:
        # Fallback for numerical issues
        sigma = np.eye(n_vars)
        warnings.warn("Numerical issues in parameter extraction, using identity covariance")

    return mu, sigma
