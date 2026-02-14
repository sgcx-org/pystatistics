"""
Little's MCAR Test.

Implementation of Little's (1988) test for Missing Completely at Random (MCAR).

Reference:
    Little, R.J.A. (1988). A test of missing completely at random for
    multivariate data with missing values. JASA, 83(404), 1198-1202.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings

from pystatistics.mvnmle.patterns import PatternInfo, identify_missingness_patterns


@dataclass
class MCARTestResult:
    """Result of Little's MCAR test."""
    statistic: float
    df: int
    p_value: float
    rejected: bool
    alpha: float
    patterns: List[PatternInfo]
    n_patterns: int
    n_patterns_used: int
    ml_mean: np.ndarray
    ml_cov: np.ndarray
    convergence_warnings: List[str]

    def summary(self) -> str:
        """Generate human-readable summary of test results."""
        summary_lines = [
            "Little's MCAR Test Results",
            "=" * 40,
            f"Test statistic (chi-sq): {self.statistic:.4f}",
            f"Degrees of freedom: {self.df}",
            f"P-value: {self.p_value:.4f}",
            f"",
            f"Decision at alpha={self.alpha}: {'Reject MCAR' if self.rejected else 'Fail to reject MCAR'}",
            f"",
            f"Number of patterns: {self.n_patterns}",
        ]

        if self.convergence_warnings:
            summary_lines.append("\nWarnings:")
            for warning in self.convergence_warnings:
                summary_lines.append(f"  - {warning}")

        return "\n".join(summary_lines)


def regularized_inverse(matrix: np.ndarray,
                       condition_threshold: float = 1e12,
                       regularization: float = 1e-8) -> Tuple[np.ndarray, bool]:
    """
    Compute inverse with regularization for near-singular matrices.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to invert
    condition_threshold : float
        Maximum acceptable condition number
    regularization : float
        Regularization parameter to add to diagonal

    Returns
    -------
    inv_matrix : np.ndarray
        Inverted matrix
    was_regularized : bool
        Whether regularization was applied
    """
    cond = np.linalg.cond(matrix)

    if cond < condition_threshold:
        try:
            return np.linalg.inv(matrix), False
        except np.linalg.LinAlgError:
            pass

    # Need regularization
    n = matrix.shape[0]
    reg_matrix = matrix + regularization * np.eye(n)

    try:
        return np.linalg.inv(reg_matrix), True
    except np.linalg.LinAlgError:
        # Use eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        min_eigenval = np.max(eigenvals) * regularization
        eigenvals_reg = np.maximum(eigenvals, min_eigenval)
        inv_matrix = eigenvecs @ np.diag(1/eigenvals_reg) @ eigenvecs.T
        return inv_matrix, True


def little_mcar_test(data,
                     alpha: float = 0.05,
                     verbose: bool = False) -> MCARTestResult:
    """
    Little's test for Missing Completely at Random (MCAR).

    Parameters
    ----------
    data : array-like, shape (n_observations, n_variables)
        Data matrix with missing values as np.nan.
    alpha : float, default=0.05
        Significance level
    verbose : bool, default=False
        Print detailed progress

    Returns
    -------
    MCARTestResult
    """
    # Import mlest here to avoid circular imports
    from pystatistics.mvnmle.solvers import mlest

    # Input conversion
    if hasattr(data, 'values'):
        data_array = np.asarray(data.values, dtype=float)
    else:
        data_array = np.asarray(data, dtype=float)

    if data_array.ndim != 2:
        raise ValueError("Data must be 2-dimensional")

    n_obs, n_vars = data_array.shape

    # Step 1: Get ML estimates
    if verbose:
        print("Step 1: Computing ML estimates...")

    try:
        ml_result = mlest(data_array, verbose=False)
        mu_ml = ml_result.muhat
        sigma_ml = ml_result.sigmahat
    except Exception as e:
        raise RuntimeError(f"ML estimation failed: {e}")

    # Step 2: Identify missingness patterns
    if verbose:
        print("Step 2: Identifying missingness patterns...")

    patterns = identify_missingness_patterns(data_array)

    # Step 3: Compute test statistic
    test_statistic = 0.0
    convergence_warnings = []
    n_patterns_used = 0

    for pattern in patterns:
        if pattern.n_observed == 0:
            continue

        obs_idx = pattern.observed_indices
        n_k = pattern.n_cases

        y_bar_k = np.mean(pattern.data, axis=0)
        mu_obs_k = mu_ml[obs_idx]
        sigma_obs_k = sigma_ml[np.ix_(obs_idx, obs_idx)]

        try:
            sigma_inv_k, was_regularized = regularized_inverse(sigma_obs_k)

            if was_regularized:
                msg = f"Pattern {pattern.pattern_id}: Covariance regularized (near-singular)"
                convergence_warnings.append(msg)

        except Exception as e:
            msg = f"Pattern {pattern.pattern_id}: Failed to invert covariance: {e}"
            convergence_warnings.append(msg)
            continue

        diff = y_bar_k - mu_obs_k
        contribution = n_k * (diff @ sigma_inv_k @ diff)
        test_statistic += contribution
        n_patterns_used += 1

    # Step 4: Degrees of freedom
    df = sum(p.n_observed for p in patterns) - n_vars

    # Handle edge cases
    if len(patterns) == 1 and patterns[0].n_observed == n_vars:
        return MCARTestResult(
            statistic=0.0,
            df=0,
            p_value=1.0,
            rejected=False,
            alpha=alpha,
            patterns=patterns,
            n_patterns=1,
            n_patterns_used=0,
            ml_mean=mu_ml,
            ml_cov=sigma_ml,
            convergence_warnings=["No missing data - MCAR test not applicable"]
        )

    if df <= 0:
        raise ValueError(f"Invalid degrees of freedom: {df}")

    # Step 5: P-value
    p_value = 1 - stats.chi2.cdf(test_statistic, df)
    rejected = p_value < alpha

    return MCARTestResult(
        statistic=test_statistic,
        df=df,
        p_value=p_value,
        rejected=rejected,
        alpha=alpha,
        patterns=patterns,
        n_patterns=len(patterns),
        n_patterns_used=n_patterns_used,
        ml_mean=mu_ml,
        ml_cov=sigma_ml,
        convergence_warnings=convergence_warnings
    )
