"""
Base objective function class with R-compatible pattern ordering.

This implementation exactly replicates R's mvnmle behavior, including the
critical pattern ordering where data is sorted by pattern codes.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class PatternData:
    """Data structure for a single missingness pattern."""
    pattern_id: int                   # Unique identifier
    n_obs: int                        # Number of observations with this pattern
    data: np.ndarray                  # Data matrix (n_obs x n_observed_vars)
    observed_indices: np.ndarray      # Indices of observed variables
    missing_indices: np.ndarray       # Indices of missing variables
    pattern_start: int                # Start index in sorted data
    pattern_end: int                  # End index in sorted data


class MLEObjectiveBase:
    """
    Base class for maximum likelihood estimation objective functions.

    Handles:
    1. R-compatible data sorting (mysort algorithm)
    2. Pattern extraction with correct ordering
    3. Initial parameter computation
    4. Subclass interface for objective/gradient computation

    Critical: Exact replication of R mvnmle behavior.
    """

    def __init__(self, data: np.ndarray,
                 skip_validation: bool = False):
        """
        Initialize objective function with data.

        Parameters
        ----------
        data : np.ndarray, shape (n_obs, n_vars)
            Input data matrix with missing values as np.nan
        skip_validation : bool
            If True, skip data validation (assumes caller validated)
        """
        if not skip_validation:
            self._validate_data(data)

        # Store original data
        self.original_data = np.asarray(data, dtype=np.float64)
        self.n_obs, self.n_vars = self.original_data.shape

        # Parameter dimensions
        self.n_mean_params = self.n_vars
        self.n_cov_params = self.n_vars * (self.n_vars + 1) // 2
        self.n_params = self.n_mean_params + self.n_cov_params

        # Apply R's mysort algorithm
        self._apply_mysort()

        # Extract patterns WITHOUT reordering (keep R's sort order)
        self.patterns = self._extract_patterns()
        self.n_patterns = len(self.patterns)

        # Compute sample statistics for subclasses
        self.sample_mean = np.nanmean(self.original_data, axis=0)
        self.sample_cov = self._compute_sample_covariance()

        # Check if data is complete (no missing values)
        self.is_complete = not np.any(np.isnan(self.original_data))

    def _validate_data(self, data: np.ndarray) -> None:
        """Validate input data matrix."""
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)

        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")

        if data.shape[0] < 2:
            raise ValueError(f"Need at least 2 observations, got {data.shape[0]}")

        if data.shape[1] < 1:
            raise ValueError(f"Need at least 1 variable, got {data.shape[1]}")

        # Check for all-missing rows or columns
        if np.any(np.all(np.isnan(data), axis=1)):
            raise ValueError("Data contains rows with all missing values")

        if np.any(np.all(np.isnan(data), axis=0)):
            raise ValueError("Data contains columns with all missing values")

    def _apply_mysort(self) -> None:
        """
        Apply R's mysort algorithm to sort data by missingness patterns.

        CRITICAL for R compatibility - must match exactly.
        Creates pattern codes using powers of 2 and sorts observations.
        """
        # Create presence/absence matrix (1 = observed, 0 = missing)
        self.presence_absence = (~np.isnan(self.original_data)).astype(int)

        # Create pattern codes using powers of 2 (R's approach)
        powers = 2 ** np.arange(self.n_vars - 1, -1, -1)
        pattern_codes = self.presence_absence @ powers

        # Sort data by pattern codes
        sort_indices = np.argsort(pattern_codes)
        self.sorted_data = self.original_data[sort_indices]
        self.sorted_patterns = self.presence_absence[sort_indices]
        self.sorted_codes = pattern_codes[sort_indices]

        # Get unique patterns and their frequencies
        unique_codes, indices, counts = np.unique(
            self.sorted_codes,
            return_index=True,
            return_counts=True
        )

        self.pattern_frequencies = counts
        self.pattern_start_indices = indices

    def _extract_patterns(self) -> List[PatternData]:
        """
        Extract pattern data from sorted dataset.

        Returns
        -------
        List[PatternData]
            Pattern data structures in R's expected order
        """
        patterns = []

        for i, (start_idx, count) in enumerate(zip(self.pattern_start_indices,
                                                   self.pattern_frequencies)):
            end_idx = start_idx + count

            # Get pattern mask for this group
            pattern_mask = self.sorted_patterns[start_idx]
            observed_indices = np.where(pattern_mask == 1)[0]
            missing_indices = np.where(pattern_mask == 0)[0]

            # Extract data for this pattern (only observed variables)
            pattern_data = self.sorted_data[start_idx:end_idx]
            if len(observed_indices) > 0:
                observed_data = pattern_data[:, observed_indices]
            else:
                observed_data = np.empty((count, 0))

            patterns.append(PatternData(
                pattern_id=i,
                n_obs=count,
                data=observed_data,
                observed_indices=observed_indices,
                missing_indices=missing_indices,
                pattern_start=start_idx,
                pattern_end=end_idx
            ))

        return patterns

    def _compute_sample_covariance(self) -> np.ndarray:
        """
        Compute sample covariance using pairwise deletion with proper regularization.

        Matches R's approach for getting initial parameter estimates.
        Uses only pairs of observations where both variables are observed.

        Returns
        -------
        np.ndarray, shape (n_vars, n_vars)
            Sample covariance matrix (guaranteed positive definite)
        """
        cov = np.zeros((self.n_vars, self.n_vars))

        for i in range(self.n_vars):
            for j in range(i, self.n_vars):
                # Get pairs where both variables are observed
                mask = ~(np.isnan(self.original_data[:, i]) |
                        np.isnan(self.original_data[:, j]))

                n_complete = np.sum(mask)

                if n_complete > 1:
                    xi = self.original_data[mask, i]
                    xj = self.original_data[mask, j]

                    # Compute covariance with bias correction
                    mean_xi = np.mean(xi)
                    mean_xj = np.mean(xj)

                    if i == j:
                        # Variance on diagonal
                        cov[i, i] = np.mean((xi - mean_xi) ** 2)
                    else:
                        # Covariance off diagonal
                        cov_ij = np.mean((xi - mean_xi) * (xj - mean_xj))
                        cov[i, j] = cov_ij
                        cov[j, i] = cov_ij
                elif i == j:
                    # Not enough data for this variable, use unit variance
                    cov[i, i] = 1.0

        # Proper regularization for positive definiteness
        try:
            eigenvals = np.linalg.eigvalsh(cov)
            min_eigenval = np.min(eigenvals)
            max_eigenval = np.max(eigenvals)

            if min_eigenval <= 0:
                # Add enough regularization to ensure positive definiteness
                regularization = max(0.01, 0.01 * max_eigenval, abs(min_eigenval) + 0.01)
                cov += regularization * np.eye(self.n_vars)

                # Shrink off-diagonal elements to improve conditioning
                shrink_factor = 0.95
                for i in range(self.n_vars):
                    for j in range(i + 1, self.n_vars):
                        cov[i, j] *= shrink_factor
                        cov[j, i] *= shrink_factor

            elif min_eigenval < 1e-4:
                # Even if positive definite, ensure reasonable conditioning
                regularization = 1e-4
                cov += regularization * np.eye(self.n_vars)

        except np.linalg.LinAlgError:
            # If eigenvalue computation fails, use diagonal matrix
            cov = np.diag(np.maximum(np.diag(cov), 1.0))

        return cov

    def get_initial_parameters(self) -> np.ndarray:
        """Get initial parameter values. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement get_initial_parameters")

    def compute_objective(self, theta: np.ndarray) -> float:
        """Compute objective function value. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement compute_objective")

    def compute_gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradient of objective function.

        Default implementation uses finite differences (R-compatible).
        Subclasses may override with analytical gradients.
        """
        eps = 1e-8
        grad = np.zeros_like(theta)
        f0 = self.compute_objective(theta)

        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            f_plus = self.compute_objective(theta_plus)
            grad[i] = (f_plus - f0) / eps

        return grad

    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Extract mu, Sigma, and log-likelihood from parameter vector. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement extract_parameters")
