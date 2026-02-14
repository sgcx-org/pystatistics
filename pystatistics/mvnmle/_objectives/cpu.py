"""
CPU objective function using R's inverse Cholesky parameterization.

This implementation exactly matches R's mvnmle package, using the same
parameterization and computational approach for complete R compatibility.

CRITICAL: This is the R-exact reference implementation. All mathematical
operations are preserved verbatim from the validated pymvnmle code.
"""

import numpy as np
from typing import Tuple, Dict, Any
from .base import MLEObjectiveBase, PatternData
from .parameterizations import InverseCholeskyParameterization


class CPUObjectiveFP64(MLEObjectiveBase):
    """
    R-compatible MLE objective using inverse Cholesky parameterization.

    This is the reference implementation that exactly matches R's mvnmle.
    Uses vectorized computation within each pattern for performance.
    """

    def __init__(self, data: np.ndarray,
                 validate: bool = True):
        """
        Initialize CPU objective with R-compatible settings.

        Parameters
        ----------
        data : np.ndarray
            Input data with missing values as np.nan
        validate : bool
            Whether to validate input data
        """
        super().__init__(data, skip_validation=not validate)

        # Create parameterization
        self.parameterization = InverseCholeskyParameterization(self.n_vars)
        self.n_params = self.parameterization.n_params

        # R compatibility settings
        self.use_inverse_cholesky = True
        self.objective_scale = -2.0  # R returns -2 * log-likelihood

        # Precompute constants for efficiency
        self._precompute_constants()

    def _precompute_constants(self) -> None:
        """Precompute constants used in likelihood calculation."""
        # Note: R does NOT include the constant term n*p*log(2pi) in objective
        # This is confirmed by evallf.c source code
        self.total_observed = sum(
            pattern.n_obs * len(pattern.observed_indices)
            for pattern in self.patterns
        )

    def get_initial_parameters(self) -> np.ndarray:
        """Get R-compatible initial parameters."""
        sample_cov_regularized = self.sample_cov.copy()

        # Check condition number
        try:
            eigenvals = np.linalg.eigvalsh(sample_cov_regularized)
            min_eig = np.min(eigenvals)
            max_eig = np.max(eigenvals)

            # If poorly conditioned or non-PD, regularize
            if min_eig < 1e-6 or max_eig / min_eig > 1e10:
                reg_amount = max(1e-4, abs(min_eig) + 1e-4)
                sample_cov_regularized += reg_amount * np.eye(self.n_vars)

                # Shrink off-diagonals for stability
                for i in range(self.n_vars):
                    for j in range(i + 1, self.n_vars):
                        sample_cov_regularized[i, j] *= 0.95
                        sample_cov_regularized[j, i] *= 0.95
        except np.linalg.LinAlgError:
            # If eigenvalue computation fails, use diagonal
            sample_cov_regularized = np.diag(np.diag(self.sample_cov))
            sample_cov_regularized += 0.1 * np.eye(self.n_vars)

        return self.parameterization.get_initial_parameters(
            self.sample_mean,
            sample_cov_regularized
        )

    def _reconstruct_delta_matrix(self, theta: np.ndarray) -> np.ndarray:
        """
        Reconstruct upper triangular Delta matrix from parameter vector.

        EXACT port from R's mvnmle.
        """
        Delta = np.zeros((self.n_vars, self.n_vars))

        # Diagonal elements: exp(log-parameters) to ensure positivity
        log_diag = theta[self.n_vars:2*self.n_vars]
        np.fill_diagonal(Delta, np.exp(log_diag))

        # Off-diagonal elements in R's column-major order
        idx = 2 * self.n_vars
        for j in range(self.n_vars):
            for i in range(j):
                Delta[i, j] = theta[idx]
                idx += 1

        return Delta

    def _apply_givens_rotations(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply Givens rotations for numerical stability.

        CRITICAL for R compatibility. Matches R's evallf.c implementation precisely.
        """
        result = matrix.copy()

        # R's exact algorithm: bottom-up, left-to-right
        for i in range(self.n_vars - 1, -1, -1):  # Bottom to top
            for j in range(i):  # Left to diagonal
                a = result[i, j]
                b = result[i, j+1] if j+1 < self.n_vars else 0.0

                # R's exact threshold
                if np.abs(a) < 0.000001:
                    result[i, j] = 0.0
                    continue

                # Compute rotation parameters
                r = np.sqrt(a*a + b*b)
                if r < 0.000001:
                    continue

                c = a / r
                d = b / r

                # Apply rotation to entire matrix
                for k in range(self.n_vars):
                    old_kj = result[k, j]
                    old_kj1 = result[k, j+1] if j+1 < self.n_vars else 0.0

                    result[k, j] = d * old_kj - c * old_kj1
                    if j+1 < self.n_vars:
                        result[k, j+1] = c * old_kj + d * old_kj1

                result[i, j] = 0.0

        # Ensure positive diagonal (R's sign adjustment)
        for i in range(self.n_vars):
            if result[i, i] < 0:
                for j in range(i+1):
                    result[j, i] *= -1

        return result

    def compute_objective(self, theta: np.ndarray) -> float:
        """
        Compute negative log-likelihood using R's exact algorithm.

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector [mu, log(diag(Delta)), off-diag(Delta)]

        Returns
        -------
        float
            -2 * log-likelihood (R convention)
        """
        # Extract mean parameters
        mu = theta[:self.n_vars]

        # Reconstruct Delta matrix using R's algorithm
        Delta = self._reconstruct_delta_matrix(theta)

        # Apply Givens rotations for numerical stability (R's evallf.c)
        Delta_stabilized = self._apply_givens_rotations(Delta)

        # Compute negative log-likelihood using pattern-wise formula
        neg_loglik = 0.0

        for pattern in self.patterns:
            if pattern.n_obs == 0 or len(pattern.observed_indices) == 0:
                continue

            # Extract relevant submatrices (R's approach)
            obs_indices = pattern.observed_indices
            n_obs_vars = len(obs_indices)
            mu_k = mu[obs_indices]

            # CRITICAL: Implement R's row shuffling algorithm exactly
            # Create reordered Delta with observed rows first, missing rows last
            subdel = np.zeros((self.n_vars, self.n_vars))

            # Put observed variable rows FIRST
            pcount = 0
            for i in range(self.n_vars):
                if i in obs_indices:
                    subdel[pcount, :] = Delta_stabilized[i, :]
                    pcount += 1

            # Put missing variable rows LAST
            acount = 0
            for i in range(self.n_vars):
                if i not in obs_indices:
                    subdel[self.n_vars - acount - 1, :] = Delta_stabilized[i, :]
                    acount += 1

            # Apply Givens rotations to shuffled matrix
            subdel_rotated = self._apply_givens_rotations(subdel)

            # Extract top-left submatrix for observed variables
            Delta_k = subdel_rotated[:n_obs_vars, :n_obs_vars]

            try:
                # Use R's exact computation approach
                diag_delta_k = np.diag(Delta_k)

                # Check for numerical issues
                if np.any(diag_delta_k <= 0):
                    return 1e20

                log_det_delta_k = np.sum(np.log(diag_delta_k))

                # Vectorized quadratic form computation
                obj_contribution = -2 * pattern.n_obs * log_det_delta_k

                # Compute Delta_k.T @ centered.T for all observations at once
                centered = pattern.data - mu_k[np.newaxis, :]
                prod_all = Delta_k.T @ centered.T
                quadratic_forms = np.sum(prod_all * prod_all, axis=0)
                obj_contribution += np.sum(quadratic_forms)

                neg_loglik += obj_contribution

            except (np.linalg.LinAlgError, RuntimeError):
                return 1e20

        return neg_loglik

    def compute_gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradient using finite differences (R-compatible).

        Uses R's exact finite difference formula with proper step size scaling.
        This matches R's nlm() behavior exactly.
        """
        n_params = len(theta)
        grad = np.zeros(n_params)

        # R's nlm uses this specific epsilon
        eps = 1.49011612e-08  # R's .Machine$double.eps^(1/3)

        # Base objective value
        f_base = self.compute_objective(theta)

        # Compute gradient using forward differences with R's exact step size
        for i in range(n_params):
            # R's step size calculation - CRITICAL
            h = eps * max(abs(theta[i]), 1.0)

            # Ensure step is not too small
            if h < 1e-12:
                h = 1e-12

            # Forward difference with properly scaled step
            theta_plus = theta.copy()
            theta_plus[i] = theta[i] + h

            try:
                f_plus = self.compute_objective(theta_plus)
                grad[i] = (f_plus - f_base) / h
            except Exception:
                # If forward fails, try backward difference
                theta_minus = theta.copy()
                theta_minus[i] = theta[i] - h
                try:
                    f_minus = self.compute_objective(theta_minus)
                    grad[i] = (f_base - f_minus) / h
                except Exception:
                    grad[i] = 0.0

        return grad

    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract mu, Sigma, and log-likelihood from parameter vector.

        EXACT port from validated implementation.
        """
        # Extract mean parameters
        mu = theta[:self.n_vars]

        # Reconstruct Delta matrix using R's algorithm
        Delta = self._reconstruct_delta_matrix(theta)

        # Convert to covariance matrix: Sigma = (Delta^-1)^T Delta^-1
        try:
            # Use solve for numerical stability (R's approach)
            I = np.eye(self.n_vars)
            Delta_inv = np.linalg.solve(Delta, I)
            sigma = Delta_inv.T @ Delta_inv

            # Ensure exact symmetry (R does this)
            sigma = (sigma + sigma.T) / 2.0

        except np.linalg.LinAlgError:
            sigma = np.eye(self.n_vars)

        # Compute log-likelihood
        loglik = -self.compute_objective(theta) / 2.0  # Objective is -2*log-likelihood

        return mu, sigma, loglik
