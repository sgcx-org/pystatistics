"""
CPU objective function using R's inverse Cholesky parameterization.

This implementation exactly matches R's mvnmle package, using the same
parameterization and computational approach for complete R compatibility.

CRITICAL: This is the R-exact reference implementation. All mathematical
operations are preserved verbatim from the validated pymvnmle code.
"""

import numpy as np
from typing import Tuple, Dict, Any
from ._r_port_base import RPortObjectiveBase, PatternData
from .parameterizations import InverseCholeskyParameterization
from pystatistics.core.exceptions import NumericalError

# Large finite penalty returned by ``compute_objective`` when the R-exact Givens
# reconstruction cannot be evaluated at a given ``theta`` (e.g. a non-positive
# Delta diagonal under extreme column-scale disparity). It is a valid "steer
# away" penalty for the optimizer, but it is NOT a log-likelihood — callers
# reporting a loglik MUST recognise it rather than returning ``-sentinel/2``.
_OBJECTIVE_FAILURE_SENTINEL: float = 1e20


class CPUObjectiveFP64(RPortObjectiveBase):
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
        """Get R-compatible initial parameters.

        Uses sample_cov (already regularized by base class if needed)
        as the starting point for optimization. Starting values do not
        affect the final MLE result.
        """
        return self.parameterization.get_initial_parameters(
            self.sample_mean,
            self.sample_cov.copy()
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
                    return _OBJECTIVE_FAILURE_SENTINEL

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
                return _OBJECTIVE_FAILURE_SENTINEL

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
                except Exception as exc:
                    # A silently zeroed component would let the optimizer
                    # report convergence at a non-stationary point (Rule 1:
                    # fail loud). compute_objective normally absorbs numerical
                    # breakdown into its finite failure sentinel, so reaching
                    # here means something genuinely unexpected broke.
                    raise NumericalError(
                        f"Gradient evaluation failed in both forward and "
                        f"backward differences at parameter index {i}."
                    ) from exc

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
            # Delta is singular at the reported optimum, so the fitted
            # covariance is undefined. Never substitute a fabricated value
            # (Rule 1): fill with NaN, which the downstream degeneracy guard
            # rejects loudly (SingularMatrixError, or a warning plus
            # converged=False under force=True).
            sigma = np.full((self.n_vars, self.n_vars), np.nan)

        # Compute log-likelihood. compute_objective returns the large finite
        # FAILURE SENTINEL when its R-exact Givens reconstruction breaks down at
        # this theta (non-positive Delta diagonal under extreme column-scale
        # disparity). That is a valid optimizer penalty but NOT a log-likelihood:
        # report NaN rather than -sentinel/2 (a spurious, enormous "loglik").
        # The backend surfaces the non-finite loglik loudly and marks the fit
        # not-converged.
        neg2ll = self.compute_objective(theta)
        loglik = (
            float("nan")
            if neg2ll >= _OBJECTIVE_FAILURE_SENTINEL
            else -neg2ll / 2.0
        )

        return mu, sigma, loglik
