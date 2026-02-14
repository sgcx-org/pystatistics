"""
Parameterizations for covariance matrices in MLE optimization.

Different parameterizations have different numerical properties and are
suited for different optimization algorithms and precision levels.
"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class CovarianceParameterization(ABC):
    """Abstract base class for covariance parameterizations."""
    
    def __init__(self, n_vars: int):
        """
        Initialize parameterization.
        
        Parameters
        ----------
        n_vars : int
            Number of variables (dimension of covariance matrix)
        """
        self.n_vars = n_vars
        self.n_mean_params = n_vars
        self.n_cov_params = n_vars + (n_vars * (n_vars - 1)) // 2
        self.n_params = self.n_mean_params + self.n_cov_params
    
    @abstractmethod
    def pack(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Pack mean and covariance into parameter vector."""
        pass
    
    @abstractmethod
    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack parameter vector into mean and covariance."""
        pass
    
    def get_initial_parameters(self, sample_mean: np.ndarray, 
                              sample_cov: np.ndarray) -> np.ndarray:
        """
        Get initial parameter vector from sample statistics.
        
        NO REGULARIZATION HERE - let objectives handle it.
        
        Parameters
        ----------
        sample_mean : np.ndarray
            Sample mean vector
        sample_cov : np.ndarray
            Sample covariance matrix
            
        Returns
        -------
        np.ndarray
            Initial parameter vector
        """
        return self.pack(sample_mean, sample_cov)


class InverseCholeskyParameterization(CovarianceParameterization):
    """
    Inverse Cholesky parameterization (R-compatible).
    
    Parameters: [μ, log(diag(Δ)), off-diag(Δ)]
    where Σ = (Δ^{-1})^T Δ^{-1} and Δ is upper triangular.
    
    This matches R's mvnmle package exactly.
    """
    
    def pack(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Pack using inverse Cholesky decomposition."""
        # Compute Δ where Σ = (Δ^{-1})^T Δ^{-1}
        L = np.linalg.cholesky(sigma)
        delta = np.linalg.inv(L).T
        
        # Extract parameters
        log_diag = np.log(np.diag(delta))
        
        # Extract UPPER triangular elements (column-wise, R style)
        # Delta is upper triangular, so we extract i < j
        tril_elements = []
        for j in range(self.n_vars):
            for i in range(j):  # i < j for upper triangle
                tril_elements.append(delta[i, j])
        
        return np.concatenate([mu, log_diag, tril_elements])
    
    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack from inverse Cholesky parameters."""
        mu = theta[:self.n_vars]
        
        # Reconstruct Δ matrix (upper triangular)
        delta = np.zeros((self.n_vars, self.n_vars))
        
        # Diagonal elements
        np.fill_diagonal(delta, np.exp(theta[self.n_vars:2*self.n_vars]))
        
        # Off-diagonal elements (column-wise, R style)
        idx = 2 * self.n_vars
        for j in range(self.n_vars):
            for i in range(j):  # i < j for upper triangle
                delta[i, j] = theta[idx]
                idx += 1
        
        # Compute Σ = (Δ^{-1})^T Δ^{-1}
        delta_inv = np.linalg.inv(delta)
        sigma = delta_inv.T @ delta_inv
        
        return mu, sigma


class CholeskyParameterization(CovarianceParameterization):
    """
    Standard Cholesky parameterization.
    
    Parameters: [μ, log(diag(L)), off-diag(L)]
    where Σ = LL^T and L is lower triangular.
    
    More natural for autodiff and GPU computation.
    """
    
    def pack(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Pack using Cholesky decomposition."""
        L = np.linalg.cholesky(sigma)
        
        # Log of diagonal for unconstrained optimization
        log_diag = np.log(np.diag(L))
        
        # Extract lower triangular elements
        tril_idx = np.tril_indices(self.n_vars, k=-1)
        tril_elements = L[tril_idx]
        
        return np.concatenate([mu, log_diag, tril_elements])
    
    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack from Cholesky parameters."""
        mu = theta[:self.n_vars]
        
        # Reconstruct L matrix (lower triangular)
        L = np.zeros((self.n_vars, self.n_vars))
        
        # Diagonal elements
        np.fill_diagonal(L, np.exp(theta[self.n_vars:2*self.n_vars]))
        
        # Off-diagonal elements (lower triangle)
        idx = 2 * self.n_vars
        tril_idx = np.tril_indices(self.n_vars, k=-1)
        if len(tril_idx[0]) > 0:
            L[tril_idx] = theta[idx:]
        
        # Compute Σ = LL^T
        sigma = L @ L.T
        
        return mu, sigma


class BoundedCholeskyParameterization(CovarianceParameterization):
    """
    Bounded Cholesky parameterization for FP32 stability.
    
    Uses sigmoid/tanh transformations to naturally bound parameters:
    - Diagonal elements bounded to [var_min, var_max]
    - Off-diagonal elements bounded to prevent ill-conditioning
    
    Essential for FP32 GPU optimization to prevent numerical explosion.
    """
    
    def __init__(self, n_vars: int,
                 var_min: float = 0.001,
                 var_max: float = 100.0,
                 corr_max: float = 0.95):
        """
        Initialize bounded parameterization.
        
        Parameters
        ----------
        n_vars : int
            Number of variables
        var_min : float
            Minimum variance
        var_max : float
            Maximum variance
        corr_max : float
            Maximum absolute correlation
        """
        super().__init__(n_vars)
        self.var_min = var_min
        self.var_max = var_max
        self.corr_max = corr_max
    
    def _sigmoid(self, x: np.ndarray, low: float, high: float) -> np.ndarray:
        """Map R to [low, high] using sigmoid."""
        return low + (high - low) / (1 + np.exp(-x))
    
    def _inverse_sigmoid(self, y: np.ndarray, low: float, high: float) -> np.ndarray:
        """Inverse sigmoid transformation."""
        # Clip to avoid log(0)
        y_clipped = np.clip((y - low) / (high - low), 1e-10, 1 - 1e-10)
        return np.log(y_clipped / (1 - y_clipped))
    
    def pack(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Pack with inverse transformations."""
        L = np.linalg.cholesky(sigma)
        
        # Transform diagonal to unbounded
        diag_vals = np.diag(L) ** 2  # Variances
        diag_vals = np.clip(diag_vals, self.var_min, self.var_max)
        diag_unbounded = self._inverse_sigmoid(diag_vals, self.var_min, self.var_max)
        
        # Transform off-diagonal to unbounded
        tril_idx = np.tril_indices(self.n_vars, k=-1)
        if len(tril_idx[0]) > 0:
            L_tril = L[tril_idx]
            # Normalize by sqrt of diagonal products (correlation scale)
            i, j = tril_idx
            normalizer = np.sqrt(L[i, i] * L[j, j])
            corr_vals = np.clip(L_tril / normalizer, -self.corr_max * 0.999, self.corr_max * 0.999)
            tril_unbounded = np.arctanh(corr_vals / self.corr_max)
        else:
            tril_unbounded = np.array([])
        
        return np.concatenate([mu, diag_unbounded, tril_unbounded])
    
    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack with forward transformations."""
        mu = theta[:self.n_vars]
        
        # Reconstruct L matrix
        L = np.zeros((self.n_vars, self.n_vars))
        
        # Diagonal: transform from unbounded to [var_min, var_max]
        diag_unbounded = theta[self.n_vars:2*self.n_vars]
        diag_vars = self._sigmoid(diag_unbounded, self.var_min, self.var_max)
        np.fill_diagonal(L, np.sqrt(diag_vars))
        
        # Off-diagonal: transform from unbounded to bounded
        idx = 2 * self.n_vars
        tril_idx = np.tril_indices(self.n_vars, k=-1)
        if len(tril_idx[0]) > 0:
            tril_unbounded = theta[idx:]
            corr_vals = self.corr_max * np.tanh(tril_unbounded)
            
            # Scale by diagonal
            i, j = tril_idx
            L[tril_idx] = corr_vals * np.sqrt(L[i, i] * L[j, j])
        
        # Compute Σ = LL^T
        sigma = L @ L.T
        
        return mu, sigma


class MatrixLogParameterization(CovarianceParameterization):
    """
    Matrix logarithm parameterization.
    
    Parameters: [μ, vech(log(Σ))]
    
    Ensures positive definiteness but can be computationally expensive.
    """
    
    def pack(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Pack using matrix logarithm."""
        from scipy.linalg import logm
        
        log_sigma = logm(sigma)
        
        # Extract lower triangular part (including diagonal)
        tril_idx = np.tril_indices(self.n_vars)
        log_sigma_vec = log_sigma[tril_idx]
        
        return np.concatenate([mu, log_sigma_vec])
    
    def unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack using matrix exponential."""
        from scipy.linalg import expm
        
        mu = theta[:self.n_vars]
        
        # Reconstruct log(Σ)
        log_sigma = np.zeros((self.n_vars, self.n_vars))
        tril_idx = np.tril_indices(self.n_vars)
        log_sigma[tril_idx] = theta[self.n_vars:]
        
        # Make symmetric
        log_sigma = log_sigma + log_sigma.T - np.diag(np.diag(log_sigma))
        
        # Compute Σ = exp(log(Σ))
        sigma = expm(log_sigma)
        
        return mu, sigma


def get_parameterization(name: str, n_vars: int, **kwargs) -> CovarianceParameterization:
    """
    Factory function for creating parameterizations.
    
    Parameters
    ----------
    name : str
        Name of parameterization ('inverse_cholesky', 'cholesky', 'bounded_cholesky', 'matrix_log')
    n_vars : int
        Number of variables
    **kwargs
        Additional parameters for specific parameterizations
        
    Returns
    -------
    CovarianceParameterization
        Parameterization instance
    """
    name_lower = name.lower()
    
    if name_lower in ['inverse_cholesky', 'inverse-cholesky', 'inv_chol']:
        return InverseCholeskyParameterization(n_vars)
    elif name_lower in ['cholesky', 'chol']:
        return CholeskyParameterization(n_vars)
    elif name_lower in ['bounded_cholesky', 'bounded-cholesky', 'bounded']:
        return BoundedCholeskyParameterization(n_vars, **kwargs)
    elif name_lower in ['matrix_log', 'matrix-log', 'logm']:
        return MatrixLogParameterization(n_vars)
    else:
        raise ValueError(f"Unknown parameterization: {name}")


def convert_parameters(theta: np.ndarray, 
                       from_param: CovarianceParameterization,
                       to_param: CovarianceParameterization) -> np.ndarray:
    """
    Convert parameters between different parameterizations.
    
    Parameters
    ----------
    theta : np.ndarray
        Parameter vector in source parameterization
    from_param : CovarianceParameterization
        Source parameterization
    to_param : CovarianceParameterization
        Target parameterization
        
    Returns
    -------
    np.ndarray
        Parameter vector in target parameterization
    """
    # Unpack from source
    mu, sigma = from_param.unpack(theta)
    
    # Pack into target
    return to_param.pack(mu, sigma)