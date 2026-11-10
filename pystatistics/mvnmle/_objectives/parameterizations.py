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
