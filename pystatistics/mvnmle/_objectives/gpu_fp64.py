"""
GPU objective function using standard Cholesky parameterization with FP64.

For data center GPUs (A100, H100, V100, RTX 5070 Ti) with FP64 support.
Enables Newton-CG optimization with analytical Hessians.
CUDA only - MPS does not support FP64.
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Dict, Any
from .base import MLEObjectiveBase, PatternData
from .parameterizations import CholeskyParameterization


class GPUObjectiveFP64(MLEObjectiveBase):
    """
    GPU-accelerated MLE objective with FP64 precision.

    Designed for data center GPUs with FP64 support.
    Enables Newton-CG optimization with analytical second-order derivatives.
    CUDA only.
    """

    def __init__(self, data: np.ndarray,
                 device: Optional[str] = None,
                 validate: bool = True):
        """
        Initialize GPU FP64 objective.

        Parameters
        ----------
        data : np.ndarray, shape (n_obs, n_vars)
            Input data with missing values as np.nan
        device : str or None
            PyTorch device (must be CUDA for FP64)
        validate : bool
            Whether to validate input data
        """
        super().__init__(data, skip_validation=not validate)

        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for GPU objectives. "
                "Install with: pip install torch"
            )

        # Create parameterization (standard Cholesky for GPU)
        self.parameterization = CholeskyParameterization(self.n_vars)
        self.n_params = self.parameterization.n_params

        # Select device (must be CUDA for FP64)
        self.device = self._select_device(device)

        # FP64 settings
        self.dtype = torch.float64
        self.eps = 1e-12  # Tighter epsilon for FP64

        # Transfer pattern data to GPU
        self._prepare_gpu_data()

    def _select_device(self, requested_device: Optional[str]) -> Any:
        """Select appropriate GPU device for FP64."""
        torch = self.torch

        if requested_device:
            device = torch.device(requested_device)
            if device.type == 'mps':
                raise RuntimeError(
                    "MPS does not support FP64. Use GPUObjectiveFP32 for MPS, "
                    "or use CUDA for FP64."
                )
            if device.type != 'cuda':
                warnings.warn(
                    f"FP64 objective requested on {device.type}. "
                    f"Performance may be poor. Consider using FP32 objective."
                )
            return device

        # Auto-select: prefer CUDA for FP64
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            warnings.warn(
                "No CUDA device available for FP64. "
                "Using CPU fallback (will be very slow)."
            )
            return torch.device('cpu')

    def _prepare_gpu_data(self) -> None:
        """Transfer pattern data to GPU with FP64 precision."""
        torch = self.torch

        self.gpu_patterns = []
        for pattern in self.patterns:
            gpu_pattern = {
                'n_obs': pattern.n_obs,
                'n_observed': len(pattern.observed_indices),
                'observed_indices': torch.tensor(
                    pattern.observed_indices,
                    device=self.device,
                    dtype=torch.long
                ),
                'data': torch.tensor(
                    pattern.data,
                    device=self.device,
                    dtype=self.dtype
                )
            }
            self.gpu_patterns.append(gpu_pattern)

    def get_initial_parameters(self) -> np.ndarray:
        """Get initial parameters using standard Cholesky."""
        return self.parameterization.get_initial_parameters(
            self.sample_mean,
            self.sample_cov
        )

    def compute_objective(self, theta: np.ndarray) -> float:
        """
        Compute -2 * log-likelihood using GPU with FP64.

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector

        Returns
        -------
        float
            -2 * log-likelihood value
        """
        torch = self.torch

        theta_gpu = torch.tensor(theta, device=self.device, dtype=self.dtype)
        obj_value = self._torch_objective(theta_gpu)

        return obj_value.item()

    def _torch_objective(self, theta_gpu: Any) -> Any:
        """PyTorch implementation of objective function."""
        torch = self.torch

        mu_gpu, sigma_gpu = self._unpack_gpu(theta_gpu)

        obj_value = torch.zeros(1, device=self.device, dtype=self.dtype)

        for gpu_pattern in self.gpu_patterns:
            if gpu_pattern['n_observed'] == 0:
                continue

            obs_idx = gpu_pattern['observed_indices']
            mu_k = mu_gpu[obs_idx]
            sigma_k = sigma_gpu[obs_idx][:, obs_idx]

            contrib = self._compute_pattern_contribution_gpu(
                gpu_pattern, mu_k, sigma_k
            )

            obj_value = obj_value + contrib

        return obj_value.squeeze()

    def _compute_pattern_contribution_gpu(self, pattern: Dict,
                                         mu_k: Any,
                                         sigma_k: Any) -> Any:
        """
        Compute pattern contribution with FP64 precision.

        CRITICAL: No constant term n*p*log(2pi) to match CPU.
        Computes: n_k * [log|Sigma_k| + tr(Sigma_k^-1 * S_k)]
        """
        torch = self.torch

        n_obs = pattern['n_obs']

        # Log determinant term
        L_k = torch.linalg.cholesky(sigma_k)
        log_det_term = 2.0 * torch.sum(torch.log(torch.diag(L_k)))

        # Compute sample covariance
        data_centered = pattern['data'] - mu_k
        S_k = (data_centered.T @ data_centered) / n_obs

        # Trace term
        X = torch.linalg.solve(sigma_k, S_k)
        trace_term = torch.trace(X)

        # Total contribution: n_obs * [log|Sigma| + tr(Sigma^-1 S)]
        total_contribution = n_obs * (log_det_term + trace_term)

        return total_contribution

    def _unpack_gpu(self, theta_gpu: Any) -> Tuple[Any, Any]:
        """Unpack parameters on GPU with FP64."""
        torch = self.torch
        n = self.n_vars

        mu = theta_gpu[:n]

        # Reconstruct L matrix (lower triangular)
        L = torch.zeros((n, n), device=self.device, dtype=self.dtype)

        # Diagonal elements (exponentiated)
        L.diagonal().copy_(torch.exp(theta_gpu[n:2*n]))

        # Off-diagonal elements
        idx = 2 * n
        for j in range(n):
            for i in range(j + 1, n):
                L[i, j] = theta_gpu[idx]
                idx += 1

        # Compute Sigma = LL'
        sigma = L @ L.T

        # Ensure symmetry
        sigma = 0.5 * (sigma + sigma.T)

        return mu, sigma

    def compute_gradient(self, theta: np.ndarray) -> np.ndarray:
        """Compute gradient using automatic differentiation."""
        torch = self.torch

        theta_gpu = torch.tensor(
            theta,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )

        obj_value = self._torch_objective(theta_gpu)

        obj_value.backward()

        return theta_gpu.grad.cpu().numpy()

    def compute_hessian(self, theta: np.ndarray) -> np.ndarray:
        """Compute Hessian matrix using automatic differentiation."""
        torch = self.torch

        theta_gpu = torch.tensor(
            theta,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )

        obj_value = self._torch_objective(theta_gpu)
        grad = torch.autograd.grad(
            obj_value, theta_gpu, create_graph=True
        )[0]

        n = len(theta)
        hessian = torch.zeros((n, n), device=self.device, dtype=self.dtype)

        for i in range(n):
            grad2 = torch.autograd.grad(
                grad[i], theta_gpu, retain_graph=True
            )[0]
            hessian[i] = grad2

        hessian = 0.5 * (hessian + hessian.T)

        return hessian.cpu().numpy()

    def extract_parameters(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Extract mean and covariance from parameter vector."""
        mu, sigma = self.parameterization.unpack(theta)
        neg2_loglik = self.compute_objective(theta)
        loglik = -0.5 * neg2_loglik
        return mu, sigma, loglik

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.device.type == 'cuda':
            self.torch.cuda.empty_cache()
