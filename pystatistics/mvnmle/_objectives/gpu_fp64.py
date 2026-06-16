"""
GPU objective function using standard Cholesky parameterization with FP64.

For data center GPUs (A100, H100, V100, RTX 5070 Ti) with FP64 support.
Enables Newton-CG optimization with analytical Hessians.
CUDA only - MPS does not support FP64.
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Any
from .base import MLEObjectiveBase, PatternData
from .parameterizations import CholeskyParameterization
from ._batched_cholesky import (
    build_batched_constants,
    to_torch,
    batched_neg2_loglik,
    unpack_cholesky,
    objective_value,
    accumulate_gradient,
    auto_chunk_size,
)


class GPUObjectiveFP64(MLEObjectiveBase):
    """
    GPU-accelerated MLE objective with FP64 precision.

    Designed for data center GPUs with FP64 support.
    Enables Newton-CG optimization with analytical second-order derivatives.
    CUDA only.
    """

    def __init__(self, data: np.ndarray,
                 device: Optional[str] = None,
                 validate: bool = True,
                 chunk_size: Optional[int] = None):
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
        chunk_size : int or None
            Missingness patterns processed per chunk in the objective/gradient;
            ``None`` auto-sizes to a memory budget. See GPUObjectiveFP32.
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

        # Patterns processed per chunk (auto-sized to a memory budget unless set).
        self.chunk_size = (chunk_size if chunk_size
                           else auto_chunk_size(self.n_vars, 8))

        # Transfer pattern data to GPU
        self._prepare_gpu_data()

    def _select_device(self, requested_device: Optional[str]) -> Any:
        """Select appropriate GPU device for FP64.

        Note: The caller (solvers._get_backend) handles the backend='gpu' vs
        'auto' distinction before instantiating this class. When backend='gpu',
        the solver calls select_device('gpu') which raises RuntimeError if no
        GPU is available — so this class is never constructed with an invalid
        explicit GPU request. The warnings below are therefore only reachable
        in 'auto' mode, where silent CPU fallback is the intended behavior.
        """
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

        # Auto-select: prefer CUDA for FP64. Silent fallback to CPU is
        # acceptable here because this path is only reached via backend='auto'.
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            warnings.warn(
                "No CUDA device available for FP64. "
                "Using CPU fallback (will be very slow)."
            )
            return torch.device('cpu')

    def _prepare_gpu_data(self) -> None:
        """Precompute padded per-pattern sufficient statistics on the GPU (FP64).

        Built once; reused by every batched objective/gradient evaluation.
        """
        consts = build_batched_constants(self.patterns, self.n_vars)
        self._consts = to_torch(consts, self.torch, self.device, self.dtype)

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

        with torch.no_grad():
            theta_gpu = torch.tensor(theta, device=self.device, dtype=self.dtype)
            mu_gpu, sigma_gpu = self._unpack_gpu(theta_gpu)
            obj_value = objective_value(torch, mu_gpu, sigma_gpu,
                                        self._consts, self.eps, self.chunk_size)

        return obj_value.item()

    def _torch_objective(self, theta_gpu: Any) -> Any:
        """PyTorch implementation of the objective (batched over patterns).

        CRITICAL: no n*p*log(2pi) constant, to match CPU; each pattern
        contributes n_k * [log|Sigma_k| + tr(Sigma_k^-1 * M_k)].
        """
        mu_gpu, sigma_gpu = self._unpack_gpu(theta_gpu)
        return batched_neg2_loglik(self.torch, mu_gpu, sigma_gpu,
                                   self._consts, self.eps)

    def _unpack_gpu(self, theta_gpu: Any) -> Tuple[Any, Any]:
        """Unpack standard-Cholesky parameters into ``(mu, Sigma)`` on the GPU.

        Delegates to the shared :func:`unpack_cholesky` so the FP64 and FP32
        paths reconstruct Sigma identically (and identically to the canonical
        ``CholeskyParameterization.unpack``).
        """
        return unpack_cholesky(self.torch, theta_gpu, self.n_vars)

    def compute_gradient(self, theta: np.ndarray) -> np.ndarray:
        """Compute gradient using automatic differentiation (chunked over
        patterns to bound peak memory)."""
        torch = self.torch

        theta_gpu = torch.tensor(
            theta,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True
        )

        grad_tensor, _ = accumulate_gradient(
            torch, theta_gpu, self._unpack_gpu,
            self._consts, self.eps, self.chunk_size)

        return grad_tensor.cpu().numpy()

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
