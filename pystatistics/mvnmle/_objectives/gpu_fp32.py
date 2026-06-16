"""
GPU FP32 objective function using standard Cholesky parameterization.

Optimized for consumer GPUs (RTX series) and Apple Metal (MPS).
Uses PyTorch autodiff for analytical gradients.
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Any
from pystatistics.core.exceptions import NumericalError

from .base import MLEObjectiveBase
from .parameterizations import (
    CholeskyParameterization,
    BoundedCholeskyParameterization
)
from ._batched_cholesky import (
    build_batched_constants,
    to_torch,
    batched_neg2_loglik,
    unpack_cholesky,
)


class GPUObjectiveFP32(MLEObjectiveBase):
    """
    GPU-accelerated MLE objective using FP32 precision.

    Designed for consumer GPUs where FP64 is severely limited.
    Uses standard Cholesky parameterization for autodiff compatibility.
    Supports both CUDA and MPS (Apple Silicon).
    """

    def __init__(self, data: np.ndarray,
                 device: Optional[str] = None,
                 use_bounded: bool = False):
        """
        Initialize GPU FP32 objective.

        Parameters
        ----------
        data : np.ndarray
            Input data with missing values as np.nan
        device : str or None
            Device to use ('cuda', 'mps', or None for auto)
        use_bounded : bool
            Whether to use bounded parameterization for stability
        """
        super().__init__(data, skip_validation=False)

        # Import PyTorch
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for GPU objectives. "
                "Install with: pip install torch"
            )

        # Create parameterization
        if use_bounded:
            self.parameterization = BoundedCholeskyParameterization(
                self.n_vars,
                var_min=0.01,
                var_max=100.0,
                corr_max=0.95
            )
        else:
            self.parameterization = CholeskyParameterization(self.n_vars)

        self.n_params = self.parameterization.n_params

        # Select device
        self.device = self._select_device(device)

        # FP32 settings
        self.dtype = torch.float32
        self.eps = 1e-6  # Looser epsilon for FP32

        # Transfer pattern data to GPU
        self._prepare_gpu_data()

    def _select_device(self, requested_device: Optional[str]) -> Any:
        """Select appropriate GPU device.

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
                # Verify MPS is available
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    warnings.warn("MPS not available, falling back to CPU")
                    return torch.device('cpu')
            return device

        # Auto-select: silent fallback to CPU is acceptable here because this
        # code path is only reached when backend='auto' (best available).
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            warnings.warn("No GPU available, using CPU (will be slow)")
            return torch.device('cpu')

    def _prepare_gpu_data(self) -> None:
        """Precompute padded per-pattern sufficient statistics on the GPU.

        Built once (patterns are fixed across optimiser iterations); every
        objective/gradient evaluation then reuses them in a single batched
        Cholesky instead of looping over patterns.
        """
        consts = build_batched_constants(self.patterns, self.n_vars)
        self._consts = to_torch(consts, self.torch, self.device, self.dtype)

    def get_initial_parameters(self) -> np.ndarray:
        """Get initial parameters — raises on ill-conditioned covariance."""
        sample_cov_regularized = self.sample_cov.copy()

        try:
            eigenvals = np.linalg.eigvalsh(sample_cov_regularized)
            min_eig = np.min(eigenvals)
            max_eig = np.max(eigenvals)

            if min_eig < 1e-6 or max_eig / min_eig > 1e10:
                raise NumericalError(
                    f"Initial sample covariance is ill-conditioned "
                    f"(min eigenvalue={min_eig:.2e}, condition number={max_eig/min_eig:.2e}). "
                    f"Check for collinear variables, remove constant columns, "
                    f"or scale your data before fitting."
                )
        except np.linalg.LinAlgError as e:
            raise NumericalError(
                f"Eigenvalue decomposition of the sample covariance failed: {e}. "
                f"Check for collinear variables, remove constant columns, "
                f"or scale your data before fitting."
            ) from e

        return self.parameterization.get_initial_parameters(
            self.sample_mean,
            sample_cov_regularized
        )

    def compute_objective(self, theta: np.ndarray) -> float:
        """
        Compute -2 * log-likelihood using GPU.

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
            obj_value = self._torch_objective(theta_gpu)

        return obj_value.item()

    def _torch_objective(self, theta_gpu: Any) -> Any:
        """
        PyTorch implementation of the objective function.

        Returns the scalar objective (-2 * log-likelihood, for R compatibility),
        evaluated over all missingness patterns in a single batched Cholesky.

        CRITICAL: matches the CPU objective — no n*p*log(2pi) constant; each
        pattern contributes n_k * log|Sigma_k| + tr(Sigma_k^-1 * M_k).
        """
        mu_gpu, sigma_gpu = self._unpack_gpu(theta_gpu)
        return batched_neg2_loglik(self.torch, mu_gpu, sigma_gpu,
                                   self._consts, self.eps)

    def _unpack_gpu(self, theta_gpu: Any) -> Tuple[Any, Any]:
        """
        Unpack parameters on GPU maintaining gradient flow.

        The standard-Cholesky case delegates to the shared
        :func:`unpack_cholesky` so the FP32 and FP64 paths reconstruct Sigma
        identically (and identically to ``CholeskyParameterization.unpack``).
        The bounded parameterization keeps its own sigmoid/tanh reconstruction,
        but reuses the same row-major off-diagonal ordering.
        """
        torch = self.torch
        n = self.n_vars

        if not isinstance(self.parameterization, BoundedCholeskyParameterization):
            return unpack_cholesky(torch, theta_gpu, n)

        # Bounded Cholesky: diagonal via sigmoid, off-diagonal via tanh.
        mu = theta_gpu[:n]
        L = torch.zeros((n, n), device=self.device, dtype=self.dtype)

        diag_unbounded = theta_gpu[n:2*n]
        var_min = self.parameterization.var_min
        var_max = self.parameterization.var_max
        diag_vars = var_min + (var_max - var_min) * torch.sigmoid(diag_unbounded)
        L.diagonal().copy_(torch.sqrt(diag_vars))

        idx = 2 * n
        tril_indices = torch.tril_indices(n, n, offset=-1, device=self.device)
        if len(tril_indices[0]) > 0:
            tril_unbounded = theta_gpu[idx:]
            corr_max = self.parameterization.corr_max
            corr_vals = corr_max * torch.tanh(tril_unbounded)

            i, j = tril_indices
            L[tril_indices[0], tril_indices[1]] = corr_vals * torch.sqrt(L[i, i] * L[j, j])

        sigma = torch.matmul(L, L.T)
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

        grad_tensor = torch.autograd.grad(
            outputs=obj_value,
            inputs=theta_gpu,
            create_graph=False,
            retain_graph=False
        )[0]

        return grad_tensor.cpu().numpy()

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
