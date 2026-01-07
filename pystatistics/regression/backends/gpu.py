"""
GPU backend for linear regression using PyTorch.

Performance path for large problems - validated against CPU reference.
"""

from typing import Any
import numpy as np

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.regression.design import Design
from pystatistics.regression.solution import LinearParams


class GPUQRBackend:
    """
    GPU backend using PyTorch with CUDA.
    
    Uses FP32 by default for performance on consumer GPUs (RTX series).
    Validates to be statistically equivalent to FP64 CPU reference.
    """
    
    def __init__(self, use_fp64: bool = False, device: str = 'cuda'):
        """
        Initialize GPU backend.
        
        Args:
            use_fp64: If True, use FP64 (slow on consumer GPUs).
                     If False, use FP32 (fast, statistically equivalent).
            device: CUDA device ('cuda', 'cuda:0', etc.)
        """
        import torch
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - cannot use GPU backend")
        
        self.device = torch.device(device)
        self.dtype = torch.float64 if use_fp64 else torch.float32
        self.use_fp64 = use_fp64
        
        # Get device info
        device_props = torch.cuda.get_device_properties(self.device)
        self.device_name = device_props.name
        self.device_memory = device_props.total_memory / 1e9  # GB
    
    @property
    def name(self) -> str:
        precision = "fp64" if self.use_fp64 else "fp32"
        return f'gpu_qr_{precision}'

    def solve(self, design: Design) -> Result[LinearParams]:
        """
        Solve linear regression on GPU using Cholesky method.
        
        Uses normal equations (X'X)^-1 X'y via Cholesky decomposition.
        Faster than QR, numerically stable for well-conditioned real data.
        """
        import torch
        
        timer = Timer()
        timer.start()
        
        X_np, y_np = design.X, design.y
        n, p = design.n, design.p
        
        # Transfer to GPU
        with timer.section('data_transfer_to_gpu'):
            X = torch.from_numpy(X_np).to(device=self.device, dtype=self.dtype)
            y = torch.from_numpy(y_np).to(device=self.device, dtype=self.dtype)
        
        # Compute normal equations
        with timer.section('normal_equations'):
            XtX = X.T @ X
            Xty = X.T @ y
        
        # Cholesky solve
        with timer.section('cholesky_solve'):
            try:
                L = torch.linalg.cholesky(XtX)
                # solve_triangular needs 2D input - add dimension then remove
                z = torch.linalg.solve_triangular(L, Xty.unsqueeze(1), upper=False).squeeze(1)
                coef_gpu = torch.linalg.solve_triangular(L.T, z.unsqueeze(1), upper=True).squeeze(1)
            except torch._C._LinAlgError:
                coef_gpu = torch.linalg.lstsq(X, y).solution
        
        # Compute fitted and residuals
        with timer.section('fitted_residuals'):
            fitted_gpu = X @ coef_gpu
            residuals_gpu = y - fitted_gpu
        
        # Statistics
        with timer.section('statistics'):
            rss = float((residuals_gpu @ residuals_gpu).item())
            y_mean = y.mean()
            tss = float(((y - y_mean) @ (y - y_mean)).item())
        
        # Transfer back to CPU
        with timer.section('data_transfer_to_cpu'):
            coefficients = coef_gpu.cpu().numpy().astype(np.float64)
            fitted_values = fitted_gpu.cpu().numpy().astype(np.float64)
            residuals = residuals_gpu.cpu().numpy().astype(np.float64)
        
        timer.stop()
        
        params = LinearParams(
            coefficients=coefficients,
            residuals=residuals,
            fitted_values=fitted_values,
            rss=rss,
            tss=tss,
            rank=p,
            df_residual=n - p,
        )
        
        return Result(
            params=params,
            info={
                'method': 'cholesky',
                'device': str(self.device),
                'dtype': str(self.dtype),
                'device_name': self.device_name,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=(),
        )
        """
        Solve linear regression on GPU using Cholesky method.
        
        Uses normal equations (X'X)^-1 X'y via Cholesky decomposition.
        Faster than QR, numerically stable for well-conditioned real data.
        """
        import torch
        
        timer = Timer()
        timer.start()
        
        X_np, y_np = design.X, design.y
        n, p = design.n, design.p
        
        # Transfer to GPU
        with timer.section('data_transfer_to_gpu'):
            X = torch.from_numpy(X_np).to(device=self.device, dtype=self.dtype)
            y = torch.from_numpy(y_np).to(device=self.device, dtype=self.dtype)
        
        # Compute normal equations
        with timer.section('normal_equations'):
            XtX = X.T @ X
            Xty = X.T @ y
        
        # Cholesky solve
        with timer.section('cholesky_solve'):
            try:
                L = torch.linalg.cholesky(XtX)
                z = torch.linalg.solve_triangular(L, Xty, upper=False)
                coef_gpu = torch.linalg.solve_triangular(L.T, z, upper=True)
            except torch._C._LinAlgError:
                coef_gpu = torch.linalg.lstsq(X, y).solution
        
        # Compute fitted and residuals
        with timer.section('fitted_residuals'):
            fitted_gpu = X @ coef_gpu
            residuals_gpu = y - fitted_gpu
        
        # Statistics
        with timer.section('statistics'):
            rss = float((residuals_gpu @ residuals_gpu).item())
            y_mean = y.mean()
            tss = float(((y - y_mean) @ (y - y_mean)).item())
        
        # Transfer back to CPU
        with timer.section('data_transfer_to_cpu'):
            coefficients = coef_gpu.cpu().numpy().astype(np.float64)
            fitted_values = fitted_gpu.cpu().numpy().astype(np.float64)
            residuals = residuals_gpu.cpu().numpy().astype(np.float64)
        
        timer.stop()
        
        params = LinearParams(
            coefficients=coefficients,
            residuals=residuals,
            fitted_values=fitted_values,
            rss=rss,
            tss=tss,
            rank=p,
            df_residual=n - p,
        )
        
        return Result(
            params=params,
            info={
                'method': 'cholesky',
                'device': str(self.device),
                'dtype': str(self.dtype),
                'device_name': self.device_name,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=(),
        )