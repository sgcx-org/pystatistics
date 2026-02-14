"""
GPU backend for linear regression using PyTorch.

Performance path for large problems - validated against CPU reference.
Supports CUDA (Linux/Windows) and MPS (macOS Apple Silicon).
"""

from typing import Any
import numpy as np

from pystatistics.core.result import Result
from pystatistics.core.exceptions import NumericalError
from pystatistics.core.compute.timing import Timer
from pystatistics.regression.design import Design
from pystatistics.regression.solution import LinearParams

# Condition number threshold above which Cholesky on normal equations
# is numerically unreliable. cond(X'X) = cond(X)^2, so at cond(X) = 1e6,
# cond(X'X) = 1e12 — near float64 epsilon and past float32 usability.
GPU_CONDITION_THRESHOLD = 1e6


class GPUQRBackend:
    """
    GPU backend using PyTorch for linear regression.

    Uses Cholesky decomposition on normal equations (X'X)^-1 X'y.
    Faster than QR for well-conditioned problems, but squares the
    condition number — ill-conditioned matrices are detected and refused
    unless force=True.

    FP32 by default for performance on consumer GPUs (RTX series).
    Supports CUDA and MPS (Apple Silicon).
    """

    def __init__(self, use_fp64: bool = False, device: str = 'cuda'):
        """
        Initialize GPU backend.

        Args:
            use_fp64: If True, use FP64 (slow on consumer GPUs).
                     If False, use FP32 (fast, statistically equivalent).
            device: GPU device type ('cuda', 'cuda:0', 'mps')
        """
        import torch

        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA not available. Install PyTorch with CUDA support, "
                    "or use backend='cpu'."
                )
            self.device = torch.device(device)
            self.dtype = torch.float64 if use_fp64 else torch.float32
            self.use_fp64 = use_fp64
            props = torch.cuda.get_device_properties(self.device)
            self.device_name = props.name
            self.device_memory = props.total_memory / 1e9

        elif device == 'mps':
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                raise RuntimeError(
                    "MPS not available. Requires macOS with Apple Silicon "
                    "and PyTorch with MPS support."
                )
            if use_fp64:
                raise RuntimeError(
                    "MPS does not support float64. Use use_fp64=False "
                    "or use backend='cpu' for double precision."
                )
            self.device = torch.device('mps')
            self.dtype = torch.float32
            self.use_fp64 = False
            self.device_name = 'Apple Silicon GPU (MPS)'
            self.device_memory = None

        else:
            raise ValueError(
                f"Unknown GPU device: {device!r}. Use 'cuda' or 'mps'."
            )

    @property
    def name(self) -> str:
        precision = "fp64" if self.use_fp64 else "fp32"
        return f'gpu_qr_{precision}'

    def solve(self, design: Design, force: bool = False) -> Result[LinearParams]:
        """
        Solve linear regression on GPU using Cholesky method.

        Uses normal equations (X'X)^-1 X'y via Cholesky decomposition.
        Faster than QR, numerically stable for well-conditioned real data.

        Args:
            design: Regression design (X matrix + y vector)
            force: If True, proceed even if the design matrix is
                ill-conditioned. If False (default), raises NumericalError
                when condition number exceeds threshold.

        Returns:
            Result[LinearParams]

        Raises:
            NumericalError: If design matrix is ill-conditioned and
                force=False.
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

        # Condition number check via singular values
        # svdvals not implemented on MPS — fall back to CPU for this check
        with timer.section('condition_check'):
            try:
                sv = torch.linalg.svdvals(X)
            except (NotImplementedError, RuntimeError):
                sv = torch.linalg.svdvals(X.cpu()).to(X.device)
            sv_min = float(sv[-1].item())
            sv_max = float(sv[0].item())
            cond = sv_max / sv_min if sv_min > 0 else float('inf')

        if cond > GPU_CONDITION_THRESHOLD and not force:
            timer.stop()
            raise NumericalError(
                f"Design matrix is ill-conditioned (condition number: {cond:.2e}). "
                f"Cholesky decomposition on normal equations likely to be "
                f"numerically unstable.\n"
                f"Options:\n"
                f"  - Use backend='cpu' for QR decomposition "
                f"(slower, numerically stable)\n"
                f"  - Use ridge regression (different estimator)\n"
                f"  - Pass force=True to proceed with Cholesky anyway"
            )

        # Compute normal equations
        with timer.section('normal_equations'):
            XtX = X.T @ X
            Xty = X.T @ y

        # Cholesky solve with lstsq fallback
        warnings_list = []
        with timer.section('cholesky_solve'):
            cholesky_succeeded = False
            try:
                L = torch.linalg.cholesky(XtX)
                z = torch.linalg.solve_triangular(
                    L, Xty.unsqueeze(1), upper=False
                ).squeeze(1)
                coef_gpu = torch.linalg.solve_triangular(
                    L.T, z.unsqueeze(1), upper=True
                ).squeeze(1)
                cholesky_succeeded = True
                effective_rank = p
            except torch._C._LinAlgError:
                # Cholesky failed — X'X not positive definite
                # lstsq may not be supported on MPS; fall back to CPU
                if self.device.type == 'mps':
                    lstsq_result = torch.linalg.lstsq(
                        X.cpu(), y.cpu()
                    )
                    coef_gpu = lstsq_result.solution.to(self.device)
                else:
                    lstsq_result = torch.linalg.lstsq(X, y)
                    coef_gpu = lstsq_result.solution

                # Determine rank from singular values (already computed)
                threshold = max(n, p) * torch.finfo(self.dtype).eps * sv_max
                effective_rank = int((sv > threshold).sum().item())

        if effective_rank < p:
            warnings_list.append(
                f"Design matrix is rank-deficient (rank={effective_rank}, "
                f"p={p}). Coefficients may not be unique."
            )

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
            rank=effective_rank,
            df_residual=n - effective_rank,
        )

        return Result(
            params=params,
            info={
                'method': 'cholesky' if cholesky_succeeded else 'lstsq',
                'device': str(self.device),
                'dtype': str(self.dtype),
                'device_name': self.device_name,
                'condition_number': cond,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=tuple(warnings_list),
        )
