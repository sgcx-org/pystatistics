"""
GPU backend for Generalized Linear Models via IRLS.

Same IRLS algorithm as cpu_glm.py, but the weighted least squares (WLS)
inner step is performed on GPU using PyTorch. Each iteration:
    1. Compute working response z and weights w (GPU)
    2. Form √w·X and √w·z (GPU)
    3. Solve WLS via torch.linalg.lstsq (GPU)
    4. Update η = X @ β, μ = linkinv(η) (GPU)
    5. Compute deviance on GPU, check convergence on CPU (scalar)

Final coefficients, fitted values, and residuals are transferred back
to CPU as float64 numpy arrays. All intermediate IRLS computations
use float32 for performance on consumer GPUs.

Supports CUDA and MPS (Apple Silicon).
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.regression.design import Design
from pystatistics.regression.families import Family
from pystatistics.regression.solution import GLMParams


class GPUIRLSBackend:
    """GPU backend using IRLS with torch.linalg.lstsq inner solve.

    Matches the CPU IRLS algorithm but leverages GPU parallelism for
    the matrix operations. Uses FP32 by default (FP64 not supported on MPS).

    For the WLS step, we solve min_β ||√w·z - √w·X·β||² via lstsq.
    Standard errors are computed on CPU from X'WX after IRLS converges.
    """

    def __init__(self, device: str = 'cuda'):
        """Initialize GPU IRLS backend.

        Args:
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
            self.dtype = torch.float32
            props = torch.cuda.get_device_properties(self.device)
            self.device_name = props.name

        elif device == 'mps':
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                raise RuntimeError(
                    "MPS not available. Requires macOS with Apple Silicon "
                    "and PyTorch with MPS support."
                )
            self.device = torch.device('mps')
            self.dtype = torch.float32
            self.device_name = 'Apple Silicon GPU (MPS)'

        else:
            raise ValueError(
                f"Unknown GPU device: {device!r}. Use 'cuda' or 'mps'."
            )

    @property
    def name(self) -> str:
        return 'gpu_irls_fp32'

    def solve(
        self,
        design: Design,
        family: Family,
        tol: float = 1e-8,
        max_iter: int = 25,
    ) -> Result[GLMParams]:
        """Run IRLS on GPU to fit the GLM.

        Args:
            design: Design object with X and y
            family: GLM family specification
            tol: Convergence tolerance (relative deviance change)
            max_iter: Maximum IRLS iterations

        Returns:
            Result[GLMParams] with coefficients, deviance, residuals, etc.
        """
        import torch

        timer = Timer()
        timer.start()

        X_np, y_np = design.X.astype(np.float64), design.y.astype(np.float64)
        n, p = design.n, design.p
        link = family.link

        # Prior weights
        wt_np = np.ones(n, dtype=np.float64)

        warnings_list: list[str] = []

        # ------------------------------------------------------------------
        # Transfer to GPU
        # ------------------------------------------------------------------
        with timer.section('data_transfer_to_gpu'):
            X_gpu = torch.from_numpy(X_np).to(device=self.device, dtype=self.dtype)
            y_gpu = torch.from_numpy(y_np).to(device=self.device, dtype=self.dtype)
            wt_gpu = torch.ones(n, device=self.device, dtype=self.dtype)

        # ------------------------------------------------------------------
        # Initialize μ and η (on CPU using family/link, then transfer)
        # ------------------------------------------------------------------
        with timer.section('initialize'):
            mu_np = family.initialize(y_np)
            eta_np = link.link(mu_np)
            mu_gpu = torch.from_numpy(mu_np).to(device=self.device, dtype=self.dtype)
            eta_gpu = torch.from_numpy(eta_np).to(device=self.device, dtype=self.dtype)

        # ------------------------------------------------------------------
        # IRLS loop
        # ------------------------------------------------------------------
        converged = False
        # Initial deviance on CPU (family methods use numpy)
        dev_old = family.deviance(y_np, mu_np, wt_np)
        dev_new = dev_old

        with timer.section('irls'):
            for iteration in range(1, max_iter + 1):
                # Transfer μ, η to CPU for family/link computations
                mu_cpu = mu_gpu.cpu().numpy().astype(np.float64)
                eta_cpu = eta_gpu.cpu().numpy().astype(np.float64)

                # Working quantities (CPU, using family/link)
                mu_eta_val = link.mu_eta(eta_cpu)    # dμ/dη
                var_mu = family.variance(mu_cpu)

                # Working response: z = η + (y - μ) / (dμ/dη)
                z_np = eta_cpu + (y_np - mu_cpu) / mu_eta_val

                # Working weights: w = wt * (dμ/dη)² / V(μ)
                w_np = wt_np * (mu_eta_val ** 2) / var_mu
                w_np = np.maximum(w_np, 1e-30)

                # Transfer to GPU
                z_gpu = torch.from_numpy(z_np).to(device=self.device, dtype=self.dtype)
                w_gpu = torch.from_numpy(w_np).to(device=self.device, dtype=self.dtype)

                # WLS via lstsq: transform to √w·X and √w·z
                sqrt_w_gpu = torch.sqrt(w_gpu)
                X_tilde = X_gpu * sqrt_w_gpu.unsqueeze(1)
                z_tilde = z_gpu * sqrt_w_gpu

                # Solve via lstsq (GPU)
                # MPS may not support lstsq — fall back to CPU if needed
                try:
                    lstsq_result = torch.linalg.lstsq(
                        X_tilde, z_tilde.unsqueeze(1)
                    )
                    coef_gpu = lstsq_result.solution.squeeze(1)
                except (NotImplementedError, RuntimeError):
                    # Fallback: solve on CPU
                    lstsq_result = torch.linalg.lstsq(
                        X_tilde.cpu(), z_tilde.unsqueeze(1).cpu()
                    )
                    coef_gpu = lstsq_result.solution.squeeze(1).to(self.device)

                # Update η = X @ β and μ = linkinv(η)
                eta_gpu = X_gpu @ coef_gpu

                # Transfer η to CPU for linkinv
                eta_cpu = eta_gpu.cpu().numpy().astype(np.float64)
                mu_cpu = link.linkinv(eta_cpu)
                mu_gpu = torch.from_numpy(mu_cpu).to(
                    device=self.device, dtype=self.dtype
                )

                # Compute deviance on CPU
                dev_new = family.deviance(y_np, mu_cpu, wt_np)

                # R's convergence criterion
                if abs(dev_new - dev_old) / (abs(dev_old) + 0.1) < tol:
                    converged = True
                    break

                dev_old = dev_new

        if not converged:
            warnings_list.append(
                f"IRLS did not converge in {max_iter} iterations "
                f"(deviance={dev_new:.6f})"
            )

        n_iter = iteration if converged else max_iter
        dev = dev_new

        # ------------------------------------------------------------------
        # Transfer final results to CPU (float64)
        # ------------------------------------------------------------------
        with timer.section('data_transfer_to_cpu'):
            coefficients = coef_gpu.cpu().numpy().astype(np.float64)
            eta_final = (X_gpu @ coef_gpu).cpu().numpy().astype(np.float64)

        # Recompute μ on CPU at float64 for maximum accuracy
        mu_final = link.linkinv(eta_final)

        # ------------------------------------------------------------------
        # Null deviance (intercept-only, matching R's glm.fit)
        # ------------------------------------------------------------------
        with timer.section('null_deviance'):
            null_deviance = self._null_deviance(y_np, wt_np, family)

        # ------------------------------------------------------------------
        # Dispersion
        # ------------------------------------------------------------------
        final_rank = p  # GPU lstsq doesn't give rank easily
        df_residual = n - final_rank
        if family.dispersion_is_fixed:
            dispersion = 1.0
        else:
            dispersion = dev / df_residual if df_residual > 0 else float('nan')

        # ------------------------------------------------------------------
        # AIC
        # ------------------------------------------------------------------
        with timer.section('aic'):
            aic = family.aic(y_np, mu_final, wt_np, final_rank, dispersion)

        # ------------------------------------------------------------------
        # Residuals (CPU, float64)
        # ------------------------------------------------------------------
        with timer.section('residuals'):
            resid_response = y_np - mu_final

            var_mu = family.variance(mu_final)
            resid_pearson = resid_response / np.sqrt(var_mu)

            resid_deviance = self._deviance_residuals(
                y_np, mu_final, wt_np, family
            )

            mu_eta_final = link.mu_eta(eta_final)
            resid_working = (y_np - mu_final) / mu_eta_final

        # ------------------------------------------------------------------
        # X'WX for SE computation (compute on CPU at float64 for accuracy)
        # ------------------------------------------------------------------
        with timer.section('XtWX'):
            w_final = wt_np * (mu_eta_final ** 2) / family.variance(mu_final)
            w_final = np.maximum(w_final, 1e-30)
            XtWX = (X_np * w_final[:, np.newaxis]).T @ X_np

        timer.stop()

        # ------------------------------------------------------------------
        # Build result
        # ------------------------------------------------------------------
        params = GLMParams(
            coefficients=coefficients,
            fitted_values=mu_final,
            linear_predictor=eta_final,
            residuals_working=resid_working,
            residuals_deviance=resid_deviance,
            residuals_pearson=resid_pearson,
            residuals_response=resid_response,
            deviance=dev,
            null_deviance=null_deviance,
            aic=aic,
            dispersion=dispersion,
            rank=final_rank,
            df_residual=df_residual,
            df_null=n - 1,
            n_iter=n_iter,
            converged=converged,
            family_name=family.name,
            link_name=link.name,
        )

        return Result(
            params=params,
            info={
                'method': 'irls_lstsq_gpu',
                'device': str(self.device),
                'dtype': str(self.dtype),
                'device_name': self.device_name,
                'XtWX': XtWX,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=tuple(warnings_list),
        )

    # ------------------------------------------------------------------
    # Helpers (same logic as CPU backend)
    # ------------------------------------------------------------------

    @staticmethod
    def _null_deviance(
        y: NDArray, wt: NDArray, family: Family
    ) -> float:
        """Compute null deviance matching R's glm.fit().

        R's glm.fit() with intercept=FALSE uses mu_null = linkinv(0).
        """
        n = len(y)
        link = family.link
        eta_null = np.zeros(n, dtype=np.float64)
        mu_null = link.linkinv(eta_null)
        return family.deviance(y, mu_null, wt)

    @staticmethod
    def _deviance_residuals(
        y: NDArray, mu: NDArray, wt: NDArray, family: Family
    ) -> NDArray:
        """Compute signed deviance residuals."""
        sign = np.sign(y - mu)

        if family.name == 'gaussian':
            d = (y - mu) ** 2
        elif family.name == 'binomial':
            mu_c = np.clip(mu, 1e-10, 1 - 1e-10)
            with np.errstate(divide='ignore', invalid='ignore'):
                term1 = np.where(y > 0, y * np.log(y / mu_c), 0.0)
                term2 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu_c)), 0.0)
            d = 2.0 * (term1 + term2)
        elif family.name == 'poisson':
            mu_c = np.maximum(mu, 1e-10)
            with np.errstate(divide='ignore', invalid='ignore'):
                term = np.where(y > 0, y * np.log(y / mu_c), 0.0)
            d = 2.0 * (term - (y - mu_c))
        else:
            n = len(y)
            d = np.zeros(n, dtype=np.float64)
            ones = np.ones(1, dtype=np.float64)
            for i in range(n):
                yi = y[i:i+1]
                mui = mu[i:i+1]
                d[i] = family.deviance(yi, mui, ones)

        return sign * np.sqrt(np.maximum(wt * d, 0.0))
