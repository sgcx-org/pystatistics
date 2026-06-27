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
from pystatistics.core.exceptions import NumericalError, ValidationError
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

    def __init__(self, device: str = 'cuda', use_fp64: bool = False):
        """Initialize GPU IRLS backend.

        Args:
            device: GPU device type ('cuda', 'cuda:0', 'mps')
            use_fp64: Run the IRLS in float64. Valid only on CUDA (Metal/MPS has
                no float64). This is a *correctness* path — numerically equivalent
                to the CPU reference, but slow on consumer GPUs (fast double
                precision needs a data-center card). The default float32 path is
                the speed path. Selected via backend='gpu_fp64'.
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
            props = torch.cuda.get_device_properties(self.device)
            self.device_name = props.name

        elif device == 'mps':
            if use_fp64:
                from pystatistics.core.compute.backend import FP64_REQUIRES_CUDA_MSG
                raise RuntimeError(FP64_REQUIRES_CUDA_MSG)
            self.device = torch.device('mps')
            self.dtype = torch.float32
            self.device_name = 'Apple Silicon GPU (MPS)'

        else:
            raise ValidationError(
                f"Unknown GPU device: {device!r}. Use 'cuda' or 'mps'."
            )
        self.use_fp64 = use_fp64

    @property
    def name(self) -> str:
        return 'gpu_irls_fp64' if self.use_fp64 else 'gpu_irls_fp32'

    def solve(
        self,
        design: Design,
        family: Family,
        tol: float = 1e-8,
        max_iter: int = 25,
        force: bool = False,
        penalty: 'NDArray[np.float64] | None' = None,
    ) -> Result[GLMParams]:
        """Run IRLS on GPU to fit the GLM.

        ``penalty`` (optional, length p): an L2 ridge penalty added to the diagonal
        of XᵀWX each iteration (``diag(penalty)``). It both regularizes the fit and
        makes the float32 Cholesky well-conditioned — the intended way to run a
        stable GLM on the GPU at very large scale. The caller is responsible for
        standardizing the design and back-transforming coefficients (see
        ``solvers._fit_glm_ridge``); the design passed here is the standardized one.

        Args:
            design: Design object with X and y
            family: GLM family specification
            tol: Convergence tolerance (relative deviance change)
            max_iter: Maximum IRLS iterations
            force: If True, return the GPU result even when IRLS does not
                converge in float32 (the caller accepts a possibly-inaccurate
                fit). If False (default), a non-converged fit raises
                NumericalError instead of returning unreliable coefficients.

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
            # Optional ridge penalty added to the XᵀWX diagonal (stabilizes fp32).
            pen_gpu = None
            if penalty is not None:
                pen_gpu = torch.from_numpy(
                    np.asarray(penalty, dtype=np.float64)
                ).to(device=self.device, dtype=self.dtype)
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
        solve_failed = False
        # Initial deviance on CPU (family methods use numpy)
        dev_old = family.deviance(y_np, mu_np, wt_np)
        dev_new = dev_old

        # float64 host-side state for μ/η. The ridge path maintains these across
        # iterations (mixed precision: float32 solve, float64 working quantities)
        # so the convergence signal isn't degraded by a float32 GPU round-trip.
        mu_cpu = mu_np.astype(np.float64)
        eta_cpu = eta_np.astype(np.float64)

        # Effective convergence tolerance. The deviance is a sum of n terms, so in
        # float32 the relative change cannot fall below a round-off floor ~√n·eps —
        # the fit then sits in a tiny benign 2-cycle around the true solution. For
        # the RIDGE path the penalty guarantees the float32 fit is accurate, so we
        # accept convergence at this float32 floor rather than demand an unreachable
        # float64 tolerance. Unpenalized GLM (where float32 non-convergence signals
        # a genuinely bad, ill-conditioned fit) keeps the strict tolerance and fails
        # loud. float64 (CUDA) keeps the strict tolerance — it can reach it.
        eff_tol = tol
        if penalty is not None and self.dtype == torch.float32:
            fp32_floor = float(np.sqrt(n)) * float(torch.finfo(torch.float32).eps)
            eff_tol = max(tol, 2.0 * fp32_floor)

        with timer.section('irls'):
            for iteration in range(1, max_iter + 1):
                # Transfer μ, η to CPU for family/link computations. The ridge path
                # already holds them in float64 on the host (updated below), so it
                # skips this float32 round-trip that would truncate them.
                if pen_gpu is None:
                    mu_cpu = mu_gpu.cpu().numpy().astype(np.float64)
                    eta_cpu = eta_gpu.cpu().numpy().astype(np.float64)

                # Working quantities (CPU, using family/link)
                mu_eta_val = link.mu_eta(eta_cpu)    # dμ/dη
                var_mu = family.variance(mu_cpu)

                # Working response: z = η + (y - μ) / (dμ/dη)
                z_np = eta_cpu + (y_np - mu_cpu) / mu_eta_val

                # Working weights: w = wt * (dμ/dη)² / V(μ)
                w_np = wt_np * (mu_eta_val ** 2) / var_mu
                # NUMERICAL GUARD: prevents division by zero in weighted regression
                w_np = np.maximum(w_np, 1e-30)

                # Transfer to GPU
                z_gpu = torch.from_numpy(z_np).to(device=self.device, dtype=self.dtype)
                w_gpu = torch.from_numpy(w_np).to(device=self.device, dtype=self.dtype)

                # WLS via the weighted normal equations (Xᵀ W X) β = Xᵀ W z,
                # solved by Cholesky + two triangular solves. This is
                # mathematically the same weighted least-squares step as the
                # √w·X / √w·z lstsq formulation, but uses only Cholesky and
                # triangular solves — which ARE supported on MPS (lstsq is NOT,
                # so the old path failed outright on Apple Silicon) and are far
                # cheaper than a full lstsq on the n×p system on CUDA. The p×p
                # Gram solve also stays on-device, removing the lstsq's larger
                # work and (on MPS) its unsupported-op fallback.
                wX_gpu = w_gpu.unsqueeze(1) * X_gpu          # W X  (n×p)
                XtWX = X_gpu.T @ wX_gpu                       # p×p
                XtWz = X_gpu.T @ (w_gpu * z_gpu)             # p
                if pen_gpu is not None:
                    # Ridge: add λ to the diagonal. Raises the smallest eigenvalue,
                    # so the float32 Cholesky is well-conditioned and stable.
                    XtWX = XtWX + torch.diag(pen_gpu)
                try:
                    L = torch.linalg.cholesky(XtWX)
                    sol = torch.linalg.solve_triangular(
                        L, XtWz.unsqueeze(1), upper=False)
                    coef_gpu = torch.linalg.solve_triangular(
                        L.T, sol, upper=True).squeeze(1)
                except torch._C._LinAlgError:
                    # The float32 weighted normal equations are not
                    # positive-definite — typically an ill-conditioned design at
                    # large n in float32. Stop and fall back to the exact CPU path
                    # below rather than raising; the caller still gets a result.
                    solve_failed = True
                    break

                # Update η = X @ β and μ = linkinv(η).
                if pen_gpu is not None:
                    # Ridge path: the penalty makes the float32 solve accurate, so
                    # the coefficients are genuinely fp32-precision. Evaluate η in
                    # float64 on the host (X is well-conditioned here) so the
                    # convergence signal — the deviance — is not polluted by float32
                    # matmul noise. This is what lets a ridge GLM converge cleanly
                    # on the GPU at very large n. (Plain, unpenalized GLM keeps the
                    # float32 η below, so an ill-conditioned fit still fails loud
                    # rather than converging to a masked-wrong point.)
                    coef_cpu = coef_gpu.detach().cpu().numpy().astype(np.float64)
                    eta_cpu = X_np @ coef_cpu
                    eta_gpu = torch.from_numpy(eta_cpu).to(
                        device=self.device, dtype=self.dtype)
                else:
                    eta_gpu = X_gpu @ coef_gpu
                    eta_cpu = eta_gpu.cpu().numpy().astype(np.float64)

                mu_cpu = link.linkinv(eta_cpu)
                mu_gpu = torch.from_numpy(mu_cpu).to(
                    device=self.device, dtype=self.dtype
                )

                # Compute deviance on CPU
                dev_new = family.deviance(y_np, mu_cpu, wt_np)

                # R's convergence criterion (eff_tol = strict tol, or the float32
                # round-off floor for the ridge path — see eff_tol above).
                if abs(dev_new - dev_old) / (abs(dev_old) + 0.1) < eff_tol:
                    converged = True
                    break

                dev_old = dev_new

        # A broken inner solve leaves no usable coefficients, so it always raises
        # (force cannot salvage it). Plain non-convergence raises unless force=True.
        if solve_failed or (not converged and not force):
            # The float32 GPU IRLS did not reach the convergence tolerance, or the
            # inner solve broke down. This happens for log-link families
            # (Poisson/Gamma) at large sample sizes, where the float32 Cholesky on
            # the normal equations XᵀWX is not stable enough — most often on Apple
            # Silicon (MPS), which has no float64. The non-converged float32
            # coefficients are NOT a usable fit. Fail loud and lay out the honest
            # options, rather than silently returning wrong numbers, silently
            # switching to the CPU, or papering over it with numerical tricks.
            # (This mirrors the GPU OLS backend's ill-conditioning message.)
            reason = ("the float32 Cholesky inner solve broke down (XᵀWX not "
                      "positive-definite in float32)"
                      if solve_failed
                      else f"IRLS did not converge within {max_iter} iterations in "
                           "float32")
            raise NumericalError(
                f"GPU GLM did not produce a reliable fit: {reason}. This is "
                f"expected for log-link families (Poisson/Gamma) at large sample "
                f"sizes, where float32 Cholesky on the normal equations is "
                f"unstable (especially on Apple Silicon / MPS, which has no "
                f"float64).\n"
                f"Options:\n"
                f"  - Use backend='cpu' for a correct double-precision fit "
                f"(slower)\n"
                f"  - Use a ridge-penalized fit (a different, better-conditioned "
                f"estimator)\n"
                f"  - Pass force=True to return the float32 GPU fit anyway "
                f"(fast, but may be inaccurate / unstable)"
            )

        if not converged:
            # force=True: the caller explicitly accepts a possibly-inaccurate
            # float32 fit. Return it, but flag it loudly on the result.
            warnings_list.append(
                f"GPU GLM IRLS did not converge in {max_iter} float32 iterations; "
                f"force=True so the (possibly inaccurate) GPU fit is returned. "
                f"Deviance={dev_new:.6f}."
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
            # NUMERICAL GUARD: prevents division by zero in weighted regression
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
                'method': 'ridge_irls_gpu' if penalty is not None else 'irls_cholesky_gpu',
                'penalized': penalty is not None,
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
        elif family.name == 'Gamma':
            mu_c = np.maximum(mu, 1e-10)
            y_safe = np.maximum(y, 1e-10)
            d = 2.0 * ((y - mu_c) / mu_c - np.log(y_safe / mu_c))
        elif family.name == 'negative.binomial':
            theta = family.theta
            mu_c = np.maximum(mu, 1e-10)
            with np.errstate(divide='ignore', invalid='ignore'):
                term1 = np.where(y > 0, y * np.log(y / mu_c), 0.0)
                term2 = (y + theta) * np.log((y + theta) / (mu_c + theta))
            d = 2.0 * (term1 - term2)
        else:
            n = len(y)
            d = np.zeros(n, dtype=np.float64)
            ones = np.ones(1, dtype=np.float64)
            for i in range(n):
                yi = y[i:i+1]
                mui = mu[i:i+1]
                d[i] = family.deviance(yi, mui, ones)

        return sign * np.sqrt(np.maximum(wt * d, 0.0))
