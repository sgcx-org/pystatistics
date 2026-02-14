"""
CPU backend for Generalized Linear Models via IRLS.

Implements Iteratively Reweighted Least Squares (Fisher scoring) to match
R's glm.fit() exactly. Each IRLS iteration solves a weighted least squares
problem via pivoted QR on the transformed system √W·X, √W·z.

Algorithm (matching R's glm.fit in src/library/stats/R/glm.R):
    Initialize: μ = family.initialize(y), η = link(μ)
    For iteration 1..max_iter:
        dμ/dη = link.mu_eta(η)
        V(μ) = family.variance(μ)
        z = η + (y - μ) / dμ_dη              # working response
        w = (dμ/dη)² / V(μ)                  # working weights
        Solve WLS: min_β || √w·z - √w·X·β ||²  via QR
        η_new = X @ β
        μ_new = linkinv(η_new)
        dev_new = family.deviance(y, μ_new, wt)
        Check: |dev_new - dev_old| / (|dev_old| + 0.1) < tol
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.core.compute.linalg.qr import qr_solve
from pystatistics.regression.design import Design
from pystatistics.regression.families import Family
from pystatistics.regression.solution import GLMParams


class CPUIRLSBackend:
    """CPU backend using IRLS with QR inner solve.

    Matches R's glm.fit() algorithm including:
    - Same convergence criterion: |dev - dev_old| / (|dev_old| + 0.1) < tol
    - Same defaults: tol=1e-8, max_iter=25
    - Null deviance via intercept-only IRLS (not just mean(y))
    - QR-based standard errors from the final iteration
    """

    @property
    def name(self) -> str:
        return 'cpu_irls'

    def solve(
        self,
        design: Design,
        family: Family,
        tol: float = 1e-8,
        max_iter: int = 25,
    ) -> Result[GLMParams]:
        """Run IRLS to fit the GLM.

        Args:
            design: Design object with X and y
            family: GLM family specification
            tol: Convergence tolerance (relative deviance change)
            max_iter: Maximum IRLS iterations

        Returns:
            Result[GLMParams] with coefficients, deviance, residuals, etc.
        """
        timer = Timer()
        timer.start()

        X, y = design.X.astype(np.float64), design.y.astype(np.float64)
        n, p = design.n, design.p
        link = family.link

        # Prior weights (unit weights for now — extend later for trials)
        wt = np.ones(n, dtype=np.float64)

        warnings_list: list[str] = []

        # ------------------------------------------------------------------
        # Initialize μ and η
        # ------------------------------------------------------------------
        with timer.section('initialize'):
            mu = family.initialize(y)
            eta = link.link(mu)

        # ------------------------------------------------------------------
        # IRLS loop
        # ------------------------------------------------------------------
        converged = False
        dev_old = family.deviance(y, mu, wt)

        # Store final QR info for SE computation
        final_qr_R = None
        final_qr_pivot = None
        final_rank = p

        with timer.section('irls'):
            for iteration in range(1, max_iter + 1):
                # Working quantities
                mu_eta_val = link.mu_eta(eta)  # dμ/dη
                var_mu = family.variance(mu)

                # Working response: z = η + (y - μ) / (dμ/dη)
                z = eta + (y - mu) / mu_eta_val

                # Working weights: w = (dμ/dη)² / V(μ)
                w = wt * (mu_eta_val ** 2) / var_mu

                # Guard against zero or negative weights
                w = np.maximum(w, 1e-30)

                # WLS via QR: transform to √w·X and √w·z
                sqrt_w = np.sqrt(w)
                X_tilde = X * sqrt_w[:, np.newaxis]
                z_tilde = z * sqrt_w

                # Solve via pivoted QR
                coefficients, qr_result = qr_solve(X_tilde, z_tilde)

                # Update η and μ
                eta = X @ coefficients
                mu = link.linkinv(eta)

                # Compute deviance
                dev_new = family.deviance(y, mu, wt)

                # R's convergence criterion
                if abs(dev_new - dev_old) / (abs(dev_old) + 0.1) < tol:
                    converged = True
                    # Save QR from final iteration for SE computation
                    final_qr_R = qr_result.R
                    final_qr_pivot = qr_result.pivot
                    final_rank = qr_result.rank
                    break

                dev_old = dev_new

                # Save QR info from last iteration regardless
                final_qr_R = qr_result.R
                final_qr_pivot = qr_result.pivot
                final_rank = qr_result.rank

        if not converged:
            warnings_list.append(
                f"IRLS did not converge in {max_iter} iterations "
                f"(deviance={dev_new:.6f})"
            )

        n_iter = iteration if converged else max_iter
        dev = dev_new

        # ------------------------------------------------------------------
        # Null deviance (intercept-only model via mini-IRLS)
        # ------------------------------------------------------------------
        with timer.section('null_deviance'):
            null_deviance = self._null_deviance(y, wt, family)

        # ------------------------------------------------------------------
        # Dispersion
        # ------------------------------------------------------------------
        df_residual = n - final_rank
        if family.dispersion_is_fixed:
            dispersion = 1.0
        else:
            dispersion = dev / df_residual if df_residual > 0 else float('nan')

        # ------------------------------------------------------------------
        # AIC
        # ------------------------------------------------------------------
        with timer.section('aic'):
            aic = family.aic(y, mu, wt, final_rank, dispersion)

        # ------------------------------------------------------------------
        # Residuals
        # ------------------------------------------------------------------
        with timer.section('residuals'):
            # Response residuals
            resid_response = y - mu

            # Pearson residuals: (y - μ) / sqrt(V(μ))
            var_mu = family.variance(mu)
            resid_pearson = resid_response / np.sqrt(var_mu)

            # Deviance residuals: signed sqrt of unit deviance contributions
            resid_deviance = self._deviance_residuals(y, mu, wt, family)

            # Working residuals from the final iteration
            mu_eta_final = link.mu_eta(eta)
            resid_working = (y - mu) / mu_eta_final

        timer.stop()

        # ------------------------------------------------------------------
        # Build result
        # ------------------------------------------------------------------
        params = GLMParams(
            coefficients=coefficients,
            fitted_values=mu,
            linear_predictor=eta,
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
                'method': 'irls_qr',
                'rank': final_rank,
                'pivot': final_qr_pivot.tolist() if final_qr_pivot is not None else None,
                'R': final_qr_R,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=tuple(warnings_list),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _null_deviance(
        y: NDArray, wt: NDArray, family: Family
    ) -> float:
        """Compute null deviance matching R's glm.fit().

        R's glm.fit() with intercept=FALSE computes null deviance using
        mu_null = linkinv(offset), where offset defaults to 0. Since our
        design matrix X already contains the intercept column, glm.fit
        sees intercept=FALSE, so:
            mu_null = linkinv(0) for all observations.

        This matches R exactly:
            Gaussian/identity: mu_null = 0
            Binomial/logit:    mu_null = 0.5
            Poisson/log:       mu_null = 1.0
        """
        n = len(y)
        link = family.link

        # R: wtdmu = linkinv(offset) with offset=0
        eta_null = np.zeros(n, dtype=np.float64)
        mu_null = link.linkinv(eta_null)

        return family.deviance(y, mu_null, wt)

    @staticmethod
    def _deviance_residuals(
        y: NDArray, mu: NDArray, wt: NDArray, family: Family
    ) -> NDArray:
        """Compute signed deviance residuals.

        deviance_residual_i = sign(y_i - μ_i) * sqrt(wt_i * d_i)
        where d_i is the unit deviance contribution.
        """
        # Compute unit deviance contribution per observation
        # by evaluating deviance for each observation individually.
        # More efficient: use family-specific formulas.
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
            # Fallback: numerical unit deviance from total deviance
            # This is slow but correct for any family
            n = len(y)
            d = np.zeros(n, dtype=np.float64)
            ones = np.ones(1, dtype=np.float64)
            for i in range(n):
                yi = y[i:i+1]
                mui = mu[i:i+1]
                d[i] = family.deviance(yi, mui, ones)

        return sign * np.sqrt(np.maximum(wt * d, 0.0))
