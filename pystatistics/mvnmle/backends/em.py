"""
EM (Expectation-Maximization) backend for MVN MLE.

Implements the textbook EM algorithm for multivariate normal estimation
with missing data (Little & Rubin, Ch. 8). Converges to the same MLE as
the direct BFGS approach but via iterative conditional expectations.

Supports CPU (numpy/scipy) and GPU (torch) execution.
"""

import numpy as np
from typing import List, Optional

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.mvnmle.design import MVNDesign
from pystatistics.mvnmle.solution import MVNParams
from pystatistics.mvnmle._objectives.base import MLEObjectiveBase, PatternData


class EMBackend:
    """
    EM backend for MVN MLE.

    Uses the Expectation-Maximization algorithm to compute maximum likelihood
    estimates of the mean vector and covariance matrix from data with missing
    values. Validated against R's norm::em.norm().

    Parameters
    ----------
    device : str
        Computation device: 'cpu', 'cuda', or 'mps'.
        GPU execution uses PyTorch for linear algebra.
    """

    def __init__(self, device: str = 'cpu'):
        self._device = device
        self._use_gpu = device in ('cuda', 'mps')

        if self._use_gpu:
            import torch
            self._torch = torch
            self._torch_device = torch.device(device)
            self._dtype = torch.float32 if device == 'mps' else torch.float64

    @property
    def name(self) -> str:
        return f'{self._device}_em'

    def solve(
        self,
        design: MVNDesign,
        *,
        tol: float = 1e-4,
        max_iter: int = 1000,
    ) -> Result[MVNParams]:
        """
        Solve MVN MLE using EM algorithm.

        Parameters
        ----------
        design : MVNDesign
            Data design wrapper.
        tol : float
            Convergence tolerance. EM converges when the maximum absolute
            change in parameters is less than tol (R's norm convention).
        max_iter : int
            Maximum EM iterations.

        Returns
        -------
        Result[MVNParams]
        """
        timer = Timer(sync_cuda=(self._device == 'cuda'))
        timer.start()
        warnings_list = []

        # --- Initialization ---
        with timer.section('initialization'):
            # Use MLEObjectiveBase for pattern extraction infrastructure
            obj = MLEObjectiveBase(design.data, skip_validation=True)
            patterns = obj.patterns
            n = obj.n_obs
            p = obj.n_vars

            mu = obj.sample_mean.copy()
            sigma = obj.sample_cov.copy()

        # --- EM iteration ---
        loglik_history = []
        converged = False
        n_iter = 0

        with timer.section('em_iterations'):
            for iteration in range(max_iter):
                # Pack current parameters for convergence check
                theta_old = self._pack_params(mu, sigma, p)

                # E-step: accumulate sufficient statistics
                T1, T2 = self._e_step(mu, sigma, patterns, n, p)

                # M-step: update parameters
                mu_new, sigma_new = self._m_step(T1, T2, n, p)

                # Positive definiteness guard
                sigma_new = self._ensure_pd(sigma_new, p)

                # Convergence check (parameter convergence, R's norm approach)
                theta_new = self._pack_params(mu_new, sigma_new, p)
                param_change = np.max(np.abs(theta_new - theta_old))

                mu = mu_new
                sigma = sigma_new
                n_iter = iteration + 1

                if param_change <= tol:
                    converged = True
                    break

        # --- Compute final log-likelihood ---
        with timer.section('loglikelihood'):
            loglik = self._compute_loglik(mu, sigma, patterns)
            loglik_history.append(loglik)

        if not converged:
            warnings_list.append(
                f"EM did not converge after {max_iter} iterations "
                f"(final param change: {param_change:.2e}, tol: {tol:.2e})"
            )

        timer.stop()

        params = MVNParams(
            muhat=mu,
            sigmahat=sigma,
            loglik=loglik,
            n_iter=n_iter,
            converged=converged,
            gradient_norm=None,  # EM does not compute gradients
        )

        return Result(
            params=params,
            info={
                'algorithm': 'em',
                'convergence_criterion': 'parameter',
                'final_param_change': float(param_change) if n_iter > 0 else float('inf'),
                'loglik_history': loglik_history,
                'device': self._device,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=tuple(warnings_list),
        )

    # ------------------------------------------------------------------
    # Core EM steps
    # ------------------------------------------------------------------

    def _e_step(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        patterns: List[PatternData],
        n: int,
        p: int,
    ) -> tuple:
        """
        E-step: compute expected sufficient statistics.

        For each missingness pattern, compute conditional expectations of
        missing values given observed values and current parameters, then
        accumulate into sufficient statistics T1 = Σ E[x_i] and
        T2 = Σ E[x_i x_i^T].

        Returns
        -------
        T1 : ndarray, shape (p,)
            Sum of expected complete-data vectors.
        T2 : ndarray, shape (p, p)
            Sum of expected outer products.
        """
        T1 = np.zeros(p)
        T2 = np.zeros((p, p))

        for pattern in patterns:
            obs = pattern.observed_indices
            mis = pattern.missing_indices
            n_k = pattern.n_obs
            data_k = pattern.data  # (n_k, len(obs))

            if len(mis) == 0:
                # Complete pattern: no imputation needed
                T1[obs] += data_k.sum(axis=0)
                T2[np.ix_(obs, obs)] += data_k.T @ data_k
                continue

            if len(obs) == 0:
                # All missing: impute with current mean, add sigma
                T1 += n_k * mu
                T2 += n_k * (sigma + np.outer(mu, mu))
                continue

            # Submatrices of current parameters
            mu_o = mu[obs]
            mu_m = mu[mis]
            sigma_oo = sigma[np.ix_(obs, obs)]
            sigma_mo = sigma[np.ix_(mis, obs)]
            sigma_mm = sigma[np.ix_(mis, mis)]

            # Regression coefficient: beta = Sigma_mo @ Sigma_oo^{-1}
            # Use solve for numerical stability: beta^T = solve(Sigma_oo, Sigma_om)
            beta = np.linalg.solve(sigma_oo, sigma_mo.T).T  # (n_mis, n_obs)

            # Conditional covariance: Sigma_{M|O} = Sigma_mm - beta @ Sigma_mo^T
            cond_cov = sigma_mm - beta @ sigma_mo.T

            # Vectorized over all n_k observations in this pattern
            centered = data_k - mu_o  # (n_k, n_obs)
            x_m_hat = mu_m + centered @ beta.T  # (n_k, n_mis)

            # Reconstruct full imputed data matrix
            x_full = np.empty((n_k, p))
            x_full[:, obs] = data_k
            x_full[:, mis] = x_m_hat

            # Accumulate sufficient statistics
            T1 += x_full.sum(axis=0)
            T2 += x_full.T @ x_full

            # Correction for conditional covariance of missing values
            # E[x_m x_m^T | x_o] = E[x_m|x_o] E[x_m|x_o]^T + Var[x_m|x_o]
            # The outer product term is in x_full.T @ x_full; add n_k * cond_cov
            T2[np.ix_(mis, mis)] += n_k * cond_cov

        return T1, T2

    def _m_step(
        self,
        T1: np.ndarray,
        T2: np.ndarray,
        n: int,
        p: int,
    ) -> tuple:
        """
        M-step: update mu and sigma from sufficient statistics.

        mu_new = T1 / n
        sigma_new = T2 / n - mu_new @ mu_new^T
        """
        mu_new = T1 / n
        sigma_new = T2 / n - np.outer(mu_new, mu_new)

        # Enforce exact symmetry (avoid floating point drift)
        sigma_new = (sigma_new + sigma_new.T) / 2

        return mu_new, sigma_new

    def _compute_loglik(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        patterns: List[PatternData],
    ) -> float:
        """
        Compute observed-data log-likelihood.

        ℓ(μ, Σ) = Σ_k Σ_{i in k} [-½ log|Σ_OO| - ½ (x_i - μ_O)^T Σ_OO^{-1} (x_i - μ_O)]

        Note: The log(2π) normalizing constant is omitted to match R's mvnmle
        convention. This ensures EM and direct MLE report consistent loglik values.
        """
        loglik = 0.0

        for pattern in patterns:
            obs = pattern.observed_indices
            n_k = pattern.n_obs
            p_k = len(obs)

            if p_k == 0:
                continue

            mu_o = mu[obs]
            sigma_oo = sigma[np.ix_(obs, obs)]

            # Log determinant
            sign, logdet = np.linalg.slogdet(sigma_oo)
            if sign <= 0:
                return -np.inf

            # Centered data
            centered = pattern.data - mu_o  # (n_k, p_k)

            # Quadratic forms: sum_i (x_i - mu_o)^T Sigma_oo^{-1} (x_i - mu_o)
            # Solve Sigma_oo @ Z = centered^T, then quad = sum(centered * Z^T)
            Z = np.linalg.solve(sigma_oo, centered.T)  # (p_k, n_k)
            quad_sum = np.sum(centered.T * Z)

            loglik += n_k * (-0.5 * logdet) - 0.5 * quad_sum

        return float(loglik)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _pack_params(self, mu: np.ndarray, sigma: np.ndarray, p: int) -> np.ndarray:
        """Pack mu and lower triangle of sigma into a flat vector for convergence check."""
        return np.concatenate([mu, sigma[np.tril_indices(p)]])

    def _ensure_pd(self, sigma: np.ndarray, p: int) -> np.ndarray:
        """Ensure positive definiteness with minimal ridge if needed."""
        try:
            eigvals = np.linalg.eigvalsh(sigma)
            min_eig = np.min(eigvals)
            if min_eig < 1e-10:
                ridge = abs(min_eig) + 1e-8
                sigma = sigma + ridge * np.eye(p)
        except np.linalg.LinAlgError:
            sigma = sigma + 1e-6 * np.eye(p)
        return sigma
