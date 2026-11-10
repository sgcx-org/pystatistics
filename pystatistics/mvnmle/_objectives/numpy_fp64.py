"""
Forward-Cholesky MVN MLE objective in float64 NumPy (no PyTorch).

The same estimator as ``GPUObjectiveFP64`` — log-Cholesky parameters, one
batched Cholesky over the padded per-pattern covariance blocks, closed-form
gradient — implemented with NumPy only. It is the direct-method objective used
when PyTorch is not installed.

Input contract: ``data`` is a 2-D float array with ``np.nan`` marking missing
values, at least two rows, and no all-missing row or column (validated unless
``validate=False``).
"""

from typing import Optional, Tuple

import numpy as np

from .base import MLEObjectiveBase
from .parameterizations import CholeskyParameterization
from ._batched_cholesky import auto_chunk_size, build_batched_constants
from ._batched_cholesky_np import neg2_loglik_np, value_and_gradient_np


class NumpyObjectiveFP64(MLEObjectiveBase):
    """Forward-Cholesky ``-2 log L`` objective and analytic gradient (NumPy).

    Parameters
    ----------
    data : np.ndarray, shape (n_obs, n_vars)
        Input data with missing values as ``np.nan``.
    validate : bool
        Whether to validate ``data``.
    chunk_size : int or None
        Missingness patterns processed per chunk; ``None`` auto-sizes to a
        memory budget.
    """

    # Diagonal jitter coefficient (scale-relative; see ``jitter_scale``).
    # Same value as the PyTorch FP64 objective so the two evaluate the same
    # function.
    EPS = 1e-12

    def __init__(self, data: np.ndarray, validate: bool = True,
                 chunk_size: Optional[int] = None):
        super().__init__(data, skip_validation=not validate)
        self.parameterization = CholeskyParameterization(self.n_vars)
        self.n_params = self.parameterization.n_params
        self.eps = self.EPS
        self.chunk_size = (chunk_size if chunk_size
                           else auto_chunk_size(self.n_vars, 8))
        self._consts = build_batched_constants(
            self.patterns, self.n_vars, var_scale=self.jitter_scale())

    def get_initial_parameters(self) -> np.ndarray:
        """Initial ``theta`` from the pairwise-complete sample moments."""
        return self.parameterization.get_initial_parameters(
            self.sample_mean, self.sample_cov)

    def compute_objective(self, theta: np.ndarray) -> float:
        """``-2 * log-likelihood`` (without the ``2*pi`` constant)."""
        theta = np.asarray(theta, dtype=np.float64)
        return neg2_loglik_np(theta, self._consts, self.n_vars, self.eps,
                              self.chunk_size)

    def compute_value_and_gradient(self, theta: np.ndarray
                                   ) -> Tuple[float, np.ndarray]:
        """Objective and closed-form gradient from one factorisation."""
        theta = np.asarray(theta, dtype=np.float64)
        return value_and_gradient_np(theta, self._consts, self.n_vars,
                                     self.eps, self.chunk_size)

    def compute_gradient(self, theta: np.ndarray) -> np.ndarray:
        """Closed-form gradient of the objective."""
        return self.compute_value_and_gradient(theta)[1]

    def extract_parameters(self, theta: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return ``(mu, Sigma, log-likelihood)`` at ``theta``."""
        mu, sigma = self.parameterization.unpack(theta)
        return mu, sigma, -0.5 * self.compute_objective(theta)
