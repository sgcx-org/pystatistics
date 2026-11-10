"""
Shared data preparation for the MVN missing-data objectives.

``MLEObjectiveBase`` validates the data matrix, groups its rows by
missingness pattern (``_pattern_grouping``) and computes starting values
(``_start_values``). The forward-Cholesky direct objectives subclass it; the
EM and monotone solvers use it for its patterns and starting values.

Input contract: ``data`` is a 2-D array with ``np.nan`` marking missing
values, at least two rows and one column, and no row or column that is
entirely missing. Violations raise ``ValidationError`` unless
``skip_validation=True``, in which case the caller vouches for them.
"""

from typing import List, Tuple

import numpy as np

from pystatistics.core.exceptions import ValidationError

from ._pattern_grouping import PatternData, group_rows_by_pattern
from ._start_values import available_case_moments, positive_definite_start

__all__ = ['MLEObjectiveBase', 'PatternData']


class MLEObjectiveBase:
    """Data preparation shared by the MVN missing-data estimators.

    Attributes
    ----------
    original_data : np.ndarray, shape (n_obs, n_vars)
        The data as float64, rows in their original order.
    patterns : list of PatternData
        Rows grouped by missingness pattern; see ``_pattern_grouping`` for
        the (deterministic) pattern order.
    n_observed_scalars : int
        Number of observed scalar values. The optimisation driver divides the
        summed objective by it so that the gradient tolerance does not depend
        on the size of the dataset (see ``backends/_optimize.py``).
    sample_mean : np.ndarray, shape (n_vars,)
        Available-case means.
    sample_cov_raw : np.ndarray, shape (n_vars, n_vars)
        Pairwise-complete covariance, exactly as estimated (it may be
        indefinite). Guards that screen the data for collinearity or scale
        problems use this matrix.
    sample_cov : np.ndarray, shape (n_vars, n_vars)
        Positive definite starting covariance derived from
        ``sample_cov_raw``; see ``_start_values``. It only determines where
        iteration starts.
    is_complete : bool
        True when the data have no missing values.
    """

    def __init__(self, data: np.ndarray, skip_validation: bool = False):
        if not skip_validation:
            self._validate_data(data)

        self.original_data = np.asarray(data, dtype=np.float64)
        self.n_obs, self.n_vars = self.original_data.shape

        self.n_mean_params = self.n_vars
        self.n_cov_params = self.n_vars * (self.n_vars + 1) // 2
        self.n_params = self.n_mean_params + self.n_cov_params

        self.patterns: List[PatternData] = group_rows_by_pattern(
            self.original_data)
        self.n_patterns = len(self.patterns)

        observed = ~np.isnan(self.original_data)
        self.n_observed_scalars = int(observed.sum())
        self.is_complete = bool(observed.all())

        self.sample_mean, self.sample_cov_raw = available_case_moments(
            self.original_data)
        self.sample_cov = positive_definite_start(self.sample_cov_raw)

    @staticmethod
    def _validate_data(data: np.ndarray) -> None:
        """Raise ``ValidationError`` unless ``data`` meets the input contract."""
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValidationError(f"Data must be 2D, got shape {data.shape}")
        if data.shape[0] < 2:
            raise ValidationError(
                f"Need at least 2 observations, got {data.shape[0]}")
        if data.shape[1] < 1:
            raise ValidationError(
                f"Need at least 1 variable, got {data.shape[1]}")

        missing = np.isnan(np.asarray(data, dtype=np.float64))
        if np.any(missing.all(axis=1)):
            raise ValidationError("Data contains rows with all missing values")
        if np.any(missing.all(axis=0)):
            raise ValidationError(
                "Data contains columns with all missing values")

    def jitter_scale(self) -> np.ndarray:
        """Per-variable scale for the diagonal jitter: the observed variances.

        The batched objectives add ``eps * scale_j`` to the diagonal of each
        pattern's covariance block, so the jitter is relative to each
        variable's own scale. A variance that is not positive and finite
        (rejected upstream by the degeneracy guards unless forced) falls back
        to scale 1.0.
        """
        var = np.diag(self.sample_cov_raw).copy()
        bad = ~np.isfinite(var) | (var <= 0.0)
        var[bad] = 1.0
        return var

    # --- Subclass interface -------------------------------------------------

    def get_initial_parameters(self) -> np.ndarray:
        """Initial parameter vector. Subclasses must implement."""
        raise NotImplementedError(
            "Subclasses must implement get_initial_parameters")

    def compute_objective(self, theta: np.ndarray) -> float:
        """Objective value at ``theta``. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement compute_objective")

    def compute_gradient(self, theta: np.ndarray) -> np.ndarray:
        """Gradient at ``theta``. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement compute_gradient")

    def extract_parameters(self, theta: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, float]:
        """``(mu, Sigma, log-likelihood)`` at ``theta``. Subclasses must
        implement."""
        raise NotImplementedError(
            "Subclasses must implement extract_parameters")
