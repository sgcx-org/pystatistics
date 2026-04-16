"""
Solution wrapper for multinomial logistic regression results.

MultinomialSolution wraps a Result[MultinomialParams] and exposes
user-friendly properties with an R-style summary() method matching
the output format of R's nnet::multinom().
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatistics.core.result import Result
from pystatistics.multinomial._common import MultinomialParams


class MultinomialSolution:
    """Multinomial logistic regression solution.

    Wraps a Result[MultinomialParams] and provides convenient access
    to model quantities: coefficients, standard errors, z-values,
    p-values, predictions, and model fit statistics.

    The reference class is the last class (highest code), matching
    R's nnet::multinom() convention. Coefficients are reported for
    each non-reference class relative to the reference.

    Attributes:
        _result: The underlying Result[MultinomialParams] object.
    """

    __slots__ = ("_result",)

    def __init__(self, _result: Result[MultinomialParams]) -> None:
        """Initialize from a Result[MultinomialParams].

        Args:
            _result: The fitted model result. Must contain a
                MultinomialParams payload.
        """
        self._result = _result

    # -- Coefficient access --

    @property
    def coefficient_matrix(self) -> NDArray[np.floating[Any]]:
        """Coefficient matrix of shape (J-1, p).

        One row per non-reference class, one column per predictor.
        """
        return self._result.params.coefficient_matrix

    @property
    def coef(self) -> dict[str, dict[str, float]]:
        """Coefficients as nested dict: class -> predictor -> value.

        Excludes the reference class (whose coefficients are all zero
        by definition).

        Returns:
            Dict mapping each non-reference class name to a dict
            mapping predictor names to coefficient values.
        """
        params = self._result.params
        result: dict[str, dict[str, float]] = {}
        for j in range(params.n_classes - 1):
            cls_name = params.class_names[j]
            row: dict[str, float] = {}
            for k, feat_name in enumerate(params.feature_names):
                row[feat_name] = float(params.coefficient_matrix[j, k])
            result[cls_name] = row
        return result

    # -- Standard errors and inference --

    @property
    def standard_errors(self) -> NDArray[np.floating[Any]]:
        """Standard error matrix of shape (J-1, p).

        Derived from the diagonal of the variance-covariance matrix.
        """
        params = self._result.params
        p = params.coefficient_matrix.shape[1]
        n_nonref = params.n_classes - 1

        se_flat = np.sqrt(np.maximum(np.diag(params.vcov), 0.0))
        return se_flat.reshape(n_nonref, p)

    @property
    def z_values(self) -> NDArray[np.floating[Any]]:
        """Z-value (Wald statistic) matrix of shape (J-1, p).

        Computed as coefficient / standard_error.
        """
        se = self.standard_errors
        # Avoid division by zero: where SE is 0, z is 0
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, self.coefficient_matrix / se, 0.0)
        return z

    @property
    def p_values(self) -> NDArray[np.floating[Any]]:
        """Two-sided p-value matrix of shape (J-1, p).

        Computed from the z-values using the standard normal distribution.
        """
        return 2.0 * stats.norm.sf(np.abs(self.z_values))

    # -- Predictions --

    @property
    def fitted_probs(self) -> NDArray[np.floating[Any]]:
        """Predicted probability matrix of shape (n, J).

        Each row sums to 1. Columns correspond to class_names in order.
        """
        return self._result.params.fitted_probs

    @property
    def predicted_class(self) -> NDArray[np.intp]:
        """Predicted class codes of shape (n,).

        The predicted class for each observation is the one with
        the highest fitted probability (argmax).
        """
        return np.argmax(self._result.params.fitted_probs, axis=1)

    # -- Model fit statistics --

    @property
    def log_likelihood(self) -> float:
        """Maximized log-likelihood of the fitted model."""
        return self._result.params.log_likelihood

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        return self._result.params.aic

    @property
    def deviance(self) -> float:
        """Residual deviance (-2 * log_likelihood)."""
        return self._result.params.deviance

    @property
    def null_deviance(self) -> float:
        """Null deviance (-2 * null_log_likelihood)."""
        return self._result.params.null_deviance

    @property
    def pseudo_r_squared(self) -> float:
        """McFadden's pseudo R-squared.

        Defined as 1 - (log_likelihood / null_log_likelihood).
        Values closer to 1 indicate better fit.
        """
        null_ll = self._result.params.null_deviance / -2.0
        if null_ll == 0.0:
            return 0.0
        return 1.0 - (self.log_likelihood / null_ll)

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self._result.params.n_obs

    @property
    def n_classes(self) -> int:
        """Number of response classes J."""
        return self._result.params.n_classes

    @property
    def class_names(self) -> tuple[str, ...]:
        """Class labels in order (last is reference)."""
        return self._result.params.class_names

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Predictor names matching columns of X."""
        return self._result.params.feature_names

    @property
    def converged(self) -> bool:
        """Whether the optimizer converged."""
        return self._result.params.converged

    @property
    def n_iter(self) -> int:
        """Number of optimizer iterations."""
        return self._result.params.n_iter

    # -- Display --

    def summary(self) -> str:
        """Generate R-style summary of the fitted model.

        Returns a formatted string matching the style of R's
        summary.multinom() output, showing coefficients, standard
        errors, and z-values for each non-reference class.

        Returns:
            Formatted summary string.
        """
        params = self._result.params
        lines: list[str] = []

        lines.append("Multinomial Logistic Regression")
        lines.append("")

        ref_name = params.class_names[-1]
        lines.append("Coefficients:")

        se_matrix = self.standard_errors
        z_matrix = self.z_values

        for j in range(params.n_classes - 1):
            cls_name = params.class_names[j]
            lines.append(
                f'  class "{cls_name}" (vs reference "{ref_name}"):'
            )

            # Header
            lines.append(
                f"  {'':>15s} {'Value':>12s} {'Std. Error':>12s} "
                f"{'z value':>12s}"
            )

            for k, feat_name in enumerate(params.feature_names):
                coef_val = params.coefficient_matrix[j, k]
                se_val = se_matrix[j, k]
                z_val = z_matrix[j, k]
                lines.append(
                    f"  {feat_name:>15s} {coef_val:>12.4f} "
                    f"{se_val:>12.4f} {z_val:>12.4f}"
                )

            lines.append("")

        lines.append(f"Residual Deviance: {params.deviance:.2f}")
        lines.append(f"AIC: {params.aic:.2f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Short representation of the solution."""
        return (
            f"MultinomialSolution("
            f"n_obs={self.n_obs}, "
            f"n_classes={self.n_classes}, "
            f"deviance={self.deviance:.2f}, "
            f"aic={self.aic:.2f}, "
            f"converged={self.converged})"
        )
