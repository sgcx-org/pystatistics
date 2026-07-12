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

from pystatistics.core.result import Result, SolutionReprMixin
from pystatistics.core.exceptions import ValidationError
from pystatistics.core.validation import check_array, check_2d, check_finite
from pystatistics.multinomial._common import MultinomialParams
from pystatistics.multinomial._likelihood import compute_probs


class MultinomialSolution(SolutionReprMixin):
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

    __slots__ = ("_result", "_conf_level")

    def __init__(self, _result: Result[MultinomialParams],
                 _conf_level: float = 0.95) -> None:
        """Initialize from a Result[MultinomialParams].

        Args:
            _result: The fitted model result. Must contain a
                MultinomialParams payload.
            _conf_level: Confidence level for ``conf_int`` (default 0.95).
        """
        self._result = _result
        self._conf_level = _conf_level

    # -- Coefficient access --

    @property
    def coefficient_matrix(self) -> NDArray[np.floating[Any]]:
        """Coefficient matrix of shape (J-1, p).

        One row per non-reference class, one column per predictor.
        """
        return self._result.params.coefficient_matrix

    @property
    def coefficients(self) -> NDArray[np.floating[Any]]:
        """Alias for ``coefficient_matrix``: the (J-1, p) coefficient matrix.

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
    def vcov(self) -> NDArray[np.floating[Any]]:
        """Variance-covariance matrix of the estimated coefficients.

        Shape ``((J-1)*p, (J-1)*p)``, ordered to match
        ``coefficient_matrix.ravel()`` (row-major over (class, predictor)). The
        diagonal reshaped to ``(J-1, p)`` gives the squared standard errors.
        """
        return self._result.params.vcov

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

    @property
    def conf_level(self) -> float:
        """Confidence level for ``conf_int`` (default 0.95)."""
        return self._conf_level

    @property
    def warnings(self) -> tuple[str, ...]:
        """Non-fatal warnings emitted during the fit (parity with
        ``OrdinalSolution.warnings``)."""
        return self._result.warnings

    @property
    def backend_name(self) -> str:
        """Backend identifier (parity with ``OrdinalSolution.backend_name``)."""
        return self._result.backend_name

    @property
    def info(self) -> dict[str, Any]:
        """Result metadata dictionary (parity with ``OrdinalSolution.info``)."""
        return self._result.info

    @property
    def timing(self) -> dict[str, float] | None:
        """Execution timing, or None (parity with ``OrdinalSolution.timing``)."""
        return self._result.timing

    @property
    def conf_int(self) -> NDArray[np.floating[Any]]:
        """Wald confidence intervals, shape (J-1, p, 2).

        ``coef ± z * se`` per (non-reference class, predictor), with the normal
        quantile for ``conf_level`` (multinomial inference is asymptotic-normal).
        The trailing axis is [lower, upper]; ``exp(conf_int)`` gives
        relative-risk-ratio intervals.
        """
        z = stats.norm.ppf((1.0 + self._conf_level) / 2.0)
        coef = self.coefficient_matrix
        se = self.standard_errors
        return np.stack([coef - z * se, coef + z * se], axis=-1)

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

    def predict(self, X: NDArray[np.floating[Any]], *,
                kind: str = "class") -> NDArray[Any]:
        """Predict for a design matrix, matching R's predict.multinom.

        Args:
            X: Design matrix of shape (n, p) — INCLUDING the intercept column if
                the model was fit with one (the caller supplies the intercept for
                multinom, so the prediction design must match the fitted design).
                p must equal the fitted number of predictors.
            kind: 'class' (default) returns the most probable class code per row
                (shape (n,)); 'probs' returns the (n, J) class-probability matrix
                (each row sums to 1, columns in code order).

        Returns:
            Class codes (kind='class') or a probability matrix (kind='probs').

        Raises:
            ValidationError: If kind is invalid or X's column count does not match
                the fitted model.
        """
        if kind not in ("class", "probs"):
            raise ValidationError(
                f"kind must be 'class' or 'probs', got {kind!r}")
        X_arr = check_array(X, "X")
        check_finite(X_arr, "X")
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        check_2d(X_arr, "X")
        params = self._result.params
        p = params.coefficient_matrix.shape[1]
        if X_arr.shape[1] != p:
            raise ValidationError(
                f"X has {X_arr.shape[1]} columns but the model was fit with {p} "
                f"predictors (include the intercept column if the model has one)")
        probs = compute_probs(
            params.coefficient_matrix.ravel(), X_arr, params.n_classes)
        return probs if kind == "probs" else np.argmax(probs, axis=1)

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
    def category_names(self) -> tuple[str, ...]:
        """Response category labels in order (last is reference)."""
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
