"""
Ordinal regression solution type.

OrdinalSolution wraps Result[OrdinalParams] and provides R's
summary.polr-style output, along with convenient property accessors
for coefficients, thresholds, standard errors, z-values, and p-values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pystatistics.core.result import Result
from pystatistics.ordinal._common import OrdinalParams


@dataclass
class OrdinalSolution:
    """
    User-facing ordinal regression results.

    Wraps Result[OrdinalParams] and provides convenient access to
    fitted model quantities. The summary() method produces output
    matching R's summary(polr(...)).

    Attributes
    ----------
    _result : Result[OrdinalParams]
        Internal result envelope.
    _names : list[str]
        Predictor variable names.
    _level_names : list[str]
        Ordered category labels.
    """

    _result: Result[OrdinalParams]
    _names: list[str]
    _level_names: list[str]

    # -- Coefficient accessors ------------------------------------------------

    @property
    def coef(self) -> dict[str, float]:
        """
        Slope coefficients as a name-to-value dictionary.

        Returns:
            Dict mapping predictor names to their fitted beta values.
        """
        p = self._result.params
        return dict(zip(self._names, p.coefficients))

    @property
    def coefficients(self) -> NDArray[np.floating[Any]]:
        """
        Slope coefficient vector beta.

        Returns:
            Array of shape (p,) with fitted slope parameters.
        """
        return self._result.params.coefficients

    @property
    def thresholds(self) -> dict[str, float]:
        """
        Threshold parameters with level-pair labels.

        Returns:
            Dict mapping threshold labels (e.g. 'low|medium') to
            their fitted alpha values.
        """
        p = self._result.params
        labels = _threshold_labels(self._level_names)
        return dict(zip(labels, p.thresholds))

    @property
    def threshold_values(self) -> NDArray[np.floating[Any]]:
        """
        Raw threshold parameter vector alpha.

        Returns:
            Array of shape (K-1,) with ordered threshold values.
        """
        return self._result.params.thresholds

    # -- Variance-covariance and standard errors ------------------------------

    @property
    def vcov(self) -> NDArray[np.floating[Any]]:
        """
        Full variance-covariance matrix.

        Ordered as [thresholds, coefficients], shape (K-1+p, K-1+p).
        This is on the raw (unconstrained) parameterization for
        thresholds.

        Returns:
            Variance-covariance matrix.
        """
        return self._result.params.vcov

    @property
    def standard_errors(self) -> NDArray[np.floating[Any]]:
        """
        Standard errors for slope coefficients beta.

        Returns:
            Array of shape (p,) with SE(beta).
        """
        p = self._result.params
        n_thresh = p.n_levels - 1
        diag = np.diag(p.vcov)
        beta_var = diag[n_thresh:]
        return np.sqrt(np.maximum(beta_var, 0.0))

    @property
    def threshold_standard_errors(self) -> NDArray[np.floating[Any]]:
        """
        Standard errors for threshold parameters.

        These are on the raw (unconstrained) parameterization.

        Returns:
            Array of shape (K-1,) with SE(raw_threshold).
        """
        p = self._result.params
        n_thresh = p.n_levels - 1
        diag = np.diag(p.vcov)
        thresh_var = diag[:n_thresh]
        return np.sqrt(np.maximum(thresh_var, 0.0))

    # -- Test statistics -------------------------------------------------------

    @property
    def z_values(self) -> NDArray[np.floating[Any]]:
        """
        Wald z-statistics for slope coefficients: beta / SE(beta).

        Returns:
            Array of shape (p,).
        """
        se = self.standard_errors
        se_safe = np.where(se > 0, se, np.nan)
        return self._result.params.coefficients / se_safe

    @property
    def p_values(self) -> NDArray[np.floating[Any]]:
        """
        Two-sided p-values for slope coefficients from z-statistics.

        Returns:
            Array of shape (p,).
        """
        z = self.z_values
        return 2.0 * norm.sf(np.abs(z))

    @property
    def threshold_z_values(self) -> NDArray[np.floating[Any]]:
        """
        Wald z-statistics for thresholds: alpha / SE(alpha).

        Note: these are for the raw (unconstrained) parameterization.

        Returns:
            Array of shape (K-1,).
        """
        se = self.threshold_standard_errors
        se_safe = np.where(se > 0, se, np.nan)
        return self._result.params.thresholds / se_safe

    # -- Model fit statistics --------------------------------------------------

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return self._result.params.aic

    @property
    def deviance(self) -> float:
        """Residual deviance (-2 * log-likelihood)."""
        return self._result.params.deviance

    @property
    def log_likelihood(self) -> float:
        """Maximized log-likelihood."""
        return self._result.params.log_likelihood

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self._result.params.n_obs

    @property
    def converged(self) -> bool:
        """Whether the optimizer converged."""
        return self._result.params.converged

    @property
    def n_iter(self) -> int:
        """Number of optimizer iterations."""
        return self._result.params.n_iter

    @property
    def method(self) -> str:
        """Link function method name."""
        return self._result.params.method

    # -- Metadata accessors ----------------------------------------------------

    @property
    def info(self) -> dict[str, Any]:
        """Result metadata dictionary."""
        return self._result.info

    @property
    def timing(self) -> dict[str, float] | None:
        """Execution timing, or None."""
        return self._result.timing

    @property
    def backend_name(self) -> str:
        """Backend identifier."""
        return self._result.backend_name

    @property
    def warnings(self) -> tuple[str, ...]:
        """Non-fatal warnings."""
        return self._result.warnings

    # -- Summary ---------------------------------------------------------------

    def summary(self) -> str:
        """
        Format results as R's summary.polr output.

        Produces a table of coefficients with values, standard errors,
        and z-values, followed by a table of intercepts (thresholds),
        and model fit statistics.

        Returns:
            Formatted string matching R's print.summary.polr style.
        """
        p = self._result.params
        n_thresh = p.n_levels - 1
        se_beta = self.standard_errors
        z_beta = self.z_values
        se_thresh = self.threshold_standard_errors
        z_thresh = self.threshold_z_values

        lines: list[str] = []

        # Coefficients table
        lines.append("Coefficients:")

        # Compute column widths for alignment
        coef_rows = []
        for i, name in enumerate(self._names):
            coef_rows.append((
                name,
                p.coefficients[i],
                se_beta[i],
                z_beta[i],
            ))

        if coef_rows:
            lines.append(_format_coef_table(coef_rows))

        lines.append("")

        # Intercepts (thresholds) table
        lines.append("Intercepts:")
        thresh_labels = _threshold_labels(self._level_names)
        thresh_rows = []
        for j in range(n_thresh):
            thresh_rows.append((
                thresh_labels[j],
                p.thresholds[j],
                se_thresh[j],
                z_thresh[j],
            ))

        if thresh_rows:
            lines.append(_format_coef_table(thresh_rows))

        lines.append("")

        # Model fit
        lines.append(f"Residual Deviance: {p.deviance:.2f}")
        lines.append(f"AIC: {p.aic:.2f}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Compact string representation."""
        p = self._result.params
        return (
            f"OrdinalSolution(method={p.method!r}, "
            f"n_obs={p.n_obs}, n_levels={p.n_levels}, "
            f"deviance={p.deviance:.2f}, aic={p.aic:.2f})"
        )


# -- Formatting helpers --------------------------------------------------------

def _threshold_labels(level_names: list[str]) -> list[str]:
    """
    Create threshold labels from level names.

    Args:
        level_names: Ordered category labels of length K.

    Returns:
        List of K-1 labels like 'low|medium', 'medium|high'.
    """
    labels = []
    for j in range(len(level_names) - 1):
        labels.append(f"{level_names[j]}|{level_names[j + 1]}")
    return labels


def _format_coef_table(
    rows: list[tuple[str, float, float, float]],
) -> str:
    """
    Format a coefficient table with aligned columns.

    Args:
        rows: List of (name, value, std_error, z_value) tuples.

    Returns:
        Formatted table string with header and aligned columns.
    """
    header = ("", "Value", "Std. Error", "z value")
    col_widths = [len(h) for h in header]

    # Format numbers
    formatted = []
    for name, val, se, z in rows:
        val_s = f"{val:.4f}"
        se_s = f"{se:.4f}"
        z_s = f"{z:.4f}"
        formatted.append((name, val_s, se_s, z_s))
        col_widths[0] = max(col_widths[0], len(name))
        col_widths[1] = max(col_widths[1], len(val_s))
        col_widths[2] = max(col_widths[2], len(se_s))
        col_widths[3] = max(col_widths[3], len(z_s))

    # Build header line
    line_parts = [
        header[0].rjust(col_widths[0]),
        header[1].rjust(col_widths[1]),
        header[2].rjust(col_widths[2]),
        header[3].rjust(col_widths[3]),
    ]
    lines = ["  ".join(line_parts)]

    # Build data lines
    for name, val_s, se_s, z_s in formatted:
        parts = [
            name.ljust(col_widths[0]),
            val_s.rjust(col_widths[1]),
            se_s.rjust(col_widths[2]),
            z_s.rjust(col_widths[3]),
        ]
        lines.append("  ".join(parts))

    return "\n".join(lines)
