"""
Descriptive statistics solution types.

Contains the parameter payload and user-facing solution wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result

if TYPE_CHECKING:
    from pystatistics.descriptive.design import DescriptiveDesign


@dataclass(frozen=True)
class DescriptiveParams:
    """
    Parameter payload for descriptive statistics.

    All fields are optional (None if not computed). describe() populates all;
    individual functions populate only their specific fields.
    """
    # Per-column statistics: arrays of shape (p,)
    mean: NDArray[np.floating[Any]] | None = None
    variance: NDArray[np.floating[Any]] | None = None
    sd: NDArray[np.floating[Any]] | None = None
    skewness: NDArray[np.floating[Any]] | None = None
    kurtosis: NDArray[np.floating[Any]] | None = None

    # Matrices: shape (p, p)
    covariance_matrix: NDArray[np.floating[Any]] | None = None
    correlation_pearson: NDArray[np.floating[Any]] | None = None
    correlation_spearman: NDArray[np.floating[Any]] | None = None
    correlation_kendall: NDArray[np.floating[Any]] | None = None

    # Quantiles: shape (n_probs, p)
    quantiles: NDArray[np.floating[Any]] | None = None
    quantile_probs: NDArray[np.floating[Any]] | None = None
    quantile_type: int | None = None

    # Summary table: shape (6, p) — rows: Min, Q1, Median, Mean, Q3, Max
    summary_table: NDArray[np.floating[Any]] | None = None

    # Missing data bookkeeping
    n_complete: int | None = None
    pairwise_n: NDArray[np.integer[Any]] | None = None


@dataclass
class DescriptiveSolution:
    """
    User-facing descriptive statistics results.

    Wraps Result[DescriptiveParams] and provides convenient accessors.
    """
    _result: Result[DescriptiveParams]
    _design: 'DescriptiveDesign'

    # --- Per-column statistics ---

    @property
    def mean(self) -> NDArray[np.floating[Any]] | None:
        """Per-column means, shape (p,)."""
        return self._result.params.mean

    @property
    def variance(self) -> NDArray[np.floating[Any]] | None:
        """Per-column variance (Bessel-corrected, n-1), shape (p,)."""
        return self._result.params.variance

    @property
    def sd(self) -> NDArray[np.floating[Any]] | None:
        """Per-column standard deviation, shape (p,)."""
        return self._result.params.sd

    @property
    def skewness(self) -> NDArray[np.floating[Any]] | None:
        """Per-column skewness (bias-adjusted), shape (p,)."""
        return self._result.params.skewness

    @property
    def kurtosis(self) -> NDArray[np.floating[Any]] | None:
        """Per-column excess kurtosis (bias-adjusted), shape (p,)."""
        return self._result.params.kurtosis

    # --- Matrices ---

    @property
    def covariance_matrix(self) -> NDArray[np.floating[Any]] | None:
        """Covariance matrix (Bessel-corrected), shape (p, p)."""
        return self._result.params.covariance_matrix

    @property
    def correlation_matrix(self) -> NDArray[np.floating[Any]] | None:
        """Returns whichever correlation matrix was computed (Pearson first)."""
        p = self._result.params
        if p.correlation_pearson is not None:
            return p.correlation_pearson
        if p.correlation_spearman is not None:
            return p.correlation_spearman
        if p.correlation_kendall is not None:
            return p.correlation_kendall
        return None

    @property
    def correlation_pearson(self) -> NDArray[np.floating[Any]] | None:
        """Pearson correlation matrix, shape (p, p)."""
        return self._result.params.correlation_pearson

    @property
    def correlation_spearman(self) -> NDArray[np.floating[Any]] | None:
        """Spearman rank correlation matrix, shape (p, p)."""
        return self._result.params.correlation_spearman

    @property
    def correlation_kendall(self) -> NDArray[np.floating[Any]] | None:
        """Kendall tau-b correlation matrix, shape (p, p)."""
        return self._result.params.correlation_kendall

    # --- Quantiles ---

    @property
    def quantiles(self) -> NDArray[np.floating[Any]] | None:
        """Quantile values, shape (n_probs, p)."""
        return self._result.params.quantiles

    @property
    def quantile_probs(self) -> NDArray[np.floating[Any]] | None:
        """Probabilities used for quantile computation."""
        return self._result.params.quantile_probs

    @property
    def quantile_type(self) -> int | None:
        """R quantile type used (1-9)."""
        return self._result.params.quantile_type

    # --- Summary ---

    @property
    def summary_table(self) -> NDArray[np.floating[Any]] | None:
        """Six-number summary (6, p): Min, Q1, Median, Mean, Q3, Max."""
        return self._result.params.summary_table

    # --- Missing data ---

    @property
    def n_complete(self) -> int | None:
        """Number of complete (no-NaN) observations used."""
        return self._result.params.n_complete

    @property
    def pairwise_n(self) -> NDArray[np.integer[Any]] | None:
        """Pairwise observation counts, shape (p, p)."""
        return self._result.params.pairwise_n

    # --- Metadata ---

    @property
    def columns(self) -> tuple[str, ...] | None:
        """Column names from the design."""
        return self._design.columns

    @property
    def info(self) -> dict[str, Any]:
        return self._result.info

    @property
    def timing(self) -> dict[str, float] | None:
        return self._result.timing

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    def summary(self) -> str:
        """R-style summary output."""
        lines = []

        if self.summary_table is not None:
            table = self.summary_table
            p = table.shape[1]
            cols = self.columns or tuple(f"V{i+1}" for i in range(p))
            row_labels = ["Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max."]

            # Column widths
            col_widths = []
            for j in range(p):
                width = max(
                    len(cols[j]),
                    max(len(f"{table[i, j]:.6f}") for i in range(6))
                )
                col_widths.append(width)

            label_width = max(len(lbl) for lbl in row_labels)

            # Header
            header = " " * (label_width + 2)
            header += "  ".join(c.rjust(w) for c, w in zip(cols, col_widths))
            lines.append(header)

            # Rows
            for i, label in enumerate(row_labels):
                row = label.ljust(label_width) + "  "
                row += "  ".join(
                    f"{table[i, j]:.6f}".rjust(w) for j, w in enumerate(col_widths)
                )
                lines.append(row)
        elif self.mean is not None:
            p = len(self.mean)
            cols = self.columns or tuple(f"V{i+1}" for i in range(p))
            lines.append("Descriptive Statistics:")
            for j, col in enumerate(cols):
                parts = [f"  {col}:"]
                if self.mean is not None:
                    parts.append(f"mean={self.mean[j]:.6f}")
                if self.sd is not None:
                    parts.append(f"sd={self.sd[j]:.6f}")
                if self.variance is not None:
                    parts.append(f"var={self.variance[j]:.6f}")
                lines.append(", ".join(parts))

        return "\n".join(lines)

    def __repr__(self) -> str:
        p = self._design.p
        n = self._design.n
        computed = []
        params = self._result.params
        if params.mean is not None:
            computed.append("mean")
        if params.variance is not None:
            computed.append("var")
        if params.covariance_matrix is not None:
            computed.append("cov")
        if params.correlation_pearson is not None:
            computed.append("cor_pearson")
        if params.correlation_spearman is not None:
            computed.append("cor_spearman")
        if params.correlation_kendall is not None:
            computed.append("cor_kendall")
        if params.quantiles is not None:
            computed.append("quantiles")
        if params.summary_table is not None:
            computed.append("summary")
        if params.skewness is not None:
            computed.append("skewness")
        if params.kurtosis is not None:
            computed.append("kurtosis")

        stats_str = ", ".join(computed) if computed else "none"
        return f"DescriptiveSolution(n={n}, p={p}, computed=[{stats_str}])"
