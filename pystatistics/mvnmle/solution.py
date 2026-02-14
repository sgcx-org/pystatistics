"""
MVN MLE solution types.

Contains the parameter payload and user-facing solution wrapper.
"""

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result

if TYPE_CHECKING:
    from pystatistics.mvnmle.design import MVNDesign


@dataclass(frozen=True)
class MVNParams:
    """
    Parameter payload for MVN MLE.

    Immutable data computed by backends.
    """
    muhat: NDArray[np.floating[Any]]
    sigmahat: NDArray[np.floating[Any]]
    loglik: float
    n_iter: int
    converged: bool
    gradient_norm: float | None = None


@dataclass
class MVNSolution:
    """
    User-facing MVN MLE results.

    Wraps the backend Result and provides convenient accessors
    for all MVN estimation outputs.
    """
    _result: Result[MVNParams]
    _design: 'MVNDesign'

    @property
    def muhat(self) -> NDArray[np.floating[Any]]:
        """Estimated mean vector."""
        return self._result.params.muhat

    @property
    def sigmahat(self) -> NDArray[np.floating[Any]]:
        """Estimated covariance matrix."""
        return self._result.params.sigmahat

    @property
    def loglik(self) -> float:
        """Log-likelihood at the estimated parameters."""
        return self._result.params.loglik

    @property
    def converged(self) -> bool:
        """Whether the optimization converged."""
        return self._result.params.converged

    @property
    def n_iter(self) -> int:
        """Number of iterations."""
        return self._result.params.n_iter

    @property
    def gradient_norm(self) -> float | None:
        """Final gradient norm, if available."""
        return self._result.params.gradient_norm

    @property
    def correlation_matrix(self) -> NDArray[np.floating[Any]]:
        """Correlation matrix derived from estimated covariance."""
        sigma = self.sigmahat
        d = np.sqrt(np.diag(sigma))
        d = np.where(d > 0, d, 1.0)
        corr = sigma / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        return corr

    @property
    def standard_deviations(self) -> NDArray[np.floating[Any]]:
        """Standard deviations from estimated covariance."""
        return np.sqrt(np.diag(self.sigmahat))

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        p = self._design.p
        n_params = p + p * (p + 1) // 2
        return -2 * self.loglik + 2 * n_params

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        p = self._design.p
        n = self._design.n
        n_params = p + p * (p + 1) // 2
        return -2 * self.loglik + n_params * np.log(n)

    @property
    def info(self) -> dict[str, Any]:
        """Backend metadata."""
        return self._result.info

    @property
    def timing(self) -> dict[str, float] | None:
        """Execution timing breakdown."""
        return self._result.timing

    @property
    def backend_name(self) -> str:
        """Name of the backend that produced this result."""
        return self._result.backend_name

    @property
    def warnings(self) -> tuple[str, ...]:
        """Non-fatal warnings from computation."""
        return self._result.warnings

    def summary(self) -> str:
        """Generate summary output."""
        lines = [
            "MVN MLE Results",
            "=" * 60,
            f"Observations: {self._design.n}",
            f"Variables: {self._design.p}",
            f"Missing rate: {self._design.missing_rate:.1%}",
            f"Converged: {self.converged}",
            f"Iterations: {self.n_iter}",
            f"Log-likelihood: {self.loglik:.6f}",
            f"AIC: {self.aic:.2f}",
            f"BIC: {self.bic:.2f}",
            "",
            "Estimated Means:",
            "-" * 60,
        ]

        for i, mu_i in enumerate(self.muhat):
            lines.append(f"  mu[{i}]: {mu_i:12.6f}")

        lines.extend([
            "",
            "Estimated Standard Deviations:",
            "-" * 60,
        ])

        for i, sd_i in enumerate(self.standard_deviations):
            lines.append(f"  sd[{i}]: {sd_i:12.6f}")

        lines.extend([
            "",
            "Estimated Correlation Matrix:",
            "-" * 60,
        ])

        corr = self.correlation_matrix
        for i in range(corr.shape[0]):
            row_str = "  " + " ".join(f"{corr[i, j]:8.4f}" for j in range(corr.shape[1]))
            lines.append(row_str)

        lines.append("-" * 60)
        lines.append(f"Backend: {self.backend_name}")
        if self.timing:
            lines.append(f"Time: {self.timing.get('total_seconds', 0):.4f}s")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'muhat': self.muhat.tolist(),
            'sigmahat': self.sigmahat.tolist(),
            'loglik': self.loglik,
            'converged': self.converged,
            'n_iter': self.n_iter,
            'n': self._design.n,
            'p': self._design.p,
            'missing_rate': self._design.missing_rate,
            'backend': self.backend_name,
        }

    def __repr__(self) -> str:
        return (
            f"MVNSolution(n={self._design.n}, p={self._design.p}, "
            f"converged={self.converged}, loglik={self.loglik:.4f})"
        )
