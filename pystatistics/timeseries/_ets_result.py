"""
ETS result containers.

Defines the immutable :class:`ETSParams` payload and the
:class:`ETSSolution` envelope returned by the fitting engine in
``_ets_fit.py`` and by public ``ets()`` selection in ``_ets_select.py``.
See ``_ets_fit.py`` for the estimation conventions (parameter space,
log-likelihood reporting) documented on these attributes.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from pystatistics.core.result import Result, SolutionReprMixin
from pystatistics.timeseries._ets_models import ETSSpec


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ETSParams:
    """Immutable parameter payload for a fitted ETS model.

    Attributes
    ----------
    spec : ETSSpec
        The fitted model specification.
    alpha : float
        Level smoothing parameter.
    beta : float or None
        Trend smoothing parameter (``None`` if no trend).
    gamma : float or None
        Seasonal smoothing parameter (``None`` if no season).
    phi : float or None
        Damping parameter (``None`` if not damped).
    init_level : float
        Estimated initial level.
    init_trend : float or None
        Estimated initial trend (``None`` if no trend).
    init_season : NDArray or None
        Estimated initial seasonal indices (``None`` if no season).
    fitted_values : NDArray
        One-step-ahead fitted values, length *n*.
    residuals : NDArray
        Residuals, length *n*.
    states : NDArray
        Full state history, shape ``(n + 1, n_states)``.
    log_likelihood : float
        Maximised **full Gaussian** log-likelihood.  R's ``forecast::ets``
        reports the concentrated pseudo-log-likelihood
        ``-0.5*n*log(SSE)`` instead; this value equals R's plus the
        constant ``0.5*n*[log(n/(2*pi)) - 1]`` (sample-size only, so all
        model *comparisons* on the same data are unaffected).
    aic : float
        Akaike Information Criterion, ``-2*log_likelihood + 2*k`` with the
        full-Gaussian log-likelihood above.  Differs from
        ``forecast::ets``'s printed AIC by ``-n*[log(n/(2*pi)) - 1]``;
        AIC *differences and rankings* between models match R exactly.
    aicc : float
        Corrected AIC (for small samples); same convention note as `aic`.
    bic : float
        Bayesian Information Criterion; same convention note as `aic`.
    mse : float
        Mean squared error of residuals.
    mae : float
        Mean absolute error of residuals.
    n_obs : int
        Number of observations.
    n_params : int
        Total number of estimated parameters (smoothing + free initial
        states + sigma^2).  Seasonal models estimate ``m - 1`` initial
        seasonal states — the remaining one is fixed by the
        normalisation, as in R ``forecast::ets`` — so the count matches
        R's.
    converged : bool
        Whether the optimiser converged.
    """

    spec: ETSSpec
    alpha: float
    beta: float | None
    gamma: float | None
    phi: float | None
    init_level: float
    init_trend: float | None
    init_season: NDArray | None
    fitted_values: NDArray
    residuals: NDArray
    states: NDArray
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    mse: float
    mae: float
    n_obs: int
    n_params: int
    converged: bool


@dataclass
class ETSSolution(SolutionReprMixin):
    """
    Result from fitting an ETS model.

    Wraps a :class:`Result` ``[ETSParams]`` envelope; every datum is
    exposed via a read-only ``@property`` so the public attribute surface
    is unchanged from the previous flat dataclass.
    """

    _result: Result[ETSParams]

    @property
    def spec(self) -> ETSSpec:
        return self._result.params.spec

    @property
    def alpha(self) -> float:
        return self._result.params.alpha

    @property
    def beta(self) -> float | None:
        return self._result.params.beta

    @property
    def gamma(self) -> float | None:
        return self._result.params.gamma

    @property
    def phi(self) -> float | None:
        return self._result.params.phi

    @property
    def init_level(self) -> float:
        return self._result.params.init_level

    @property
    def init_trend(self) -> float | None:
        return self._result.params.init_trend

    @property
    def init_season(self) -> NDArray | None:
        return self._result.params.init_season

    @property
    def fitted_values(self) -> NDArray:
        return self._result.params.fitted_values

    @property
    def residuals(self) -> NDArray:
        return self._result.params.residuals

    @property
    def states(self) -> NDArray:
        return self._result.params.states

    @property
    def log_likelihood(self) -> float:
        """Full Gaussian log-likelihood.

        R's ``forecast::ets`` reports the concentrated pseudo-
        log-likelihood ``-0.5*n*log(SSE)``; this value equals R's plus
        the deterministic constant ``0.5*n*[log(n/(2*pi)) - 1]``.  Model
        comparisons on the same data are identical either way.
        """
        return self._result.params.log_likelihood

    @property
    def aic(self) -> float:
        """AIC under the full-Gaussian log-likelihood.

        Differs from ``forecast::ets``'s printed AIC by the constant
        ``-n*[log(n/(2*pi)) - 1]``; AIC differences and model rankings
        match R exactly (same parameter count ``k``).
        """
        return self._result.params.aic

    @property
    def aicc(self) -> float:
        """AICc; same convention note as :attr:`aic`."""
        return self._result.params.aicc

    @property
    def bic(self) -> float:
        """BIC; same convention note as :attr:`aic`."""
        return self._result.params.bic

    @property
    def mse(self) -> float:
        return self._result.params.mse

    @property
    def mae(self) -> float:
        return self._result.params.mae

    @property
    def n_obs(self) -> int:
        return self._result.params.n_obs

    @property
    def n_params(self) -> int:
        return self._result.params.n_params

    @property
    def converged(self) -> bool:
        return self._result.params.converged

    @property
    def info(self) -> dict:
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
        """
        Return a human-readable summary matching R's ``forecast::ets()`` style.

        Returns
        -------
        str
            Multi-line summary.
        """
        lines = [
            self.spec.name,
            "",
            "  Smoothing parameters:",
            f"    alpha = {self.alpha:.4f}",
        ]
        if self.beta is not None:
            lines.append(f"    beta  = {self.beta:.4f}")
        if self.gamma is not None:
            lines.append(f"    gamma = {self.gamma:.4f}")
        if self.phi is not None:
            lines.append(f"    phi   = {self.phi:.4f}")
        lines.append("")
        lines.append("  Initial states:")
        lines.append(f"    l = {self.init_level:.4f}")
        if self.init_trend is not None:
            lines.append(f"    b = {self.init_trend:.4f}")
        if self.init_season is not None:
            s_str = ", ".join(f"{v:.4f}" for v in self.init_season)
            lines.append(f"    s = [{s_str}]")
        lines.extend([
            "",
            f"  sigma^2: {self.mse:.4f}",
            "",
            f"  Log-likelihood: {self.log_likelihood:.2f}",
            f"  AIC:  {self.aic:.2f}",
            f"  AICc: {self.aicc:.2f}",
            f"  BIC:  {self.bic:.2f}",
            "",
            f"  MSE: {self.mse:.4f}",
            f"  MAE: {self.mae:.4f}",
            f"  n = {self.n_obs}, k = {self.n_params}",
            f"  Converged: {self.converged}",
        ])
        return "\n".join(lines)



