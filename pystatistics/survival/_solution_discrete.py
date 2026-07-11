"""
Discrete-time survival solution wrapper.

DiscreteTimeSolution wraps a Result[DiscreteTimeParams] and exposes
user-friendly properties for the logistic regression on person-period data.
"""

from __future__ import annotations

from pystatistics.core.result import Result, SolutionReprMixin
from pystatistics.survival._common import DiscreteTimeParams


class DiscreteTimeSolution(SolutionReprMixin):
    """Discrete-time survival model solution.

    Properties mirror the logistic regression on person-period data.
    """

    __slots__ = ('_result', '_names', '_conf_level')

    def __init__(self, _result: Result[DiscreteTimeParams],
                 _names: tuple[str, ...] | None = None,
                 _conf_level: float = 0.95) -> None:
        self._result = _result
        self._names = _names
        self._conf_level = _conf_level

    @property
    def coefficients(self):
        return self._result.params.coefficients

    @property
    def coef(self) -> dict[str, float]:
        """Named coefficient mapping."""
        names = self._names or tuple(f"x{i}" for i in range(len(self.coefficients)))
        return dict(zip(names, self.coefficients.tolist()))

    @property
    def hr(self) -> dict[str, float]:
        """Named hazard ratio mapping (exp(coef))."""
        names = self._names or tuple(f"x{i}" for i in range(len(self.hazard_ratios)))
        return dict(zip(names, self.hazard_ratios.tolist()))

    @property
    def standard_errors(self):
        return self._result.params.standard_errors

    @property
    def z_values(self):
        return self._result.params.z_values

    @property
    def p_values(self):
        return self._result.params.p_values

    @property
    def conf_level(self) -> float:
        """Confidence level for ``conf_int`` (default 0.95)."""
        return self._conf_level

    @property
    def conf_int(self):
        """Wald confidence intervals for the covariate coefficients, shape (p, 2).

        ``coef ± z * se`` on the (discrete-time log-hazard) coefficient scale,
        with the normal quantile for ``conf_level`` (the person-period logistic
        fit is asymptotic-normal). ``exp(conf_int)`` gives discrete-time
        hazard-ratio intervals.
        """
        import numpy as np
        from scipy import stats

        z = stats.norm.ppf((1.0 + self._conf_level) / 2.0)
        coef = self._result.params.coefficients
        se = self._result.params.standard_errors
        return np.column_stack([coef - z * se, coef + z * se])

    @property
    def hazard_ratios(self):
        return self._result.params.hazard_ratios

    @property
    def baseline_hazard(self):
        return self._result.params.baseline_hazard

    @property
    def interval_labels(self):
        return self._result.params.interval_labels

    @property
    def person_period_n(self) -> int:
        return self._result.params.person_period_n

    @property
    def n_intervals(self) -> int:
        return self._result.params.n_intervals

    @property
    def n_observations(self) -> int:
        return self._result.params.n_observations

    @property
    def n_events(self) -> int:
        return self._result.params.n_events

    @property
    def glm_deviance(self) -> float:
        return self._result.params.glm_deviance

    @property
    def glm_aic(self) -> float:
        return self._result.params.glm_aic

    @property
    def converged(self) -> bool:
        """Whether the person-period binomial IRLS converged.

        ``False`` means the underlying logistic fit hit the iteration cap (or,
        for an all-censored design with no events, that no fit ran) — treat the
        coefficients as unreliable. Mirrors ``CoxSolution.converged``.
        """
        return self._result.params.converged

    @property
    def n_iter(self) -> int:
        """IRLS iterations used by the person-period binomial GLM.

        Mirrors ``CoxSolution.n_iter``. A value at the solver's iteration cap
        alongside ``converged == False`` indicates a fit that did not settle.
        """
        return self._result.params.n_iter

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def timing(self):
        return self._result.timing

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    def summary(self) -> str:
        """Summary of discrete-time survival model."""
        lines = []
        lines.append("Call: discrete_time()")
        lines.append("")
        lines.append(
            f"  n={self.n_observations}, events={self.n_events}, "
            f"intervals={self.n_intervals}, "
            f"person-period rows={self.person_period_n}"
        )
        lines.append("")

        p = len(self.coefficients)
        names = self._names or tuple(f"x{i}" for i in range(p))
        max_name_len = max(len(n) for n in names)
        col_w = max(max_name_len, 10)

        lines.append(
            f"  {'':{col_w}s}  {'coef':>10s}  {'exp(coef)':>10s}  "
            f"{'se(coef)':>10s}  {'z':>10s}  {'Pr(>|z|)':>12s}"
        )
        for i, name in enumerate(names):
            lines.append(
                f"  {name:>{col_w}s}  {self.coefficients[i]:10.6f}  "
                f"{self.hazard_ratios[i]:10.6f}  "
                f"{self.standard_errors[i]:10.6f}  "
                f"{self.z_values[i]:10.4f}  "
                f"{self.p_values[i]:12.4g}"
            )

        lines.append("")
        lines.append(f"  Deviance: {self.glm_deviance:.4f}")
        lines.append(f"  AIC: {self.glm_aic:.4f}")
        lines.append(f"  Converged: {self.converged} ({self.n_iter} iterations)")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DiscreteTimeSolution(n={self.n_observations}, "
            f"events={self.n_events}, "
            f"intervals={self.n_intervals})"
        )
