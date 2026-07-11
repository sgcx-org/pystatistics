"""
Cox proportional hazards solution wrapper.

CoxSolution wraps a Result[CoxParams] and exposes user-friendly properties
with an R-style summary() method mirroring coxph() output.
"""

from __future__ import annotations

from pystatistics.core.result import Result, SolutionReprMixin
from pystatistics.survival._common import CoxParams


class CoxSolution(SolutionReprMixin):
    """Cox proportional hazards solution.

    Properties mirror R's coxph() output.
    """

    __slots__ = ('_result', '_names', '_design')

    def __init__(self, _result: Result[CoxParams], _names: tuple[str, ...] | None = None,
                 _design=None) -> None:
        self._result = _result
        self._names = _names
        # The fitted SurvivalDesign, retained for post-fit diagnostics that
        # need the data (cox_zph). Same pattern as GLMSolution._design.
        self._design = _design

    @property
    def coefficients(self):
        return self._result.params.coefficients

    @property
    def coef(self) -> dict[str, float]:
        """Named coefficient mapping (like R's coef())."""
        names = self._names or tuple(f"x{i}" for i in range(len(self.coefficients)))
        return dict(zip(names, self.coefficients.tolist()))

    @property
    def hr(self) -> dict[str, float]:
        """Named hazard ratio mapping (exp(coef))."""
        names = self._names or tuple(f"x{i}" for i in range(len(self.hazard_ratios)))
        return dict(zip(names, self.hazard_ratios.tolist()))

    @property
    def hazard_ratios(self):
        return self._result.params.hazard_ratios

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
    def loglik(self):
        return self._result.params.loglik

    @property
    def concordance(self) -> float:
        return self._result.params.concordance

    @property
    def n_events(self) -> int:
        return self._result.params.n_events

    @property
    def n_observations(self) -> int:
        return self._result.params.n_observations

    @property
    def n_iter(self) -> int:
        return self._result.params.n_iter

    @property
    def converged(self) -> bool:
        return self._result.params.converged

    @property
    def ties(self) -> str:
        return self._result.params.ties

    @property
    def conf_level(self) -> float:
        """Confidence level for ``conf_int`` (default 0.95)."""
        return self._result.params.conf_level

    @property
    def n_strata(self) -> int:
        """Number of strata (1 for an unstratified fit)."""
        return self._result.params.n_strata

    @property
    def robust(self) -> bool:
        """Whether ``standard_errors`` are the sandwich (robust) estimator."""
        return self._result.params.robust

    @property
    def naive_standard_errors(self):
        """Model-based SEs when ``robust`` — else the same as
        ``standard_errors``."""
        naive = self._result.params.naive_standard_errors
        return naive if naive is not None else self.standard_errors

    @property
    def conf_int(self):
        """Wald confidence intervals for the coefficients, shape (p, 2).

        ``coef ± z * se`` on the coefficient (log-hazard-ratio) scale, where
        ``z`` is the normal quantile for ``conf_level`` (Cox inference is
        asymptotic-normal). ``exp(conf_int)`` gives hazard-ratio intervals.
        """
        import numpy as np
        from scipy import stats

        z = stats.norm.ppf((1.0 + self.conf_level) / 2.0)
        coef = self.coefficients
        se = self.standard_errors
        return np.column_stack([coef - z * se, coef + z * se])

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
        """R-style summary of Cox PH fit."""
        import numpy as np

        p = len(self.coefficients)
        names = self._names or tuple(f"x{i}" for i in range(p))
        max_name_len = max(len(n) for n in names)
        col_w = max(max_name_len, 10)

        lines = []
        lines.append("Call: coxph()")
        lines.append("")
        strata_note = f", {self.n_strata} strata" if self.n_strata > 1 else ""
        lines.append(
            f"  n= {self.n_observations}, "
            f"number of events= {self.n_events}{strata_note}"
        )
        lines.append("")

        # Coefficient table. When robust, show both the model-based se(coef)
        # and the robust se (as R's summary.coxph does), with z/p from robust.
        se_label = "robust se" if self.robust else "se(coef)"
        header = (f"  {'':{col_w}s}  {'coef':>10s}  {'exp(coef)':>10s}  "
                  f"{'se(coef)':>10s}  ")
        if self.robust:
            header += f"{'robust se':>10s}  "
        header += f"{'z':>10s}  {'Pr(>|z|)':>12s}"
        lines.append(header)
        naive_se = self.naive_standard_errors
        for i, name in enumerate(names):
            row = (f"  {name:>{col_w}s}  {self.coefficients[i]:10.6f}  "
                   f"{self.hazard_ratios[i]:10.6f}  "
                   f"{naive_se[i]:10.6f}  ")
            if self.robust:
                row += f"{self.standard_errors[i]:10.6f}  "
            row += f"{self.z_values[i]:10.4f}  {self.p_values[i]:12.4g}"
            lines.append(row)

        # HR confidence intervals (like R's summary.coxph), at conf_level.
        ci = np.exp(self.conf_int)                 # hazard-ratio scale
        lvl = f"{self.conf_level:.2f}".lstrip("0")  # e.g. ".95"
        lines.append("")
        lines.append(
            f"  {'':{col_w}s}  {'exp(coef)':>10s}  {'exp(-coef)':>10s}  "
            f"{'lower ' + lvl:>10s}  {'upper ' + lvl:>10s}"
        )
        for i, name in enumerate(names):
            lines.append(
                f"  {name:>{col_w}s}  {self.hazard_ratios[i]:10.4f}  "
                f"{1.0 / self.hazard_ratios[i]:10.4f}  "
                f"{ci[i, 0]:10.4f}  {ci[i, 1]:10.4f}"
            )

        lines.append("")
        lines.append(
            f"  Concordance= {self.concordance:.4f}"
        )
        lr_stat = 2 * (self.loglik[1] - self.loglik[0])
        lines.append(
            f"  Likelihood ratio test= {lr_stat:.4f} on {p} df"
        )
        lines.append(
            f"  n= {self.n_observations}, "
            f"number of events= {self.n_events}"
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CoxSolution(n={self.n_observations}, "
            f"events={self.n_events}, "
            f"concordance={self.concordance:.4f})"
        )
