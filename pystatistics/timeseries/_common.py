"""
Common result types for time series analysis.

Library-standard "Solution wraps Result[Params]" pattern:
- ACFSolution / ACFParams: ACF / PACF results
- StationaritySolution / StationarityParams: stationarity test results

Each ``*Params`` is a frozen dataclass holding only the computed data
fields. Each ``*Solution`` wraps a :class:`Result` envelope and exposes
every datum via ``@property`` plus the shared metadata accessors
(``.info``, ``.timing``, ``.backend_name``, ``.warnings``).
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from pystatistics.core.result import Result, SolutionReprMixin


@dataclass(frozen=True)
class ACFParams:
    """Immutable parameter payload for ACF / PACF results.

    Attributes
    ----------
    acf : NDArray
        Autocorrelation values at each lag. For ACF, includes lag 0 (= 1.0).
        For PACF, starts at lag 1 (matching R's pacf()).
    lags : NDArray
        Lag indices corresponding to each acf value.
    n_obs : int
        Number of observations in the original series.
    conf_level : float
        Confidence level used for the confidence bands.
    ci_upper : NDArray
        Upper confidence bound per lag.
    ci_lower : NDArray
        Lower confidence bound per lag.
    type : str
        'correlation' for ACF or 'partial' for PACF.
    """

    acf: NDArray
    lags: NDArray
    n_obs: int
    conf_level: float
    ci_upper: NDArray
    ci_lower: NDArray
    type: str


@dataclass
class ACFSolution(SolutionReprMixin):
    """
    Result from autocorrelation or partial autocorrelation computation.

    Wraps a :class:`Result` ``[ACFParams]`` envelope; every datum is
    exposed via a read-only ``@property`` so the public attribute
    surface is unchanged from the previous flat dataclass.

    Attributes
    ----------
    acf : NDArray
        Autocorrelation values at each lag. For ACF, includes lag 0 (= 1.0).
        For PACF, starts at lag 1 (matching R's pacf()).
    lags : NDArray
        Lag indices corresponding to each acf value.
    n_obs : int
        Number of observations in the original series.
    conf_level : float
        Confidence level used for the confidence bands.
    ci_upper : NDArray
        Upper confidence bound per lag.
    ci_lower : NDArray
        Lower confidence bound per lag.
    type : str
        'correlation' for ACF or 'partial' for PACF.
    """

    _result: Result[ACFParams]

    @property
    def acf(self) -> NDArray:
        return self._result.params.acf

    @property
    def lags(self) -> NDArray:
        return self._result.params.lags

    @property
    def n_obs(self) -> int:
        return self._result.params.n_obs

    @property
    def conf_level(self) -> float:
        return self._result.params.conf_level

    @property
    def ci_upper(self) -> NDArray:
        return self._result.params.ci_upper

    @property
    def ci_lower(self) -> NDArray:
        return self._result.params.ci_lower

    @property
    def type(self) -> str:
        return self._result.params.type

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
        Return a human-readable summary of the ACF/PACF result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        label = "Autocorrelation" if self.type == "correlation" else "Partial Autocorrelation"
        lines = [
            f"{label} Function",
            f"  Observations: {self.n_obs}",
            f"  Max lag:      {int(self.lags[-1])}",
            f"  Conf. level:  {self.conf_level}",
            "",
            "  Lag    ACF",
            "  ---    -------",
        ]
        for lag_val, acf_val in zip(self.lags, self.acf):
            lines.append(f"  {int(lag_val):>3}    {acf_val:>8.4f}")
        return "\n".join(lines)



@dataclass(frozen=True)
class StationarityParams:
    """Immutable parameter payload for a stationarity test.

    Attributes
    ----------
    statistic : float
        Test statistic value.
    p_value : float
        p-value of the test. For KPSS, may be truncated to the table range
        (0.01 to 0.10).
    method : str
        Name of the test, e.g. 'Augmented Dickey-Fuller', 'KPSS'.
    alternative : str
        Alternative hypothesis description, e.g. 'stationary' for ADF,
        'unit root' for KPSS.
    n_lags : int
        Number of lags used in the test.
    n_obs : int
        Number of observations used (after differencing/lag adjustments).
    critical_values : dict[str, float] | None
        Critical values at standard significance levels,
        e.g. {'1%': -3.43, '5%': -2.86, '10%': -2.57}.
    """

    statistic: float
    p_value: float
    method: str
    alternative: str
    n_lags: int
    n_obs: int
    critical_values: dict[str, float] | None


@dataclass
class StationaritySolution(SolutionReprMixin):
    """
    Result from a stationarity test (ADF, KPSS, PP).

    Wraps a :class:`Result` ``[StationarityParams]`` envelope; every datum
    is exposed via a read-only ``@property`` so the public attribute
    surface is unchanged from the previous flat dataclass.

    Attributes
    ----------
    statistic : float
        Test statistic value.
    p_value : float
        p-value of the test. For KPSS, may be truncated to the table range
        (0.01 to 0.10).
    method : str
        Name of the test, e.g. 'Augmented Dickey-Fuller', 'KPSS'.
    alternative : str
        Alternative hypothesis description, e.g. 'stationary' for ADF,
        'unit root' for KPSS.
    n_lags : int
        Number of lags used in the test.
    n_obs : int
        Number of observations used (after differencing/lag adjustments).
    critical_values : dict[str, float] | None
        Critical values at standard significance levels,
        e.g. {'1%': -3.43, '5%': -2.86, '10%': -2.57}.
    """

    _result: Result[StationarityParams]

    @property
    def statistic(self) -> float:
        return self._result.params.statistic

    @property
    def p_value(self) -> float:
        return self._result.params.p_value

    @property
    def method(self) -> str:
        return self._result.params.method

    @property
    def alternative(self) -> str:
        return self._result.params.alternative

    @property
    def n_lags(self) -> int:
        return self._result.params.n_lags

    @property
    def n_obs(self) -> int:
        return self._result.params.n_obs

    @property
    def critical_values(self) -> dict[str, float] | None:
        return self._result.params.critical_values

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
        Return a human-readable summary of the stationarity test result.

        Returns
        -------
        str
            Multi-line summary string matching R-style output.
        """
        lines = [
            self.method,
            "",
            f"  Test statistic:  {self.statistic:.4f}",
        ]
        # For KPSS, indicate truncation if p-value is at boundary
        if "KPSS" in self.method:
            if self.p_value <= 0.01:
                lines.append(f"  p-value:         < 0.01")
            elif self.p_value >= 0.10:
                lines.append(f"  p-value:         > 0.10")
            else:
                lines.append(f"  p-value:         {self.p_value:.4f}")
        else:
            lines.append(f"  p-value:         {self.p_value:.4f}")
        lines.extend([
            f"  Alternative:     {self.alternative}",
            f"  Lags used:       {self.n_lags}",
            f"  Observations:    {self.n_obs}",
        ])
        if self.critical_values is not None:
            lines.append("")
            lines.append("  Critical values:")
            for level, value in sorted(self.critical_values.items()):
                lines.append(f"    {level:>5s}  {value:.4f}")
        return "\n".join(lines)

