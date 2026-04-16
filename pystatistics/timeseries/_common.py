"""
Common result types for time series analysis.

Defines frozen dataclasses for ACF/PACF results and stationarity test results.
All result types are immutable and carry full diagnostic information.
"""

from __future__ import annotations

from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass(frozen=True)
class ACFResult:
    """
    Result from autocorrelation or partial autocorrelation computation.

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
class StationarityResult:
    """
    Result from a stationarity test (ADF, KPSS, PP).

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
