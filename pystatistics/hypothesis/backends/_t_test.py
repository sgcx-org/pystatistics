"""
t-test implementation matching R's t.test().

Supports one-sample, two-sample (Welch and pooled), and paired tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from scipy import stats as sp_stats

from pystatistics.hypothesis._common import HTestParams

if TYPE_CHECKING:
    from pystatistics.hypothesis.design import HypothesisDesign


def t_one_sample(design: HypothesisDesign) -> tuple[HTestParams, list[str]]:
    """One-sample t-test: H0: mean(x) = mu."""
    x = design.x
    mu = design.mu
    alternative = design.alternative
    conf_level = design.conf_level
    warnings_list: list[str] = []

    n = len(x)
    mean_x = np.mean(x)
    var_x = np.var(x, ddof=1)
    se = np.sqrt(var_x / n)
    df = float(n - 1)

    if se == 0.0:
        warnings_list.append("data are essentially constant")
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat = float((mean_x - mu) / se)
        p_value = _t_pvalue(t_stat, df, alternative)

    ci = _t_conf_int(mean_x, se, df, conf_level, alternative)

    return HTestParams(
        statistic=t_stat,
        statistic_name="t",
        parameter={"df": df},
        p_value=p_value,
        conf_int=ci,
        conf_level=conf_level,
        estimate={"mean of x": float(mean_x)},
        null_value={"mean": mu},
        alternative=alternative,
        method="One Sample t-test",
        data_name=design.data_name,
    ), warnings_list


def t_paired(design: HypothesisDesign) -> tuple[HTestParams, list[str]]:
    """Paired t-test: H0: mean(x - y) = mu.

    Note: design.x already contains the paired differences (NaN pairs removed).
    """
    d = design.x
    mu = design.mu
    alternative = design.alternative
    conf_level = design.conf_level
    warnings_list: list[str] = []

    n = len(d)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    se = np.sqrt(var_d / n)
    df = float(n - 1)

    if se == 0.0:
        warnings_list.append("data are essentially constant")
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat = float((mean_d - mu) / se)
        p_value = _t_pvalue(t_stat, df, alternative)

    ci = _t_conf_int(mean_d, se, df, conf_level, alternative)

    return HTestParams(
        statistic=t_stat,
        statistic_name="t",
        parameter={"df": df},
        p_value=p_value,
        conf_int=ci,
        conf_level=conf_level,
        estimate={"mean difference": float(mean_d)},
        null_value={"mean difference": mu},
        alternative=alternative,
        method="Paired t-test",
        data_name=design.data_name,
    ), warnings_list


def t_two_sample(design: HypothesisDesign) -> tuple[HTestParams, list[str]]:
    """Two-sample t-test: Welch (default) or pooled."""
    x = design.x
    y = design.y
    mu = design.mu
    var_equal = design.var_equal
    alternative = design.alternative
    conf_level = design.conf_level
    warnings_list: list[str] = []

    n1, n2 = len(x), len(y)
    mean1, mean2 = np.mean(x), np.mean(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    diff = mean1 - mean2

    if var_equal:
        # Pooled (Student's) t-test
        df = float(n1 + n2 - 2)
        sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / df
        se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        method = " Two Sample t-test"
    else:
        # Welch t-test (default, matches R)
        v1 = var1 / n1
        v2 = var2 / n2
        se = np.sqrt(v1 + v2)
        # Welch-Satterthwaite degrees of freedom (fractional, DO NOT round)
        if se == 0.0:
            df = 0.0
        else:
            df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
        method = "Welch Two Sample t-test"

    if se == 0.0:
        warnings_list.append("data are essentially constant")
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat = float((diff - mu) / se)
        p_value = _t_pvalue(t_stat, df, alternative)

    ci = _t_conf_int(diff, se, df, conf_level, alternative)

    return HTestParams(
        statistic=t_stat,
        statistic_name="t",
        parameter={"df": df},
        p_value=p_value,
        conf_int=ci,
        conf_level=conf_level,
        estimate={"mean of x": float(mean1), "mean of y": float(mean2)},
        null_value={"difference in means": mu},
        alternative=alternative,
        method=method,
        data_name=design.data_name,
    ), warnings_list


# --- Helpers ---

def _t_pvalue(t_stat: float, df: float, alternative: str) -> float:
    """Compute p-value from t distribution."""
    if np.isnan(t_stat) or np.isnan(df) or df <= 0:
        return np.nan
    if alternative == "two.sided":
        return float(2.0 * sp_stats.t.sf(abs(t_stat), df))
    elif alternative == "less":
        return float(sp_stats.t.cdf(t_stat, df))
    else:  # greater
        return float(sp_stats.t.sf(t_stat, df))


def _t_conf_int(
    estimate: float,
    se: float,
    df: float,
    conf_level: float,
    alternative: str,
) -> np.ndarray:
    """Compute confidence interval for t-test."""
    alpha = 1.0 - conf_level

    if np.isnan(se) or np.isnan(df) or df <= 0 or se == 0.0:
        return np.array([np.nan, np.nan])

    if alternative == "two.sided":
        t_crit = sp_stats.t.ppf(1.0 - alpha / 2.0, df)
        return np.array([estimate - t_crit * se, estimate + t_crit * se])
    elif alternative == "less":
        t_crit = sp_stats.t.ppf(1.0 - alpha, df)
        return np.array([-np.inf, estimate + t_crit * se])
    else:  # greater
        t_crit = sp_stats.t.ppf(1.0 - alpha, df)
        return np.array([estimate - t_crit * se, np.inf])
