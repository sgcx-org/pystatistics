"""
F-test for equality of two variances, matching R's var.test().

Supports:
- Two-sample F-test
- One-sided and two-sided alternatives
- Confidence interval for variance ratio
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from scipy import stats as sp_stats

from pystatistics.hypothesis._common import HTestParams

if TYPE_CHECKING:
    from pystatistics.hypothesis.design import HypothesisDesign


def var_test(design: HypothesisDesign) -> tuple[HTestParams, list[str]]:
    """F-test for equality of two variances, matching R's var.test()."""
    x = design.x
    y = design.y
    ratio = design.ratio
    alternative = design.alternative
    conf_level = design.conf_level
    warnings_list: list[str] = []

    nx = len(x)
    ny = len(y)
    df_x = float(nx - 1)
    df_y = float(ny - 1)

    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))

    # F statistic: ratio of sample variances divided by null ratio
    f_stat = (var_x / var_y) / ratio

    # p-value
    if alternative == "two.sided":
        p_value = 2.0 * min(
            float(sp_stats.f.cdf(f_stat, df_x, df_y)),
            float(sp_stats.f.sf(f_stat, df_x, df_y)),
        )
    elif alternative == "less":
        p_value = float(sp_stats.f.cdf(f_stat, df_x, df_y))
    else:  # greater
        p_value = float(sp_stats.f.sf(f_stat, df_x, df_y))

    # Confidence interval for the ratio of variances
    alpha = 1.0 - conf_level
    if alternative == "two.sided":
        ci_lo = f_stat / float(sp_stats.f.ppf(1.0 - alpha / 2.0, df_x, df_y))
        ci_hi = f_stat / float(sp_stats.f.ppf(alpha / 2.0, df_x, df_y))
    elif alternative == "less":
        ci_lo = 0.0
        ci_hi = f_stat / float(sp_stats.f.ppf(alpha, df_x, df_y))
    else:  # greater
        ci_lo = f_stat / float(sp_stats.f.ppf(1.0 - alpha, df_x, df_y))
        ci_hi = float('inf')

    estimate = {"ratio of variances": float(var_x / var_y)}
    null_value = {"ratio of variances": ratio}
    parameter = {"num df": df_x, "denom df": df_y}

    method = "F test to compare two variances"

    return HTestParams(
        statistic=float(f_stat),
        statistic_name="F",
        parameter=parameter,
        p_value=p_value,
        conf_int=np.array([ci_lo, ci_hi]),
        conf_level=conf_level,
        estimate=estimate,
        null_value=null_value,
        alternative=alternative,
        method=method,
        data_name=design.data_name,
    ), warnings_list
