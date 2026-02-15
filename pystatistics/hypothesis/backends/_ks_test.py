"""
Kolmogorov-Smirnov test implementation matching R's ks.test().

Supports:
- One-sample test against a theoretical distribution
- Two-sample test
- Exact and asymptotic p-values
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from scipy import stats as sp_stats

from pystatistics.hypothesis._common import HTestParams

if TYPE_CHECKING:
    from pystatistics.hypothesis.design import HypothesisDesign


def ks_test(design: HypothesisDesign) -> tuple[HTestParams, list[str]]:
    """Kolmogorov-Smirnov test matching R's ks.test()."""
    x = design.x
    alternative = design.alternative
    warnings_list: list[str] = []

    if design.y is not None:
        return _ks_two_sample(x, design.y, alternative, design.data_name,
                               warnings_list)
    elif design.distribution is not None:
        return _ks_one_sample(x, design.distribution, design.dist_params,
                               alternative, design.data_name, warnings_list)
    else:
        # Default: test against standard normal
        return _ks_one_sample(x, "norm", {}, alternative,
                               design.data_name, warnings_list)


def _ks_two_sample(
    x: np.ndarray, y: np.ndarray,
    alternative: str, data_name: str,
    warnings_list: list[str],
) -> tuple[HTestParams, list[str]]:
    """Two-sample KS test."""
    # Check for ties
    combined = np.concatenate([x, y])
    if len(np.unique(combined)) < len(combined):
        warnings_list.append(
            "p-value will be approximate in the presence of ties"
        )

    # Map alternative for scipy
    scipy_alt = alternative.replace(".", "-")
    stat, p_value = sp_stats.ks_2samp(x, y, alternative=scipy_alt)

    # R uses D, D+, D- for the statistic name
    if alternative == "two.sided":
        stat_name = "D"
    elif alternative == "less":
        stat_name = "D^-"
    else:
        stat_name = "D^+"

    # R uses exact test for n.x * n.y < 10000 (regardless of ties)
    # scipy.stats.ks_2samp also uses exact for small samples automatically
    nx, ny = len(x), len(y)
    if nx * ny < 10000:
        method = "Exact two-sample Kolmogorov-Smirnov test"
    else:
        method = "Asymptotic two-sample Kolmogorov-Smirnov test"

    return HTestParams(
        statistic=float(stat),
        statistic_name=stat_name,
        parameter=None,
        p_value=float(p_value),
        conf_int=None,
        conf_level=0.95,
        estimate=None,
        null_value=None,
        alternative=alternative,
        method=method,
        data_name=data_name,
    ), warnings_list


def _ks_one_sample(
    x: np.ndarray, distribution: str,
    dist_params: dict, alternative: str,
    data_name: str, warnings_list: list[str],
) -> tuple[HTestParams, list[str]]:
    """One-sample KS test against a theoretical distribution."""
    # Map distribution name to scipy CDF
    cdf = _get_cdf(distribution, dist_params)

    # Map alternative for scipy
    scipy_alt = alternative.replace(".", "-")
    stat, p_value = sp_stats.kstest(x, cdf, alternative=scipy_alt)

    if alternative == "two.sided":
        stat_name = "D"
    elif alternative == "less":
        stat_name = "D^-"
    else:
        stat_name = "D^+"

    method = "Exact one-sample Kolmogorov-Smirnov test"

    return HTestParams(
        statistic=float(stat),
        statistic_name=stat_name,
        parameter=None,
        p_value=float(p_value),
        conf_int=None,
        conf_level=0.95,
        estimate=None,
        null_value=None,
        alternative=alternative,
        method=method,
        data_name=data_name,
    ), warnings_list


def _get_cdf(distribution: str, params: dict):
    """Map distribution name to scipy CDF function."""
    # Map R distribution names to scipy
    dist_map = {
        "norm": sp_stats.norm,
        "pnorm": sp_stats.norm,
        "unif": sp_stats.uniform,
        "punif": sp_stats.uniform,
        "exp": sp_stats.expon,
        "pexp": sp_stats.expon,
    }

    dist_name = distribution.lower()
    if dist_name not in dist_map:
        raise ValueError(
            f"Unknown distribution: {distribution!r}. "
            f"Supported: {list(dist_map.keys())}"
        )

    dist = dist_map[dist_name]

    # Map R parameter names to scipy
    if dist_name in ("norm", "pnorm"):
        loc = params.get("mean", 0.0)
        scale = params.get("sd", 1.0)
        return lambda x: dist.cdf(x, loc=loc, scale=scale)
    elif dist_name in ("unif", "punif"):
        a = params.get("min", 0.0)
        b = params.get("max", 1.0)
        return lambda x: dist.cdf(x, loc=a, scale=b - a)
    elif dist_name in ("exp", "pexp"):
        rate = params.get("rate", 1.0)
        return lambda x: dist.cdf(x, scale=1.0 / rate)
    else:
        return dist.cdf
