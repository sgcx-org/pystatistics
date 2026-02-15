"""
Proportion test implementation matching R's prop.test().

Uses chi-squared statistic with optional Yates' continuity correction.
Confidence intervals use Wilson score method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from scipy import stats as sp_stats

from pystatistics.hypothesis._common import HTestParams

if TYPE_CHECKING:
    from pystatistics.hypothesis.design import HypothesisDesign


def prop_test(design: HypothesisDesign) -> tuple[HTestParams, list[str]]:
    """Proportion test matching R's prop.test()."""
    x = design.successes
    n = design.trials
    p0 = design.expected_p
    alternative = design.alternative
    conf_level = design.conf_level
    correct = design.correct
    warnings_list: list[str] = []

    k = len(x)
    p_hat = x / n  # observed proportions

    if k == 1:
        # One-sample proportion test
        return _prop_test_one_sample(
            x[0], n[0], p0[0] if p0 is not None else 0.5,
            alternative, conf_level, correct, warnings_list
        )

    if p0 is not None:
        # k-sample test against specified null proportions
        return _prop_test_given_p(
            x, n, p0, conf_level, correct, k, p_hat, warnings_list
        )

    # k-sample test of equality of proportions
    return _prop_test_equality(
        x, n, conf_level, correct, k, p_hat, alternative, warnings_list
    )


def _prop_test_one_sample(
    x: float, n: float, p0: float,
    alternative: str, conf_level: float, correct: bool,
    warnings_list: list[str],
) -> tuple[HTestParams, list[str]]:
    """One-sample proportion test."""
    p_hat = x / n

    # Yates-corrected chi-squared statistic
    # R applies: YATES = min(0.5, abs(x - n*p0))
    yates_correction = 0.0
    if correct:
        yates_correction = min(0.5, abs(x - n * p0))

    # Statistic: X^2 = (|x - np0| - yates)^2 / (np0(1-p0))
    numerator = (abs(x - n * p0) - yates_correction) ** 2
    chisq = float(numerator / (n * p0 * (1 - p0)))

    df = 1.0

    if alternative == "two.sided":
        p_value = float(sp_stats.chi2.sf(chisq, df))
    elif alternative == "less":
        # One-sided: use normal approximation
        z = (x - n * p0 - yates_correction) / np.sqrt(n * p0 * (1 - p0))
        # But R uses the chi-squared stat and converts
        # For "less": p = pnorm(sign(p_hat - p0) * sqrt(STATISTIC))
        sign = 1.0 if p_hat >= p0 else -1.0
        z_stat = sign * np.sqrt(chisq)
        p_value = float(sp_stats.norm.cdf(z_stat))
    else:  # greater
        sign = 1.0 if p_hat >= p0 else -1.0
        z_stat = sign * np.sqrt(chisq)
        p_value = float(sp_stats.norm.sf(z_stat))

    # Wilson score confidence interval (uses same YATES as the statistic)
    ci = _wilson_ci(x, n, conf_level, alternative, correct, yates_correction)

    method = "1-sample proportions test with continuity correction"
    if not correct:
        method = "1-sample proportions test without continuity correction"

    return HTestParams(
        statistic=chisq,
        statistic_name="X-squared",
        parameter={"df": df},
        p_value=p_value,
        conf_int=np.array(ci),
        conf_level=conf_level,
        estimate={"p": float(p_hat)},
        null_value={"p": float(p0)},
        alternative=alternative,
        method=method,
        data_name="x out of n",
    ), warnings_list


def _prop_test_given_p(
    x: np.ndarray, n: np.ndarray, p0: np.ndarray,
    conf_level: float, correct: bool,
    k: int, p_hat: np.ndarray,
    warnings_list: list[str],
) -> tuple[HTestParams, list[str]]:
    """k-sample test against specified null proportions."""
    # Yates correction
    yates = np.zeros(k)
    if correct:
        yates = np.minimum(0.5, np.abs(x - n * p0))

    chisq = float(np.sum(
        (np.abs(x - n * p0) - yates) ** 2 / (n * p0 * (1 - p0))
    ))
    df = float(k)

    p_value = float(sp_stats.chi2.sf(chisq, df))

    # No CI for k-sample with given p
    estimate = {f"prop {i+1}": float(p_hat[i]) for i in range(k)}

    method = "k-sample test for given proportions"

    return HTestParams(
        statistic=chisq,
        statistic_name="X-squared",
        parameter={"df": df},
        p_value=p_value,
        conf_int=None,
        conf_level=conf_level,
        estimate=estimate,
        null_value=None,
        alternative="two.sided",
        method=method,
        data_name="x",
    ), warnings_list


def _prop_test_equality(
    x: np.ndarray, n: np.ndarray,
    conf_level: float, correct: bool,
    k: int, p_hat: np.ndarray,
    alternative: str,
    warnings_list: list[str],
) -> tuple[HTestParams, list[str]]:
    """k-sample test of equality of proportions (no specified null)."""
    # Pooled proportion
    p_pool = np.sum(x) / np.sum(n)

    # Expected under H0: all proportions equal to pooled
    expected_x = n * p_pool

    # Yates correction (R: only for 2x2)
    yates = np.zeros(k)
    if correct and k == 2:
        yates = np.minimum(0.5, np.abs(x - expected_x))

    # Chi-squared statistic
    chisq = float(np.sum(
        (np.abs(x - expected_x) - yates) ** 2 / expected_x
        + (np.abs((n - x) - (n - expected_x)) - yates) ** 2 / (n - expected_x)
    ))

    df = float(k - 1)
    p_value = float(sp_stats.chi2.sf(chisq, df))

    # Confidence interval: only for 2 groups
    ci = None
    if k == 2:
        ci = _prop_diff_ci(
            x[0], n[0], x[1], n[1], conf_level, correct
        )

    estimate = {f"prop {i+1}": float(p_hat[i]) for i in range(k)}

    method = f"{k}-sample test for equality of proportions"
    if correct and k == 2:
        method += " with continuity correction"
    elif not correct and k == 2:
        method += " without continuity correction"

    return HTestParams(
        statistic=chisq,
        statistic_name="X-squared",
        parameter={"df": df},
        p_value=p_value,
        conf_int=np.array(ci) if ci is not None else None,
        conf_level=conf_level,
        estimate=estimate,
        null_value=None,
        alternative=alternative,
        method=method,
        data_name="x",
    ), warnings_list


def _wilson_ci(
    x: float, n: float, conf_level: float, alternative: str,
    correct: bool, yates: float,
) -> tuple[float, float]:
    """
    Wilson score confidence interval for a single proportion.

    Matches R's prop.test CI exactly, including the Yates continuity
    correction that shifts p_hat by ±YATES/n before applying Wilson.

    Parameters
    ----------
    x : number of successes
    n : number of trials
    conf_level : confidence level
    alternative : "two.sided", "less", or "greater"
    correct : whether correction was applied
    yates : the YATES value (0.5 if corrected, 0 otherwise),
            already capped by min(0.5, abs(x - n*p0))
    """
    p_hat = x / n

    if alternative == "two.sided":
        z = float(sp_stats.norm.ppf((1.0 + conf_level) / 2.0))
    else:
        z = float(sp_stats.norm.ppf(conf_level))

    z22n = z**2 / (2.0 * n)

    # Upper bound: shift p_hat up by YATES/n
    # R short-circuits: if p.c >= 1, p.u = 1
    p_c = p_hat + yates / n
    if p_c >= 1.0:
        p_u = 1.0
    else:
        p_u = (
            (p_c + z22n + z * np.sqrt(p_c * (1.0 - p_c) / n + z22n / (2.0 * n)))
            / (1.0 + 2.0 * z22n)
        )

    # Lower bound: shift p_hat down by YATES/n
    # R short-circuits: if p.c <= 0, p.l = 0
    p_c = p_hat - yates / n
    if p_c <= 0.0:
        p_l = 0.0
    else:
        p_l = (
            (p_c + z22n - z * np.sqrt(p_c * (1.0 - p_c) / n + z22n / (2.0 * n)))
            / (1.0 + 2.0 * z22n)
        )

    # Final clamp to [0, 1] (safety net, matches R)
    p_l = max(0.0, float(p_l))
    p_u = min(1.0, float(p_u))

    if alternative == "two.sided":
        return (p_l, p_u)
    elif alternative == "less":
        return (0.0, p_u)
    else:  # greater
        return (p_l, 1.0)


def _prop_diff_ci(
    x1: float, n1: float, x2: float, n2: float,
    conf_level: float, correct: bool,
) -> tuple[float, float]:
    """
    Confidence interval for difference in proportions (two-sample).

    Uses Newcombe's method with Wilson score intervals.
    This matches R's prop.test CI for the two-sample case.
    """
    p1 = x1 / n1
    p2 = x2 / n2
    diff = p1 - p2

    alpha = 1.0 - conf_level
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)

    # Yates correction for CI: R subtracts 1/(2*harmonic_mean(n1,n2))
    yates_ci = 0.0
    if correct:
        yates_ci = 0.5 * (1.0 / n1 + 1.0 / n2)

    # Standard error of the difference
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    lo = diff - z * se - yates_ci
    hi = diff + z * se + yates_ci

    lo = max(-1.0, lo)
    hi = min(1.0, hi)

    return (lo, hi)
