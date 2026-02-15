"""
Chi-squared test implementation matching R's chisq.test().

Supports:
- Independence test (contingency table)
- Goodness-of-fit test (observed vs expected)
- Yates continuity correction (2x2 tables)
- Monte Carlo simulation for p-value
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from scipy import stats as sp_stats

from pystatistics.hypothesis._common import HTestParams

if TYPE_CHECKING:
    from pystatistics.hypothesis.design import HypothesisDesign


def chisq_independence(design: HypothesisDesign) -> tuple[HTestParams, list[str]]:
    """Chi-squared test of independence for a contingency table."""
    table = design.table.copy()
    correct = design.correct
    simulate = design.simulate_p_value
    B = design.n_monte_carlo
    warnings_list: list[str] = []

    nrow, ncol = table.shape
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    total = table.sum()

    # Expected counts: E[i,j] = row_sum[i] * col_sum[j] / total
    expected = np.outer(row_sums, col_sums) / total

    # Yates correction: only for 2x2 tables
    yates = correct and nrow == 2 and ncol == 2

    if yates:
        diff = np.abs(table - expected) - 0.5
        diff = np.maximum(diff, 0.0)
        chisq = float(np.sum(diff ** 2 / expected))
        method = (
            "Pearson's Chi-squared test with Yates' continuity correction"
        )
    else:
        chisq = float(np.sum((table - expected) ** 2 / expected))
        method = "Pearson's Chi-squared test"

    df = float((nrow - 1) * (ncol - 1))

    # Warning for small expected counts
    if np.any(expected < 5):
        warnings_list.append(
            "Chi-squared approximation may be incorrect"
        )

    if simulate:
        # Monte Carlo p-value
        p_value = _monte_carlo_independence(table, chisq, B)
        method += (
            f" with simulated p-value\n\t(based on {B} replicates)"
        )
        parameter = None  # R returns NA for df when simulated
    else:
        p_value = float(sp_stats.chi2.sf(chisq, df))
        parameter = {"df": df}

    # Residuals
    residuals = (table - expected) / np.sqrt(expected)

    # Standardized residuals
    row_prop = row_sums / total
    col_prop = col_sums / total
    v = np.outer(1.0 - row_prop, 1.0 - col_prop)
    stdres = residuals / np.sqrt(v)

    return HTestParams(
        statistic=chisq,
        statistic_name="X-squared",
        parameter=parameter,
        p_value=p_value,
        conf_int=None,
        conf_level=0.95,
        estimate=None,
        null_value=None,
        alternative="two.sided",
        method=method,
        data_name=design.data_name,
        extras={
            "observed": table,
            "expected": expected,
            "residuals": residuals,
            "stdres": stdres,
        },
    ), warnings_list


def chisq_gof(design: HypothesisDesign) -> tuple[HTestParams, list[str]]:
    """Chi-squared goodness-of-fit test."""
    observed = design.x
    p = design.expected_p
    simulate = design.simulate_p_value
    B = design.n_monte_carlo
    warnings_list: list[str] = []

    n = np.sum(observed)
    k = len(observed)

    if p is None:
        p = np.ones(k) / k
    else:
        p = np.asarray(p, dtype=np.float64)
        if design.rescale_p:
            p = p / np.sum(p)

    expected = n * p
    chisq = float(np.sum((observed - expected) ** 2 / expected))
    df = float(k - 1)

    if np.any(expected < 5):
        warnings_list.append(
            "Chi-squared approximation may be incorrect"
        )

    if simulate:
        p_value = _monte_carlo_gof(observed, p, chisq, B)
        method = (
            "Chi-squared test for given probabilities with simulated p-value"
            f"\n\t(based on {B} replicates)"
        )
        parameter = None
    else:
        p_value = float(sp_stats.chi2.sf(chisq, df))
        method = "Chi-squared test for given probabilities"
        parameter = {"df": df}

    return HTestParams(
        statistic=chisq,
        statistic_name="X-squared",
        parameter=parameter,
        p_value=p_value,
        conf_int=None,
        conf_level=0.95,
        estimate=None,
        null_value=None,
        alternative="two.sided",
        method=method,
        data_name=design.data_name,
        extras={
            "observed": observed.copy(),
            "expected": expected,
        },
    ), warnings_list


def _monte_carlo_independence(
    table: np.ndarray, observed_stat: float, B: int
) -> float:
    """Monte Carlo p-value for independence test using random tables."""
    from scipy.stats import random_table

    row_sums = table.sum(axis=1).astype(int)
    col_sums = table.sum(axis=0).astype(int)
    expected = np.outer(row_sums, col_sums) / float(table.sum())

    dist = random_table(row_sums, col_sums)

    count = 0
    for _ in range(B):
        sim_table = dist.rvs()
        sim_stat = np.sum((sim_table - expected) ** 2 / expected)
        if sim_stat >= observed_stat - 1e-12:
            count += 1

    return (count + 1) / (B + 1)


def _monte_carlo_gof(
    observed: np.ndarray, p: np.ndarray, observed_stat: float, B: int
) -> float:
    """Monte Carlo p-value for GOF test."""
    rng = np.random.default_rng()
    n = int(np.sum(observed))
    expected = n * p

    count = 0
    for _ in range(B):
        sim = rng.multinomial(n, p).astype(np.float64)
        sim_stat = np.sum((sim - expected) ** 2 / expected)
        if sim_stat >= observed_stat - 1e-12:
            count += 1

    return (count + 1) / (B + 1)


def _r2dtable(
    row_sums: np.ndarray,
    col_sums: np.ndarray,
    total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a random contingency table with fixed marginals.

    Uses multinomial row-sampling (Patefield's algorithm simplified).
    """
    nrow = len(row_sums)
    ncol = len(col_sums)
    table = np.zeros((nrow, ncol), dtype=np.float64)

    remaining_col = col_sums.astype(np.float64).copy()
    remaining_total = float(total)

    for i in range(nrow - 1):
        ri = int(row_sums[i])
        probs = remaining_col / remaining_total
        row = rng.multinomial(ri, probs).astype(np.float64)
        table[i] = row
        remaining_col -= row
        remaining_total -= ri

    # Last row is determined
    table[nrow - 1] = remaining_col
    return table
