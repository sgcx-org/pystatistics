"""
Wilcoxon rank-sum and signed-rank test matching R's wilcox.test().

Supports:
- Signed-rank test (one-sample or paired)
- Rank-sum test (two-sample, Mann-Whitney U)
- Exact distribution for small n without ties
- Normal approximation with tie correction and continuity correction
- Hodges-Lehmann confidence interval
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from scipy import stats as sp_stats

from pystatistics.hypothesis._common import HTestParams

if TYPE_CHECKING:
    from pystatistics.hypothesis.design import HypothesisDesign


def wilcox_signed_rank(design: HypothesisDesign) -> tuple[HTestParams, list[str]]:
    """Wilcoxon signed-rank test (one-sample or paired)."""
    x = design.x
    mu = design.mu
    alternative = design.alternative
    conf_level = design.conf_level
    correct = design.correct
    exact_flag = design.exact
    compute_ci = design.compute_wilcox_ci
    warnings_list: list[str] = []

    # Differences from mu
    d = x - mu

    # Remove zeros (R: "zeroes in 'x - mu'")
    nonzero_mask = d != 0.0
    n_zeros = int(np.sum(~nonzero_mask))
    if n_zeros > 0:
        warnings_list.append(
            "cannot compute exact p-value with zeroes"
        )
    d = d[nonzero_mask]
    n = len(d)

    if n == 0:
        # All differences are zero
        return HTestParams(
            statistic=0.0,
            statistic_name="V",
            parameter=None,
            p_value=1.0,
            conf_int=None,
            conf_level=conf_level,
            estimate=None,
            null_value={"location": mu},
            alternative=alternative,
            method="Wilcoxon signed rank test",
            data_name=design.data_name,
        ), warnings_list

    # Rank |differences|
    abs_d = np.abs(d)
    ranks = sp_stats.rankdata(abs_d, method='average')

    # Check for ties in ranks
    has_ties = len(np.unique(ranks)) < len(ranks)
    if has_ties:
        warnings_list.append(
            "cannot compute exact p-value with ties"
        )

    # Test statistic V = sum of positive ranks
    V = float(np.sum(ranks[d > 0]))

    # Determine if exact
    use_exact = False
    if exact_flag is True and not has_ties and n_zeros == 0:
        use_exact = True
    elif exact_flag is None and not has_ties and n_zeros == 0 and n < 50:
        use_exact = True

    if use_exact:
        p_value = _signed_rank_exact_p(V, n, alternative)
        method = "Wilcoxon signed rank exact test"
    else:
        p_value = _signed_rank_normal_p(V, n, ranks, alternative, correct)
        method = "Wilcoxon signed rank test with continuity correction"
        if not correct:
            method = "Wilcoxon signed rank test"

    # Confidence interval and pseudomedian via Walsh averages
    ci = None
    estimate = None
    if compute_ci:
        walsh = _walsh_averages(x - mu)
        pseudomedian = float(np.median(walsh)) + mu
        estimate = {"(pseudo)median": pseudomedian}

        if alternative == "two.sided":
            alpha = 1.0 - conf_level
            lo = float(np.percentile(walsh, 100 * alpha / 2)) + mu
            hi = float(np.percentile(walsh, 100 * (1 - alpha / 2))) + mu
            ci = (lo, hi)
        elif alternative == "less":
            alpha = 1.0 - conf_level
            hi = float(np.percentile(walsh, 100 * (1 - alpha))) + mu
            ci = (float('-inf'), hi)
        else:  # greater
            alpha = 1.0 - conf_level
            lo = float(np.percentile(walsh, 100 * alpha)) + mu
            ci = (lo, float('inf'))

    return HTestParams(
        statistic=V,
        statistic_name="V",
        parameter=None,
        p_value=float(p_value),
        conf_int=np.array(ci) if ci is not None else None,
        conf_level=conf_level,
        estimate=estimate,
        null_value={"location": mu},
        alternative=alternative,
        method=method,
        data_name=design.data_name,
    ), warnings_list


def wilcox_rank_sum(design: HypothesisDesign) -> tuple[HTestParams, list[str]]:
    """Wilcoxon rank-sum test (Mann-Whitney U)."""
    x = design.x
    y = design.y
    mu = design.mu
    alternative = design.alternative
    conf_level = design.conf_level
    correct = design.correct
    exact_flag = design.exact
    compute_ci = design.compute_wilcox_ci
    warnings_list: list[str] = []

    nx = len(x)
    ny = len(y)

    # Pool and rank
    combined = np.concatenate([x - mu, y])
    ranks = sp_stats.rankdata(combined, method='average')
    W = float(np.sum(ranks[:nx])) - nx * (nx + 1) / 2.0

    # Check for ties
    has_ties = len(np.unique(ranks)) < len(ranks)
    if has_ties:
        warnings_list.append(
            "cannot compute exact p-value with ties"
        )

    # Determine if exact
    use_exact = False
    if exact_flag is True and not has_ties:
        use_exact = True
    elif exact_flag is None and not has_ties and nx < 50 and ny < 50:
        use_exact = True

    if use_exact:
        # Use scipy's exact Mann-Whitney
        stat_scipy, p_two = sp_stats.mannwhitneyu(
            x, y + mu, alternative='two-sided', method='exact'
        )
        if alternative == "two.sided":
            p_value = p_two
        elif alternative == "less":
            _, p_value = sp_stats.mannwhitneyu(
                x, y + mu, alternative='less', method='exact'
            )
        else:
            _, p_value = sp_stats.mannwhitneyu(
                x, y + mu, alternative='greater', method='exact'
            )
        method = "Wilcoxon rank sum exact test"
    else:
        p_value = _rank_sum_normal_p(W, nx, ny, ranks, alternative, correct)
        method = "Wilcoxon rank sum test with continuity correction"
        if not correct:
            method = "Wilcoxon rank sum test"

    # Hodges-Lehmann CI for location shift
    ci = None
    estimate = None
    if compute_ci:
        diffs = np.subtract.outer(x, y).ravel()
        diffs.sort()
        estimate = {"difference in location": float(np.median(diffs))}

        if alternative == "two.sided":
            alpha = 1.0 - conf_level
            lo = float(np.percentile(diffs, 100 * alpha / 2))
            hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
            ci = (lo, hi)
        elif alternative == "less":
            alpha = 1.0 - conf_level
            hi = float(np.percentile(diffs, 100 * (1 - alpha)))
            ci = (float('-inf'), hi)
        else:  # greater
            alpha = 1.0 - conf_level
            lo = float(np.percentile(diffs, 100 * alpha))
            ci = (lo, float('inf'))

    return HTestParams(
        statistic=W,
        statistic_name="W",
        parameter=None,
        p_value=float(p_value),
        conf_int=np.array(ci) if ci is not None else None,
        conf_level=conf_level,
        estimate=estimate,
        null_value={"location shift": mu},
        alternative=alternative,
        method=method,
        data_name=design.data_name,
    ), warnings_list


def _signed_rank_exact_p(V: float, n: int, alternative: str) -> float:
    """Exact p-value for signed-rank test using scipy."""
    # scipy.stats.wilcoxon uses a different statistic convention
    # We compute using the distribution directly
    # The signed-rank statistic V has support 0..n(n+1)/2
    # Under H0, the distribution is symmetric around n(n+1)/4

    # Use scipy's distribution (available in recent scipy)
    try:
        # scipy.stats.wilcoxon gives exact p for small n
        # But we need to handle the V statistic properly
        # V = sum of positive ranks; T+ in R
        max_V = n * (n + 1) / 2.0
        mean_V = max_V / 2.0

        # Use scipy's signedrank distribution if available
        dist = sp_stats.signaltonoise  # dummy - not available
    except AttributeError:
        pass

    # Generate exact distribution via enumeration for small n
    # Each of n ranks can be + or -, giving 2^n equally likely outcomes
    if n <= 20:
        # Enumerate all 2^n sign assignments
        all_sums = np.zeros(2**n)
        ranks = np.arange(1, n + 1, dtype=np.float64)
        for i in range(2**n):
            signs = np.array([(i >> j) & 1 for j in range(n)], dtype=np.float64)
            all_sums[i] = np.sum(signs * ranks)

        if alternative == "two.sided":
            # p = P(V >= |V - E[V]| + E[V]) + P(V <= E[V] - |V - E[V]|)
            # Equivalently, p = 2 * min(P(V >= v), P(V <= v))
            # But R uses: p_geq + p_leq where they overlap at v
            mean_V = n * (n + 1) / 4.0
            p_geq = np.mean(all_sums >= V - 1e-12)
            p_leq = np.mean(all_sums <= V + 1e-12)
            p_value = min(2.0 * min(p_geq, p_leq), 1.0)
        elif alternative == "less":
            p_value = float(np.mean(all_sums <= V + 1e-12))
        else:  # greater
            p_value = float(np.mean(all_sums >= V - 1e-12))

        return p_value

    # Fallback to normal approximation for large n
    return _signed_rank_normal_p(V, n, np.arange(1, n + 1, dtype=float),
                                  alternative, True)


def _signed_rank_normal_p(
    V: float, n: int, ranks: np.ndarray,
    alternative: str, correct: bool,
) -> float:
    """Normal approximation p-value for signed-rank test."""
    mean_V = n * (n + 1) / 4.0
    # Variance with tie correction
    # var = n(n+1)(2n+1)/24 - sum(t^3 - t)/48
    # where t is the tie group size
    var_V = n * (n + 1) * (2 * n + 1) / 24.0

    # Tie correction
    unique, counts = np.unique(ranks, return_counts=True)
    tie_correction = np.sum(counts**3 - counts) / 48.0
    var_V -= tie_correction

    if var_V <= 0:
        return 1.0

    sd_V = np.sqrt(var_V)

    # Continuity correction
    cc = 0.5 if correct else 0.0

    if alternative == "two.sided":
        z = (abs(V - mean_V) - cc) / sd_V
        p_value = 2.0 * sp_stats.norm.sf(z)
    elif alternative == "less":
        z = (V - mean_V + cc) / sd_V
        p_value = float(sp_stats.norm.cdf(z))
    else:  # greater
        z = (V - mean_V - cc) / sd_V
        p_value = float(sp_stats.norm.sf(z))

    return min(p_value, 1.0)


def _rank_sum_normal_p(
    W: float, nx: int, ny: int, ranks: np.ndarray,
    alternative: str, correct: bool,
) -> float:
    """Normal approximation p-value for rank-sum test."""
    N = nx + ny
    mean_W = nx * ny / 2.0

    # Variance with tie correction
    # var = nx*ny/12 * (N+1 - sum(t^3-t)/(N*(N-1)))
    unique, counts = np.unique(ranks, return_counts=True)
    tie_sum = np.sum(counts**3 - counts)

    var_W = nx * ny / 12.0 * (N + 1 - tie_sum / (N * (N - 1)))

    if var_W <= 0:
        return 1.0

    sd_W = np.sqrt(var_W)

    # Continuity correction
    cc = 0.5 if correct else 0.0

    if alternative == "two.sided":
        z = (abs(W - mean_W) - cc) / sd_W
        p_value = 2.0 * sp_stats.norm.sf(z)
    elif alternative == "less":
        z = (W - mean_W + cc) / sd_W
        p_value = float(sp_stats.norm.cdf(z))
    else:  # greater
        z = (W - mean_W - cc) / sd_W
        p_value = float(sp_stats.norm.sf(z))

    return min(p_value, 1.0)


def _walsh_averages(d: np.ndarray) -> np.ndarray:
    """
    Compute Walsh averages: (d[i] + d[j]) / 2 for all i <= j.

    Used for Hodges-Lehmann pseudomedian and CI for signed-rank test.
    """
    n = len(d)
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((d[i] + d[j]) / 2.0)
    return np.array(walsh)
