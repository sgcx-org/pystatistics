"""Hodges-Lehmann confidence intervals for the Wilcoxon tests, matching R's
``wilcox.test`` exactly in BOTH regimes.

R does NOT use empirical percentiles of the Walsh averages. It uses:

- **Exact regime** (small n, no ties): invert the exact test via the order
  statistics of the (sorted) Walsh averages / pairwise differences, with the
  cut index given by the exact null quantile ``qsignrank`` / ``qwilcox``.
- **Approximate regime** (ties or large n): invert the tie-corrected normal
  approximation with a continuity correction by root-finding on the shift
  ``d`` (R's ``uniroot`` on ``W(d) - z_q``).

This module provides the exact null quantiles (built from the exact null count
distributions, verified against R's ``dsignrank`` / ``dwilcox``) and both CI
constructions. Estimates (pseudomedian / median of differences) are unchanged --
only the interval endpoints are corrected.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import brentq


# --------------------------------------------------------------------------
# Exact null count distributions (integer counts; probabilities = counts/total)
# --------------------------------------------------------------------------

def _signrank_counts(n: int) -> np.ndarray:
    """Counts of the signed-rank null: number of sign subsets of {1..n} summing
    to each value 0..n(n+1)/2. Sums to 2**n. Matches R's dsignrank * 2**n."""
    poly = np.array([1.0])
    for r in range(1, n + 1):
        term = np.zeros(r + 1)
        term[0] = 1.0
        term[r] = 1.0
        poly = np.convolve(poly, term)
    return poly


@lru_cache(maxsize=None)
def _wilcox_counts(m: int, n: int) -> tuple:
    """Counts of the Mann-Whitney U null (Gaussian binomial [m+n, m]_q), degrees
    0..m*n. Sums to C(m+n, m). Matches R's dwilcox * C(m+n, m).

    Pascal identity for Gaussian binomials:
        [a+b, a]_q = [a+b-1, a-1]_q + q**a [a+b-1, a]_q
    """
    if m == 0 or n == 0:
        return (1.0,)
    left = _wilcox_counts(m - 1, n)          # [a+b-1, a-1]_q
    right = _wilcox_counts(m, n - 1)         # q**a [a+b-1, a]_q  (shift by a=m)
    out = [0.0] * (m * n + 1)
    for i, v in enumerate(left):
        out[i] += v
    for i, v in enumerate(right):
        out[i + m] += v
    return tuple(out)


def _quantile_from_counts(counts: np.ndarray, p: float) -> int:
    """R-style discrete lower quantile: smallest k with P(X <= k) >= p."""
    cdf = np.cumsum(counts) / counts.sum()
    # searchsorted 'left' -> first index where cdf[idx] >= p (guard fp noise)
    return int(np.searchsorted(cdf, p - 1e-12, side="left"))


def qsignrank(p: float, n: int) -> int:
    return _quantile_from_counts(_signrank_counts(n), p)


def qwilcox(p: float, m: int, n: int) -> int:
    return _quantile_from_counts(np.array(_wilcox_counts(m, n)), p)


# --------------------------------------------------------------------------
# Exact-regime CIs (order-statistic inversion)
# --------------------------------------------------------------------------

def _exact_ci_from_sorted(sorted_vals: np.ndarray, qu: int, total: int,
                          alternative: str) -> tuple[float, float]:
    """R's exact endpoint selection from a sorted array of the N=total Walsh
    averages / pairwise diffs; qu is the null quantile cut (1-based in R)."""
    if qu == 0:
        qu = 1
    ql = total - qu
    if alternative == "two-sided":
        return (float(sorted_vals[qu - 1]), float(sorted_vals[ql]))
    if alternative == "greater":
        return (float(sorted_vals[qu - 1]), float("inf"))
    return (float("-inf"), float(sorted_vals[ql]))         # less


def signed_rank_exact_ci(d: np.ndarray, conf_level: float,
                         alternative: str) -> tuple[float, float]:
    # R drops zero differences before the signed-rank CI (wilcox.test source).
    d = np.asarray(d, dtype=np.float64)
    d = d[d != 0.0]
    n = len(d)
    walsh_sorted = np.sort(np.add.outer(d, d)[np.tril_indices(n)] / 2.0)
    alpha = 1.0 - conf_level
    total = n * (n + 1) // 2
    p = alpha / 2 if alternative == "two-sided" else alpha
    qu = qsignrank(p, n)
    return _exact_ci_from_sorted(walsh_sorted, qu, total, alternative)


def rank_sum_exact_ci(diffs_sorted: np.ndarray, m: int, n: int,
                      conf_level: float, alternative: str) -> tuple[float, float]:
    alpha = 1.0 - conf_level
    total = m * n
    p = alpha / 2 if alternative == "two-sided" else alpha
    qu = qwilcox(p, m, n)
    return _exact_ci_from_sorted(diffs_sorted, qu, total, alternative)


# --------------------------------------------------------------------------
# Approximate-regime CIs (R's uniroot inversion of the normal approximation)
# --------------------------------------------------------------------------

def _root(f, lo: float, hi: float) -> float:
    """Bracketed root of a monotone step function; fall back to the endpoint
    whose sign dominates (R returns the boundary when no interior sign change)."""
    flo, fhi = f(lo), f(hi)
    if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
        return hi if abs(fhi) < abs(flo) else lo
    return float(brentq(f, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=200))


def rank_sum_approx_ci(x: np.ndarray, y: np.ndarray, conf_level: float,
                       alternative: str, correct: bool) -> tuple[float, float]:
    """Match R's approximate two-sample CI: invert the tie-corrected normal
    approximation W(d) with a continuity correction (wilcox.test source)."""
    nx, ny = len(x), len(y)
    alpha = 1.0 - conf_level

    def W(d: float) -> float:
        dr = sp_stats.rankdata(np.concatenate([x - d, y]))
        _, cnt = np.unique(dr, return_counts=True)
        sigma = np.sqrt((nx * ny / 12.0) * (
            (nx + ny + 1) - np.sum(cnt**3 - cnt) / ((nx + ny) * (nx + ny - 1))))
        z = np.sum(dr[:nx]) - nx * (nx + 1) / 2.0 - nx * ny / 2.0
        corr = (np.sign(z) * 0.5) if correct else 0.0
        if sigma == 0:
            return np.nan
        return (z - corr) / sigma

    mumin, mumax = float(np.min(x) - np.max(y)), float(np.max(x) - np.min(y))
    if alternative == "two-sided":
        zq = sp_stats.norm.ppf(1 - alpha / 2)
        lo = _root(lambda d: W(d) - zq, mumin, mumax)
        hi = _root(lambda d: W(d) + zq, mumin, mumax)
        return (lo, hi)
    zq = sp_stats.norm.ppf(1 - alpha)
    if alternative == "greater":
        return (_root(lambda d: W(d) - zq, mumin, mumax), float("inf"))
    return (float("-inf"), _root(lambda d: W(d) + zq, mumin, mumax))


def signed_rank_approx_ci(d: np.ndarray, conf_level: float, alternative: str,
                          correct: bool) -> tuple[float, float]:
    """Match R's approximate signed-rank CI: invert the tie-corrected normal
    approximation on the shift, over the range of Walsh averages."""
    # R drops zero differences before the signed-rank CI (wilcox.test source).
    d = np.asarray(d, dtype=np.float64)
    d = d[d != 0.0]
    n = len(d)
    alpha = 1.0 - conf_level

    def W(delta: float) -> float:
        dd = d - delta
        zer = dd[dd != 0.0]
        r = sp_stats.rankdata(np.abs(zer))
        _, cnt = np.unique(r, return_counts=True)
        s = np.sum(r[zer > 0])
        nn = len(zer)
        sigma = np.sqrt(nn * (nn + 1) * (2 * nn + 1) / 24.0
                        - np.sum(cnt**3 - cnt) / 48.0)
        z = s - nn * (nn + 1) / 4.0
        corr = (np.sign(z) * 0.5) if correct else 0.0
        if sigma == 0:
            return np.nan
        return (z - corr) / sigma

    walsh = np.add.outer(d, d)[np.tril_indices(n)] / 2.0
    mumin, mumax = float(np.min(walsh)), float(np.max(walsh))
    if alternative == "two-sided":
        zq = sp_stats.norm.ppf(1 - alpha / 2)
        lo = _root(lambda dd: W(dd) - zq, mumin, mumax)
        hi = _root(lambda dd: W(dd) + zq, mumin, mumax)
        return (lo, hi)
    zq = sp_stats.norm.ppf(1 - alpha)
    if alternative == "greater":
        return (_root(lambda dd: W(dd) - zq, mumin, mumax), float("inf"))
    return (float("-inf"), _root(lambda dd: W(dd) + zq, mumin, mumax))
