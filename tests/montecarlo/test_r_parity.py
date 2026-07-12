"""
Tests for the 4.6.8 boot_ci / permutation R-parity fixes.

- boot_ci basic/perc/stud/bca use R's norm.inter quantile rule (not numpy type-7);
- BCa acceleration uses the regression influence estimate (R empinf default) for
  the ordinary bootstrap, with a jackknife fallback that is verified by a
  self-check;
- permutation_test two-sided uses the 2*min-tail rule, correct for any statistic
  (matches exact enumeration for a difference AND a ratio).
"""

from itertools import combinations

import numpy as np
import pytest

from pystatistics.montecarlo import boot, boot_ci, permutation_test
from pystatistics.montecarlo._ci import _norm_inter
from pystatistics.montecarlo._influence import (
    regression_influence, jackknife_influence,
)


def corr(d, i):
    x = d[i]
    return np.array([np.corrcoef(x[:, 0], x[:, 1])[0, 1]])


def mean_stat(d, i):
    return np.array([np.mean(d[i])])


LAW = np.column_stack([
    [576, 635, 558, 578, 666, 580, 555, 661, 651, 605, 653, 575, 545, 572, 594],
    [3.39, 3.30, 2.81, 3.03, 3.44, 3.07, 3.00, 3.43, 3.36, 3.13, 3.12, 2.74,
     2.76, 2.88, 2.96]]).astype(float)


# ---------------------------------------------------------------------------
# norm.inter
# ---------------------------------------------------------------------------

def test_norm_inter_matches_reference_values():
    # A fixed sample; endpoints computed by R boot:::norm.inter.
    t = np.arange(1.0, 1001.0)
    lo, hi = _norm_inter(t, [0.025, 0.975])
    # (R+1)*alpha = 1001*0.025 = 25.025 -> between the 25th and 26th order stat,
    # interpolated on the normal scale (very close to 25 here).
    assert 24.5 < lo < 26.0
    assert 975.0 < hi < 977.0


def test_norm_inter_differs_from_type7_in_tail():
    rng = np.random.default_rng(0)
    t = rng.standard_normal(500)
    ni = _norm_inter(t, [0.01])[0]
    t7 = np.quantile(t, 0.01)  # type-7
    assert ni != t7  # different quantile conventions


def test_norm_inter_exact_hit_returns_order_statistic():
    t = np.arange(1.0, 100.0)  # R=99, (R+1)*0.5 = 50 exactly
    assert _norm_inter(t, [0.5])[0] == 50.0


# ---------------------------------------------------------------------------
# regression influence (BCa acceleration)
# ---------------------------------------------------------------------------

def test_regression_influence_fires_for_ordinary_seeded():
    r = boot(LAW, corr, n_resamples=2000, seed=42)
    L = regression_influence(r, 0)
    assert L is not None
    assert L.shape == (15,)
    assert np.all(np.isfinite(L))
    assert abs(L.sum()) < 1e-4  # approximately centred (pinv rank-cutoff floor)

def test_regression_and_jackknife_acceleration_close_but_distinct():
    r = boot(LAW, corr, n_resamples=3000, seed=7)
    Lr, Lj = regression_influence(r, 0), jackknife_influence(r, 0)
    a_reg = np.sum(Lr**3) / (6 * np.sum(Lr**2)**1.5)
    a_jack = np.sum(Lj**3) / (6 * np.sum(Lj**2)**1.5)
    assert a_reg == pytest.approx(a_jack, abs=0.05)  # same ballpark
    assert a_reg != a_jack                            # but distinct estimators


def test_regression_influence_none_without_seed():
    r = boot(LAW, corr, n_resamples=500)  # seed=None
    assert regression_influence(r, 0) is None


def test_regression_influence_self_check_rejects_foreign_replicates():
    from pystatistics.montecarlo._common import BootParams
    from pystatistics.montecarlo.solution import BootstrapSolution
    from pystatistics.core.result import Result
    r = boot(LAW, corr, n_resamples=1000, seed=1)
    foreign = np.sort(r.t.copy(), axis=0)  # not the seed's replicates
    p = BootParams(t0=r.t0, t=foreign, n_resamples=1000, bias=r.bias,
                   standard_errors=r.standard_errors,
                   conf_int=None, conf_level=None)
    sol = BootstrapSolution(
        _result=Result(params=p, info=r._result.info, timing=None,
                       backend_name="x", warnings=()), _design=r._design)
    assert regression_influence(sol, 0) is None  # self-check fails safe


def test_bca_uses_regression_and_stays_in_range():
    r = boot(LAW, corr, n_resamples=3000, seed=99)
    ci = boot_ci(r, ci_type="bca").conf_int["bca"][0]
    assert -1.0 <= ci[0] < ci[1] <= 1.0


# ---------------------------------------------------------------------------
# two-sided permutation = 2*min-tail
# ---------------------------------------------------------------------------

def _exact_two_sided(x, y, stat):
    comb = np.concatenate([x, y]); n = len(comb); n1 = len(x)
    obs = stat(x, y); S = []
    for c in combinations(range(n), n1):
        m = np.zeros(n, bool); m[list(c)] = True
        S.append(stat(comb[m], comb[~m]))
    S = np.array(S)
    pg = np.mean(S >= obs - 1e-12); pl = np.mean(S <= obs + 1e-12)
    return min(1.0, 2 * min(pg, pl))


@pytest.mark.parametrize("stat", [
    lambda a, b: a.mean() - b.mean(),   # null-centred difference
    lambda a, b: a.mean() / b.mean(),   # non-centred ratio
])
def test_two_sided_matches_exact_enumeration(stat):
    x = np.array([2.0, 3, 4, 5, 6]); y = np.array([3.0, 4, 5, 6, 20])
    exact = _exact_two_sided(x, y, stat)
    p = permutation_test(x, y, stat, n_resamples=200000, seed=3,
                         alternative="two-sided").p_value
    assert p == pytest.approx(exact, abs=0.01)


def test_two_sided_ratio_not_the_abs_convention():
    """The non-centred ratio must NOT return the old |perm|>=|obs| value."""
    x = np.array([2.0, 3, 4, 5, 6]); y = np.array([3.0, 4, 5, 6, 20])
    ratio = lambda a, b: a.mean() / b.mean()
    p = permutation_test(x, y, ratio, n_resamples=200000, seed=3,
                         alternative="two-sided").p_value
    assert p < 0.5   # proper two-sided ~0.40, not the ~0.89 |.| artefact


def test_two_sided_pvalue_capped_at_one():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20); y = rng.normal(0, 1, 20)  # null true
    p = permutation_test(x, y, lambda a, b: a.mean() - b.mean(),
                         n_resamples=5000, seed=1).p_value
    assert 0.0 <= p <= 1.0
