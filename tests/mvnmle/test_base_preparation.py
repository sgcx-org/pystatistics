"""
Tests for the shared MVN data preparation.

Covers ``_objectives/_pattern_grouping.py`` (grouping rows by missingness
pattern), ``_objectives/_start_values.py`` (available-case moments and the
positive-definite start) and ``_objectives/base.py`` (``MLEObjectiveBase``),
and checks that the estimators built on them give the results recorded before
this preparation was rewritten (``references/pre_base_rewrite_fits.json``):

- the objective at a fixed parameter vector is unchanged to rounding, and
  does not depend on the order of the rows or of the patterns,
- converged fits are unchanged to optimizer tolerance on the NumPy path, the
  PyTorch FP64 CPU path and (when a GPU is present) the FP32 path,
- the stored results of the removed inverse-Cholesky reference optimizer are
  reproduced,
- data with more than 50 and more than 64 variables are grouped and fitted.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.core.exceptions import NumericalError, ValidationError
from pystatistics.mvnmle import datasets, mlest
from pystatistics.mvnmle._objectives._pattern_grouping import (
    group_rows_by_pattern,
)
from pystatistics.mvnmle._objectives._start_values import (
    MIN_CORRELATION_EIGENVALUE,
    available_case_moments,
    positive_definite_start,
)
from pystatistics.mvnmle._objectives.base import MLEObjectiveBase
from pystatistics.mvnmle._objectives.numpy_fp64 import NumpyObjectiveFP64
from pystatistics.mvnmle.backends.numpy_cholesky import NumpyCholeskyMLEBackend
from pystatistics.mvnmle.design import MVNDesign

REFERENCES = Path(__file__).parent / "references"

with open(REFERENCES / "pre_base_rewrite_fits.json") as _f:
    PRE_CHANGE = json.load(_f)['problems']


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _gpu_device():
    """'cuda', 'mps' or None."""
    if not _torch_available():
        return None
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return None


needs_torch = pytest.mark.skipif(not _torch_available(), reason="needs PyTorch")


def make_problem(n, p, missing_rate, seed):
    """Seeded MVN sample with MCAR missingness (no all-missing rows).

    Built from ``standard_normal`` draws and an explicit Cholesky factor, so
    the data are the same on every platform up to rounding
    (``Generator.multivariate_normal`` factorises with an SVD whose sign
    conventions depend on the LAPACK build)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    sigma = A @ A.T + p * np.eye(p)
    mu = rng.standard_normal(p)
    X = mu + rng.standard_normal((n, p)) @ np.linalg.cholesky(sigma).T
    X[rng.random((n, p)) < missing_rate] = np.nan
    return X[~np.isnan(X).all(axis=1)]


def block_missing(n, p, seed):
    """Independent normal columns with three blocks of missingness, including
    at the highest column indices."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    q = n // 4
    X[0:q, 0:3] = np.nan
    X[q:2 * q, p - 3:p] = np.nan
    X[2 * q:3 * q, p - 1] = np.nan
    return X


def problem_data(name):
    """Input matrix of a recorded problem: a bundled dataset, or the stored
    array (the data themselves are stored, not regenerated, so every platform
    fits exactly the matrix the recorded values belong to)."""
    source = PRE_CHANGE[name]['data']
    if source.startswith('datasets.'):
        return np.asarray(getattr(datasets, source.split('.', 1)[1]),
                          dtype=float)
    filename, key = source.split(':')
    with np.load(REFERENCES / filename) as stored:
        return np.array(stored[key], dtype=float)


def brute_force_groups(X):
    """Reference grouping straight from the definition: a dict keyed by the
    tuple of missing flags."""
    groups = {}
    for i, row in enumerate(np.isnan(X)):
        groups.setdefault(tuple(bool(v) for v in row), []).append(i)
    return groups


# ---------------------------------------------------------------------------
# Pattern grouping
# ---------------------------------------------------------------------------

class TestPatternGrouping:
    @pytest.mark.parametrize('p', [1, 3, 8, 9, 50, 51, 63, 64, 65, 130])
    def test_partition_matches_definition(self, p):
        rng = np.random.default_rng(p)
        X = rng.standard_normal((300, p))
        X[rng.random((300, p)) < 0.03] = np.nan
        X[0:40, p - 1] = np.nan      # missingness at the highest index
        X[40:80, 0] = np.nan

        patterns = group_rows_by_pattern(X)
        expected = brute_force_groups(X)

        assert len(patterns) == len(expected)
        seen = np.concatenate([pat.row_indices for pat in patterns])
        assert sorted(seen.tolist()) == list(range(300))
        for pat in patterns:
            key = tuple(bool(v) for v in np.isnan(X[pat.row_indices[0]]))
            assert pat.row_indices.tolist() == expected[key]   # original order
            assert pat.n_obs == len(expected[key])
            assert not np.isnan(pat.data).any()
            np.testing.assert_array_equal(
                pat.data, X[np.ix_(pat.row_indices, pat.observed_indices)])
            np.testing.assert_array_equal(
                np.sort(np.concatenate(
                    [pat.observed_indices, pat.missing_indices])),
                np.arange(p))
            assert np.isnan(X[np.ix_(pat.row_indices,
                                     pat.missing_indices)]).all()

    def test_documented_pattern_order(self):
        nan = np.nan
        X = np.array([
            [nan, 1.0, nan],     # missing {0, 2}
            [1.0, 1.0, nan],     # missing {2}
            [1.0, 1.0, 1.0],     # complete
            [nan, 1.0, 1.0],     # missing {0}
            [1.0, nan, nan],     # missing {1, 2}
            [1.0, nan, 1.0],     # missing {1}
        ])
        missing_sets = [tuple(pat.missing_indices.tolist())
                        for pat in group_rows_by_pattern(X)]
        # Fewest missing first; ties lexicographic on the missing-indicator
        # vector (False < True, variable 0 most significant).
        assert missing_sets == [(), (2,), (1,), (0,), (1, 2), (0, 2)]
        assert [pat.pattern_id for pat in group_rows_by_pattern(X)] == list(
            range(6))

    def test_order_does_not_depend_on_row_order(self):
        X = make_problem(200, 6, 0.3, 0)
        shuffled = X[np.random.default_rng(1).permutation(X.shape[0])]
        a = [pat.missing_indices.tolist() for pat in group_rows_by_pattern(X)]
        b = [pat.missing_indices.tolist()
             for pat in group_rows_by_pattern(shuffled)]
        assert a == b

    def test_patterns_differing_only_in_a_high_column_stay_apart(self):
        p = 80
        X = np.arange(200.0)[:, None] + np.zeros((200, p))
        X[0:100, p - 1] = np.nan
        patterns = group_rows_by_pattern(X)
        assert [pat.n_obs for pat in patterns] == [100, 100]
        assert [len(pat.observed_indices) for pat in patterns] == [p, p - 1]

    def test_complete_data_is_one_pattern(self):
        patterns = group_rows_by_pattern(np.ones((5, 4)))
        assert len(patterns) == 1
        assert patterns[0].n_obs == 5
        assert patterns[0].missing_indices.size == 0

    def test_invalid_input_raises(self):
        with pytest.raises(ValidationError, match="must be 2D"):
            group_rows_by_pattern(np.arange(4.0))
        with pytest.raises(ValidationError, match="no rows"):
            group_rows_by_pattern(np.empty((0, 3)))


# ---------------------------------------------------------------------------
# Starting values
# ---------------------------------------------------------------------------

class TestAvailableCaseMoments:
    def test_matches_definition(self):
        X = make_problem(150, 5, 0.3, 3)
        mean, cov = available_case_moments(X)
        p = X.shape[1]
        for j in range(p):
            assert mean[j] == pytest.approx(np.nanmean(X[:, j]), rel=1e-13)
            for k in range(p):
                both = ~np.isnan(X[:, j]) & ~np.isnan(X[:, k])
                xj, xk = X[both, j], X[both, k]
                expected = np.mean((xj - xj.mean()) * (xk - xk.mean()))
                assert cov[j, k] == pytest.approx(expected, rel=1e-11,
                                                  abs=1e-13)

    def test_complete_data_gives_ml_covariance(self):
        X = np.random.default_rng(4).standard_normal((80, 4)) + 1e6
        mean, cov = available_case_moments(X)
        np.testing.assert_allclose(mean, X.mean(axis=0), rtol=1e-14)
        # Large common offset: centring keeps the covariance accurate.
        np.testing.assert_allclose(
            cov, np.cov(X, rowvar=False, bias=True), rtol=1e-8)

    def test_pair_never_observed_together_has_zero_covariance(self):
        nan = np.nan
        X = np.array([[1.0, nan], [2.0, nan], [4.0, nan],
                      [nan, 3.0], [nan, 5.0], [nan, 9.0]])
        _, cov = available_case_moments(X)
        assert cov[0, 1] == 0.0 and cov[1, 0] == 0.0
        assert cov[0, 0] == pytest.approx(np.var([1.0, 2.0, 4.0]))
        assert cov[1, 1] == pytest.approx(np.var([3.0, 5.0, 9.0]))

    def test_all_missing_column_raises(self):
        X = np.ones((4, 2))
        X[:, 1] = np.nan
        with pytest.raises(ValidationError, match="columns with all missing"):
            available_case_moments(X)

    def test_one_dimensional_input_raises(self):
        with pytest.raises(ValidationError, match="must be 2D"):
            available_case_moments(np.arange(3.0))


class TestPositiveDefiniteStart:
    INDEFINITE = np.array([[1.0, 0.9, -0.9],
                           [0.9, 1.0, 0.9],
                           [-0.9, 0.9, 1.0]])

    def test_positive_definite_input_is_returned_unchanged(self):
        cov = np.array([[4.0, 1.0], [1.0, 9.0]])
        np.testing.assert_array_equal(positive_definite_start(cov), cov)

    def test_indefinite_input_reaches_the_eigenvalue_floor(self):
        scale = np.array([1e-3, 1.0, 1e3])
        cov = self.INDEFINITE * np.outer(scale, scale)
        assert np.linalg.eigvalsh(cov)[0] < 0

        start = positive_definite_start(cov)
        np.testing.assert_allclose(np.diag(start), np.diag(cov), rtol=1e-14)
        corr = start / np.outer(scale, scale)
        assert np.linalg.eigvalsh(corr)[0] == pytest.approx(
            MIN_CORRELATION_EIGENVALUE, rel=1e-6)
        np.linalg.cholesky(start)      # factorises

    def test_rule_is_invariant_to_units(self):
        scale = np.array([1e-4, 3.0, 2e5])
        base = positive_definite_start(self.INDEFINITE)
        scaled = positive_definite_start(
            self.INDEFINITE * np.outer(scale, scale))
        np.testing.assert_allclose(
            scaled, base * np.outer(scale, scale), rtol=1e-12)

    def test_shrinkage_only_moves_correlations_towards_zero(self):
        start = positive_definite_start(self.INDEFINITE)
        off = ~np.eye(3, dtype=bool)
        ratio = start[off] / self.INDEFINITE[off]
        assert np.all((ratio > 0) & (ratio < 1))
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)

    def test_degenerate_variance_is_replaced_by_one(self):
        cov = np.array([[2.0, 0.0, 0.3], [0.0, 0.0, 0.0], [0.3, 0.0, 1.0]])
        start = positive_definite_start(cov)
        assert start[1, 1] == 1.0
        assert np.all(start[1, [0, 2]] == 0.0)
        np.linalg.cholesky(start)

    def test_non_finite_input_raises(self):
        with pytest.raises(NumericalError, match="non-finite"):
            positive_definite_start(np.array([[1.0, np.nan], [np.nan, 1.0]]))

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValidationError, match="square"):
            positive_definite_start(np.ones((2, 3)))
        with pytest.raises(ValidationError, match="min_eigenvalue"):
            positive_definite_start(np.eye(2), min_eigenvalue=1.5)


class TestObjectiveBase:
    def test_attributes(self):
        X = make_problem(120, 4, 0.25, 5)
        obj = MLEObjectiveBase(X)
        assert (obj.n_obs, obj.n_vars) == X.shape
        assert obj.n_patterns == len(brute_force_groups(X))
        assert obj.n_observed_scalars == int((~np.isnan(X)).sum())
        assert not obj.is_complete
        mean, cov = available_case_moments(X)
        np.testing.assert_array_equal(obj.sample_mean, mean)
        np.testing.assert_array_equal(obj.sample_cov_raw, cov)
        np.linalg.cholesky(obj.sample_cov)
        np.testing.assert_array_equal(obj.jitter_scale(), np.diag(cov))

    def test_start_is_positive_definite_when_raw_moments_are_not(self):
        X = np.asarray(datasets.missvals, dtype=float)
        obj = MLEObjectiveBase(X)
        assert np.linalg.eigvalsh(obj.sample_cov_raw)[0] < 0
        assert np.linalg.eigvalsh(obj.sample_cov)[0] > 0
        np.testing.assert_allclose(
            np.diag(obj.sample_cov), np.diag(obj.sample_cov_raw), rtol=1e-14)

    @pytest.mark.parametrize('bad,match', [
        (np.arange(4.0), "must be 2D"),
        (np.ones((1, 3)), "at least 2 observations"),
        (np.ones((3, 0)), "at least 1 variable"),
        (np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]]),
         "rows with all missing"),
        (np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]]),
         "columns with all missing"),
    ])
    def test_invalid_data_raises(self, bad, match):
        with pytest.raises(ValidationError, match=match):
            MLEObjectiveBase(bad)

    def test_abstract_interface_raises(self):
        obj = MLEObjectiveBase(np.ones((3, 2)) + np.eye(3, 2))
        theta = np.zeros(obj.n_params)
        for call in (obj.get_initial_parameters,
                     lambda: obj.compute_objective(theta),
                     lambda: obj.compute_gradient(theta),
                     lambda: obj.extract_parameters(theta)):
            with pytest.raises(NotImplementedError):
                call()


# ---------------------------------------------------------------------------
# The objective and the fits are what they were before the rewrite
# ---------------------------------------------------------------------------

class TestObjectiveUnchanged:
    # The recorded values come from one machine (macOS arm64, Accelerate).
    # Measured there the change is <= 3e-14; the allowance covers a different
    # BLAS summing in a different order at the ill-conditioned missvals point.
    RTOL = 1e-9

    @pytest.mark.parametrize('name', sorted(PRE_CHANGE))
    def test_value_at_fixed_theta_matches_pre_change(self, name):
        rec = PRE_CHANGE[name]
        theta = np.asarray(rec['theta'])
        expected = rec['objective_at_theta']
        obj = NumpyObjectiveFP64(problem_data(name))
        assert abs(obj.compute_objective(theta) - expected) <= (
            self.RTOL * abs(expected))

    @needs_torch
    @pytest.mark.parametrize('name', sorted(PRE_CHANGE))
    def test_torch_value_at_fixed_theta_matches_pre_change(self, name):
        from pystatistics.mvnmle._objectives.gpu_fp64 import GPUObjectiveFP64

        rec = PRE_CHANGE[name]
        theta = np.asarray(rec['theta'])
        expected = rec['objective_at_theta_torch']
        obj = GPUObjectiveFP64(problem_data(name), device='cpu')
        assert abs(obj.compute_objective(theta) - expected) <= (
            self.RTOL * abs(expected))

    def test_value_does_not_depend_on_row_order(self):
        X = make_problem(400, 6, 0.25, 6)
        shuffled = X[np.random.default_rng(2).permutation(X.shape[0])]
        a, b = NumpyObjectiveFP64(X), NumpyObjectiveFP64(shuffled)
        theta = a.get_initial_parameters() + 0.05
        f_a, g_a = a.compute_value_and_gradient(theta)
        f_b, g_b = b.compute_value_and_gradient(theta)
        assert abs(f_a - f_b) <= 1e-12 * abs(f_a)
        np.testing.assert_allclose(g_b, g_a, rtol=1e-9, atol=1e-9)

    def test_value_does_not_depend_on_pattern_order(self):
        from pystatistics.mvnmle._objectives._batched_cholesky import (
            build_batched_constants,
        )
        from pystatistics.mvnmle._objectives._batched_cholesky_np import (
            neg2_loglik_np,
        )

        obj = NumpyObjectiveFP64(make_problem(400, 6, 0.25, 7))
        theta = obj.get_initial_parameters() + 0.05
        reversed_consts = build_batched_constants(
            obj.patterns[::-1], obj.n_vars, var_scale=obj.jitter_scale())
        f = obj.compute_objective(theta)
        f_reversed = neg2_loglik_np(theta, reversed_consts, obj.n_vars, obj.eps)
        assert abs(f - f_reversed) <= 1e-12 * abs(f)


def _assert_fit_matches(result_params, expected, *, loglik_rtol, atol, rtol):
    assert abs(result_params.loglik - expected['loglik']) <= (
        loglik_rtol * abs(expected['loglik']))
    np.testing.assert_allclose(
        result_params.muhat, expected['muhat'], rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        result_params.sigmahat, expected['sigmahat'], rtol=rtol, atol=atol)


class TestFitsUnchanged:
    """Converged fits agree with the pre-rewrite fits to optimizer tolerance.

    ``missvals`` (n=13, p=5) has a flat likelihood ridge: its log-likelihood
    is reproduced to 1e-8 but the estimates only to about 1e-3, as in the
    comparisons with R elsewhere in this suite.

    The tolerances are the optimizer's, not rounding: a different BLAS can
    take a slightly different BFGS path to the same gradient tolerance.
    """

    FP64_TOL = {'missvals': dict(loglik_rtol=1e-8, atol=2e-3, rtol=2e-3)}
    DEFAULT_FP64_TOL = dict(loglik_rtol=1e-8, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize('name', sorted(PRE_CHANGE))
    def test_numpy_path(self, name):
        rec = PRE_CHANGE[name]
        res = NumpyCholeskyMLEBackend().solve(
            MVNDesign.from_array(problem_data(name)), max_iter=rec['max_iter'])
        assert res.params.converged
        _assert_fit_matches(res.params, rec['forward_cholesky_fp64'],
                            **self.FP64_TOL.get(name, self.DEFAULT_FP64_TOL))

    @needs_torch
    @pytest.mark.parametrize('name', sorted(PRE_CHANGE))
    def test_torch_fp64_cpu_path(self, name):
        from pystatistics.mvnmle.backends.gpu import DirectMLEBackend

        rec = PRE_CHANGE[name]
        res = DirectMLEBackend(device='cpu', use_fp64=True).solve(
            MVNDesign.from_array(problem_data(name)), max_iter=rec['max_iter'])
        assert res.params.converged
        _assert_fit_matches(res.params, rec['forward_cholesky_fp64'],
                            **self.FP64_TOL.get(name, self.DEFAULT_FP64_TOL))

    @pytest.mark.skipif(_gpu_device() != 'mps',
                        reason="FP32 reference values were recorded on MPS")
    @pytest.mark.parametrize(
        'name', sorted(n for n in PRE_CHANGE
                       if 'forward_cholesky_fp32_mps' in PRE_CHANGE[n]))
    def test_fp32_gpu_path(self, name):
        from pystatistics.mvnmle.backends.gpu import DirectMLEBackend

        rec = PRE_CHANGE[name]
        res = DirectMLEBackend(device='mps', use_fp64=False).solve(
            MVNDesign.from_array(problem_data(name)), max_iter=rec['max_iter'])
        assert res.params.converged
        _assert_fit_matches(res.params, rec['forward_cholesky_fp32_mps'],
                            loglik_rtol=1e-6, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize('name', ['apple', 'bench_n500_p5'])
    def test_iteration_count_kept_when_start_needs_no_repair(self, name):
        # Available-case moments are positive definite here, so the start is
        # the same matrix as before and the optimizer takes the same path
        # (identical count on the recording machine; a couple of iterations
        # of slack for a different BLAS).
        rec = PRE_CHANGE[name]
        res = NumpyCholeskyMLEBackend().solve(
            MVNDesign.from_array(problem_data(name)), max_iter=rec['max_iter'])
        assert abs(res.params.n_iter
                   - rec['forward_cholesky_fp64']['n_iter']) <= 2


class TestAgreementWithFormerReferenceOptimizer:
    """The removed inverse-Cholesky reference optimizer's stored results (a
    finite-difference BFGS on a different parameterization) are reproduced by
    the forward-Cholesky estimator: the same maximum, to optimizer tolerance."""

    @pytest.mark.parametrize(
        'name', sorted(n for n in PRE_CHANGE if 'reference' in PRE_CHANGE[n]))
    def test_numpy_fit_matches_stored_reference_fit(self, name):
        rec = PRE_CHANGE[name]
        expected = rec['reference']
        res = NumpyCholeskyMLEBackend().solve(
            MVNDesign.from_array(problem_data(name)), max_iter=rec['max_iter'])
        assert res.params.converged
        assert abs(res.params.loglik - expected['loglik']) < 1e-5
        np.testing.assert_allclose(res.params.muhat, expected['muhat'],
                                   rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(res.params.sigmahat, expected['sigmahat'],
                                   rtol=5e-3, atol=5e-3)


class TestPatternDiagnosticsUnchanged:
    """``identify_missingness_patterns`` is built on the shared grouping; its
    public output and Little's MCAR test, which consumes it, are what they
    were (statistics recorded before the change)."""

    # name -> (statistic, df, p_value) recorded before the rewrite
    MCAR = {
        'apple': (7.357854458961007, 1, 0.006677042025383067),
        'missvals': (11.160688852048725, 6, 0.08353530901018048),
        'bench_n500_p5': (68.06869015048711, 72, 0.6094633682864238),
    }

    @pytest.mark.parametrize('name', sorted(MCAR))
    def test_little_mcar_statistic_and_p_value(self, name):
        from pystatistics.mvnmle import little_mcar_test

        statistic, df, p_value = self.MCAR[name]
        res = little_mcar_test(problem_data(name))
        assert res.df == df
        assert res.statistic == pytest.approx(statistic, rel=1e-6)
        assert res.p_value == pytest.approx(p_value, rel=1e-5)

    def test_patterns_sorted_by_frequency_with_documented_tie_break(self):
        from pystatistics.mvnmle.patterns import identify_missingness_patterns

        X = make_problem(60, 6, 0.3, 9)      # many equally frequent patterns
        patterns = identify_missingness_patterns(X)
        keys = [(-p.n_cases, tuple(p.pattern_vector.tolist()))
                for p in patterns]
        assert keys == sorted(keys)
        assert len(set(k[0] for k in keys)) < len(keys)   # ties are present
        assert [p.pattern_id for p in patterns] == list(
            range(1, len(patterns) + 1))
        assert sum(p.n_cases for p in patterns) == X.shape[0]
        assert sum(p.percent_cases for p in patterns) == pytest.approx(100.0)
        for p in patterns:
            assert not np.isnan(p.data).any()
            np.testing.assert_array_equal(
                np.flatnonzero(p.pattern_vector == 1), p.observed_indices)
            np.testing.assert_array_equal(
                np.flatnonzero(p.pattern_vector == 0), p.missing_indices)


# ---------------------------------------------------------------------------
# Wide data
# ---------------------------------------------------------------------------

class TestWideData:
    """More than 50 and more than 64 variables: no limit from the grouping.

    The fits use L-BFGS-B: with a few thousand parameters the default BFGS
    spends seconds on its dense inverse-Hessian updates, which is not what
    these tests are about.
    """

    @pytest.mark.parametrize('p', [52, 66])
    def test_numpy_fit(self, p):
        X = block_missing(400, p, seed=p)
        res = NumpyCholeskyMLEBackend().solve(
            MVNDesign.from_array(X), method='L-BFGS-B')
        em = mlest(X, method='em')
        assert res.params.converged
        assert abs(res.params.loglik - em.loglik) <= 1e-6 * abs(em.loglik)
        np.testing.assert_allclose(res.params.muhat, em.muhat, atol=5e-3)

    @needs_torch
    @pytest.mark.parametrize('p', [52, 66])
    def test_torch_fp64_fit_matches_numpy(self, p):
        from pystatistics.mvnmle.backends.gpu import DirectMLEBackend

        design = MVNDesign.from_array(block_missing(400, p, seed=p))
        res_np = NumpyCholeskyMLEBackend().solve(design, method='L-BFGS-B')
        res_t = DirectMLEBackend(device='cpu', use_fp64=True).solve(
            design, method='L-BFGS-B')
        assert res_t.params.converged
        assert abs(res_np.params.loglik - res_t.params.loglik) <= (
            1e-9 * abs(res_t.params.loglik))
        np.testing.assert_allclose(
            res_np.params.sigmahat, res_t.params.sigmahat, atol=1e-5)

    @pytest.mark.parametrize('method', ['em', 'monotone'])
    def test_other_estimators_accept_wide_data(self, method):
        p = 66
        X = np.random.default_rng(0).standard_normal((300, p))
        X[0:100, p - 2:p] = np.nan       # monotone: last two columns
        X[100:150, p - 1] = np.nan
        res = mlest(X, method=method)
        assert np.isfinite(res.loglik)
        assert res.sigmahat.shape == (p, p)
