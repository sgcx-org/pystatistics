"""Integration tests for the mice() solver, chain, and solution."""

import numpy as np
import pytest

from pystatistics.mice import datasets, mice, MICEDesign
from pystatistics.mice.solution import MICESolution


@pytest.fixture
def small_missing():
    complete = datasets.make_gaussian_complete(60, seed=1)
    return datasets.make_mcar(complete, 0.2, seed=2)


class TestBasicRun:
    def test_returns_solution(self, small_missing):
        sol = mice(small_missing, m=3, maxit=4, seed=0)
        assert isinstance(sol, MICESolution)
        assert sol.m == 3
        assert sol.maxit == 4

    def test_completed_datasets_have_no_nan(self, small_missing):
        sol = mice(small_missing, m=3, maxit=3, seed=0)
        for d in sol.completed_datasets():
            assert not np.isnan(d).any()
            assert d.shape == small_missing.shape

    def test_observed_values_preserved(self, small_missing):
        sol = mice(small_missing, m=2, maxit=3, seed=0)
        observed = ~np.isnan(small_missing)
        for d in sol.completed_datasets():
            np.testing.assert_array_equal(
                d[observed], small_missing[observed]
            )

    def test_iteration_yields_m_datasets(self, small_missing):
        sol = mice(small_missing, m=4, maxit=2, seed=0)
        assert len(list(sol)) == 4

    def test_pmm_imputes_observed_values_only(self, small_missing):
        # PMM donors: every imputed value must equal some observed value of
        # its column.
        sol = mice(small_missing, m=2, maxit=3, method="pmm", seed=0)
        for j in sol.incomplete_columns:
            observed = set(np.round(small_missing[~np.isnan(small_missing[:, j]), j], 10))
            imp = sol.imputations(j)
            for v in imp.ravel():
                assert round(float(v), 10) in observed


class TestDeterminism:
    def test_same_seed_identical(self, small_missing):
        a = mice(small_missing, m=3, maxit=4, seed=123)
        b = mice(small_missing, m=3, maxit=4, seed=123)
        for da, db in zip(a.completed_datasets(), b.completed_datasets()):
            np.testing.assert_array_equal(da, db)

    def test_different_seed_differs(self, small_missing):
        a = mice(small_missing, m=3, maxit=4, seed=1)
        b = mice(small_missing, m=3, maxit=4, seed=2)
        # At least one imputed cell should differ.
        diff = any(
            not np.array_equal(da, db)
            for da, db in zip(a.completed_datasets(), b.completed_datasets())
        )
        assert diff

    def test_norm_method_runs_and_is_deterministic(self, small_missing):
        a = mice(small_missing, m=2, maxit=3, method="norm", seed=5)
        b = mice(small_missing, m=2, maxit=3, method="norm", seed=5)
        for da, db in zip(a.completed_datasets(), b.completed_datasets()):
            np.testing.assert_array_equal(da, db)


class TestImputationsAccessor:
    def test_imputations_shape(self, small_missing):
        sol = mice(small_missing, m=3, maxit=2, seed=0)
        j = sol.incomplete_columns[0]
        n_mis = int(np.count_nonzero(np.isnan(small_missing[:, j])))
        assert sol.imputations(j).shape == (3, n_mis)

    def test_imputations_unknown_column_raises(self, small_missing):
        sol = mice(small_missing, m=2, maxit=2, seed=0)
        complete_cols = set(range(small_missing.shape[1])) - set(sol.incomplete_columns)
        if complete_cols:
            with pytest.raises(Exception):
                sol.imputations(next(iter(complete_cols)))


class TestConvergenceTrace:
    def test_trace_shapes(self, small_missing):
        sol = mice(small_missing, m=3, maxit=5, seed=0)
        n_incomplete = len(sol.incomplete_columns)
        assert sol.chain_mean.shape == (3, 5, n_incomplete)
        assert sol.chain_var.shape == (3, 5, n_incomplete)

    def test_trace_is_finite(self, small_missing):
        sol = mice(small_missing, m=2, maxit=5, seed=0)
        assert np.all(np.isfinite(sol.chain_mean))
        assert np.all(np.isfinite(sol.chain_var))


class TestRecovery:
    def test_imputed_means_near_truth_on_mcar(self):
        # MCAR + correct linear model: pooled imputed column means should land
        # near the complete-data means (sanity, not exact).
        complete = datasets.make_gaussian_complete(400, seed=7)
        miss = datasets.make_mcar(complete, 0.25, seed=8)
        sol = mice(miss, m=10, maxit=10, method="norm", seed=9)
        stacked = np.mean(sol.completed_datasets(), axis=0)  # (n, p) avg over m
        col_means_imp = stacked.mean(axis=0)
        col_means_true = complete.mean(axis=0)
        np.testing.assert_allclose(col_means_imp, col_means_true, atol=0.15)


class TestSolverValidation:
    def test_seed_is_required(self, small_missing):
        with pytest.raises(TypeError):
            mice(small_missing, m=3, maxit=3)  # no seed

    @pytest.mark.parametrize("bad_m", [0, -1])
    def test_bad_m_rejected(self, small_missing, bad_m):
        with pytest.raises(ValueError):
            mice(small_missing, m=bad_m, maxit=3, seed=0)

    @pytest.mark.parametrize("bad_it", [0, -2])
    def test_bad_maxit_rejected(self, small_missing, bad_it):
        with pytest.raises(ValueError):
            mice(small_missing, m=2, maxit=bad_it, seed=0)

    def test_no_missing_rejected(self):
        complete = datasets.make_gaussian_complete(30, seed=0)
        with pytest.raises(ValueError, match="no missing"):
            mice(complete, m=2, maxit=2, seed=0)

    def test_gpu_backend_dispatch_is_explicit(self, small_missing):
        # backend='gpu' must either run on a real GPU or fail loud — never
        # silently downgrade to CPU.
        try:
            import torch

            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = False

        if cuda:
            sol = mice(small_missing, m=2, maxit=2, seed=0, backend="gpu")
            assert "gpu" in sol.backend_name
        else:
            with pytest.raises((RuntimeError, NotImplementedError)):
                mice(small_missing, m=2, maxit=2, seed=0, backend="gpu")

    def test_accepts_prebuilt_design(self, small_missing):
        design = MICEDesign.from_array(small_missing, method="norm")
        sol = mice(design, m=2, maxit=2, seed=0)
        assert sol.m == 2

    def test_design_ignores_method_with_warning(self, small_missing):
        design = MICEDesign.from_array(small_missing)
        with pytest.warns(UserWarning):
            mice(design, m=2, maxit=2, seed=0, method="norm")
