"""
Tests for describe(), var(), and the module skeleton.

Phase 1: Mean, Variance, SD with missing data handling.
"""

import numpy as np
import pytest

from pystatistics.descriptive import (
    describe, var, cov, cor, summary,
    DescriptiveDesign, DescriptiveSolution,
)


class TestDescriptiveDesign:
    """Test DescriptiveDesign construction and validation."""

    def test_from_array_2d(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        design = DescriptiveDesign.from_array(data)
        assert design.n == 3
        assert design.p == 2
        assert design.columns is None
        np.testing.assert_array_equal(design.data, data)

    def test_from_array_1d_reshaped(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        design = DescriptiveDesign.from_array(data)
        assert design.n == 5
        assert design.p == 1
        assert design.data.shape == (5, 1)

    def test_from_array_with_nan(self):
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        design = DescriptiveDesign.from_array(data)
        assert design.has_missing
        assert design.n_missing == 1

    def test_from_array_no_nan(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        design = DescriptiveDesign.from_array(data)
        assert not design.has_missing
        assert design.n_missing == 0

    def test_rejects_inf(self):
        data = np.array([[1.0, np.inf], [3.0, 4.0]])
        with pytest.raises(Exception, match="infinite"):
            DescriptiveDesign.from_array(data)

    def test_rejects_empty(self):
        with pytest.raises(Exception):
            DescriptiveDesign.from_array(np.array([]).reshape(0, 1))

    def test_repr(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        design = DescriptiveDesign.from_array(data)
        r = repr(design)
        assert "n=2" in r
        assert "p=2" in r


class TestVar:
    """Test var() — Bessel correction is the critical R-matching detail."""

    def test_bessel_correction_basic(self):
        """var([1,2,3]) must be 1.0 (n-1 denom), not 0.6667 (n denom)."""
        result = var(np.array([1.0, 2.0, 3.0]), backend='cpu')
        np.testing.assert_allclose(result.variance, [1.0], rtol=1e-12)

    def test_var_matches_numpy_ddof1(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 3))
        result = var(data, backend='cpu')
        expected = np.var(data, axis=0, ddof=1)
        np.testing.assert_allclose(result.variance, expected, rtol=1e-12)

    def test_var_single_value(self):
        """Variance of a single value: R returns NA (NaN)."""
        result = var(np.array([5.0]), backend='cpu')
        assert np.isnan(result.variance[0])

    def test_var_constant(self):
        """Variance of constant data should be 0."""
        result = var(np.array([3.0, 3.0, 3.0, 3.0]), backend='cpu')
        np.testing.assert_allclose(result.variance, [0.0], atol=1e-15)

    def test_var_2d_returns_cov(self):
        """R var() on a matrix returns cov()."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = var(data, backend='cpu')
        assert result.covariance_matrix is not None
        assert result.covariance_matrix.shape == (2, 2)

    def test_var_with_nan_everything(self):
        """use='everything' propagates NaN."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = var(data, use='everything', backend='cpu')
        assert np.isnan(result.variance[0])

    def test_var_with_nan_complete_obs(self):
        """use='complete.obs' ignores NaN rows."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = var(data, use='complete.obs', backend='cpu')
        expected = np.var([1.0, 2.0, 4.0, 5.0], ddof=1)
        np.testing.assert_allclose(result.variance, [expected], rtol=1e-12)


class TestMeanBasic:
    """Test mean computation through describe()."""

    def test_mean_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Use var() which also computes mean in Phase 1
        result = var(data, backend='cpu')
        # Mean is not computed by var(), use describe() for that
        # For now, test through a targeted solve
        from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend
        design = DescriptiveDesign.from_array(data)
        be = CPUDescriptiveBackend()
        r = be.solve(design, compute={'mean'}, use='everything')
        np.testing.assert_allclose(r.params.mean, [3.0], rtol=1e-12)

    def test_mean_2d(self):
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend
        design = DescriptiveDesign.from_array(data)
        be = CPUDescriptiveBackend()
        r = be.solve(design, compute={'mean'}, use='everything')
        np.testing.assert_allclose(r.params.mean, [2.0, 20.0], rtol=1e-12)

    def test_mean_with_nan_everything(self):
        """use='everything' propagates NaN in mean."""
        data = np.array([1.0, 2.0, np.nan, 4.0])
        from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend
        design = DescriptiveDesign.from_array(data)
        be = CPUDescriptiveBackend()
        r = be.solve(design, compute={'mean'}, use='everything')
        assert np.isnan(r.params.mean[0])

    def test_mean_with_nan_complete_obs(self):
        data = np.array([1.0, 2.0, np.nan, 4.0])
        from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend
        design = DescriptiveDesign.from_array(data)
        be = CPUDescriptiveBackend()
        r = be.solve(design, compute={'mean'}, use='complete.obs')
        np.testing.assert_allclose(r.params.mean, [7.0 / 3.0], rtol=1e-12)

    def test_mean_with_nan_pairwise(self):
        """Pairwise for column-wise stats: ignores NaN per column."""
        data = np.array([[1.0, np.nan], [2.0, 20.0], [np.nan, 30.0]])
        from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend
        design = DescriptiveDesign.from_array(data)
        be = CPUDescriptiveBackend()
        r = be.solve(design, compute={'mean'}, use='pairwise.complete.obs')
        np.testing.assert_allclose(r.params.mean, [1.5, 25.0], rtol=1e-12)


class TestSD:
    """Test standard deviation."""

    def test_sd_basic(self):
        from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        design = DescriptiveDesign.from_array(data)
        be = CPUDescriptiveBackend()
        r = be.solve(design, compute={'sd'}, use='everything')
        expected = np.std(data, ddof=1)
        np.testing.assert_allclose(r.params.sd, [expected], rtol=1e-12)

    def test_sd_is_sqrt_var(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 3))
        from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend
        design = DescriptiveDesign.from_array(data)
        be = CPUDescriptiveBackend()
        r = be.solve(design, compute={'var', 'sd'}, use='everything')
        np.testing.assert_allclose(r.params.sd, np.sqrt(r.params.variance), rtol=1e-12)


class TestSolution:
    """Test DescriptiveSolution wrapper."""

    def test_repr(self):
        from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        design = DescriptiveDesign.from_array(data)
        be = CPUDescriptiveBackend()
        r = be.solve(design, compute={'mean', 'var'}, use='everything')
        sol = DescriptiveSolution(_result=r, _design=design)
        rep = repr(sol)
        assert "n=3" in rep
        assert "p=2" in rep
        assert "mean" in rep

    def test_backend_name(self):
        result = var(np.array([1.0, 2.0, 3.0]), backend='cpu')
        assert result.backend_name == 'cpu_descriptive'

    def test_timing_populated(self):
        result = var(np.array([1.0, 2.0, 3.0]), backend='cpu')
        assert result.timing is not None
        assert 'total_seconds' in result.timing
