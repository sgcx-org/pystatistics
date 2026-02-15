"""
Tests for jackknife influence values.

Used by BCa confidence intervals. Validates against known results.
"""

import numpy as np
import pytest

from pystatistics.montecarlo import boot
from pystatistics.montecarlo._influence import jackknife_influence


def mean_stat(data, indices):
    """Bootstrap statistic: sample mean."""
    return np.array([np.mean(data[indices])])


class TestJackknifeInfluence:
    """Tests for jackknife influence values."""

    def test_influence_shape(self):
        """Influence values have shape (n,)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = boot(data, mean_stat, R=100, seed=42)
        L = jackknife_influence(result, stat_index=0)
        assert L.shape == (5,)

    def test_influence_sum_zero(self):
        """Influence values should sum to approximately zero."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = boot(data, mean_stat, R=100, seed=42)
        L = jackknife_influence(result, stat_index=0)

        # Sum of influence values should be close to zero
        assert abs(np.sum(L)) < 1e-10

    def test_influence_mean_known(self):
        """For the mean, influence values are proportional to (x_i - x_bar)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n = len(data)
        result = boot(data, mean_stat, R=100, seed=42)
        L = jackknife_influence(result, stat_index=0)

        # For the mean: theta_{-i} = (n*x_bar - x_i) / (n-1)
        # mean_jack = mean of theta_{-i} = x_bar
        # L_i = (n-1) * (x_bar - theta_{-i}) = x_i - x_bar
        x_bar = np.mean(data)
        expected_L = data - x_bar

        np.testing.assert_allclose(L, expected_L, rtol=1e-10)

    def test_influence_symmetric_data(self):
        """Symmetric data gives antisymmetric influence values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = boot(data, mean_stat, R=100, seed=42)
        L = jackknife_influence(result, stat_index=0)

        # L[0] should equal -L[4], L[1] should equal -L[3]
        assert L[0] == pytest.approx(-L[4], rel=1e-10)
        assert L[1] == pytest.approx(-L[3], rel=1e-10)

    def test_acceleration_parameter(self):
        """BCa acceleration parameter from influence values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = boot(data, mean_stat, R=100, seed=42)
        L = jackknife_influence(result, stat_index=0)

        # Acceleration: a = sum(L^3) / (6 * sum(L^2)^1.5)
        a = np.sum(L ** 3) / (6.0 * np.sum(L ** 2) ** 1.5)

        # For symmetric data, a should be close to 0
        assert abs(a) < 0.01

    def test_skewed_data_nonzero_acceleration(self):
        """Skewed data gives nonzero acceleration parameter."""
        # Exponential-like data (right-skewed)
        data = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
        result = boot(data, mean_stat, R=100, seed=42)
        L = jackknife_influence(result, stat_index=0)

        a = np.sum(L ** 3) / (6.0 * np.sum(L ** 2) ** 1.5)

        # Should be nonzero for skewed data
        assert abs(a) > 0.01
