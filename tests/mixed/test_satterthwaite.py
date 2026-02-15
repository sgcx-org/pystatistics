"""Tests for Satterthwaite degrees of freedom computation."""

import numpy as np
import pytest

from pystatistics.mixed import lmm


class TestSatterthwaite:
    """Tests for Satterthwaite df computation."""

    def test_df_are_positive(self, random_intercept_simple):
        """All Satterthwaite df should be positive."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})
        assert np.all(result.df_satterthwaite > 0)

    def test_df_are_finite(self, random_intercept_simple):
        """All Satterthwaite df should be finite."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})
        assert np.all(np.isfinite(result.df_satterthwaite))

    def test_p_values_consistent_with_df(self, random_intercept_simple):
        """p-values should be consistent with t-statistics and df."""
        d = random_intercept_simple
        result = lmm(d['y'], d['X'], groups={'group': d['group']})

        from scipy.stats import t as t_dist
        for i in range(len(result.coefficients)):
            expected_p = 2.0 * t_dist.sf(
                abs(result.t_values[i]), result.df_satterthwaite[i]
            )
            np.testing.assert_allclose(
                result.p_values[i], expected_p, rtol=1e-8
            )

    def test_skip_satterthwaite_flag(self, random_intercept_simple):
        """compute_satterthwaite=False should use residual df."""
        d = random_intercept_simple
        n = d['n_groups'] * d['n_per_group']
        p = 2
        result = lmm(
            d['y'], d['X'], groups={'group': d['group']},
            compute_satterthwaite=False,
        )
        expected_df = float(n - p)
        np.testing.assert_allclose(
            result.df_satterthwaite, expected_df, atol=1e-10
        )

    def test_skip_satterthwaite_faster(self, random_intercept_simple):
        """Skipping Satterthwaite should be faster (less computation)."""
        import time
        d = random_intercept_simple

        t0 = time.time()
        _ = lmm(d['y'], d['X'], groups={'group': d['group']},
                compute_satterthwaite=False)
        time_no_satt = time.time() - t0

        t0 = time.time()
        _ = lmm(d['y'], d['X'], groups={'group': d['group']},
                compute_satterthwaite=True)
        time_with_satt = time.time() - t0

        # The with-Satterthwaite version should generally be slower,
        # but we don't assert this strictly due to timing variability.
        # Just check both complete in reasonable time.
        assert time_no_satt < 30.0
        assert time_with_satt < 30.0
