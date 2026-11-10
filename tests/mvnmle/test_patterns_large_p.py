"""
Regression tests: pattern identification must stay exact for p >= 65.

A grouping keyed on fixed-width integer pattern codes silently overflows once
the number of variables exceeds the code's width (numpy integer overflow
raises no warning), collapsing distinct missingness patterns onto one code;
that once corrupted ``analyze_patterns`` and, through it, Little's MCAR test
from p = 65. Patterns are now grouped by exact mask equality (see
test_pattern_codes_large_p.py for the estimator-side tests); these tests pin
the public-API siblings.
"""

import numpy as np
import pytest

from pystatistics.mvnmle import analyze_patterns, little_mcar_test


def _two_pattern_data(p: int, n_per: int = 60, seed: int = 0) -> np.ndarray:
    """n_per rows missing only column 0 + n_per fully observed rows."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((2 * n_per, p))
    data[:n_per, 0] = np.nan
    return data


class TestPatternCodesLargeP:
    @pytest.mark.parametrize("p", [64, 65, 70, 80])
    def test_distinct_patterns_not_merged(self, p):
        data = _two_pattern_data(p)
        patterns = analyze_patterns(data)
        assert len(patterns) == 2, (
            f"p={p}: expected 2 distinct missingness patterns, got "
            f"{len(patterns)} — pattern codes collided (int64 overflow)."
        )
        sizes = sorted(pat.n_cases for pat in patterns)
        assert sizes == [60, 60]

    @pytest.mark.parametrize("p", [65, 70])
    def test_no_nan_leaks_into_observed_data(self, p):
        data = _two_pattern_data(p)
        for pat in analyze_patterns(data):
            assert np.all(np.isfinite(pat.data)), (
                f"p={p}: NaN leaked into a pattern's observed data — "
                f"rows were grouped under a mask they do not match."
            )

    def test_small_p_behaviour_unchanged(self):
        data = _two_pattern_data(5)
        patterns = analyze_patterns(data)
        assert len(patterns) == 2
        assert sorted(pat.n_cases for pat in patterns) == [60, 60]


class TestLittleMCARLargeP:
    def test_mcar_df_correct_at_p70(self):
        """df = sum(p_k) - p over the true patterns.

        With three patterns at p=70 (complete, missing col 0, missing col 1):
        sum(p_k) = 70 + 69 + 69 = 208, so df = 208 - 70 = 138. Under the
        overflow, patterns collided and df came out wrong (or the statistic
        went NaN) with no error.
        """
        rng = np.random.default_rng(1)
        p = 70
        data = rng.standard_normal((240, p))
        data[:60, 0] = np.nan
        data[60:120, 1] = np.nan
        result = little_mcar_test(data)
        assert result.df == 138
        assert np.isfinite(result.statistic)
        assert 0.0 <= result.p_value <= 1.0
