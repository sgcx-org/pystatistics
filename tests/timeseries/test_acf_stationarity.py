"""
Tests for time series Phase 7A: ACF/PACF, differencing, and stationarity tests.

Validates against known analytical properties and R behavior.
Uses deterministic seeds for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.timeseries import (
    acf,
    pacf,
    diff,
    ndiffs,
    adf_test,
    kpss_test,
    ACFResult,
    StationarityResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def white_noise():
    """White noise series with known seed."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, size=200)


@pytest.fixture
def ar1_series():
    """AR(1) process with phi=0.8, n=500."""
    rng = np.random.default_rng(123)
    n = 500
    x = np.zeros(n)
    phi = 0.8
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, 1)
    return x, phi


@pytest.fixture
def ar2_series():
    """AR(2) process with phi1=0.5, phi2=0.3, n=500."""
    rng = np.random.default_rng(456)
    n = 500
    x = np.zeros(n)
    phi1, phi2 = 0.5, 0.3
    for t in range(2, n):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + rng.normal(0, 1)
    return x, phi1, phi2


@pytest.fixture
def random_walk():
    """Random walk (unit root) process, n=500."""
    rng = np.random.default_rng(789)
    increments = rng.normal(0, 1, size=500)
    return np.cumsum(increments)


# ---------------------------------------------------------------------------
# ACF Tests
# ---------------------------------------------------------------------------

class TestACF:
    """Tests for the autocorrelation function."""

    def test_white_noise_lag0_is_one(self, white_noise):
        """ACF at lag 0 must always be 1.0."""
        result = acf(white_noise)
        assert result.acf[0] == 1.0

    def test_white_noise_acf_near_zero(self, white_noise):
        """For white noise, ACF at lags > 0 should be near zero."""
        result = acf(white_noise)
        # All values beyond lag 0 should be within CI bounds
        for k in range(1, len(result.acf)):
            assert abs(result.acf[k]) < 0.2, (
                f"ACF at lag {k} = {result.acf[k]}, expected near 0"
            )

    def test_ar1_geometric_decay(self, ar1_series):
        """AR(1) ACF should decay approximately as phi^k."""
        x, phi = ar1_series
        result = acf(x, max_lag=10)
        for k in range(1, 6):
            expected = phi ** k
            assert abs(result.acf[k] - expected) < 0.15, (
                f"ACF at lag {k} = {result.acf[k]:.4f}, expected ~{expected:.4f}"
            )

    def test_max_lag_default(self):
        """Default max_lag should be floor(10*log10(n))."""
        rng = np.random.default_rng(100)
        for n in [50, 100, 200, 500]:
            x = rng.normal(0, 1, size=n)
            result = acf(x)
            expected_max_lag = int(np.floor(10.0 * np.log10(n)))
            assert int(result.lags[-1]) == expected_max_lag

    def test_max_lag_too_large_raises(self, white_noise):
        """max_lag >= n should raise ValidationError."""
        n = len(white_noise)
        with pytest.raises(ValidationError, match="must be < n"):
            acf(white_noise, max_lag=n)

    def test_ci_bands_contain_zero_white_noise(self, white_noise):
        """CI bounds should contain zero for white noise at all lags."""
        result = acf(white_noise)
        for i in range(len(result.ci_upper)):
            assert result.ci_lower[i] < 0 < result.ci_upper[i]

    def test_lag0_always_included(self, white_noise):
        """Lags array should always start at 0."""
        result = acf(white_noise)
        assert result.lags[0] == 0

    def test_short_series(self):
        """ACF should work on short series (n < 10)."""
        x = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        result = acf(x, max_lag=2)
        assert result.acf[0] == 1.0
        assert len(result.acf) == 3

    def test_type_is_correlation(self, white_noise):
        """Result type should be 'correlation'."""
        result = acf(white_noise)
        assert result.type == "correlation"

    def test_result_is_acfresult(self, white_noise):
        """Return type should be ACFResult."""
        result = acf(white_noise)
        assert isinstance(result, ACFResult)

    def test_n_obs_matches(self, white_noise):
        """n_obs should match input length."""
        result = acf(white_noise)
        assert result.n_obs == len(white_noise)

    def test_conf_level_stored(self, white_noise):
        """conf_level should be stored in result."""
        result = acf(white_noise, conf_level=0.99)
        assert result.conf_level == 0.99

    def test_demean_false(self):
        """With demean=False, no mean subtraction occurs."""
        x = np.array([10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0])
        result_demean = acf(x, max_lag=3, demean=True)
        result_no_demean = acf(x, max_lag=3, demean=False)
        # Results should differ when mean is not zero
        assert not np.allclose(result_demean.acf[1:], result_no_demean.acf[1:])

    def test_summary_runs(self, white_noise):
        """summary() should return a non-empty string."""
        result = acf(white_noise, max_lag=5)
        s = result.summary()
        assert isinstance(s, str)
        assert "Autocorrelation" in s

    def test_negative_max_lag_raises(self, white_noise):
        """Negative max_lag should raise."""
        with pytest.raises(ValidationError):
            acf(white_noise, max_lag=-1)

    def test_nan_input_raises(self):
        """NaN in input should raise ValidationError."""
        x = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValidationError, match="non-finite"):
            acf(x)

    def test_single_observation_raises(self):
        """Series with fewer than 2 observations should raise."""
        with pytest.raises(ValidationError, match="at least 2"):
            acf(np.array([1.0]))


# ---------------------------------------------------------------------------
# PACF Tests
# ---------------------------------------------------------------------------

class TestPACF:
    """Tests for the partial autocorrelation function."""

    def test_ar1_pacf_lag1(self, ar1_series):
        """For AR(1), PACF at lag 1 should approximate phi."""
        x, phi = ar1_series
        result = pacf(x, max_lag=10)
        assert abs(result.acf[0] - phi) < 0.1, (
            f"PACF at lag 1 = {result.acf[0]:.4f}, expected ~{phi}"
        )

    def test_ar1_pacf_cutoff(self, ar1_series):
        """For AR(1), PACF at lags > 1 should be near zero."""
        x, _ = ar1_series
        result = pacf(x, max_lag=10)
        for k in range(1, len(result.acf)):
            assert abs(result.acf[k]) < 0.15, (
                f"PACF at lag {k + 1} = {result.acf[k]:.4f}, expected near 0"
            )

    def test_ar2_pacf_nonzero_lags12(self, ar2_series):
        """For AR(2), PACF at lags 1 and 2 should be nonzero."""
        x, phi1, phi2 = ar2_series
        result = pacf(x, max_lag=10)
        assert abs(result.acf[0]) > 0.1, "PACF at lag 1 should be nonzero"
        assert abs(result.acf[1]) > 0.1, "PACF at lag 2 should be nonzero"

    def test_ar2_pacf_cutoff(self, ar2_series):
        """For AR(2), PACF at lags > 2 should be approximately zero."""
        x, _, _ = ar2_series
        result = pacf(x, max_lag=10)
        for k in range(2, len(result.acf)):
            assert abs(result.acf[k]) < 0.15, (
                f"PACF at lag {k + 1} = {result.acf[k]:.4f}, expected near 0"
            )

    def test_no_lag0(self, white_noise):
        """PACF should NOT include lag 0 (matching R)."""
        result = pacf(white_noise, max_lag=5)
        assert result.lags[0] == 1

    def test_lags_start_at_one(self, white_noise):
        """Lags array should start at 1."""
        result = pacf(white_noise, max_lag=5)
        np.testing.assert_array_equal(result.lags, np.arange(1, 6))

    def test_type_is_partial(self, white_noise):
        """Result type should be 'partial'."""
        result = pacf(white_noise)
        assert result.type == "partial"

    def test_summary_runs(self, white_noise):
        """summary() should return a non-empty string."""
        result = pacf(white_noise, max_lag=5)
        s = result.summary()
        assert isinstance(s, str)
        assert "Partial Autocorrelation" in s

    def test_nan_input_raises(self):
        """NaN in input should raise ValidationError."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        with pytest.raises(ValidationError, match="non-finite"):
            pacf(x)

    def test_short_series_raises(self):
        """Series with fewer than 3 observations should raise for PACF."""
        with pytest.raises(ValidationError, match="at least 3"):
            pacf(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Differencing Tests
# ---------------------------------------------------------------------------

class TestDiff:
    """Tests for the diff function."""

    def test_simple_diff(self):
        """diff of [1,3,6,10] should be [2,3,4]."""
        result = diff(np.array([1.0, 3.0, 6.0, 10.0]))
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_lag2(self):
        """diff with lag=2 of [1,3,6,10] should be [5,7]."""
        result = diff(np.array([1.0, 3.0, 6.0, 10.0]), lag=2)
        np.testing.assert_array_almost_equal(result, [5.0, 7.0])

    def test_differences2(self):
        """diff with differences=2 applies differencing twice."""
        x = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        result = diff(x, differences=2)
        # First diff: [2, 3, 4, 5]
        # Second diff: [1, 1, 1]
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])

    def test_seasonal_diff(self):
        """Seasonal differencing with lag=4."""
        x = np.arange(1.0, 9.0)  # [1, 2, 3, 4, 5, 6, 7, 8]
        result = diff(x, lag=4)
        np.testing.assert_array_almost_equal(result, [4.0, 4.0, 4.0, 4.0])

    def test_output_length(self):
        """Output should have length n - differences * lag."""
        x = np.arange(20.0)
        r1 = diff(x, differences=1, lag=1)
        assert len(r1) == 19
        r2 = diff(x, differences=2, lag=1)
        assert len(r2) == 18
        r3 = diff(x, differences=1, lag=3)
        assert len(r3) == 17

    def test_too_short_raises(self):
        """Series too short for requested differencing should raise."""
        with pytest.raises(ValidationError, match="too short"):
            diff(np.array([1.0, 2.0]), differences=2, lag=1)

    def test_invalid_differences_raises(self):
        """differences < 1 should raise."""
        with pytest.raises(ValidationError, match="positive integer"):
            diff(np.array([1.0, 2.0, 3.0]), differences=0)

    def test_invalid_lag_raises(self):
        """lag < 1 should raise."""
        with pytest.raises(ValidationError, match="positive integer"):
            diff(np.array([1.0, 2.0, 3.0]), lag=0)

    def test_nan_input_raises(self):
        """NaN in input should raise."""
        with pytest.raises(ValidationError, match="non-finite"):
            diff(np.array([1.0, np.nan, 3.0]))


class TestNdiffs:
    """Tests for the ndiffs function."""

    def test_random_walk_needs_differencing(self, random_walk):
        """A random walk should need at least 1 difference."""
        d = ndiffs(random_walk, test="adf")
        assert d >= 1

    def test_stationary_needs_zero(self, white_noise):
        """White noise should need 0 differences."""
        d = ndiffs(white_noise, test="adf")
        assert d == 0

    def test_kpss_stationary_zero(self, white_noise):
        """White noise with KPSS test should need 0 differences."""
        d = ndiffs(white_noise, test="kpss")
        assert d == 0

    def test_invalid_test_raises(self, white_noise):
        """Invalid test name should raise."""
        with pytest.raises(ValidationError, match="must be one of"):
            ndiffs(white_noise, test="invalid")

    def test_max_d_respected(self, random_walk):
        """Result should not exceed max_d."""
        d = ndiffs(random_walk, max_d=1)
        assert d <= 1


# ---------------------------------------------------------------------------
# ADF Test
# ---------------------------------------------------------------------------

class TestADFTest:
    """Tests for the Augmented Dickey-Fuller test."""

    def test_random_walk_not_rejected(self, random_walk):
        """Random walk: fail to reject H0 (p > 0.05)."""
        result = adf_test(random_walk)
        assert result.p_value > 0.05, (
            f"ADF p-value = {result.p_value}, expected > 0.05 for random walk"
        )

    def test_stationary_ar1_rejected(self, ar1_series):
        """Stationary AR(1): reject H0 (p < 0.05)."""
        x, _ = ar1_series
        result = adf_test(x)
        assert result.p_value < 0.05, (
            f"ADF p-value = {result.p_value}, expected < 0.05 for stationary AR(1)"
        )

    def test_critical_values_keys(self, white_noise):
        """Critical values dict should have standard keys."""
        result = adf_test(white_noise)
        assert result.critical_values is not None
        assert "1%" in result.critical_values
        assert "5%" in result.critical_values
        assert "10%" in result.critical_values

    def test_regression_c_vs_ct(self, ar1_series):
        """regression='c' and 'ct' should give different statistics."""
        x, _ = ar1_series
        r_c = adf_test(x, regression="c")
        r_ct = adf_test(x, regression="ct")
        assert r_c.statistic != r_ct.statistic

    def test_n_lags_default(self):
        """Default n_lags should be floor((n-1)^(1/3))."""
        rng = np.random.default_rng(999)
        for n in [50, 100, 200]:
            x = rng.normal(0, 1, size=n)
            result = adf_test(x)
            expected_lags = int(np.floor((n - 1) ** (1.0 / 3.0)))
            assert result.n_lags == expected_lags

    def test_result_is_stationarity_result(self, white_noise):
        """Return type should be StationarityResult."""
        result = adf_test(white_noise)
        assert isinstance(result, StationarityResult)

    def test_method_name(self, white_noise):
        """Method name should indicate ADF."""
        result = adf_test(white_noise)
        assert "Dickey-Fuller" in result.method

    def test_alternative_is_stationary(self, white_noise):
        """Alternative hypothesis should be 'stationary'."""
        result = adf_test(white_noise)
        assert result.alternative == "stationary"

    def test_summary_runs(self, white_noise):
        """summary() should produce a non-empty string."""
        result = adf_test(white_noise)
        s = result.summary()
        assert isinstance(s, str)
        assert "Dickey-Fuller" in s

    def test_nan_input_raises(self):
        """NaN in input should raise."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        with pytest.raises(ValidationError, match="non-finite"):
            adf_test(x)

    def test_short_series(self):
        """Very short series (n=20) should still work."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=20)
        result = adf_test(x)
        assert isinstance(result, StationarityResult)

    def test_regression_nc(self, white_noise):
        """regression='nc' should work."""
        result = adf_test(white_noise, regression="nc")
        assert isinstance(result, StationarityResult)

    def test_invalid_regression_raises(self, white_noise):
        """Invalid regression type should raise."""
        with pytest.raises(ValidationError, match="must be one of"):
            adf_test(white_noise, regression="invalid")

    def test_too_short_raises(self):
        """Series with fewer than 3 observations should raise."""
        with pytest.raises(ValidationError, match="at least 3"):
            adf_test(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# KPSS Test
# ---------------------------------------------------------------------------

class TestKPSSTest:
    """Tests for the KPSS stationarity test."""

    def test_stationary_not_rejected(self, white_noise):
        """Stationary series: fail to reject H0 (p > 0.05)."""
        result = kpss_test(white_noise)
        assert result.p_value > 0.05, (
            f"KPSS p-value = {result.p_value}, expected > 0.05 for stationary"
        )

    def test_random_walk_rejected(self, random_walk):
        """Random walk: reject H0 (p < 0.05)."""
        result = kpss_test(random_walk)
        assert result.p_value <= 0.05, (
            f"KPSS p-value = {result.p_value}, expected <= 0.05 for random walk"
        )

    def test_result_is_stationarity_result(self, white_noise):
        """Return type should be StationarityResult."""
        result = kpss_test(white_noise)
        assert isinstance(result, StationarityResult)

    def test_method_name(self, white_noise):
        """Method should contain 'KPSS'."""
        result = kpss_test(white_noise)
        assert "KPSS" in result.method

    def test_alternative_is_unit_root(self, white_noise):
        """Alternative hypothesis should be 'unit root'."""
        result = kpss_test(white_noise)
        assert result.alternative == "unit root"

    def test_critical_values_present(self, white_noise):
        """Critical values should be present."""
        result = kpss_test(white_noise)
        assert result.critical_values is not None
        assert len(result.critical_values) > 0

    def test_pvalue_range(self, white_noise):
        """P-value should be in [0.01, 0.10] (table range)."""
        result = kpss_test(white_noise)
        assert 0.01 <= result.p_value <= 0.10

    def test_regression_ct(self, white_noise):
        """regression='ct' should work."""
        result = kpss_test(white_noise, regression="ct")
        assert "Trend" in result.method

    def test_regression_c_vs_ct(self, white_noise):
        """Level vs trend stationarity should give different statistics."""
        r_c = kpss_test(white_noise, regression="c")
        r_ct = kpss_test(white_noise, regression="ct")
        assert r_c.statistic != r_ct.statistic

    def test_summary_runs(self, white_noise):
        """summary() should produce a non-empty string."""
        result = kpss_test(white_noise)
        s = result.summary()
        assert isinstance(s, str)
        assert "KPSS" in s

    def test_summary_truncation_large_p(self, white_noise):
        """Summary should show '> 0.10' for large p-values."""
        result = kpss_test(white_noise)
        if result.p_value >= 0.10:
            assert "> 0.10" in result.summary()

    def test_nan_input_raises(self):
        """NaN in input should raise."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        with pytest.raises(ValidationError, match="non-finite"):
            kpss_test(x)

    def test_short_series(self):
        """Very short series (n=20) should work."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=20)
        result = kpss_test(x)
        assert isinstance(result, StationarityResult)

    def test_invalid_regression_raises(self, white_noise):
        """Invalid regression type should raise."""
        with pytest.raises(ValidationError, match="must be one of"):
            kpss_test(white_noise, regression="nc")

    def test_too_short_raises(self):
        """Series with fewer than 3 observations should raise."""
        with pytest.raises(ValidationError, match="at least 3"):
            kpss_test(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Cross-cutting edge case tests."""

    def test_constant_series_acf(self):
        """Constant series should have ACF = NaN for lags > 0."""
        x = np.ones(50)
        result = acf(x, max_lag=5)
        assert result.acf[0] == 1.0
        for k in range(1, len(result.acf)):
            assert np.isnan(result.acf[k])

    def test_very_short_acf(self):
        """n=2 should work for ACF."""
        x = np.array([1.0, 2.0])
        result = acf(x, max_lag=1)
        assert result.acf[0] == 1.0

    def test_very_short_adf(self):
        """n=20 ADF test should complete without error."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=20)
        result = adf_test(x)
        assert isinstance(result, StationarityResult)

    def test_very_short_kpss(self):
        """n=20 KPSS test should complete without error."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, size=20)
        result = kpss_test(x)
        assert isinstance(result, StationarityResult)

    def test_frozen_dataclass_acf(self, white_noise):
        """ACFResult should be immutable."""
        result = acf(white_noise, max_lag=5)
        with pytest.raises(AttributeError):
            result.n_obs = 999  # type: ignore[misc]

    def test_frozen_dataclass_stationarity(self, white_noise):
        """StationarityResult should be immutable."""
        result = adf_test(white_noise)
        with pytest.raises(AttributeError):
            result.p_value = 0.999  # type: ignore[misc]

    def test_list_input(self):
        """Plain Python list should be accepted as input."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = acf(x, max_lag=3)
        assert result.acf[0] == 1.0

    def test_integer_input(self):
        """Integer array should be accepted and converted to float."""
        x = np.arange(50)
        result = acf(x, max_lag=5)
        assert result.acf[0] == 1.0
