"""Tests for CoxSolution.conf_int (4.1.0).

Wald intervals on the coefficient (log-hazard-ratio) scale using the normal
quantile (Cox inference is asymptotic-normal); exp(conf_int) gives hazard-ratio
intervals matching R's summary.coxph conf.int.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.survival import coxph

# A non-degenerate two-covariate Cox problem.
_RNG = np.random.default_rng(20260627)
_N = 150
_X = _RNG.standard_normal((_N, 2))
_ETA = _X @ np.array([0.7, -0.4])
_T = -np.log(_RNG.uniform(size=_N)) / np.exp(_ETA)
_C = _RNG.exponential(2.0, size=_N)
_TIME = np.minimum(_T, _C)
_EVENT = (_T <= _C).astype(float)


def test_cox_conf_int_is_z_wald():
    sol = coxph(_TIME, _EVENT, _X)
    q = stats.norm.ppf(0.975)
    expect = np.column_stack([sol.coefficients - q * sol.standard_errors,
                              sol.coefficients + q * sol.standard_errors])
    assert sol.conf_int.shape == (2, 2)
    assert sol.conf_level == 0.95
    assert_allclose(sol.conf_int, expect, rtol=0, atol=0)


def test_cox_hr_interval_brackets_point_estimate():
    sol = coxph(_TIME, _EVENT, _X)
    hr_ci = np.exp(sol.conf_int)
    assert np.all(hr_ci[:, 0] < sol.hazard_ratios)
    assert np.all(sol.hazard_ratios < hr_ci[:, 1])


def test_conf_level_narrows_interval():
    s95 = coxph(_TIME, _EVENT, _X, conf_level=0.95)
    s90 = coxph(_TIME, _EVENT, _X, conf_level=0.90)
    assert s90.conf_level == 0.90
    assert np.all(np.diff(s90.conf_int, axis=1) < np.diff(s95.conf_int, axis=1))


def test_invalid_conf_level_raises():
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValidationError):
            coxph(_TIME, _EVENT, _X, conf_level=bad)
