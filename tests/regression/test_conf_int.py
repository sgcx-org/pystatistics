"""Tests for the .conf_int accessor on regression solutions (4.1.0).

Wald confidence intervals: OLS uses the Student-t quantile at df_residual
(matching R's confint.lm exactly); GLM uses the normal quantile for
fixed-dispersion families and t for estimated-dispersion families.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.regression import fit, Design


# Fixed design with an R reference (confint(lm(y ~ x1 + x2), level=0.95)).
_X1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
_X2 = np.array([2, 1, 4, 3, 6, 5, 8, 7, 10, 9], dtype=float)
_Y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 11.9, 14.2, 15.8, 18.1, 19.9])
_X = np.column_stack([np.ones(10), _X1, _X2])

# R: confint(lm(y ~ x1 + x2)) — rows (Intercept), x1, x2; cols 2.5%, 97.5%.
_R_OLS_CI = np.array([
    [-0.0957391278, 0.0957391278],
    [1.8155367994, 1.9044632006],
    [0.0955367994, 0.1844632006],
])


def test_ols_conf_int_matches_r_confint_lm():
    sol = fit(Design.from_arrays(_X, _Y))
    assert_allclose(sol.conf_int, _R_OLS_CI, rtol=1e-7, atol=1e-9)
    assert sol.conf_int.shape == (3, 2)
    assert sol.conf_level == 0.95


def test_ols_conf_int_is_t_wald():
    sol = fit(Design.from_arrays(_X, _Y))
    q = stats.t.ppf(0.975, sol.df_residual)
    expect = np.column_stack([sol.coefficients - q * sol.standard_errors,
                              sol.coefficients + q * sol.standard_errors])
    assert_allclose(sol.conf_int, expect, rtol=0, atol=0)


def test_glm_conf_int_is_z_wald():
    yb = (_Y > _Y.mean()).astype(float)
    sol = fit(Design.from_arrays(_X, yb), family="binomial")
    q = stats.norm.ppf(0.975)
    expect = np.column_stack([sol.coefficients - q * sol.standard_errors,
                              sol.coefficients + q * sol.standard_errors])
    assert_allclose(sol.conf_int, expect, rtol=0, atol=0)


def test_conf_level_narrows_interval():
    s95 = fit(Design.from_arrays(_X, _Y), conf_level=0.95)
    s90 = fit(Design.from_arrays(_X, _Y), conf_level=0.90)
    assert s90.conf_level == 0.90
    assert np.all(np.diff(s90.conf_int, axis=1) < np.diff(s95.conf_int, axis=1))


def test_invalid_conf_level_raises():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValidationError):
            fit(Design.from_arrays(_X, _Y), conf_level=bad)


def test_ridge_conf_int_is_nan():
    # Penalized (biased) fit reports NaN SEs, so its Wald intervals are NaN.
    sol = fit(Design.from_arrays(_X, _Y), l2=1.0)
    assert np.all(np.isnan(sol.conf_int))
