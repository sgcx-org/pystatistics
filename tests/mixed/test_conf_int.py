"""Tests for LMM/GLMM .conf_int (4.1.0).

LMM uses the Student-t quantile at each coefficient's Satterthwaite df (matching
lmerTest); GLMM uses the normal quantile (asymptotic). Shape (p, 2).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.mixed import lmm, glmm

_RNG = np.random.default_rng(20260627)
_N = 200
_G = _RNG.integers(0, 10, size=_N)
_X = np.column_stack([np.ones(_N), _RNG.standard_normal(_N)])
_Y = _X @ np.array([1.0, 0.6]) + np.take(_RNG.standard_normal(10) * 0.8, _G) \
    + _RNG.standard_normal(_N) * 0.5


def test_lmm_conf_int_uses_satterthwaite_t():
    sol = lmm(_Y, _X, {"grp": _G})
    q = stats.t.ppf(0.975, sol.df_satterthwaite)   # per-coefficient df
    expect = np.column_stack([sol.coefficients - q * sol.standard_errors,
                              sol.coefficients + q * sol.standard_errors])
    assert sol.conf_int.shape == (2, 2)
    assert sol.conf_level == 0.95
    assert_allclose(sol.conf_int, expect, rtol=0, atol=0)


def test_glmm_conf_int_is_z_wald():
    yb = (_Y > _Y.mean()).astype(float)
    sol = glmm(yb, _X, {"grp": _G}, family="binomial")
    q = stats.norm.ppf(0.975)
    expect = np.column_stack([sol.coefficients - q * sol.standard_errors,
                              sol.coefficients + q * sol.standard_errors])
    assert sol.conf_int.shape == (2, 2)
    assert_allclose(sol.conf_int, expect, rtol=0, atol=0)


def test_conf_level_narrows_interval():
    s95 = lmm(_Y, _X, {"grp": _G}, conf_level=0.95)
    s90 = lmm(_Y, _X, {"grp": _G}, conf_level=0.90)
    assert s90.conf_level == 0.90
    assert np.all(np.diff(s90.conf_int, axis=1) < np.diff(s95.conf_int, axis=1))


def test_invalid_conf_level_raises():
    with pytest.raises(ValidationError):
        lmm(_Y, _X, {"grp": _G}, conf_level=1.0)
    yb = (_Y > _Y.mean()).astype(float)
    with pytest.raises(ValidationError):
        glmm(yb, _X, {"grp": _G}, family="binomial", conf_level=0.0)
