"""Tests for OrdinalSolution.conf_int (4.1.0).

Wald intervals for the slope coefficients using the normal quantile
(proportional-odds inference is asymptotic-normal). Shape (p, 2); thresholds
are not included.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.ordinal import polr

_RNG = np.random.default_rng(11)
_N = 200
_X = np.column_stack([_RNG.standard_normal(_N), _RNG.standard_normal(_N)])
_Y = _RNG.integers(0, 4, size=_N)            # 4 ordered levels


def test_conf_int_shape_and_z_wald():
    sol = polr(_Y, _X)
    z = stats.norm.ppf(0.975)
    coef, se = sol.coefficients, sol.standard_errors
    expect = np.column_stack([coef - z * se, coef + z * se])
    assert sol.conf_int.shape == (2, 2)      # (p, 2)
    assert sol.conf_level == 0.95
    assert_allclose(sol.conf_int, expect, rtol=0, atol=0)


def test_conf_level_narrows_interval():
    s95 = polr(_Y, _X, conf_level=0.95)
    s90 = polr(_Y, _X, conf_level=0.90)
    assert s90.conf_level == 0.90
    assert np.all(np.diff(s90.conf_int, axis=1) < np.diff(s95.conf_int, axis=1))


def test_invalid_conf_level_raises():
    for bad in (0.0, 1.0, 5.0):
        with pytest.raises(ValidationError):
            polr(_Y, _X, conf_level=bad)
