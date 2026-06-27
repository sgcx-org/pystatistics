"""Tests for MultinomialSolution.conf_int (4.1.0).

Wald intervals per (non-reference class, predictor) using the normal quantile.
Shape (J-1, p, 2); trailing axis is [lower, upper].
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.multinomial import multinom

_RNG = np.random.default_rng(7)
_N = 200
_X = np.column_stack([np.ones(_N), _RNG.standard_normal(_N), _RNG.standard_normal(_N)])
_Y = _RNG.integers(0, 3, size=_N)            # 3 classes -> 2 non-reference rows


def test_conf_int_shape_and_z_wald():
    sol = multinom(_Y, _X)
    z = stats.norm.ppf(0.975)
    coef, se = sol.coefficient_matrix, sol.standard_errors
    expect = np.stack([coef - z * se, coef + z * se], axis=-1)
    assert sol.conf_int.shape == (2, 3, 2)      # (J-1, p, 2)
    assert sol.conf_level == 0.95
    assert_allclose(sol.conf_int, expect, rtol=0, atol=0)


def test_conf_level_narrows_interval():
    s95 = multinom(_Y, _X, conf_level=0.95)
    s90 = multinom(_Y, _X, conf_level=0.90)
    assert s90.conf_level == 0.90
    assert np.all(np.diff(s90.conf_int, axis=-1) < np.diff(s95.conf_int, axis=-1))


def test_invalid_conf_level_raises():
    for bad in (0.0, 1.0, -0.5):
        with pytest.raises(ValidationError):
            multinom(_Y, _X, conf_level=bad)
