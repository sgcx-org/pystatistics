"""DiscreteTimeSolution.conf_int (4.2.1) — completes the uniform .conf_int surface.

Wald intervals for the covariate coefficients using the normal quantile (the
person-period logistic fit is asymptotic-normal), matching the other coefficient
models. Shape (p, 2).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.survival import discrete_time

_RNG = np.random.default_rng(0)
_N = 2000
_X = _RNG.standard_normal((_N, 3))
_TE = -np.log(_RNG.uniform(size=_N)) / np.exp(_X @ [0.5, -0.3, 0.2])
_TC = _RNG.exponential(2, size=_N)
_TIME = np.minimum(_TE, _TC)
_EVENT = (_TE <= _TC).astype(float)
_BOUNDS = np.unique(np.quantile(np.unique(_TIME[_EVENT == 1]),
                                np.linspace(0, 1, 21)[:-1]))


def test_conf_int_shape_and_z_wald():
    sol = discrete_time(_TIME, _EVENT, _X, intervals=_BOUNDS)
    z = stats.norm.ppf(0.975)
    expect = np.column_stack([sol.coefficients - z * sol.standard_errors,
                              sol.coefficients + z * sol.standard_errors])
    assert sol.conf_int.shape == (3, 2)
    assert sol.conf_level == 0.95
    assert_allclose(sol.conf_int, expect, rtol=0, atol=0)


def test_conf_level_narrows_interval():
    s95 = discrete_time(_TIME, _EVENT, _X, intervals=_BOUNDS, conf_level=0.95)
    s90 = discrete_time(_TIME, _EVENT, _X, intervals=_BOUNDS, conf_level=0.90)
    assert s90.conf_level == 0.90
    assert np.all(np.diff(s90.conf_int, axis=1) < np.diff(s95.conf_int, axis=1))


def test_invalid_conf_level_raises():
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValidationError):
            discrete_time(_TIME, _EVENT, _X, intervals=_BOUNDS, conf_level=bad)
