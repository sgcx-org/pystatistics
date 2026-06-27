"""C2 + C3: uniform KM accessors and the not-implemented exception type.

C2 — KMSolution exposes the constitutional `.standard_errors` and `.conf_int`
alongside the legacy `.se` / `.ci_lower` / `.ci_upper`.
C3 — unimplemented features (stratified KM / Cox) raise
NotImplementedFeatureError, which is both a PyStatisticsError (uniform catch)
and a builtin NotImplementedError (backward-compatible catch).
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pystatistics.core.exceptions import (
    NotImplementedFeatureError, PyStatisticsError,
)
from pystatistics.survival import kaplan_meier, coxph

_TIME = np.array([5, 6, 6, 2, 4, 4, 3, 8, 9, 10, 7, 11, 12, 13], dtype=float)
_EVENT = np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=float)
_GROUP = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
_X = np.column_stack([np.linspace(-1, 1, 14), _GROUP.astype(float)])


# --- C2: uniform KM accessors -------------------------------------------------

def test_km_standard_errors_aliases_se():
    km = kaplan_meier(_TIME, _EVENT)
    assert_array_equal(km.standard_errors, km.se)


def test_km_conf_int_stacks_bounds():
    km = kaplan_meier(_TIME, _EVENT)
    ci = km.conf_int
    assert ci.shape == (len(km.time), 2)
    assert_array_equal(ci[:, 0], km.ci_lower)
    assert_array_equal(ci[:, 1], km.ci_upper)


# --- C3: not-implemented feature exception ------------------------------------

def test_stratified_km_raises_feature_error():
    with pytest.raises(NotImplementedFeatureError):
        kaplan_meier(_TIME, _EVENT, strata=_GROUP)


def test_stratified_cox_raises_feature_error():
    with pytest.raises(NotImplementedFeatureError):
        coxph(_TIME, _EVENT, _X, strata=_GROUP)


def test_feature_error_catchable_both_ways():
    # As a library error (uniform catch) ...
    with pytest.raises(PyStatisticsError):
        kaplan_meier(_TIME, _EVENT, strata=_GROUP)
    # ... and as the builtin (backward-compatible catch).
    with pytest.raises(NotImplementedError):
        kaplan_meier(_TIME, _EVENT, strata=_GROUP)
