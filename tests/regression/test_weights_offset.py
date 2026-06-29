"""Behavioral tests for prior weights (``weights=``) and offset (``offset=``).

Covers the boundary validation (normal / edge / failure), the unit-weight and
no-offset identities, the case-weights ≡ row-replication equivalence (a strong
R-independent correctness check), offset folding for the identity link, and the
documented gates (ridge + weights/offset). R-reference numeric parity lives in
``test_weights_offset_r_validation.py``.
"""

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.regression import fit, ridge
from pystatistics.regression._inputs import resolve_weights, resolve_offset


@pytest.fixture
def ols_data():
    rng = np.random.default_rng(0)
    n = 60
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.standard_normal(n) * 0.4
    return X, y


@pytest.fixture
def poisson_data():
    rng = np.random.default_rng(1)
    n = 80
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = rng.poisson(np.exp(X @ np.array([0.5, 0.3]))).astype(float)
    return X, y


# ----------------------------------------------------------------------
# Boundary validation: resolve_weights / resolve_offset
# ----------------------------------------------------------------------

class TestResolveWeights:
    def test_none_passthrough(self):
        assert resolve_weights(None, 5) is None

    def test_valid_returns_float64(self):
        w = resolve_weights([1, 2, 3], 3)
        assert w.dtype == np.float64
        np.testing.assert_array_equal(w, [1.0, 2.0, 3.0])

    def test_zero_weights_allowed(self):
        w = resolve_weights([0.0, 1.0, 2.0], 3)
        np.testing.assert_array_equal(w, [0.0, 1.0, 2.0])

    def test_wrong_length_raises(self):
        with pytest.raises(ValidationError, match="entries but the design has 5"):
            resolve_weights([1.0, 2.0], 5)

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            resolve_weights([1.0, -0.5, 2.0], 3)

    def test_all_zero_raises(self):
        with pytest.raises(ValidationError, match="not all be zero"):
            resolve_weights([0.0, 0.0, 0.0], 3)

    def test_non_finite_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            resolve_weights([1.0, np.nan, 2.0], 3)
        with pytest.raises(ValidationError, match="finite"):
            resolve_weights([1.0, np.inf, 2.0], 3)

    def test_2d_raises(self):
        with pytest.raises(ValidationError, match="1-dimensional"):
            resolve_weights(np.ones((3, 1)), 3)


class TestResolveOffset:
    def test_none_passthrough(self):
        assert resolve_offset(None, 5) is None

    def test_valid_returns_float64(self):
        off = resolve_offset([0.1, -0.2, 0.3], 3)
        assert off.dtype == np.float64

    def test_wrong_length_raises(self):
        with pytest.raises(ValidationError, match="entries but the design has 4"):
            resolve_offset([0.1, 0.2], 4)

    def test_non_finite_raises(self):
        with pytest.raises(ValidationError, match="finite"):
            resolve_offset([0.1, np.nan], 2)

    def test_negative_offset_allowed(self):
        # An offset is signed and unconstrained.
        off = resolve_offset([-3.0, 0.0, 5.0], 3)
        np.testing.assert_array_equal(off, [-3.0, 0.0, 5.0])


# ----------------------------------------------------------------------
# Identities: unit weights / no offset reproduce the plain fit
# ----------------------------------------------------------------------

def test_unit_weights_equal_unweighted_ols(ols_data):
    X, y = ols_data
    base = fit(X, y)
    weighted = fit(X, y, weights=np.ones(len(y)))
    np.testing.assert_allclose(weighted.coefficients, base.coefficients, rtol=1e-12)
    np.testing.assert_allclose(weighted.standard_errors, base.standard_errors, rtol=1e-10)


def test_unit_weights_equal_unweighted_glm(poisson_data):
    X, y = poisson_data
    base = fit(X, y, family="poisson")
    weighted = fit(X, y, family="poisson", weights=np.ones(len(y)))
    np.testing.assert_allclose(weighted.coefficients, base.coefficients, rtol=1e-10)
    np.testing.assert_allclose(weighted.deviance, base.deviance, rtol=1e-10)


def test_zero_offset_equal_no_offset(poisson_data):
    X, y = poisson_data
    base = fit(X, y, family="poisson")
    offset0 = fit(X, y, family="poisson", offset=np.zeros(len(y)))
    np.testing.assert_allclose(offset0.coefficients, base.coefficients, rtol=1e-10)


# ----------------------------------------------------------------------
# Strong R-independent checks: case weights ≡ row replication
# ----------------------------------------------------------------------

def test_integer_weights_equal_replication_ols(ols_data):
    X, y = ols_data
    reps = np.random.default_rng(2).integers(1, 4, len(y))
    Xr = np.repeat(X, reps, axis=0)
    yr = np.repeat(y, reps)
    weighted = fit(X, y, weights=reps.astype(float))
    replicated = fit(Xr, yr)
    # Point estimates and the (weighted) RSS coincide with row replication...
    np.testing.assert_allclose(weighted.coefficients, replicated.coefficients, rtol=1e-10)
    np.testing.assert_allclose(weighted.rss, replicated.rss, rtol=1e-10)
    # ...but the SEs do NOT: R's lm(weights=) treats prior weights as precision
    # weights with df = n − p (rows), whereas replication has df = Σw − p. The
    # weighted fit therefore reports LARGER SEs (fewer residual df). This is the
    # intended R-matching behavior, not a bug.
    assert np.all(weighted.standard_errors > replicated.standard_errors)
    assert weighted.df_residual == len(y) - X.shape[1]


def test_integer_weights_equal_replication_poisson(poisson_data):
    X, y = poisson_data
    reps = np.random.default_rng(3).integers(1, 4, len(y))
    Xr = np.repeat(X, reps, axis=0)
    yr = np.repeat(y, reps)
    weighted = fit(X, y, family="poisson", weights=reps.astype(float))
    replicated = fit(Xr, yr, family="poisson")
    np.testing.assert_allclose(weighted.coefficients, replicated.coefficients, rtol=1e-8)
    np.testing.assert_allclose(weighted.deviance, replicated.deviance, rtol=1e-8)


def test_offset_folds_into_response_for_identity(ols_data):
    # For the identity link, η = Xβ + offset means fitting (y − offset).
    X, y = ols_data
    off = np.random.default_rng(4).standard_normal(len(y))
    with_offset = fit(X, y, offset=off)
    folded = fit(X, y - off)
    np.testing.assert_allclose(with_offset.coefficients, folded.coefficients, rtol=1e-10)
    # Fitted values add the offset back, so they match the original-scale y.
    np.testing.assert_allclose(
        with_offset.fitted_values, folded.fitted_values + off, rtol=1e-10)


def test_poisson_exposure_offset_shifts_intercept(poisson_data):
    # A constant log-exposure offset c shifts the Poisson intercept by −c.
    X, y = poisson_data
    base = fit(X, y, family="poisson")
    c = 0.75
    shifted = fit(X, y, family="poisson", offset=np.full(len(y), c))
    np.testing.assert_allclose(shifted.coefficients[0], base.coefficients[0] - c, rtol=1e-8)
    np.testing.assert_allclose(shifted.coefficients[1:], base.coefficients[1:], rtol=1e-8)


# ----------------------------------------------------------------------
# Gates and failure modes
# ----------------------------------------------------------------------

def test_ridge_with_weights_raises(ols_data):
    X, y = ols_data
    with pytest.raises(NotImplementedError, match="ridge penalty"):
        fit(X, y, l2=1.0, weights=np.ones(len(y)))


def test_ridge_with_offset_raises(ols_data):
    X, y = ols_data
    with pytest.raises(NotImplementedError, match="ridge penalty"):
        fit(X, y, l2=1.0, offset=np.zeros(len(y)))


def test_ridge_wrapper_with_weights_raises(ols_data):
    X, y = ols_data
    with pytest.raises(NotImplementedError, match="ridge penalty"):
        ridge(X, y, lam=1.0, weights=np.ones(len(y)))


def test_weights_wrong_length_raises_through_fit(ols_data):
    X, y = ols_data
    with pytest.raises(ValidationError, match="entries but the design has"):
        fit(X, y, weights=np.ones(len(y) + 1))


def test_offset_wrong_length_raises_through_fit(poisson_data):
    X, y = poisson_data
    with pytest.raises(ValidationError, match="entries but the design has"):
        fit(X, y, family="poisson", offset=np.zeros(len(y) - 1))


def test_weights_change_the_fit(ols_data):
    # Sanity: non-uniform weights move the coefficients.
    X, y = ols_data
    rng = np.random.default_rng(5)
    w = rng.uniform(0.1, 5.0, len(y))
    assert not np.allclose(fit(X, y, weights=w).coefficients, fit(X, y).coefficients)
