"""
Tests for the cumulative-link observed information and vcov module.

Covers:
- The raw -> natural threshold Jacobian (shape, block structure, values).
- observed_information: symmetry, positive-definiteness at the MLE, and
  agreement with an independent dense finite-difference Hessian.
- vcov_natural: agreement with R's MASS::polr()-reported standard errors
  (the natural threshold coordinates), and slope-SE invariance.

The MASS reference values were produced with R 4.x / MASS::polr() on the
seeded dataset constructed in ``reference_data`` below.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.ordinal import polr
from pystatistics.ordinal._solver import _fit_polr
from pystatistics.regression.families import LogitLink
from pystatistics.ordinal._likelihood import cumulative_gradient
from pystatistics.ordinal._information import (
    observed_information,
    raw_to_natural_jacobian,
    vcov_natural,
)


@pytest.fixture
def reference_data():
    """4-level ordinal data with three predictors (seed 7, n=800).

    The matching ``MASS::polr(y ~ x1 + x2 + x3, method='logistic')`` fit gives
    the coefficient and threshold standard errors recorded in
    ``test_threshold_se_matches_mass`` below.
    """
    rng = np.random.default_rng(7)
    n = 800
    X = rng.standard_normal((n, 3))
    beta = np.array([1.0, -0.6, 0.4])
    z = X @ beta
    cuts = np.quantile(z, [0.3, 0.6, 0.85])
    y = np.digitize(z + rng.logistic(size=n), cuts).astype(np.intp)
    return y, X


# =========================================================================
# raw_to_natural_jacobian
# =========================================================================

class TestJacobian:
    def test_shape_and_slope_block_identity(self):
        raw = np.array([-0.7, 0.1, 0.4])  # 3 thresholds
        n_params = 5  # + 2 slopes
        jac = raw_to_natural_jacobian(raw, n_params)
        assert jac.shape == (5, 5)
        # Slope block is identity, no threshold/slope coupling.
        assert_allclose(jac[3:, 3:], np.eye(2))
        assert_allclose(jac[3:, :3], 0.0)
        assert_allclose(jac[:3, 3:], 0.0)

    def test_threshold_block_values(self):
        raw = np.array([-0.7, 0.1, 0.4])
        jac = raw_to_natural_jacobian(raw, 3)
        # d alpha_j / d raw_0 = 1 for all j; d alpha_j / d raw_k = exp(raw_k)
        # for j >= k (k >= 1), else 0.
        expected = np.array([
            [1.0, 0.0, 0.0],
            [1.0, np.exp(0.1), 0.0],
            [1.0, np.exp(0.1), np.exp(0.4)],
        ])
        assert_allclose(jac[:3, :3], expected)


# =========================================================================
# observed_information
# =========================================================================

class TestObservedInformation:
    def test_symmetric_and_pd_at_mle(self, reference_data):
        y, X = reference_data
        params, *_ = _fit_polr(y, X, LogitLink(), 4, 1e-8, 200)
        H = observed_information(params, y, X, LogitLink(), 4)
        assert_allclose(H, H.T, rtol=1e-8, atol=1e-10)
        # Positive definite at the maximum likelihood estimate.
        assert np.all(np.linalg.eigvalsh(H) > 0)

    def test_matches_dense_finite_difference(self, reference_data):
        """The d+1-evaluation Hessian matches a dense central-difference one."""
        y, X = reference_data
        params, *_ = _fit_polr(y, X, LogitLink(), 4, 1e-8, 200)
        H = observed_information(params, y, X, LogitLink(), 4)

        d = len(params)
        eps = 1e-5
        H_ref = np.empty((d, d))
        for k in range(d):
            e = np.zeros(d)
            e[k] = eps
            g_plus = cumulative_gradient(params + e, y, X, LogitLink(), 4)
            g_minus = cumulative_gradient(params - e, y, X, LogitLink(), 4)
            H_ref[:, k] = (g_plus - g_minus) / (2 * eps)
        H_ref = 0.5 * (H_ref + H_ref.T)
        assert_allclose(H, H_ref, rtol=1e-4, atol=1e-4)


# =========================================================================
# vcov_natural / MASS agreement
# =========================================================================

class TestVcovNatural:
    def test_threshold_se_matches_mass(self, reference_data):
        """Natural-coordinate threshold SEs match MASS::polr() exactly."""
        y, X = reference_data
        sol = polr(y, X, method="logistic")
        # Reference from R: MASS::polr(y ~ x1+x2+x3, method='logistic',
        # Hess=TRUE) on the same seeded data.
        mass_coef_se = np.array([0.0800, 0.0721, 0.0709])
        mass_thresh_se = np.array([0.0835, 0.0801, 0.0933])
        assert_allclose(sol.standard_errors, mass_coef_se, atol=5e-4)
        assert_allclose(sol.threshold_standard_errors, mass_thresh_se,
                        atol=5e-4)

    def test_slope_se_invariant_to_threshold_parameterization(
        self, reference_data,
    ):
        """The slope-SE block is identical in raw and natural coordinates."""
        y, X = reference_data
        params, _, _, _, vcov_raw = _fit_polr(y, X, LogitLink(), 4, 1e-8, 200)
        n_thresh = 3
        raw_thresh = params[:n_thresh]
        hess = observed_information(params, y, X, LogitLink(), 4)
        vcov_nat = vcov_natural(hess, raw_thresh)
        # Slope variances (the trailing block) are unchanged by a
        # threshold-only reparameterization.
        assert_allclose(
            np.diag(vcov_nat)[n_thresh:], np.diag(vcov_raw)[n_thresh:],
            rtol=1e-8, atol=1e-10,
        )
