"""
Tests for VA-7: glmm ``correlated=False`` — a diagonal (uncorrelated)
random-effects covariance, R's ``(… || g)``.

Validated vs lme4::glmer's ``(1 + x || g)`` on a random-intercept + random-slope
binomial model: the uncorrelated fit forces the RE correlation to exactly 0 and
matches glmer's variances/fixed effects (at the Laplace two-tier tolerance),
while the correlated fit of the same data has a clearly non-zero correlation.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from pystatistics.mixed import glmm

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIXTURES / "glmm_uncorrelated_re_glmer.json").read_text())


def _fit(ref, correlated):
    y = np.asarray(ref["y"], float)
    x = np.asarray(ref["x"], float)
    g = np.asarray(ref["g"], int)
    X = np.column_stack([np.ones(len(y)), x])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return glmm(y, X, groups={"g": g}, family="binomial",
                    random_effects={"g": ["1", "x"]}, random_data={"x": x},
                    correlated=correlated)


class TestUncorrelatedRE:
    def test_correlation_is_zero(self, ref):
        sol = _fit(ref, correlated=False)
        corr = [c.corr for c in sol.var_components if c.corr is not None]
        assert corr, "expected a correlation entry for the slope term"
        assert all(abs(c) < 1e-10 for c in corr)

    def test_matches_glmer_uncorrelated(self, ref):
        sol = _fit(ref, correlated=False)
        r = ref["uncorrelated"]
        assert np.allclose(sol.coefficients, r["fixef"], atol=1e-2)
        # variance components (order: intercept, slope)
        var = {c.name: c.variance for c in sol.var_components}
        assert abs(var["(Intercept)"] - r["var_intercept"]) < 1e-2
        assert abs(var["x"] - r["var_x"]) < 1e-2

    def test_diagonal_has_fewer_theta_than_full(self, ref):
        """The diagonal parameterisation drops the off-diagonal θ (q vs q(q+1)/2)."""
        from pystatistics.mixed._random_effects import parse_random_effects
        g = np.asarray(ref["g"], int)
        x = np.asarray(ref["x"], float)
        n = len(g)
        full = parse_random_effects({"g": g}, {"g": ["1", "x"]}, {"x": x}, n,
                                    correlated=True)
        diag = parse_random_effects({"g": g}, {"g": ["1", "x"]}, {"x": x}, n,
                                    correlated=False)
        assert full[0].theta_size == 3      # 2*(2+1)/2
        assert diag[0].theta_size == 2      # diagonal only
        assert diag[0].correlated is False

    def test_correlated_fit_has_nonzero_corr(self, ref):
        """The correlated fit of the same data recovers a real correlation —
        confirming the two parameterisations are genuinely different."""
        sol = _fit(ref, correlated=True)
        corr = [c.corr for c in sol.var_components if c.corr is not None]
        assert any(abs(c) > 0.1 for c in corr)

    def test_bool_applies_to_all_factors(self, ref):
        """`correlated=False` (a bare bool) applies to every grouping factor."""
        sol = _fit(ref, correlated=False)
        # single factor here; just assert it ran and is diagonal.
        corr = [c.corr for c in sol.var_components if c.corr is not None]
        assert all(abs(c) < 1e-10 for c in corr)
