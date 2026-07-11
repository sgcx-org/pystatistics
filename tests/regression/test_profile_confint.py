"""
Tests for VA-6b: profile-likelihood confidence intervals (R's confint.glm).

Validated vs R `confint(glm)` on the committed va5 fixture. Profile intervals
differ from Wald when the log-likelihood is asymmetric; here they match R's
profile intervals to ~1e-4 and differ from the Wald intervals.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pystatistics.regression import fit

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="module")
def va5():
    d = json.loads((_FIXTURES / "glm_links_families.json").read_text())
    x1 = np.asarray(d["x1"], float)
    x2 = np.asarray(d["x2"], float)
    X = np.column_stack([np.ones(len(x1)), x1, x2])
    return X, d


# R confint(glm(...)) profile intervals on the va5 data.
_R_BINOMIAL = np.array([
    [-0.71520545690, 0.03881320823],
    [0.05926242955, 0.99123409357],
    [-0.60752845534, 0.07189177221],
])
_R_POISSON = np.array([
    [0.3967044722, 0.6808377604],
    [0.1516666402, 0.4817434233],
    [0.1996042061, 0.4206770971],
])


class TestProfileConfInt:
    def test_binomial_matches_r(self, va5):
        X, d = va5
        m = fit(X, np.asarray(d["yb"], float), family="binomial")
        assert np.allclose(m.profile_conf_int(), _R_BINOMIAL, atol=1e-4)

    def test_poisson_matches_r(self, va5):
        X, d = va5
        m = fit(X, np.asarray(d["yc"], float), family="poisson")
        assert np.allclose(m.profile_conf_int(), _R_POISSON, atol=1e-4)

    def test_profile_differs_from_wald(self, va5):
        X, d = va5
        m = fit(X, np.asarray(d["yb"], float), family="binomial")
        prof = m.profile_conf_int()
        wald = m.conf_int
        # For a coefficient with an asymmetric likelihood the two differ.
        assert not np.allclose(prof, wald, atol=1e-3)

    def test_brackets_the_estimate(self, va5):
        X, d = va5
        m = fit(X, np.asarray(d["yc"], float), family="poisson")
        prof = m.profile_conf_int()
        beta = m.coefficients
        assert np.all(prof[:, 0] < beta)
        assert np.all(beta < prof[:, 1])

    def test_conf_level_widens(self, va5):
        X, d = va5
        m = fit(X, np.asarray(d["yb"], float), family="binomial")
        ci95 = m.profile_conf_int(0.95)
        ci99 = m.profile_conf_int(0.99)
        assert np.all(ci99[:, 0] <= ci95[:, 0] + 1e-9)
        assert np.all(ci99[:, 1] >= ci95[:, 1] - 1e-9)

    def test_gaussian_smoke(self, va5):
        """Estimated-dispersion path runs and brackets the estimate."""
        X, d = va5
        m = fit(X, np.asarray(d["yig"], float), family="gaussian")
        prof = m.profile_conf_int()
        assert prof.shape == (3, 2)
        assert np.all(prof[:, 0] < m.coefficients)
        assert np.all(m.coefficients < prof[:, 1])

    def test_invalid_conf_level_raises(self, va5):
        from pystatistics.core.exceptions import ValidationError
        X, d = va5
        m = fit(X, np.asarray(d["yc"], float), family="poisson")
        with pytest.raises(ValidationError, match="conf_level"):
            m.profile_conf_int(1.5)
