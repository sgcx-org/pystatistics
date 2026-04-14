"""
Tests for CLAUDE.md Rule 1 compliance in mvnmle module.

Verifies that ill-conditioned or singular matrices raise NumericalError
instead of silently falling back to ridge regularization.
"""

import pytest
import numpy as np
from pathlib import Path

from pystatistics.core.exceptions import NumericalError


class TestRegularizedInverse:
    """regularized_inverse() must raise, not silently regularize."""

    def test_singular_matrix_raises(self):
        from pystatistics.mvnmle.mcar_test import regularized_inverse
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(NumericalError):
            regularized_inverse(singular)

    def test_ill_conditioned_raises(self):
        from pystatistics.mvnmle.mcar_test import regularized_inverse
        ill = np.array([[1.0, 0.0], [0.0, 1e-15]])
        with pytest.raises(NumericalError):
            regularized_inverse(ill)

    def test_well_conditioned_succeeds(self):
        from pystatistics.mvnmle.mcar_test import regularized_inverse
        good = np.array([[2.0, 0.5], [0.5, 3.0]])
        inv, was_reg = regularized_inverse(good)
        assert not was_reg
        np.testing.assert_allclose(inv @ good, np.eye(2), atol=1e-10)


class TestNoIdentityFallback:
    """Parameter extraction must never return identity covariance."""

    def test_source_has_no_identity_fallback(self):
        src = (Path(__file__).parent.parent.parent
               / "pystatistics" / "mvnmle" / "_utils.py").read_text()
        assert "NumericalError" in src
        assert "sigma = np.eye(n_vars)" not in src


class TestNoSilentRegularization:
    """Source files must not silently add ridge penalties."""

    def test_wilcox_no_dead_code(self):
        src = (Path(__file__).parent.parent.parent
               / "pystatistics" / "hypothesis" / "backends"
               / "_wilcox_test.py").read_text()
        assert "signaltonoise" not in src, "Dead try/except should be removed"
