"""
Tests for CLAUDE.md Rule 1 compliance in mvnmle module.

Verifies that ill-conditioned or singular matrices are handled
*visibly* — either by raising (strict mode, ``regularize=False``)
or by falling back to a pseudo-inverse with a ``UserWarning``
(``regularize=True``, the R-compatible default). Silent fallback
is forbidden either way.
"""

import warnings

import pytest
import numpy as np
from pathlib import Path

from pystatistics.core.exceptions import NumericalError


class TestRegularizedInverse:
    """regularized_inverse() must raise in strict mode and warn in
    permissive mode — never silently regularize."""

    def test_singular_matrix_raises_in_strict_mode(self):
        from pystatistics.mvnmle.mcar_test import regularized_inverse
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(NumericalError):
            regularized_inverse(singular, regularize=False)

    def test_ill_conditioned_raises_in_strict_mode(self):
        from pystatistics.mvnmle.mcar_test import regularized_inverse
        ill = np.array([[1.0, 0.0], [0.0, 1e-15]])
        with pytest.raises(NumericalError):
            regularized_inverse(ill, regularize=False)

    def test_ill_conditioned_warns_and_uses_pinv_in_default_mode(self):
        """Default regularize=True falls back to pinv with a loud warning.

        We don't assert on the numerical content of the inverse — pinv
        intentionally zeros out singular values below its rcond
        threshold, and the whole point of regularize=True is accepting
        that degraded precision. We only pin the contract: was_reg is
        True and the user heard about it via a warning.
        """
        from pystatistics.mvnmle.mcar_test import regularized_inverse
        ill = np.array([[1.0, 0.0], [0.0, 1e-15]])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            inv, was_reg = regularized_inverse(ill)
        assert was_reg is True
        assert inv.shape == ill.shape
        assert np.all(np.isfinite(inv))
        assert any("ill-conditioned" in str(w.message) and "pseudo-inverse"
                   in str(w.message) for w in captured), (
            "Pseudo-inverse fallback must emit a UserWarning — no silent "
            "regularization."
        )

    def test_singular_matrix_warns_and_uses_pinv_in_default_mode(self):
        from pystatistics.mvnmle.mcar_test import regularized_inverse
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            inv, was_reg = regularized_inverse(singular)
        assert was_reg is True
        assert any("pseudo-inverse" in str(w.message) for w in captured)

    def test_well_conditioned_succeeds_without_warning(self):
        from pystatistics.mvnmle.mcar_test import regularized_inverse
        good = np.array([[2.0, 0.5], [0.5, 3.0]])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            inv, was_reg = regularized_inverse(good)
        assert not was_reg
        assert not any("pseudo-inverse" in str(w.message) for w in captured)
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
