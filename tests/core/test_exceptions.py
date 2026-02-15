"""
Tests for PyStatistics exception hierarchy.

Validates:
    - Inheritance chain (all exceptions catchable via PyStatisticsError)
    - Diagnostic attributes on SingularMatrixError, NotPositiveDefiniteError,
      ConvergenceError
    - str/repr work correctly
    - Default attribute values (None for optional attributes)
"""

import pytest

from pystatistics.core.exceptions import (
    ConvergenceError,
    DimensionError,
    NotPositiveDefiniteError,
    NumericalError,
    PyStatisticsError,
    SingularMatrixError,
    ValidationError,
)


# ═══════════════════════════════════════════════════════════════════════
# Inheritance hierarchy
# ═══════════════════════════════════════════════════════════════════════


class TestInheritance:
    """Every exception is catchable via PyStatisticsError."""

    def test_validation_error_is_pystatistics_error(self):
        with pytest.raises(PyStatisticsError):
            raise ValidationError("bad input")

    def test_dimension_error_is_validation_error(self):
        with pytest.raises(ValidationError):
            raise DimensionError("wrong shape")

    def test_dimension_error_is_pystatistics_error(self):
        with pytest.raises(PyStatisticsError):
            raise DimensionError("wrong shape")

    def test_numerical_error_is_pystatistics_error(self):
        with pytest.raises(PyStatisticsError):
            raise NumericalError("computation failed")

    def test_singular_matrix_error_is_numerical_error(self):
        with pytest.raises(NumericalError):
            raise SingularMatrixError("singular")

    def test_singular_matrix_error_is_pystatistics_error(self):
        with pytest.raises(PyStatisticsError):
            raise SingularMatrixError("singular")

    def test_not_positive_definite_is_numerical_error(self):
        with pytest.raises(NumericalError):
            raise NotPositiveDefiniteError("not PD")

    def test_not_positive_definite_is_pystatistics_error(self):
        with pytest.raises(PyStatisticsError):
            raise NotPositiveDefiniteError("not PD")

    def test_convergence_error_is_pystatistics_error(self):
        with pytest.raises(PyStatisticsError):
            raise ConvergenceError("did not converge", iterations=100)

    def test_convergence_error_is_not_numerical_error(self):
        """ConvergenceError inherits from PyStatisticsError, not NumericalError."""
        err = ConvergenceError("did not converge", iterations=100)
        assert not isinstance(err, NumericalError)


# ═══════════════════════════════════════════════════════════════════════
# Simple exceptions (no extra attributes)
# ═══════════════════════════════════════════════════════════════════════


class TestSimpleExceptions:
    """ValidationError, DimensionError, NumericalError carry only a message."""

    def test_pystatistics_error_message(self):
        err = PyStatisticsError("base error")
        assert str(err) == "base error"

    def test_validation_error_message(self):
        err = ValidationError("X: cannot convert to array")
        assert "cannot convert" in str(err)

    def test_dimension_error_message(self):
        err = DimensionError("X: expected 2D, got 3D")
        assert "expected 2D" in str(err)

    def test_numerical_error_message(self):
        err = NumericalError("overflow in computation")
        assert "overflow" in str(err)


# ═══════════════════════════════════════════════════════════════════════
# SingularMatrixError
# ═══════════════════════════════════════════════════════════════════════


class TestSingularMatrixError:
    """SingularMatrixError carries matrix diagnostic attributes."""

    def test_all_attributes(self):
        err = SingularMatrixError(
            "X'X is singular",
            matrix_name="X'X",
            condition_number=1e18,
            rank=3,
            expected_rank=5,
        )
        assert str(err) == "X'X is singular"
        assert err.matrix_name == "X'X"
        assert err.condition_number == 1e18
        assert err.rank == 3
        assert err.expected_rank == 5

    def test_defaults_are_none(self):
        err = SingularMatrixError("singular")
        assert err.matrix_name is None
        assert err.condition_number is None
        assert err.rank is None
        assert err.expected_rank is None

    def test_partial_attributes(self):
        err = SingularMatrixError(
            "rank deficient",
            matrix_name="design",
            rank=2,
        )
        assert err.matrix_name == "design"
        assert err.rank == 2
        assert err.condition_number is None
        assert err.expected_rank is None

    def test_catchable_with_attributes(self):
        """Attributes accessible in except block."""
        with pytest.raises(SingularMatrixError) as exc_info:
            raise SingularMatrixError(
                "singular", matrix_name="A", condition_number=1e15
            )
        assert exc_info.value.matrix_name == "A"
        assert exc_info.value.condition_number == 1e15


# ═══════════════════════════════════════════════════════════════════════
# NotPositiveDefiniteError
# ═══════════════════════════════════════════════════════════════════════


class TestNotPositiveDefiniteError:
    """NotPositiveDefiniteError carries matrix diagnostic attributes."""

    def test_all_attributes(self):
        err = NotPositiveDefiniteError(
            "Cholesky failed",
            matrix_name="covariance",
            min_eigenvalue=-0.001,
        )
        assert str(err) == "Cholesky failed"
        assert err.matrix_name == "covariance"
        assert err.min_eigenvalue == -0.001

    def test_defaults_are_none(self):
        err = NotPositiveDefiniteError("not PD")
        assert err.matrix_name is None
        assert err.min_eigenvalue is None

    def test_catchable_with_attributes(self):
        with pytest.raises(NotPositiveDefiniteError) as exc_info:
            raise NotPositiveDefiniteError(
                "not PD", matrix_name="Sigma", min_eigenvalue=-1e-8
            )
        assert exc_info.value.min_eigenvalue == pytest.approx(-1e-8)


# ═══════════════════════════════════════════════════════════════════════
# ConvergenceError
# ═══════════════════════════════════════════════════════════════════════


class TestConvergenceError:
    """ConvergenceError carries iteration diagnostics."""

    def test_all_attributes(self):
        err = ConvergenceError(
            "EM did not converge",
            iterations=500,
            final_change=1e-4,
            reason="max_iterations",
            threshold=1e-8,
        )
        assert str(err) == "EM did not converge"
        assert err.iterations == 500
        assert err.final_change == 1e-4
        assert err.reason == "max_iterations"
        assert err.threshold == 1e-8

    def test_required_iterations(self):
        """iterations is required (positional)."""
        err = ConvergenceError("failed", 42)
        assert err.iterations == 42

    def test_defaults_are_none(self):
        err = ConvergenceError("failed", iterations=10)
        assert err.final_change is None
        assert err.reason is None
        assert err.threshold is None

    def test_catchable_with_attributes(self):
        with pytest.raises(ConvergenceError) as exc_info:
            raise ConvergenceError(
                "diverging",
                iterations=5,
                reason="diverging",
                final_change=100.0,
            )
        assert exc_info.value.iterations == 5
        assert exc_info.value.reason == "diverging"
        assert exc_info.value.final_change == 100.0

    def test_zero_iterations(self):
        """Edge case: convergence fails at iteration 0."""
        err = ConvergenceError("immediate failure", iterations=0)
        assert err.iterations == 0
