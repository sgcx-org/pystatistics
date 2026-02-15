"""
Tests for the Result[P] envelope.

Validates:
    - Generic type parameter works with arbitrary payload types
    - Frozen immutability
    - Default factories (warnings, provenance)
    - has_warning() method
    - Provenance metadata contains expected version keys
"""

from dataclasses import FrozenInstanceError, dataclass

import pytest

from pystatistics.core.result import Result, _default_provenance


# ═══════════════════════════════════════════════════════════════════════
# Test payload types
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FakeParams:
    """Minimal payload for testing."""
    value: float


@dataclass(frozen=True)
class MultiFieldParams:
    """Payload with multiple fields."""
    alpha: float
    beta: float
    name: str


# ═══════════════════════════════════════════════════════════════════════
# Construction and field access
# ═══════════════════════════════════════════════════════════════════════


class TestResultConstruction:
    """Result can be created with any payload type."""

    def test_basic_creation(self):
        params = FakeParams(value=42.0)
        result = Result(
            params=params,
            info={"method": "test"},
            timing={"total_seconds": 0.01},
            backend_name="cpu",
        )
        assert result.params.value == 42.0
        assert result.info["method"] == "test"
        assert result.timing["total_seconds"] == 0.01
        assert result.backend_name == "cpu"

    def test_multi_field_params(self):
        params = MultiFieldParams(alpha=0.5, beta=1.5, name="test")
        result = Result(
            params=params,
            info={},
            timing=None,
            backend_name="gpu",
        )
        assert result.params.alpha == 0.5
        assert result.params.beta == 1.5
        assert result.params.name == "test"

    def test_timing_none(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
        )
        assert result.timing is None

    def test_timing_with_breakdown(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing={"total_seconds": 1.0, "e_step": 0.6, "m_step": 0.4},
            backend_name="cpu_em",
        )
        assert result.timing["e_step"] == 0.6
        assert result.timing["m_step"] == 0.4

    def test_info_dict_arbitrary_keys(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={"converged": True, "iterations": 23, "method": "em"},
            timing=None,
            backend_name="cpu",
        )
        assert result.info["converged"] is True
        assert result.info["iterations"] == 23


# ═══════════════════════════════════════════════════════════════════════
# Default factories
# ═══════════════════════════════════════════════════════════════════════


class TestDefaults:
    """Default values for warnings and provenance."""

    def test_warnings_default_empty(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
        )
        assert result.warnings == ()
        assert isinstance(result.warnings, tuple)

    def test_warnings_explicit(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
            warnings=("convergence slow", "condition number high"),
        )
        assert len(result.warnings) == 2
        assert "convergence slow" in result.warnings

    def test_provenance_auto_generated(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
        )
        assert "pystatistics_version" in result.provenance
        assert "numpy_version" in result.provenance

    def test_provenance_explicit_override(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
            provenance={"custom": "metadata"},
        )
        assert result.provenance == {"custom": "metadata"}
        assert "pystatistics_version" not in result.provenance


# ═══════════════════════════════════════════════════════════════════════
# Immutability
# ═══════════════════════════════════════════════════════════════════════


class TestImmutability:
    """Result is frozen — no attribute mutation allowed."""

    def test_cannot_set_params(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
        )
        with pytest.raises(FrozenInstanceError):
            result.params = FakeParams(value=2.0)

    def test_cannot_set_backend_name(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
        )
        with pytest.raises(FrozenInstanceError):
            result.backend_name = "gpu"

    def test_cannot_set_warnings(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
        )
        with pytest.raises(FrozenInstanceError):
            result.warnings = ("new warning",)

    def test_cannot_set_timing(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing={"total_seconds": 0.5},
            backend_name="cpu",
        )
        with pytest.raises(FrozenInstanceError):
            result.timing = None


# ═══════════════════════════════════════════════════════════════════════
# has_warning()
# ═══════════════════════════════════════════════════════════════════════


class TestHasWarning:
    """has_warning() checks for substring in any warning."""

    def test_no_warnings_returns_false(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
        )
        assert result.has_warning("anything") is False

    def test_exact_match(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
            warnings=("convergence slow",),
        )
        assert result.has_warning("convergence slow") is True

    def test_substring_match(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
            warnings=("convergence slow after 100 iterations",),
        )
        assert result.has_warning("convergence") is True
        assert result.has_warning("100 iterations") is True

    def test_no_match(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
            warnings=("convergence slow",),
        )
        assert result.has_warning("divergence") is False

    def test_multiple_warnings(self):
        result = Result(
            params=FakeParams(value=1.0),
            info={},
            timing=None,
            backend_name="cpu",
            warnings=("condition number high", "near singularity"),
        )
        assert result.has_warning("condition") is True
        assert result.has_warning("singularity") is True
        assert result.has_warning("convergence") is False


# ═══════════════════════════════════════════════════════════════════════
# _default_provenance()
# ═══════════════════════════════════════════════════════════════════════


class TestDefaultProvenance:
    """_default_provenance() generates version metadata."""

    def test_contains_pystatistics_version(self):
        prov = _default_provenance()
        assert "pystatistics_version" in prov
        assert isinstance(prov["pystatistics_version"], str)

    def test_contains_numpy_version(self):
        prov = _default_provenance()
        assert "numpy_version" in prov
        assert isinstance(prov["numpy_version"], str)

    def test_returns_dict(self):
        prov = _default_provenance()
        assert isinstance(prov, dict)

    def test_independent_copies(self):
        """Each call returns a new dict."""
        prov1 = _default_provenance()
        prov2 = _default_provenance()
        assert prov1 is not prov2
        assert prov1 == prov2
