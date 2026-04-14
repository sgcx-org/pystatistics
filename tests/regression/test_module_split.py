"""
Tests for regression module split (CLAUDE.md Rule 3).

Verifies that the split into _linear.py / _glm.py / _formatting.py
maintains backward compatibility via the solution.py re-export shim.
"""

import pytest
import numpy as np

from pystatistics.regression import fit


class TestModuleSplitImports:
    """Rule 3: Linear and GLM are separate modules, both importable."""

    def test_linear_from_new_path(self):
        from pystatistics.regression._linear import LinearParams, LinearSolution
        assert LinearParams is not None
        assert LinearSolution is not None

    def test_glm_from_new_path(self):
        from pystatistics.regression._glm import GLMParams, GLMSolution
        assert GLMParams is not None
        assert GLMSolution is not None

    def test_backward_compat_solution_py(self):
        """solution.py re-exports must be the same objects."""
        from pystatistics.regression._linear import LinearParams as LP1
        from pystatistics.regression._glm import GLMParams as GP1
        from pystatistics.regression.solution import LinearParams as LP2, GLMParams as GP2
        assert LP1 is LP2
        assert GP1 is GP2

    def test_formatting_shared(self):
        from pystatistics.regression._formatting import significance_stars
        assert significance_stars(0.0001) == "***"
        assert significance_stars(0.005) == "**"
        assert significance_stars(0.03) == "*"
        assert significance_stars(0.08) == "."
        assert significance_stars(0.5) == ""


class TestNamedCoefficients:
    """Verify names= kwarg and coef dict property."""

    def test_fit_with_names(self):
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(50), rng.standard_normal((50, 2))])
        y = X @ [1, 2, -1] + rng.standard_normal(50) * 0.5
        result = fit(X, y, names=["height", "weight"])
        assert "(Intercept)" in result.coef
        assert "height" in result.coef
        assert "weight" in result.coef
        assert len(result.coef) == 3

    def test_fit_with_names_exact_length(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2))
        y = X @ [2, -1] + rng.standard_normal(50)
        result = fit(X, y, names=["a", "b"])
        assert list(result.coef.keys()) == ["a", "b"]

    def test_fit_names_wrong_length_raises(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        y = rng.standard_normal(50)
        with pytest.raises(ValueError, match="names must have"):
            fit(X, y, names=["a"])

    def test_glm_coef_dict(self):
        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(100), rng.standard_normal((100, 2))])
        y = (X @ [0, 1, -1] + rng.standard_normal(100) > 0).astype(float)
        result = fit(X, y, family="binomial", names=["x1", "x2"])
        assert "(Intercept)" in result.coef
        assert "x1" in result.coef

    def test_coef_without_names_uses_generic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2))
        y = rng.standard_normal(50)
        result = fit(X, y)
        assert "B[0]" in result.coef
        assert "B[1]" in result.coef
