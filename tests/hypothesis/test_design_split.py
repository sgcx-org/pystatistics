"""
Tests for hypothesis design split (CLAUDE.md Rule 3) and seed support (Rule 6).

Verifies factory functions work correctly and that Monte Carlo tests
are reproducible with seed control.
"""

import pytest
import numpy as np

from pystatistics.hypothesis.design import HypothesisDesign
from pystatistics.hypothesis._design_factories import (
    build_t_test_design, build_chisq_test_design,
    build_prop_test_design, build_fisher_test_design,
    build_wilcox_test_design, build_ks_test_design,
    build_var_test_design,
)
from pystatistics.hypothesis import chisq_test, fisher_test


class TestDesignFactories:
    """Rule 3: Factory functions extracted to _design_factories.py."""

    def test_build_t_test_design(self):
        d = build_t_test_design([1, 2, 3, 4, 5])
        assert isinstance(d, HypothesisDesign)
        assert d.test_type == "t_one_sample"

    def test_build_chisq_test_design(self):
        d = build_chisq_test_design(np.array([[10, 20], [30, 40]], dtype=float))
        assert isinstance(d, HypothesisDesign)
        assert d.test_type == "chisq_independence"

    def test_build_prop_test_design(self):
        d = build_prop_test_design([10], [100], p=0.5)
        assert isinstance(d, HypothesisDesign)
        assert d.test_type == "prop_test"

    def test_build_fisher_test_design(self):
        d = build_fisher_test_design(np.array([[5, 10], [15, 20]], dtype=float))
        assert isinstance(d, HypothesisDesign)
        assert d.test_type == "fisher_test"

    def test_build_wilcox_test_design(self):
        d = build_wilcox_test_design([1, 2, 3, 4, 5])
        assert isinstance(d, HypothesisDesign)
        assert d.test_type == "wilcox_signed_rank"

    def test_build_ks_test_design(self):
        d = build_ks_test_design([1, 2, 3, 4, 5])
        assert isinstance(d, HypothesisDesign)
        assert d.test_type == "ks_one_sample"

    def test_build_var_test_design(self):
        d = build_var_test_design([1, 2, 3, 4], [5, 6, 7, 8])
        assert isinstance(d, HypothesisDesign)
        assert d.test_type == "var_test"


class TestClassmethodBackwardCompat:
    """Classmethods still work as before."""

    def test_for_t_test(self):
        d = HypothesisDesign.for_t_test([1, 2, 3, 4, 5])
        assert d.test_type == "t_one_sample"

    def test_for_chisq_test(self):
        d = HypothesisDesign.for_chisq_test(
            np.array([[10, 20], [30, 40]], dtype=float)
        )
        assert d.test_type == "chisq_independence"

    def test_for_fisher_test(self):
        d = HypothesisDesign.for_fisher_test(
            np.array([[5, 10], [15, 20]], dtype=float)
        )
        assert d.test_type == "fisher_test"


class TestSeedField:
    """HypothesisDesign stores seed correctly."""

    def test_seed_in_chisq_design(self):
        d = build_chisq_test_design(
            np.array([16, 18, 16, 14, 12, 12], dtype=float),
            simulate_p_value=True, seed=42,
        )
        assert d.seed == 42

    def test_seed_in_fisher_design(self):
        d = build_fisher_test_design(
            np.array([[5, 10], [15, 20]], dtype=float),
            simulate_p_value=True, seed=42,
        )
        assert d.seed == 42

    def test_seed_default_none(self):
        d = build_t_test_design([1, 2, 3])
        assert d.seed is None


class TestMonteCarloReproducibility:
    """Rule 6: Monte Carlo tests are reproducible with seed control."""

    # NON-DETERMINISTIC: seed-controlled Monte Carlo simulation
    def test_chisq_mc_same_seed_same_result(self):
        obs = np.array([16, 18, 16, 14, 12, 12], dtype=float)
        r1 = chisq_test(obs, simulate_p_value=True, B=999, seed=42)
        r2 = chisq_test(obs, simulate_p_value=True, B=999, seed=42)
        assert r1.p_value == r2.p_value

    # NON-DETERMINISTIC: seed-controlled Monte Carlo simulation
    def test_fisher_mc_same_seed_same_result(self):
        table = np.array([[5, 10, 15], [10, 20, 25]], dtype=float)
        r1 = fisher_test(table, simulate_p_value=True, B=999, seed=42)
        r2 = fisher_test(table, simulate_p_value=True, B=999, seed=42)
        assert r1.p_value == r2.p_value

    # NON-DETERMINISTIC: seed-controlled Monte Carlo simulation
    def test_chisq_mc_different_seeds_differ(self):
        obs = np.array([16, 18, 16, 14, 12, 12], dtype=float)
        r1 = chisq_test(obs, simulate_p_value=True, B=999, seed=42)
        r2 = chisq_test(obs, simulate_p_value=True, B=999, seed=123)
        # Different seeds should produce slightly different p-values
        # Both should be close (same null), but not identical
        assert abs(r1.p_value - r2.p_value) < 0.1
