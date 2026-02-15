"""
Tests for p_adjust() matching R p.adjust().

All R reference values verified against R 4.5.2.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.hypothesis import p_adjust


# --- R reference values ---
# Computed in R 4.5.2 with digits=17

PV1 = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 0.9])

R_HOLM_1 = np.array([0.006, 0.05, 0.2, 0.3, 1.0, 1.0])
R_HOCHBERG_1 = np.array([0.006, 0.05, 0.2, 0.3, 0.9, 0.9])
R_HOMMEL_1 = np.array([0.006, 0.05, 0.2, 0.3, 0.9, 0.9])
R_BH_1 = np.array([0.006, 0.03, 0.1, 0.15, 0.6, 0.9])
R_BY_1 = np.array([0.0147, 0.0735, 0.245, 0.3675, 1.0, 1.0])
R_BONFERRONI_1 = np.array([0.006, 0.06, 0.3, 0.6, 1.0, 1.0])

PV2 = np.array([0.01, 0.04, 0.03, 0.005])

R_HOLM_2 = np.array([0.03, 0.06, 0.06, 0.02])
R_BH_2 = np.array([0.02, 0.04, 0.04, 0.02])
R_BONFERRONI_2 = np.array([0.04, 0.16, 0.12, 0.02])


class TestPAdjustMethods:
    """Test each p.adjust method against R."""

    def test_holm(self):
        assert_allclose(p_adjust(PV1, method="holm"), R_HOLM_1, rtol=1e-10)

    def test_hochberg(self):
        assert_allclose(p_adjust(PV1, method="hochberg"), R_HOCHBERG_1, rtol=1e-10)

    def test_hommel(self):
        assert_allclose(p_adjust(PV1, method="hommel"), R_HOMMEL_1, rtol=1e-10)

    def test_bh(self):
        assert_allclose(p_adjust(PV1, method="BH"), R_BH_1, rtol=1e-10)

    def test_fdr_alias(self):
        """'fdr' is an alias for 'BH'."""
        assert_allclose(p_adjust(PV1, method="fdr"), R_BH_1, rtol=1e-10)

    def test_by(self):
        assert_allclose(p_adjust(PV1, method="BY"), R_BY_1, rtol=1e-10)

    def test_bonferroni(self):
        assert_allclose(
            p_adjust(PV1, method="bonferroni"), R_BONFERRONI_1, rtol=1e-10
        )

    def test_none(self):
        result = p_adjust(PV1, method="none")
        assert_allclose(result, PV1, rtol=1e-15)

    def test_holm_set2(self):
        assert_allclose(p_adjust(PV2, method="holm"), R_HOLM_2, rtol=1e-10)

    def test_bh_set2(self):
        assert_allclose(p_adjust(PV2, method="BH"), R_BH_2, rtol=1e-10)

    def test_bonferroni_set2(self):
        assert_allclose(
            p_adjust(PV2, method="bonferroni"), R_BONFERRONI_2, rtol=1e-10
        )


class TestPAdjustNParameter:
    """Test the n parameter (effective number of tests)."""

    def test_holm_n_larger(self):
        """p.adjust(c(0.01, 0.05), method='holm', n=10)."""
        result = p_adjust([0.01, 0.05], method="holm", n=10)
        assert_allclose(result, [0.1, 0.45], rtol=1e-10)

    def test_bh_n_larger(self):
        """p.adjust(c(0.01, 0.05), method='BH', n=10)."""
        result = p_adjust([0.01, 0.05], method="BH", n=10)
        assert_allclose(result, [0.1, 0.25], rtol=1e-10)

    def test_bonferroni_n_larger(self):
        result = p_adjust([0.01, 0.05], method="bonferroni", n=10)
        assert_allclose(result, [0.1, 0.5], rtol=1e-10)


class TestPAdjustEdgeCases:
    """Edge cases and validation."""

    def test_single_pvalue(self):
        """Single p-value should be unchanged (all methods)."""
        for method in ["holm", "hochberg", "hommel", "BH", "BY", "bonferroni"]:
            result = p_adjust([0.05], method=method)
            assert result[0] == pytest.approx(0.05, rel=1e-10), f"Failed for {method}"

    def test_empty_array(self):
        result = p_adjust([], method="holm")
        assert len(result) == 0

    def test_all_ones(self):
        """p-values of 1 should stay at 1."""
        result = p_adjust([1.0, 1.0, 1.0], method="bonferroni")
        assert_allclose(result, [1.0, 1.0, 1.0])

    def test_all_zeros(self):
        """p-values of 0 stay at 0."""
        result = p_adjust([0.0, 0.0, 0.0], method="BH")
        assert_allclose(result, [0.0, 0.0, 0.0])

    def test_nan_preserved(self):
        """NaN positions are preserved."""
        result = p_adjust([0.01, np.nan, 0.05], method="holm")
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])

    def test_clipping(self):
        """Adjusted values never exceed 1."""
        result = p_adjust([0.5, 0.5, 0.5], method="bonferroni")
        assert np.all(result <= 1.0)

    def test_invalid_method(self):
        with pytest.raises(Exception):
            p_adjust([0.05], method="invalid")

    def test_n_too_small(self):
        """n < len(p) should raise."""
        with pytest.raises(Exception):
            p_adjust([0.01, 0.05], method="holm", n=1)

    def test_default_is_holm(self):
        """Default method is 'holm' (not bonferroni!)."""
        result_default = p_adjust(PV1)
        result_holm = p_adjust(PV1, method="holm")
        assert_allclose(result_default, result_holm)

    def test_monotonicity_holm(self):
        """Holm adjusted p-values should be monotone non-decreasing after sorting."""
        pv = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 0.9])
        result = p_adjust(pv, method="holm")
        # After sorting by original p, adjusted should be non-decreasing
        order = np.argsort(pv)
        sorted_adj = result[order]
        assert np.all(np.diff(sorted_adj) >= -1e-15)

    def test_all_nan(self):
        """All NaN input returns all NaN."""
        result = p_adjust([np.nan, np.nan, np.nan], method="holm")
        assert np.all(np.isnan(result))
