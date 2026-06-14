"""Tests for example data and the deterministic MCAR generators."""

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.mice import datasets


class TestExample:
    def test_shape_and_missing(self):
        assert datasets.EXAMPLE.shape == (12, 3)
        assert np.isnan(datasets.EXAMPLE).any()


class TestGaussianComplete:
    def test_shape(self):
        x = datasets.make_gaussian_complete(50, seed=0)
        assert x.shape == (50, 3)

    def test_deterministic(self):
        a = datasets.make_gaussian_complete(20, seed=3)
        b = datasets.make_gaussian_complete(20, seed=3)
        np.testing.assert_array_equal(a, b)

    def test_no_missing(self):
        x = datasets.make_gaussian_complete(20, seed=0)
        assert not np.isnan(x).any()


class TestMakeMcar:
    def test_deterministic(self):
        comp = datasets.make_gaussian_complete(40, seed=1)
        a = datasets.make_mcar(comp, 0.2, seed=5)
        b = datasets.make_mcar(comp, 0.2, seed=5)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))

    def test_introduces_missing(self):
        comp = datasets.make_gaussian_complete(100, seed=1)
        miss = datasets.make_mcar(comp, 0.3, seed=5)
        assert np.isnan(miss).any()

    def test_no_all_missing_row_or_column(self):
        comp = datasets.make_gaussian_complete(60, seed=2)
        miss = datasets.make_mcar(comp, 0.5, seed=9)
        assert not np.all(np.isnan(miss), axis=1).any()
        assert not np.all(np.isnan(miss), axis=0).any()

    def test_observed_values_unchanged(self):
        comp = datasets.make_gaussian_complete(30, seed=2)
        miss = datasets.make_mcar(comp, 0.2, seed=4)
        observed = ~np.isnan(miss)
        np.testing.assert_array_equal(miss[observed], comp[observed])

    def test_protect_columns(self):
        comp = datasets.make_gaussian_complete(50, seed=2)
        miss = datasets.make_mcar(comp, 0.4, seed=4, protect_columns=(2,))
        assert not np.isnan(miss[:, 2]).any()

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_bad_prop_rejected(self, bad):
        comp = datasets.make_gaussian_complete(20, seed=0)
        with pytest.raises(ValidationError):
            datasets.make_mcar(comp, bad, seed=0)

    def test_nan_input_rejected(self):
        with pytest.raises(ValidationError):
            datasets.make_mcar(datasets.EXAMPLE, 0.2, seed=0)
