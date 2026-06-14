"""Tests for MICEDesign construction, metadata, and boundary validation."""

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.mice import datasets
from pystatistics.mice.design import MICEDesign


class TestConstruction:
    def test_from_example(self):
        d = MICEDesign.from_array(datasets.EXAMPLE)
        assert d.n == 12
        assert d.p == 3
        assert d.has_missing
        assert d.n_missing == 5

    def test_default_names_and_kinds(self):
        d = MICEDesign.from_array(datasets.EXAMPLE)
        assert d.col_names == ("x0", "x1", "x2")
        assert d.col_kinds == ("numeric", "numeric", "numeric")

    def test_incomplete_columns_detected(self):
        d = MICEDesign.from_array(datasets.EXAMPLE)
        # cols 0, 1, 2 all have at least one NaN in EXAMPLE.
        assert d.incomplete_columns == (0, 1, 2)

    def test_missing_mask_matches_nan(self):
        d = MICEDesign.from_array(datasets.EXAMPLE)
        np.testing.assert_array_equal(d.missing_mask, np.isnan(datasets.EXAMPLE))

    def test_fully_observed_column_has_empty_method(self):
        data = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, 6.0]])
        d = MICEDesign.from_array(data)
        assert d.method_for(0) == ""        # fully observed
        assert d.method_for(1) == "pmm"     # incomplete, default method

    def test_custom_column_names(self):
        d = MICEDesign.from_array(datasets.EXAMPLE, column_names=["a", "b", "c"])
        assert d.col_names == ("a", "b", "c")


class TestMethodAssignment:
    def test_default_method_is_pmm(self):
        d = MICEDesign.from_array(datasets.EXAMPLE)
        assert all(m == "pmm" for m in d.methods if m)

    def test_method_override_all(self):
        d = MICEDesign.from_array(datasets.EXAMPLE, method="norm")
        assert all(m == "norm" for m in d.methods if m)

    def test_per_column_method_by_name(self):
        d = MICEDesign.from_array(
            datasets.EXAMPLE,
            column_names=["a", "b", "c"],
            methods={"a": "norm"},
        )
        assert d.method_for(0) == "norm"
        assert d.method_for(1) == "pmm"

    def test_per_column_method_by_index(self):
        d = MICEDesign.from_array(datasets.EXAMPLE, methods={2: "norm"})
        assert d.method_for(2) == "norm"
        assert d.method_for(0) == "pmm"

    def test_per_column_method_sequence(self):
        d = MICEDesign.from_array(
            datasets.EXAMPLE, methods=["pmm", "norm", "pmm"]
        )
        assert d.method_for(1) == "norm"


class TestValidationFailures:
    def test_1d_rejected(self):
        with pytest.raises(ValidationError):
            MICEDesign.from_array(np.array([1.0, 2.0, np.nan]))

    def test_too_few_rows(self):
        with pytest.raises(ValidationError):
            MICEDesign.from_array(np.array([[1.0, 2.0]]))

    def test_inf_rejected(self):
        data = np.array([[1.0, np.inf], [2.0, 3.0], [4.0, 5.0]])
        with pytest.raises(ValidationError, match="infinite"):
            MICEDesign.from_array(data)

    def test_all_nan_row_rejected(self):
        data = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])
        with pytest.raises(ValidationError, match="all"):
            MICEDesign.from_array(data)

    def test_all_nan_column_rejected(self):
        data = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
        with pytest.raises(ValidationError, match="completely missing"):
            MICEDesign.from_array(data)

    def test_single_column_with_missing_rejected(self):
        data = np.array([[1.0], [np.nan], [3.0]])
        with pytest.raises(ValidationError, match="at least 2 variables"):
            MICEDesign.from_array(data)

    def test_unknown_method_rejected(self):
        with pytest.raises(ValidationError, match="unknown method"):
            MICEDesign.from_array(datasets.EXAMPLE, method="bogus")

    def test_non_numeric_kind_rejected(self):
        with pytest.raises(ValidationError, match="numeric"):
            MICEDesign.from_array(
                datasets.EXAMPLE, column_kinds=["numeric", "categorical", "numeric"]
            )

    def test_mismatched_column_names_length(self):
        with pytest.raises(ValidationError):
            MICEDesign.from_array(datasets.EXAMPLE, column_names=["a", "b"])

    def test_methods_unknown_column_name(self):
        with pytest.raises(ValidationError, match="unknown column"):
            MICEDesign.from_array(datasets.EXAMPLE, methods={"zzz": "pmm"})


class TestNoMissing:
    def test_complete_data_has_no_incomplete_columns(self):
        complete = datasets.make_gaussian_complete(30, seed=0)
        d = MICEDesign.from_array(complete)
        assert not d.has_missing
        assert d.incomplete_columns == ()
        assert all(m == "" for m in d.methods)
