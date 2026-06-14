"""Tests for visit-sequence construction."""

import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.mice._visit import default_visit_sequence, resolve_visit_sequence


class TestDefaultVisitSequence:
    def test_ascending_order(self):
        assert default_visit_sequence((2, 0, 1)) == (0, 1, 2)

    def test_empty(self):
        assert default_visit_sequence(()) == ()


class TestResolveVisitSequence:
    def test_none_uses_default(self):
        assert resolve_visit_sequence((1, 0), None) == (0, 1)

    def test_explicit_passthrough(self):
        assert resolve_visit_sequence((0, 1, 2), [2, 1, 0]) == (2, 1, 0)

    def test_repeats_allowed(self):
        # A variable may be visited more than once per iteration.
        assert resolve_visit_sequence((0, 1), [0, 1, 0]) == (0, 1, 0)

    def test_unknown_column_rejected(self):
        with pytest.raises(ValidationError, match="no missing values"):
            resolve_visit_sequence((0, 1), [0, 1, 5])

    def test_omitted_incomplete_column_rejected(self):
        with pytest.raises(ValidationError, match="omits"):
            resolve_visit_sequence((0, 1, 2), [0, 1])
