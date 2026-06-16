"""Pattern grouping must stay correct past 62 variables.

`MLEObjectiveBase` groups rows by missingness pattern using a powers-of-two
code. With int64 codes, `2**i` overflows for variable index >= 63, collapsing
distinct patterns onto the same code; rows with different masks then get grouped
together and the group's observed mask slices NaN (missing) cells into the
"observed" data. These tests guard the arbitrary-precision path used for
n_vars > 62.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatistics.mvnmle._objectives.base import MLEObjectiveBase


class _Obj(MLEObjectiveBase):
    # MLEObjectiveBase is abstract; we only exercise its pattern extraction.
    def get_initial_parameters(self): ...
    def compute_objective(self, theta): ...
    def compute_gradient(self, theta): ...
    def extract_parameters(self, theta): ...


def _patterns_all_finite(obj) -> bool:
    return all(np.isfinite(np.asarray(p.data)).all() for p in obj.patterns)


def _ground_truth_n_patterns(X) -> int:
    return int(np.unique(~np.isnan(X), axis=0).shape[0])


@pytest.mark.parametrize("n_vars", [40, 62, 63, 80, 120])
def test_grouping_correct_across_the_int64_boundary(n_vars):
    rng = np.random.default_rng(n_vars)
    X = rng.standard_normal((400, n_vars))
    # distinct patterns, including missingness at high indices (the overflow zone)
    X[0:100, 0:3] = np.nan
    X[100:200, n_vars - 3:n_vars] = np.nan
    X[200:280, n_vars - 1] = np.nan
    obj = _Obj(X, skip_validation=True)
    assert _patterns_all_finite(obj), f"NaN in observed data at n_vars={n_vars}"
    assert obj.n_patterns == _ground_truth_n_patterns(X)


def test_high_index_only_difference_not_merged():
    """Two patterns differing only in variable 79 must NOT collapse together
    (the exact int64-overflow collision)."""
    n_vars = 80
    X = np.zeros((200, n_vars))
    X[:] = np.arange(200)[:, None] + 1.0  # all-distinct, finite, fully observed
    X[0:100, n_vars - 1] = np.nan          # pattern A: var 79 missing
    # rows 100:200 fully observed (pattern B)
    obj = _Obj(X, skip_validation=True)
    assert obj.n_patterns == 2
    assert _patterns_all_finite(obj)
    # the pattern missing var 79 must observe exactly n_vars-1 columns
    sizes = sorted(len(p.observed_indices) for p in obj.patterns)
    assert sizes == [n_vars - 1, n_vars]
