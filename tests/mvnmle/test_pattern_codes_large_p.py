"""Pattern grouping must be correct for any number of variables.

`MLEObjectiveBase` groups rows by exact equality of their missingness masks.
A grouping keyed on a fixed-width integer code would collapse distinct
patterns once the number of variables exceeds the code's width (63 for
int64); rows with different masks would then be grouped together and the
group's observed mask would slice NaN (missing) cells into the "observed"
data. These tests guard against that across the 62/63/64 boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatistics.mvnmle._objectives.base import MLEObjectiveBase


class _Obj(MLEObjectiveBase):
    # Only the pattern grouping is exercised here.
    def get_initial_parameters(self): ...
    def compute_objective(self, theta): ...
    def compute_gradient(self, theta): ...
    def extract_parameters(self, theta): ...


def _patterns_all_finite(obj) -> bool:
    return all(np.isfinite(np.asarray(p.data)).all() for p in obj.patterns)


def _ground_truth_n_patterns(X) -> int:
    return int(np.unique(~np.isnan(X), axis=0).shape[0])


@pytest.mark.parametrize("n_vars", [40, 51, 62, 63, 64, 65, 80, 120])
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
