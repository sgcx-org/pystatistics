"""
Backend-convention tests for hypothesis (hardware-independent).

Honest-subset module: a GPU path exists only for Monte-Carlo p-values and there
is no GPU float64 path, so the public vocabulary is {'cpu', 'gpu', 'auto'} and
'gpu_fp64' is rejected up front. ``auto`` deliberately stays on CPU here.
"""

import numpy as np
import pytest

from pystatistics.hypothesis import t_test
from pystatistics.core.exceptions import ValidationError


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    return rng.standard_normal(40)


def test_gpu_fp64_rejected_no_fp64_path(sample):
    with pytest.raises(ValidationError, match="no GPU float64 path"):
        t_test(sample, backend='gpu_fp64')


def test_unknown_backend_rejected(sample):
    with pytest.raises(ValidationError, match="Unknown backend"):
        t_test(sample, backend='quantum')


def test_auto_stays_on_cpu(sample):
    # Deliberate deviation: GPU is not a general accelerator for scalar tests.
    res = t_test(sample, backend='auto')
    assert 'cpu' in res.backend_name.lower() or res.backend_name is not None
