"""
Backend-convention tests for descriptive (hardware-independent).

descriptive is an honest-subset module: it has a GPU path but no GPU float64
path, so its public backend vocabulary is {'cpu', 'gpu', 'auto'} and
'gpu_fp64' is rejected up front (no hardware needed to observe this).
"""

import numpy as np
import pytest

from pystatistics.descriptive import describe
from pystatistics.core.exceptions import ValidationError


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((50, 3))


def test_gpu_fp64_rejected_no_fp64_path(data):
    with pytest.raises(ValidationError, match="no GPU float64 path"):
        describe(data, backend='gpu_fp64')


def test_unknown_backend_rejected(data):
    with pytest.raises(ValidationError, match="Unknown backend"):
        describe(data, backend='quantum')


def test_cpu_is_default(data):
    assert 'cpu' in describe(data).backend_name
