"""
Backend-convention tests for montecarlo (hardware-independent).

Honest-subset module: bootstrap/permutation have a GPU path but no GPU float64
path (replicate counts dominate, not precision), so the public vocabulary is
{'cpu', 'gpu', 'auto'} and 'gpu_fp64' is rejected up front.
"""

import numpy as np
import pytest

from pystatistics.montecarlo import boot
from pystatistics.core.exceptions import ValidationError


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal(60)


def _mean(d, idx):
    return float(np.mean(d[idx]))


def test_gpu_fp64_rejected_no_fp64_path(data):
    with pytest.raises(ValidationError, match="no GPU float64 path"):
        boot(data, _mean, n_resamples=99, seed=1, backend='gpu_fp64')


def test_unknown_backend_rejected(data):
    with pytest.raises(ValidationError, match="Unknown backend"):
        boot(data, _mean, n_resamples=99, seed=1, backend='quantum')


def test_cpu_default(data):
    res = boot(data, _mean, n_resamples=99, seed=1)
    assert 'cpu' in res.backend_name.lower()
