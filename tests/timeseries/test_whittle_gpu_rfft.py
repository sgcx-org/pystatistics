"""Regression tests for the Whittle GPU backends' rFFT call.

torch's MPS backend routes a plain ``torch.fft.rfft`` through an
internal empty-tensor resize that emits a deprecation warning as of
torch 2.12 (and is slated to stop working). Both GPU Whittle backends
therefore pass a pre-sized ``out=`` tensor. These tests pin the two
guarantees of that fix: construction is warning-free, and the
periodogram matches the float64 numpy reference computation.
"""

import warnings

import numpy as np
import pytest


def _gpu_device():
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return None


pytestmark = pytest.mark.skipif(
    _gpu_device() is None, reason="no torch GPU (CUDA/MPS)"
)


def _series(K=8, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((K, n))
    return Y - Y.mean(axis=1, keepdims=True)


class TestWhittleGPURfft:

    def test_construction_emits_no_warnings(self):
        from pystatistics.timeseries.backends.whittle_batch_gpu import (
            BatchedWhittleGPU,
        )
        from pystatistics.timeseries.backends.whittle_gpu import (
            WhittleGPULikelihood,
        )
        Y = _series()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            BatchedWhittleGPU(Y, 1, 1, device=_gpu_device())
            WhittleGPULikelihood(Y[0], 1, 1, device=_gpu_device())
        assert not rec, [str(w.message) for w in rec]

    def test_periodogram_matches_numpy_reference(self):
        """The out= form must compute the same rFFT as the plain call:
        periodogram at the fp32 floor vs the float64 numpy reference."""
        from pystatistics.timeseries.backends.whittle_batch_gpu import (
            BatchedWhittleGPU,
        )
        from pystatistics.timeseries.backends.whittle_gpu import (
            WhittleGPULikelihood,
        )
        Y = _series()
        K, n = Y.shape
        m = (n - 1) // 2
        ref = (np.abs(np.fft.rfft(Y, axis=1)) ** 2 / n)[:, 1 : 1 + m]

        b = BatchedWhittleGPU(Y, 1, 1, device=_gpu_device())
        got = b._periodogram.cpu().numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-4)

        s = WhittleGPULikelihood(Y[0], 1, 1, device=_gpu_device())
        np.testing.assert_allclose(
            s._periodogram.cpu().numpy(), ref[0], rtol=1e-4,
        )
