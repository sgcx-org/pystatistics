"""Tests for DataSource — with focus on the device transfer API.

DataSource factory paths (from_arrays, from_tensors, from_dataframe,
from_file) are exercised by higher-level tests; this file targets the
new `.to(device)` method and its interaction with the device-aware
GPU backends.
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatistics import DataSource
from pystatistics.core.exceptions import ValidationError


def _gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


class TestDataSourceDevice:
    """`.to(device)` + `.device` property."""

    def test_default_device_is_cpu(self):
        ds = DataSource.from_arrays(X=np.ones((10, 3)))
        assert ds.device == "cpu"
        assert not ds.supports("gpu_native")

    def test_to_cpu_is_noop_for_cpu_source(self):
        ds = DataSource.from_arrays(X=np.ones((10, 3)))
        ds2 = ds.to("cpu")
        assert ds2.device == "cpu"
        # numpy arrays are preserved as numpy
        assert isinstance(ds2["X"], np.ndarray)

    def test_to_cuda_returns_new_datasource(self):
        if not _gpu_available():
            pytest.skip("no GPU available")
        import torch
        X = np.random.randn(10, 3)
        ds = DataSource.from_arrays(X=X)
        gds = ds.to("cuda")
        # Immutability: original is untouched
        assert ds.device == "cpu"
        assert isinstance(ds["X"], np.ndarray)
        # New one is device-resident
        assert gds.device.startswith("cuda")
        assert isinstance(gds["X"], torch.Tensor)
        assert gds["X"].device.type == "cuda"
        assert gds.supports("gpu_native")

    def test_roundtrip_cuda_cpu_preserves_values(self):
        if not _gpu_available():
            pytest.skip("no GPU available")
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        ds = DataSource.from_arrays(X=X, y=y)
        gds = ds.to("cuda")
        cds = gds.to("cpu")
        # Roundtrip produces numpy again
        assert isinstance(cds["X"], np.ndarray)
        assert isinstance(cds["y"], np.ndarray)
        np.testing.assert_array_equal(cds["X"], X)
        np.testing.assert_array_equal(cds["y"], y)
        assert cds.device == "cpu"

    def test_same_device_is_same_object(self):
        """Calling .to() with the current device returns self (no
        unnecessary copy)."""
        if not _gpu_available():
            pytest.skip("no GPU available")
        ds = DataSource.from_arrays(X=np.ones((10, 3)))
        gds = ds.to("cuda")
        assert gds.to("cuda") is gds

    def test_to_cuda_without_gpu_raises(self, monkeypatch):
        """When no GPU is available, `.to('cuda')` fails loudly (Rule 1
        — no silent fallback to CPU)."""
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        ds = DataSource.from_arrays(X=np.ones((10, 3)))
        with pytest.raises(RuntimeError, match="CUDA"):
            ds.to("cuda")

    def test_preserves_non_array_metadata(self):
        """Per-array auxiliary storage survives device transfer."""
        if not _gpu_available():
            pytest.skip("no GPU available")
        ds = DataSource.from_arrays(X=np.ones((10, 3)), y=np.zeros(10))
        gds = ds.to("cuda")
        assert gds.n_observations == 10
        assert gds.keys() == ds.keys()


class TestGpuDatasourceIntegration:
    """End-to-end: build a GPU DataSource and run a fit on it with zero
    extra host↔device transfer."""

    def test_pca_on_gpu_datasource_matches_cpu_at_tier(self):
        from pystatistics.multivariate import pca
        from pystatistics.core.compute.tolerances import GPU_FP32
        if not _gpu_available():
            pytest.skip("no GPU available")
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1000, 20))
        ds = DataSource.from_arrays(X=X)
        gds = ds.to("cuda")

        r_cpu = pca(X, backend="cpu")
        r_gpu = pca(gds["X"], method="gram")
        np.testing.assert_allclose(
            r_cpu.sdev, r_gpu.sdev,
            rtol=GPU_FP32.rtol, atol=GPU_FP32.atol,
        )

    def test_pca_cpu_backend_rejects_gpu_tensor(self):
        """Passing a GPU tensor with ``backend='cpu'`` raises, not
        silently moves. Rule 1: no hidden GPU→CPU migration."""
        from pystatistics.multivariate import pca
        if not _gpu_available():
            pytest.skip("no GPU available")
        X = np.random.randn(100, 5)
        gds = DataSource.from_arrays(X=X).to("cuda")
        with pytest.raises(ValidationError, match="torch.Tensor"):
            pca(gds["X"], backend="cpu")
