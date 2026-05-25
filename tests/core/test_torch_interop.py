"""Tests for the torch host/device transfer helpers.

The single invariant under test: ``to_host_f64`` casts to float64 only
*after* moving the tensor to the host, so it works on every backend —
including MPS, which has no float64 dtype and rejects an on-device
``.to(torch.float64)``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pystatistics.core.compute.torch_interop import to_host_f64


def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


@pytest.mark.parametrize("device", _devices())
def test_returns_float64_numpy_with_correct_values(device):
    src = np.array([[1.5, -2.25], [0.0, 3.75]], dtype=np.float32)
    t = torch.as_tensor(src, device=device)
    out = to_host_f64(t)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, src.astype(np.float64), rtol=0, atol=0)


@pytest.mark.parametrize("device", _devices())
def test_detaches_grad_tensor(device):
    # A tensor that requires grad cannot be sent to numpy without detach;
    # the helper must handle it (CPU is enough to exercise the detach).
    t = torch.ones(3, device=device, requires_grad=True)
    out = to_host_f64(t)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, np.ones(3))


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="requires Apple Silicon MPS",
)
def test_mps_float32_tensor_downloads_without_device_side_cast():
    # Regression guard: the naive ``.to(float64).cpu()`` raises on MPS.
    # The helper must move to host first, so this must succeed.
    t = torch.ones(4, device="mps", dtype=torch.float32)
    out = to_host_f64(t)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, np.ones(4))
