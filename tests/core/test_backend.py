"""
Tests for the canonical backend resolver (``core.compute.backend``).

This is the single source of truth for the PyStatistics backend/precision
convention, so its behavior is tested directly here and relied upon by every
module's own backend tests.
"""

import pytest

from pystatistics.core.compute import device as dev_mod
from pystatistics.core.compute import backend as bk
from pystatistics.core.compute.backend import resolve_backend, ComputeTarget
from pystatistics.core.compute.device import DeviceInfo
from pystatistics.core.exceptions import ValidationError


def _cuda_info() -> DeviceInfo:
    return DeviceInfo(
        device_type='cuda', device_index=0, name='Fake CUDA',
        memory_bytes=16 * 1024**3, compute_capability=(8, 9),
    )


def _mps_info() -> DeviceInfo:
    return DeviceInfo(
        device_type='mps', device_index=0, name='Apple Silicon GPU',
        memory_bytes=None, compute_capability=None,
    )


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(dev_mod, 'detect_gpu', lambda *a, **k: None)


@pytest.fixture
def cuda_gpu(monkeypatch):
    monkeypatch.setattr(dev_mod, 'detect_gpu', lambda *a, **k: _cuda_info())


@pytest.fixture
def mps_gpu(monkeypatch):
    monkeypatch.setattr(dev_mod, 'detect_gpu', lambda *a, **k: _mps_info())


class TestDefaults:
    def test_none_resolves_to_cpu(self, no_gpu):
        t = resolve_backend(None)
        assert t == ComputeTarget('cpu', 'cpu', True, t.device)
        assert t.use_fp64 is True and not t.is_gpu

    def test_none_with_gpu_tensor_resolves_to_gpu(self, cuda_gpu):
        t = resolve_backend(None, input_on_gpu=True)
        assert t.backend == 'gpu' and t.device_type == 'cuda'
        assert t.use_fp64 is False

    def test_cpu_never_touches_gpu_detection(self, monkeypatch):
        # CPU path must not call detect_gpu at all (no torch import cost).
        def boom(*a, **k):  # pragma: no cover - must not run
            raise AssertionError("detect_gpu called on the cpu path")
        monkeypatch.setattr(dev_mod, 'detect_gpu', boom)
        assert resolve_backend('cpu').device_type == 'cpu'


class TestGpu:
    def test_gpu_fp32_on_cuda(self, cuda_gpu):
        t = resolve_backend('gpu')
        assert t.device_type == 'cuda' and t.use_fp64 is False

    def test_gpu_fp32_on_mps(self, mps_gpu):
        t = resolve_backend('gpu')
        assert t.device_type == 'mps' and t.use_fp64 is False

    def test_gpu_fp64_on_cuda(self, cuda_gpu):
        t = resolve_backend('gpu_fp64')
        assert t.backend == 'gpu_fp64'
        assert t.device_type == 'cuda' and t.use_fp64 is True

    def test_gpu_fp64_on_mps_raises_cuda_required(self, mps_gpu):
        with pytest.raises(RuntimeError, match="requires CUDA"):
            resolve_backend('gpu_fp64')

    def test_gpu_unavailable_raises(self, no_gpu):
        with pytest.raises(RuntimeError, match="No GPU available"):
            resolve_backend('gpu')

    def test_gpu_fp64_unavailable_raises(self, no_gpu):
        with pytest.raises(RuntimeError, match="No GPU available"):
            resolve_backend('gpu_fp64')


class TestAuto:
    def test_auto_picks_cuda(self, cuda_gpu):
        t = resolve_backend('auto')
        assert t.device_type == 'cuda' and t.use_fp64 is False

    def test_auto_never_picks_mps(self, mps_gpu):
        # MPS is fp32-only and not the validated default: auto -> CPU.
        assert resolve_backend('auto').device_type == 'cpu'

    def test_auto_falls_back_to_cpu(self, no_gpu):
        assert resolve_backend('auto').device_type == 'cpu'


class TestHonestSubset:
    def test_gpu_fp64_rejected_when_unsupported(self, cuda_gpu):
        with pytest.raises(ValidationError, match="no GPU float64 path"):
            resolve_backend('gpu_fp64', supports_fp64=False)

    def test_unsupported_message_lists_valid_subset(self, cuda_gpu):
        with pytest.raises(ValidationError) as exc:
            resolve_backend('gpu_fp64', supports_fp64=False)
        msg = str(exc.value)
        assert "'gpu_fp64'" not in msg.split("Valid options:")[1]
        assert "'gpu'" in msg and "'cpu'" in msg

    def test_supported_module_still_allows_fp64_string_offline(self, no_gpu):
        # supports_fp64=True but no GPU present: the string is valid, the
        # failure is the (later) hardware check, not a vocabulary rejection.
        with pytest.raises(RuntimeError, match="No GPU available"):
            resolve_backend('gpu_fp64', supports_fp64=True)


class TestUnknown:
    def test_unknown_backend_raises_validation(self, no_gpu):
        with pytest.raises(ValidationError, match="Unknown backend"):
            resolve_backend('quantum')

    def test_unknown_backend_lists_options(self, no_gpu):
        with pytest.raises(ValidationError, match="gpu_fp64"):
            resolve_backend('cuda')  # 'cuda' is an internal device type, not public
