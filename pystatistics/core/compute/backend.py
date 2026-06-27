"""
Canonical resolution of the public ``backend=`` argument.

Single source of truth for the PyStatistics backend/precision convention
(see ``pystatistics/CONVENTIONS.md``). ``backend=`` jointly encodes *device*
and *precision*:

    'cpu'       -> CPU, float64   (the reference path)
    'gpu'       -> GPU, float32   (CUDA or MPS, auto-resolved; the speed default)
    'gpu_fp64'  -> GPU, float64   (CUDA only; a correctness path)
    'auto'      -> GPU-fp32 if CUDA is present, else CPU (never auto-MPS)

Every module routes its ``backend=`` handling through :func:`resolve_backend`
so the semantics and the error wording are byte-for-byte identical library
wide. This is the mechanical enforcement of constitution rule S0 (one name,
one meaning): there is exactly one place that decides what a backend string
means, and exactly one phrasing for each failure.

The CPU path never imports torch (``detect_gpu`` is only consulted for GPU /
auto requests), preserving the no-torch-cost guarantee for CPU-only installs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.compute import device as _device
from pystatistics.core.compute.device import DeviceInfo


#: The full public backend vocabulary. A module exposes a *subset* of this via
#: its own ``Literal`` annotation (the honest-subset rule); the resolution
#: semantics live here regardless.
PUBLIC_BACKENDS: tuple[str, ...] = ('cpu', 'gpu', 'gpu_fp64', 'auto')

PublicBackend = Literal['cpu', 'gpu', 'gpu_fp64', 'auto']


# --- Canonical messages -------------------------------------------------------
# One phrasing per failure mode, reused everywhere. Do not fork these.

FP64_REQUIRES_CUDA_MSG = (
    "backend='gpu_fp64' requires CUDA: Apple Silicon (Metal/MPS) has no "
    "float64. Use backend='gpu' (float32) on Apple Silicon, or backend='cpu' "
    "for a double-precision fit."
)

NO_GPU_MSG = (
    "No GPU available (need CUDA or MPS). Use backend='cpu', or install "
    "PyTorch with CUDA/MPS support."
)


def unknown_backend_message(backend: object, valid: Sequence[str]) -> str:
    """Canonical 'unknown backend' message listing the valid options."""
    opts = ", ".join(repr(v) for v in valid)
    return f"Unknown backend {backend!r}. Valid options: {opts}."


def fp64_unsupported_message(valid: Sequence[str]) -> str:
    """Canonical message when a module has no GPU float64 path at all."""
    opts = ", ".join(repr(v) for v in valid)
    return (
        "backend='gpu_fp64' is not available for this procedure: it has no GPU "
        f"float64 path. Use backend='cpu' for double precision. Valid options: "
        f"{opts}."
    )


@dataclass(frozen=True)
class ComputeTarget:
    """A resolved ``(device, precision)`` pair from a public backend string.

    Attributes:
        backend: the normalized public string actually selected — one of
            ``'cpu'``, ``'gpu'``, ``'gpu_fp64'`` (``'auto'`` is resolved away).
        device_type: ``'cpu'``, ``'cuda'``, or ``'mps'``.
        use_fp64: ``True`` for float64, ``False`` for float32.
        device: the underlying :class:`DeviceInfo`.
    """

    backend: str
    device_type: Literal['cpu', 'cuda', 'mps']
    use_fp64: bool
    device: DeviceInfo

    @property
    def is_gpu(self) -> bool:
        return self.device_type in ('cuda', 'mps')


def valid_backends(supports_fp64: bool) -> tuple[str, ...]:
    """The public vocabulary a module exposes, given its fp64 capability."""
    if supports_fp64:
        return PUBLIC_BACKENDS
    return tuple(b for b in PUBLIC_BACKENDS if b != 'gpu_fp64')


def resolve_backend(
    backend: str | None,
    *,
    supports_fp64: bool = True,
    input_on_gpu: bool = False,
) -> ComputeTarget:
    """Resolve a public backend string to a concrete compute target.

    Args:
        backend: the user's ``backend=`` value, or ``None`` to resolve by the
            default policy (numpy/CPU input -> ``'cpu'``; GPU-tensor input ->
            ``'gpu'`` when ``input_on_gpu`` is True).
        supports_fp64: whether *this* procedure has a CUDA float64 path. When
            ``False``, ``'gpu_fp64'`` is rejected up front (the honest-subset
            rule) rather than silently downgraded.
        input_on_gpu: ``True`` when the input is already a GPU tensor, so an
            unspecified backend resolves to ``'gpu'`` rather than ``'cpu'``.

    Returns:
        The resolved :class:`ComputeTarget`.

    Raises:
        ValidationError: unknown backend string, or ``'gpu_fp64'`` requested of
            a procedure with no float64 GPU path.
        RuntimeError: a GPU was requested but none is available, or
            ``'gpu_fp64'`` was requested on a non-CUDA GPU (MPS).
    """
    allowed = valid_backends(supports_fp64)

    if backend is None:
        backend = 'gpu' if input_on_gpu else 'cpu'

    if backend not in PUBLIC_BACKENDS:
        raise ValidationError(unknown_backend_message(backend, allowed))

    if backend == 'gpu_fp64' and not supports_fp64:
        raise ValidationError(fp64_unsupported_message(allowed))

    if backend == 'cpu':
        return ComputeTarget('cpu', 'cpu', True, _device.get_cpu_info())

    if backend == 'auto':
        # Policy: auto selects a GPU only on CUDA. MPS is float32-only and not
        # the R-validated default, so auto never picks it. (CONVENTIONS.md)
        gpu = _device.detect_gpu()
        if gpu is not None and gpu.device_type == 'cuda':
            return ComputeTarget('gpu', 'cuda', False, gpu)
        return ComputeTarget('cpu', 'cpu', True, _device.get_cpu_info())

    # Explicit GPU request: 'gpu' or 'gpu_fp64'.
    gpu = _device.detect_gpu()
    if gpu is None:
        raise RuntimeError(NO_GPU_MSG)

    if backend == 'gpu_fp64':
        if gpu.device_type != 'cuda':
            raise RuntimeError(FP64_REQUIRES_CUDA_MSG)
        return ComputeTarget('gpu_fp64', 'cuda', True, gpu)

    # backend == 'gpu' (float32, CUDA or MPS)
    return ComputeTarget('gpu', gpu.device_type, False, gpu)
