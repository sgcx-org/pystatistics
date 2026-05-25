"""Host/device transfer helpers for the PyTorch GPU backends.

One job: move tensors off a compute device into host numpy arrays
correctly across CUDA and Apple Silicon (MPS).

The single invariant this module exists to enforce:

    Cast to float64 only AFTER moving the tensor to the host.

MPS has no float64 dtype, so an on-device ``tensor.to(torch.float64)``
raises ``TypeError: Cannot convert a MPS Tensor to float64``. The
download must therefore be ``.cpu().to(torch.float64)``, never
``.to(torch.float64).cpu()``. Centralising it here means no backend
can reintroduce the device-side cast by accident (Coding Bible: make
the wrong thing hard to do accidentally).

This is also correct and lossless on CUDA: for a float64-on-device
tensor the host-side cast is a no-op; for a float32 tensor the
resulting float64 values are identical regardless of cast order.

``torch`` is imported lazily inside each function so that importing
this module never pulls torch into a CPU-only install.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def to_host_f64(tensor: Any) -> NDArray[np.float64]:
    """Download a torch tensor to a contiguous float64 numpy array.

    Detaches from autograd, moves to the host, then casts to float64
    (in that order — see the module docstring for why the order is
    load-bearing on MPS). Safe on CUDA, MPS, and CPU tensors.

    Args:
        tensor: A ``torch.Tensor`` on any device.

    Returns:
        A float64 ``numpy.ndarray`` with the tensor's values.
    """
    import torch

    return tensor.detach().cpu().to(torch.float64).numpy()
