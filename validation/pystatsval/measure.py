"""Device-aware timing with peak-memory capture.

One job: the richer sibling of :mod:`pystatsval.timing` for GPU benchmarks — warmup
+ timed repeats, but with (a) device synchronization so asynchronous CUDA/MPS kernel
dispatch cannot make the GPU look artificially fast, and (b) peak-memory capture
(GPU via the torch allocator, CPU via process RSS high-water mark). Returns the same
``(summary, last_result)`` shape as :func:`pystatsval.timing.time_call`, with the
extra ``peak_mem_bytes`` key, so a record can be built from it directly.

# NON-DETERMINISTIC: wall-clock and memory readings depend on machine state; the
# callable's RESULT is expected deterministic. We report median + spread, not a
# single shot, which is why warmup/repeats and sync live here.
"""

from __future__ import annotations

import resource
import sys
import time
from statistics import median
from typing import Any, Callable


def _resolve_gpu(device: str) -> str:
    """Map a device label to the GPU backend actually present.

    ``'gpu'`` means "whatever GPU this host has" — CUDA on an NVIDIA box, MPS on an
    Apple Silicon Mac. ``'cuda'``/``'mps'`` force a specific one; all fall back to
    ``'cpu'`` if the requested accelerator is unavailable.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        return "mps" if (mps and mps.is_available()) else "cpu"
    if device == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps and mps.is_available():
            return "mps"
    return "cpu"


def _sync(device: str) -> None:
    backend = _resolve_gpu(device)
    if backend in ("cuda", "mps"):
        import torch
        (torch.cuda if backend == "cuda" else torch.mps).synchronize()


def _gpu_peak_reset(device: str) -> None:
    backend = _resolve_gpu(device)
    if backend == "cpu":
        return
    import torch
    if backend == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif backend == "mps":
        # MPS has no peak tracker; empty the cache so the post-run reading reflects
        # this call's footprint.
        torch.mps.empty_cache()


def _gpu_peak_read(device: str) -> int | None:
    backend = _resolve_gpu(device)
    if backend == "cpu":
        return None
    import torch
    if backend == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if backend == "mps":
        return int(torch.mps.driver_allocated_memory())
    return None


def _rss_bytes() -> int:
    """Process peak resident set size in bytes (macOS reports bytes, Linux KB)."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maxrss if sys.platform == "darwin" else maxrss * 1024)


def _iqr(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n < 4:
        return s[-1] - s[0]
    return s[(3 * n) // 4] - s[n // 4]


def measure(fn: Callable[[], Any], *,
            device: str = "cpu",
            repeats: int = 5,
            warmup: int = 1,
            capture_memory: bool = True) -> tuple[dict[str, Any], Any]:
    """Time ``fn`` ``repeats`` times after ``warmup`` untimed runs, on ``device``.

    ``device`` (``'cpu'`` / ``'gpu'`` / ``'mps'`` / ``'cuda'``) controls only the
    sync + memory instrumentation, not what ``fn`` does. ``fn`` takes no args (close
    over them).

    Returns ``(summary, last_result)`` where ``summary`` has keys ``median_s``,
    ``min_s``, ``max_s``, ``mean_s``, ``iqr_s``, ``times_s``, ``reps``, and
    ``peak_mem_bytes`` (None on CPU when memory cannot be attributed). The shape is
    record-ready: pass ``summary`` as the ``wall`` argument to
    :func:`pystatsval.record.make_record`.

    Raises
    ------
    ValueError
        If ``repeats < 1`` or ``warmup < 0``.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    for _ in range(warmup):
        fn()
        _sync(device)

    times: list[float] = []
    last: Any = None
    peak: int | None = None
    for _ in range(repeats):
        if capture_memory:
            _gpu_peak_reset(device)
        rss_before = _rss_bytes() if capture_memory else 0
        t0 = time.perf_counter()
        last = fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
        if capture_memory:
            gpu_peak = _gpu_peak_read(device)
            if gpu_peak is not None:
                peak = gpu_peak if peak is None else max(peak, gpu_peak)
            else:
                peak = max(peak or 0, _rss_bytes() - rss_before)

    summary = {
        "median_s": median(times),
        "min_s": min(times),
        "max_s": max(times),
        "mean_s": sum(times) / len(times),
        "iqr_s": _iqr(times),
        "times_s": times,
        "reps": repeats,
        "peak_mem_bytes": peak,
    }
    return summary, last
