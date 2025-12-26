"""
Execution timing utilities.

Provides accurate timing for both CPU and GPU operations, handling
CUDA synchronization automatically for accurate GPU measurements.
"""

import time
from contextlib import contextmanager
from typing import Iterator


class Timer:
    """
    Accumulating timer with optional CUDA synchronization.
    
    Provides accurate timing for both CPU and GPU operations. When sync_cuda=True,
    synchronizes CUDA before each timing measurement to ensure GPU operations
    have completed.
    
    Usage:
        timer = Timer()
        timer.start()
        
        with timer.section('qr_decomposition'):
            Q, R = np.linalg.qr(X)
            
        with timer.section('back_substitution'):
            beta = solve_triangular(R, Q.T @ y)
            
        timer.stop()
        result = timer.result()
        # {'total_seconds': 0.05, 'qr_decomposition': 0.03, 'back_substitution': 0.02}
    """
    
    def __init__(self, sync_cuda: bool = False):
        """
        Initialize timer.
        
        Args:
            sync_cuda: If True, synchronize CUDA before timing measurements.
                       Required for accurate GPU timing.
        """
        self._sync_cuda = sync_cuda
        self._sections: dict[str, float] = {}
        self._start_time: float | None = None
        self._total: float | None = None
        
    def _sync(self) -> None:
        """Synchronize CUDA if enabled and available."""
        if self._sync_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
    
    def start(self) -> None:
        """Start the overall timer."""
        self._sync()
        self._start_time = time.perf_counter()
        
    def stop(self) -> None:
        """Stop the overall timer."""
        self._sync()
        if self._start_time is None:
            raise RuntimeError("Timer.stop() called before start()")
        self._total = time.perf_counter() - self._start_time
        
    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """
        Time a named section.
        
        Args:
            name: Section identifier (used as key in result dict)
            
        Yields:
            None
            
        Note:
            Sections can overlap with each other and with the total time.
            The timer does not enforce mutual exclusion.
        """
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            # Accumulate if section called multiple times
            self._sections[name] = self._sections.get(name, 0.0) + elapsed
    
    def result(self) -> dict[str, float]:
        """
        Get timing results.
        
        Returns:
            Dictionary with 'total_seconds' and all section timings
            
        Raises:
            RuntimeError: If called before stop()
        """
        if self._total is None:
            raise RuntimeError("Timer.result() called before stop()")
        
        result = {'total_seconds': self._total}
        result.update(self._sections)
        return result


@contextmanager
def timed(sync_cuda: bool = False) -> Iterator[Timer]:
    """
    Context manager for simple timing.
    
    Usage:
        with timed() as timer:
            result = expensive_computation()
        print(f"Took {timer.result()['total_seconds']:.3f}s")
    
    Args:
        sync_cuda: If True, synchronize CUDA for accurate GPU timing
        
    Yields:
        Timer instance
    """
    timer = Timer(sync_cuda=sync_cuda)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()
