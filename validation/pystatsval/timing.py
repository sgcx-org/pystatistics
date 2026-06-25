"""Repeated wall-clock timing of a callable.

One job: run a callable with warmup iterations (discarded) followed by timed
repeats, and summarise the wall-clock distribution. Used by the benchmark
runners so every engine is timed the same way.

# NON-DETERMINISTIC: wall-clock measurement is inherently non-deterministic
# (it depends on the machine's instantaneous load, the OS scheduler, GPU clock
# state, etc.). That is the point of this module — it *measures* time. The
# callable's RESULT is expected to be deterministic; only the recorded durations
# vary run to run. Warmup exists precisely to discard cold-start effects (e.g.
# first-call GPU/MPS initialisation) so the reported repeats are comparable.
"""

from __future__ import annotations

import time
from typing import Any, Callable


def time_call(fn: Callable[[], Any], *,
              warmup: int = 1,
              reps: int = 5) -> tuple[dict[str, Any], Any]:
    """Run ``fn`` ``warmup`` times (discarded) then ``reps`` timed times.

    Parameters
    ----------
    fn
        Zero-argument callable. Its return value is captured from the last timed
        repeat and returned alongside the timing summary.
    warmup
        Number of discarded warmup calls (>= 0). The first GPU/MPS call pays a
        one-time initialisation cost; at least one warmup removes it.
    reps
        Number of timed calls (>= 1). The median is the headline statistic
        (robust to occasional scheduler hiccups).

    Returns
    -------
    (summary, last_result)
        ``summary`` is a dict with keys ``median_s``, ``min_s``, ``max_s``,
        ``mean_s``, ``reps``, ``times_s`` (the raw per-rep durations).

    Raises
    ------
    ValueError
        If ``warmup`` is negative or ``reps`` is not positive.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")

    for _ in range(warmup):
        fn()

    times: list[float] = []
    last_result: Any = None
    for _ in range(reps):
        start = time.perf_counter()
        last_result = fn()
        times.append(time.perf_counter() - start)

    times_sorted = sorted(times)
    n = len(times_sorted)
    median = (times_sorted[n // 2] if n % 2
              else 0.5 * (times_sorted[n // 2 - 1] + times_sorted[n // 2]))
    summary = {
        "median_s": median,
        "min_s": times_sorted[0],
        "max_s": times_sorted[-1],
        "mean_s": sum(times) / n,
        "reps": reps,
        "times_s": times,
    }
    return summary, last_result
