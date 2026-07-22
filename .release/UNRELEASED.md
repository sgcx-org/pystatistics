# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

### Changed
- **pystatistics now ships as compiled binary wheels instead of a pure-Python
  wheel.** The performance-critical time-series and survival kernels are now
  Cython extension modules rather than Numba-JIT'd Python. Practical effects:
  - **No first-use JIT compilation.** Time-to-first-result is deterministic —
    there is no multi-second warm-up on the first call in a fresh process. This
    matters most for containers, serverless, CI, and other short-lived or
    read-only environments where a JIT cache cannot persist.
  - **Prebuilt wheels** are published for Linux (x86_64, glibc) and macOS
    (Apple Silicon) on CPython 3.11–3.13. Installing from the source
    distribution now requires a C compiler.
  - Numerical results are **unchanged, bit-for-bit**, on every supported
    platform (the kernels are verified against a reference to the last bit in
    CI on each built wheel).
- The ARIMA exact-ML fitter reuses a single scratch workspace across the
  optimizer's likelihood evaluations, removing per-evaluation array allocation.
  Fitted models are identical; warm-fit speed is on par with the previous
  Numba build while cold-start latency is eliminated.

### Removed
- **`numba` (and its `llvmlite` dependency) is no longer required.** This drops
  a large transitive dependency and removes the release-timing coupling to
  Numba's CPython support schedule.
