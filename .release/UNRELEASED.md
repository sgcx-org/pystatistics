# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **ETS fitting is dramatically faster on longer series — estimates
  unchanged.** The ETS state-space forward recursion
  (`ets_recursion` in `pystatistics/timeseries/_ets_models.py`), which the
  maximum-likelihood optimiser evaluates once per objective call, was a
  per-timestep interpreted-Python loop whose per-observation cost grew with
  series length. It is now a numba-compiled kernel
  (`ets_recursion_nb` in the new `pystatistics/timeseries/_ets_kernels.py`),
  matching the in-module pattern already used by the SARIMA Kalman filter
  (`_arima_kalman.py`) and the STL loop (`_stl_core.py`):
  `@njit(cache=True, fastmath=False)`. The recursion itself is now flat at
  ~6.7 ns/observation (was ~0.64 µs/obs and rising), a **40–96× kernel
  speedup** (39.6× at n=200, 77.5× at n=1000, 95.6× at n=5000). End-to-end
  `ets()` fit time on a length-*n* series (model `AAN`) drops from
  28.4 ms → 6.1 ms (n=200, 4.7×), 174 ms → 9.4 ms (n=1000, 18.5×), and
  323 ms → 7.0 ms (n=5000, 46×); fit time no longer grows with *n*. Against
  R's `forecast::ets` (9.0.0) on the identical series, the fit-time ratio
  falls from a gap that widened with length to roughly on par at small *n*
  and faster than R at large *n* (measured py/R ≈ 4.6× at n=200, 2.0× at
  n=1000, 0.3× at n=5000).
- **The compiled kernel reproduces the previous recursion bit-for-bit.**
  `fastmath=False` forbids floating-point reassociation, so the JIT kernel
  matches the pure-numpy reference (retained verbatim as
  `_ets_recursion_reference`) to the last bit in fp64. A new test,
  `tests/timeseries/test_ets_kernel_parity.py`, asserts exact equality
  (atol = 0) of fitted values, residuals, and the full state history across
  every ETS(error, trend, season) family and a parameter/initial-state grid
  — so fitted parameters, log-likelihood, states, AIC/AICc/BIC, and `"ZZZ"`
  model selection are identical to the previous release; only speed changes.
  numba caches the compiled kernel to disk (`cache=True`), so the one-time
  compile cost is paid only on the first run after install.
