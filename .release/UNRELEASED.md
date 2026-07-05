# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **timeseries: unified per-series failure contract for `arima_batch` across
  backends** (`pystatistics/timeseries/_arima_batch.py`, new
  `pystatistics/timeseries/_arima_batch_contract.py`). The last open defect
  from the timeseries module validation of 4.6.4: on batches where the
  Whittle optimum is non-stationary, the two backends implemented different
  failure contracts for the same estimator (a Guarantee-2 violation). The
  CPU loop raised `ConvergenceError` on the FIRST failing series (making
  the batch API useless at scale and discarding good fits — measured: the
  validation's near-unit-root ar=0.97/ma=0.3 K=64 batch has 17/64 perfectly
  good series that the old CPU path threw away), while the GPU path
  returned NON-STATIONARY AR estimates as plain numbers with only
  `converged=False` set (measured on Forge CUDA, RTX 5070 Ti, 4.6.4:
  ar_mean 1.011 / ar_max 1.054 returned where the exact fp64 Kalman
  reference is ar≈0.966; violates the RIGOR GPU corollary — an explicit
  GPU request that cannot certify a valid result must not present failure
  as a number).
  New contract, identical on every backend by construction (both paths flow
  through one shared `enforce_batch_failure_contract`):
  - a series **fails** when its backend cannot certify a valid *stationary*
    fit — CPU: the single-series fitter raised `ConvergenceError`; GPU: a
    host-side float64 AR-root check on the returned estimates (torch-build
    independent, per RIGOR R14). Deliberately NOT the per-series Adam
    `converged` flag: measured on MPS/CUDA fp32, healthy batches converge
    only ~1/64 by gradient-tol with blessed-tier-correct estimates (gating
    on the flag would falsely refuse them, A6 obligation 2), and one
    both-extreme series converged=True at a non-stationary mirror-basin
    optimum (the flag is neither necessary nor sufficient for validity).
  - **all K series fail → `ConvergenceError`** (matches the single-series
    behavior on the same input);
  - **some fail → failed rows' `ar`/`ma`/`sigma2`/`mean` are NaN,
    `converged` forced False**, a `UserWarning` naming the count is emitted
    and recorded in the solution's `.warnings` envelope — a failed fit can
    never be consumed accidentally, while the batch stays usable at scale;
  - **no failures → bit-identical passthrough** (healthy batches unchanged
    vs 4.6.4, no warning; asserted by an identity test).
  Single-series `arima(..., method='Whittle')` keeps its raise behavior
  (no partial-failure case). MA invertibility is not checked — the
  single-series Whittle contract only checks AR stationarity, and the batch
  matches that contract exactly. `summary()` now reports a
  `Failed: x/K (estimates NaN)` line and uses NaN-aware aggregates.
  Post-fix measurements: CPU partial on the near-unit-root batch returns
  17 finite rows with AR mean 0.9655 (exact reference ≈0.966); GPU (fp32)
  NaNs the 39 non-stationary rows, surviving rows AR mean 0.9733; healthy
  ar=0.6/ma=0.4 K=64 batches: no warning, estimates unchanged. New test
  module `tests/timeseries/test_arima_batch_contract.py` (contract-layer
  unit tests + CPU/GPU integration + CPU-vs-GPU semantic-parity tests);
  full timeseries suite green (559 passed) on CPU+MPS; contract confirmed
  on Forge CUDA (gpu and gpu_fp64).

- **timeseries: Whittle GPU backends no longer trigger torch's MPS rfft
  deprecation warning** (`pystatistics/timeseries/backends/whittle_gpu.py`,
  `pystatistics/timeseries/backends/whittle_batch_gpu.py`). On Apple
  Silicon with torch 2.12, every `torch.fft.rfft` call routes through an
  internal empty-tensor resize that emits "An output with one or more
  elements was resized…" (deprecated behavior slated to stop working in a
  future torch release; fires on ANY MPS rfft call, reproduced minimally —
  not specific to our shapes). Both backends now pass a pre-sized `out=`
  tensor (complex64/complex128 per precision), which avoids the deprecated
  internal path entirely rather than suppressing the warning. Values are
  bit-identical to the plain call (verified `torch.equal` on MPS);
  periodogram matches the numpy float64 reference at the fp32 floor. New
  regression tests in `tests/timeseries/test_whittle_gpu_rfft.py`
  (warning-free construction + periodogram parity); re-verified on Forge
  CUDA since the change touches the CUDA path too.
