# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- Fixed a correctness bug in the FP64 GPU MVN MLE objective's parameter
  unpacking. `GPUObjectiveFP64._unpack_gpu` (in
  `pystatistics/mvnmle/_objectives/gpu_fp64.py`) reconstructed the Cholesky
  factor's off-diagonal entries with a hand-rolled **column-major** loop, while
  the canonical `CholeskyParameterization.unpack` and the FP32 path place them
  **row-major** (via `tril_indices`). For `n_vars >= 3` the two orderings
  diverge, so the FP64 path optimised one covariance while `extract_parameters`
  reported another — the returned Σ could be wrong (~0.06 max abs error on a
  4-variable example). The discrepancy was dtype-independent (fp32 and fp64 both
  affected through this code path); it went unnoticed because FP64 GPU is
  CUDA-only (MPS blocks FP64) and was untested on Apple Silicon.
- Unified both GPU paths on a single shared reconstruction,
  `unpack_cholesky` (in `pystatistics/mvnmle/_objectives/_batched_cholesky.py`),
  so FP32 and FP64 cannot drift apart again. It builds L functionally
  (`diag_embed` for the diagonal, out-of-place `index_put` for the
  off-diagonals) — no `.diagonal().copy_()` view aliasing — and reproduces the
  canonical `CholeskyParameterization.unpack` Σ to floating tolerance in both
  precisions. The FP32 bounded-Cholesky branch keeps its own sigmoid/tanh
  reconstruction (different math) but the same row-major off-diagonal ordering.
- Added `tests/mvnmle/test_gpu_unpack_equiv.py` asserting that the shared helper
  and both objectives' `_unpack_gpu` match the canonical Σ for
  `n_vars in {3, 4, 8}` and dtype in {float32, float64} on the CPU torch device,
  plus differentiability and FP32/FP64 agreement checks.
