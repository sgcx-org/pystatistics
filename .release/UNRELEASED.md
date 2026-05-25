# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **Apple Silicon (MPS) GPU support for FP32-capable backends.** The
  `multinom`, `polr`, `gam`, and `arima`/`arima_batch` (Whittle) GPU
  backends now run on Apple Silicon GPUs via `backend='gpu'`, using FP32
  (MPS has no float64). Every operation on these paths is a native Metal
  kernel — no silent CPU fallback. Results are validated against the CPU
  reference at the `GPU_FP32` tolerance tier. `DataSource.to('mps')` now
  works (float64 arrays are downcast to float32 on transfer, since MPS
  has no float64). The CUDA FP64 path and its R-validation are unchanged.
- **`backend='auto'` never selects MPS.** On Apple Silicon, `'auto'`
  routes to the CPU (FP64, R-validated) path; MPS is opt-in only via an
  explicit `backend='gpu'`. This makes `multinom`/`polr`/`gam`/`arima`
  consistent with the existing `regression` and `mvnmle` dispatch policy.
  CUDA is still auto-selected.
- **PCA GPU remains CUDA-only by design.** PCA is fundamentally an
  SVD / symmetric-eigendecomposition problem, and neither `linalg.svd`
  (the `method='svd'` path) nor the eigendecomposition of `X'X` (the
  `method='gram'` path) has a Metal kernel — both silently fall back to
  the CPU on MPS. Rather than advertise a GPU path that isn't one,
  `pca(backend='gpu')` now raises an actionable error on Apple Silicon
  (use `backend='cpu'`, or `backend='auto'` which selects CPU on MPS).
- **MVN MLE GPU remains CUDA-only by design.** The EM algorithm's
  iterative small-step + per-pattern scatter workload is far slower on
  Metal than on the CPU, so `mlest(algorithm='em', backend='gpu')` now
  raises an actionable error on Apple Silicon (use `backend='cpu'`, or
  `backend='auto'` which routes to CPU). Direct (BFGS) GPU fitting is
  unaffected.
- **Whittle ARIMA GPU FP32 convergence.** On the FP32 GPU path, an
  L-BFGS-B `ABNORMAL_TERMINATION_IN_LNSRCH` at a stationary point (the
  line search hitting the FP32 noise floor) is now accepted rather than
  raised, matching the CPU fit at the `GPU_FP32` tier. The AR-stationarity
  check still rejects genuinely bad optima. FP64 behaviour is unchanged.
