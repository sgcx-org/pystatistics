# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **Fixed: float32 GPU binomial/Poisson/Gamma GLMs (`backend='gpu'`) — and
  therefore `survival.discrete_time(backend='gpu')` — no longer spuriously fail on
  discrete-time person-period designs at moderate scale.** On Apple Silicon (MPS)
  the float32 IRLS loop in `regression/backends/gpu_glm.py` stopped at the √n·eps
  deviance round-off floor while slowly-converging low-baseline-hazard interval
  dummies were still moving; the strict acceptance gate then (correctly) saw a
  non-stationary iterate and raised `NumericalError`, even though float32 is fully
  capable of fitting the design. The float32 path now iterates until it is
  genuinely stationary, deciding via the relative Newton decrement — the same
  affine-invariant measure the acceptance gate uses, computed cheaply each
  iteration from the XᵀWX/XᵀWz already formed on-device — and damps float32
  overshoot with monotone-descent step-halving. The strict float64 acceptance gate
  is unchanged, so genuinely float32-infeasible designs (e.g. a fine monthly grid
  whose conditioning puts the Newton decrement above the gate threshold) still fail
  loud, pointing the user to `backend='cpu'` or `backend='gpu_fp64'`. On a
  representative flchain person-period design the MPS float32 fit now matches the
  CPU float64 fit to ~1e-6 on the identifiable coefficients at yearly and quarterly
  granularity, where it previously raised. The CPU and CUDA-float64 paths are
  unchanged. New helpers in `regression/backends/_irls_step.py`
  (`relative_newton_decrement`, `step_halve`) with unit tests; new GPU regression
  tests in `tests/regression/test_irls_step.py`.
