# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **Performance (GPU direct MVN MLE): fused per-evaluation value+gradient halves
  host<->device sync overhead.** The direct (`algorithm='direct'`) GPU path runs a
  host-side optimiser (scipy L-BFGS-B) that calls the device objective once per
  accepted step. Previously each evaluation paid the host<->device synchronisation
  **twice** — once for the objective value (`compute_objective` → `.item()`) and
  once for the gradient (`compute_gradient` → `.cpu()`), with the gradient pass
  redundantly recomputing the forward Cholesky. The GPU objectives now expose
  `compute_value_and_gradient`, which computes the value and gradient in a single
  device pass (the objective value is read off the same closed-form intermediates
  the analytic gradient already forms — no extra factorisation) and returns them
  through **one** coalesced device→host copy; `run_scaled_minimize`
  (`backends/_optimize.py`) drives scipy with that single `jac=True` callable when
  available, falling back to the previous two-callable path otherwise. On Apple
  Silicon (MPS), where sync latency dominates the fit at large `p`, isolated
  per-evaluation cost at `p=100`, `n=50000` drops from ~4.68 s to ~2.48 s
  (**1.88× faster**, ~2.2 s saved per evaluation), bringing Metal end-to-end
  per-eval in line with discrete-GPU compute. CUDA and CPU paths are unaffected in
  numerics and see the same (or fewer) round-trips. Estimates are unchanged: the
  fused value/gradient match the separate paths to floating-point rounding
  (new guards in `tests/mvnmle/test_gpu_batched_equiv.py` and
  `tests/mvnmle/test_optimize.py`). New benchmark
  `benchmarks/mvnmle_gpu_per_eval.py` measures the per-eval old-vs-fused ratio so
  the saving cannot silently regress.
