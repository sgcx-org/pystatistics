# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- MICE GPU backend now supports **ordered categorical targets** (incomplete
  ordered-factor columns), imputed with the `polr` proportional-odds method on
  CUDA and Apple Silicon/MPS. The fit is a batched Newton iteration across all `m`
  chains in the unconstrained threshold parameterization (so the cumulative-logit
  thresholds stay strictly ordered by construction), with the gradient and
  observed Hessian taken by autograd, followed by the same joint threshold+slope
  posterior draw and category sampling as the CPU path. GPU results match the CPU
  `polr` to ~1e-9 in coefficients/thresholds/covariance at the true MLE.
  **With this, every categorical target type — binary, unordered, ordered — is now
  supported on GPU**, and the full mixed-type dataset matches R `mice` 3.19.0
  imputed category proportions directly on GPU (within Monte-Carlo tolerance).

- MICE GPU backend now supports **unordered categorical targets** (incomplete
  factor columns with more than two levels), imputed with the `polyreg` method on
  CUDA and Apple Silicon/MPS. The fit is a batched multinomial-logit Newton
  iteration across all `m` chains (same convex MLE as the CPU L-BFGS-B path, last
  class as reference), each step forming the multinomial block Hessian and solving
  through `cholesky_ex` plus the matmul-series triangular inverse on MPS, followed
  by the posterior draw `beta* ~ N(beta_hat, H^-1)` and an inverse-CDF category
  draw per missing cell. GPU results match the CPU `polyreg` distributionally at
  the GPU/FP32 tolerance (coefficients/probabilities agree with the true MLE to
  ~1e-6). Ordinal (`polr`) targets remain refused on GPU with a clear message.

- MICE GPU backend now supports **binary categorical targets** (incomplete
  two-level columns), imputed with the `logreg` method on CUDA and Apple
  Silicon/MPS. The fit is a batched, ridge-stabilised IRLS logistic regression
  run across all `m` imputation chains at once (Newton on the logistic
  log-likelihood with per-chain convergence freezing), followed by the same
  posterior draw `beta* ~ N(beta_hat, (X'WX)^-1)` R `mice` uses, then a Bernoulli
  draw per missing cell. The per-step solve goes through `cholesky_ex` plus the
  matmul-series triangular inverse on MPS (avoiding its slow `solve_triangular`).
  GPU results match the CPU `logreg` distributionally at the GPU/FP32 tolerance
  (coefficients agree to machine precision when the fit converges; under
  separation the predicted probabilities agree to ~5e-5). Multinomial (`polyreg`)
  and ordinal (`polr`) targets are still refused on GPU with a clear message;
  numeric targets and categorical predictors are unchanged.

- MICE GPU backend now supports **categorical predictors** for numeric-target
  imputation. Previously `backend='gpu'` refused any data containing a
  categorical column; it now treatment-dummy-encodes categorical predictor
  columns (matching the CPU path) and only refuses categorical *targets*
  (incomplete categorical columns), with a clear message. Numeric-only problems
  are unchanged. Categorical-target imputation on GPU is still pending.

- Internal (no user-facing change): unified the GPU batched triangular-factor
  inverse onto one primitive. The MVNMLE GPU objective now uses the same
  matmul-series inverse as the MICE GPU draw
  (`core.compute.linalg.batched_tri_inv_series`); the older block-recursion
  inverse (`batched_tri_inv`) was removed. The series inverse is now autograd-safe
  — a differentiable Newton step from a detached, already-accurate iterate yields
  the exact matrix-inverse gradient (matches a `solve_triangular` oracle to ~5e-16)
  — so it is a full drop-in. MVNMLE GPU results are unchanged within the GPU/FP32
  tolerance (MPS end-to-end vs CPU: max |Δmu| ~6e-5, max |ΔSigma| ~5e-4); CUDA/CPU
  paths are unaffected.
