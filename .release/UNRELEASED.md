# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **GPU MVN MLE: closed-form analytical gradient (large MPS speedup; makes wide
  data practical).** `compute_gradient` previously used reverse-mode autodiff,
  which backpropagates through `torch.linalg.cholesky`; on Apple Metal the
  Cholesky backward is pathologically slow (~20s at p=100 over ~43k patterns),
  so large-p MPS fits timed out. Replaced by the closed-form matrix gradient
  (`analytic_gradient` in `_objectives/_batched_cholesky.py`):
  `dF/dSigma_k = n_k Sigma_k^{-1} - Sigma_k^{-1} M_k Sigma_k^{-1}` and
  `dF/dmu_k = -2 n_k Sigma_k^{-1}(ybar_k - mu_k)`, formed from the forward inverse
  covariance and backpropagated only through the cheap gather + `theta -> (mu,
  Sigma)` reconstruction (no autodiff through `cholesky` or the inverse).
  Mathematically identical to autodiff (matches to ~1e-15 in FP64; tested against
  the autodiff reference). On MPS the per-gradient cost drops from ~22s to ~0.97s
  at p=100, and an end-to-end WVS p=100 fit goes from a >30-minute timeout to 193s
  (converged). CUDA and CPU benefit as well. `compute_objective` is unchanged.
