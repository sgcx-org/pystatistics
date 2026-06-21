# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- Fixed GPU MICE `polr` (ordinal/proportional-odds) imputation collapsing every
  ordered column onto a single category under (quasi-)complete separation. On
  imbalanced real-survey ordinals (a sparse extreme category ordered by a
  continuous predictor), the GPU `batched_polr_newton` fit in
  `mice/backends/_gpu_polr.py` diverged — thresholds ran to `|alpha| ~ 1e6` and
  every missing row was assigned one category, while the CPU `polr` stayed
  finite. Two complementary fixes: (1) added the same scale-aware ridge on the
  proportional-odds slopes that the CPU path uses (`0.5 * lambda * ||beta||^2`,
  thresholds unpenalised, `lambda` scaled by the mean predictor second moment),
  and (2) globalised the batched Newton with a per-chain backtracking line
  search that halves each step until the penalised negative log-likelihood
  decreases. The line search is the load-bearing part: it bounds the
  *unpenalised thresholds*, which the slope ridge alone cannot. The GPU fit now
  reproduces the CPU penalised maximum-likelihood fit (matching thresholds and
  slopes) and recovers the sparse extreme category instead of collapsing.
- Added a GPU `polr` separation regression test
  (`tests/mice/test_gpu_polr_separation.py`): on a deterministically separated
  ordinal it asserts the fitted thresholds and slopes stay finite and bounded,
  match the CPU ridged fit, and that the end-to-end imputed-category proportions
  track the CPU `polr`. Covers the FP64 device-agnostic core and the on-device
  (CUDA/MPS) FP32 path. The prior suite only exercised balanced/synthetic data,
  which never separates, so it missed this failure mode.
