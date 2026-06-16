# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **GPU MVN MLE: batched per-pattern objective (major speedup on many-pattern
  data).** The direct GPU objective (`_objectives/gpu_fp32.py`, `gpu_fp64.py`)
  previously evaluated the missing-data log-likelihood with a Python loop over
  missingness patterns — one tiny `cholesky` + solves per pattern — which
  launched thousands of small GPU kernels per optimizer step and dominated
  runtime once the pattern count grew (survey-scale data routinely has thousands
  of distinct patterns). The per-pattern loop is replaced by a single batched
  evaluation (new `_objectives/_batched_cholesky.py`): one batched `cholesky` and
  batched triangular solves over all patterns at once, using precomputed
  per-pattern sufficient statistics. Math unchanged; validated against the looped
  reference for value and autodiff gradient in FP32 and FP64.
- **FP32 numerical stability of the GPU objective.** Per-pattern second moments
  are now formed as `C_k + n_k·δδᵀ` (centered scatter plus a small mean-shift
  term) instead of `Σyyᵀ − n·μμᵀ`, avoiding catastrophic cancellation in FP32.
