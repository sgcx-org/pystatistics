# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **`mlest(algorithm='direct')` default CPU path is now substantially faster.**
  `backend='cpu'` (and the unspecified default) now route to the PyTorch
  forward-Cholesky FP64 estimator instead of the numpy inverse-Cholesky
  optimizer. On a p=10, n=2000 fit with 15% missingness it completes in ~0.08s
  versus ~100s for the previous default, while matching R's `mvnmle` to ~1e-9.
  The estimates are unchanged (both paths match R); only the speed differs.
  Requires PyTorch (the optional `pystatistics[gpu]` extra).
- **New `backend='cpu-reference'`.** Selects the numpy inverse-Cholesky
  reference optimizer explicitly. It matches R, needs no PyTorch, and is the
  recommended choice when an independent, dependency-free reference is wanted.
  Valid only with `algorithm='direct'`; combining it with `algorithm='em'` or
  `'monotone'` raises `ValueError`.
- **Automatic fallback when PyTorch is absent.** If PyTorch is not installed,
  the default `backend='cpu'` direct path falls back to the numpy reference and
  emits a `UserWarning` explaining how to get the fast path (or how to silence
  the warning by selecting `backend='cpu-reference'`). Results remain correct
  and R-validated.
- `backend='auto'` on machines without CUDA now uses the fast forward-Cholesky
  CPU path (when PyTorch is available) rather than the slower numpy optimizer.
