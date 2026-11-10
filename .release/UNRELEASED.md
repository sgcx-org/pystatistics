# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **mvnmle: NumPy implementation of the forward-Cholesky direct estimator; it
  is now what a PyTorch-less install runs.** New
  `pystatistics/mvnmle/_objectives/_batched_cholesky_np.py` (kernel),
  `_objectives/numpy_fp64.py` (`NumpyObjectiveFP64`) and
  `backends/numpy_cholesky.py` (`NumpyCholeskyMLEBackend`, `backend_name`
  `'cpu_cholesky_numpy_fp64'`). Same estimator as the PyTorch FP64 CPU path:
  log-Cholesky parameters `[mu; log diag(L); offdiag(L)]`, per-pattern
  principal submatrices padded and factorised with one batched
  `np.linalg.cholesky` per chunk, closed-form gradient, the shared
  `run_direct_solve` optimizer path with the same defaults (BFGS, per-observation
  `gtol` 1e-5, 100 iterations) and the same `info` keys
  (`parameterization='cholesky'`, `device='cpu'`, `precision='fp64'`). It was
  written from the PyTorch forward-Cholesky code and the mathematics; it does
  not use the inverse-Cholesky reference objective.
  - `solvers._fast_cpu_backend()` now returns `NumpyCholeskyMLEBackend` when
    `import torch` fails (previously `CPUMLEBackend`, the inverse-Cholesky
    reference). The `UserWarning` text changed accordingly and no longer
    suggests `solver='reference'`; its `stacklevel` was corrected (4 -> 5) so it
    points at the caller's `mlest(...)` line instead of inside `solvers.py`.
  - `solver='reference'` and `CPUMLEBackend` are unchanged and still work.
  - User impact: without PyTorch, `mlest(X)` / `backend='cpu'` / `'auto'` gives
    the same estimates as before to optimizer tolerance, far faster, and
    `result.backend_name` changes from `'cpu_bfgs_fp64'` to
    `'cpu_cholesky_numpy_fp64'` (`info['parameterization']` from
    `'inverse_cholesky'` to `'cholesky'`). With PyTorch installed nothing
    changes.
  - Measured (Apple M5 Pro, numpy 2.4.4, torch 2.11.0, scipy 1.17.1):
    objective identical and gradient within 6e-16 relative of the PyTorch FP64
    objective on seeded problems (p = 5, 10, 30). End-to-end, n=2000, p=10, 15%
    MCAR (237 patterns): NumPy 25 ms, PyTorch FP64 CPU 37 ms, inverse-Cholesky
    reference 60.8 s; NumPy vs PyTorch |dloglik| = 0, max|dmu| = 6e-12,
    max|dSigma| = 1e-10 (identical iteration counts); NumPy vs reference
    |dloglik| = 1e-5. n=5000, p=30, 10% MCAR (3236 patterns): NumPy 2.2 s,
    PyTorch 1.7 s.
