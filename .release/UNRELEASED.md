# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **BREAKING — mvnmle: the inverse-Cholesky reference optimizer is removed;
  `mlest(..., solver='reference')` now raises `ValidationError`.** A removed
  public option is a breaking API change (CONVENTIONS "Versioning policy": a
  major). Hard removal, no deprecation shim.
  - Deleted: `pystatistics/mvnmle/_objectives/cpu.py` (`CPUObjectiveFP64`),
    `pystatistics/mvnmle/backends/cpu.py` (`CPUMLEBackend`), the data
    preparation they used, and `InverseCholeskyParameterization` in
    `_objectives/parameterizations.py`, with their exports from
    `mvnmle._objectives` and `mvnmle.backends`. These were the parts of the
    package ported from the R package `mvnmle`; no ported code remains in
    `pystatistics/mvnmle/`.
  - `mlest(..., solver='reference')` raises `ValidationError` for every
    `method` and `backend`, with a message pointing to the default CPU path.
    Other `solver` values are unchanged (they are scipy optimizer methods).
  - No result can report `backend_name == 'cpu_bfgs_fp64'` or
    `info['parameterization'] == 'inverse_cholesky'` any more; the direct
    method always reports `'cholesky'`.
  - Migration: drop `solver='reference'`. The default CPU path (PyTorch when
    installed, NumPy otherwise) is the supported double-precision estimator;
    it finds the same maximum (log-likelihood within 2e-5 of the removed
    optimizer on the problems checked, including n=2000, p=10) and is far
    faster (25-37 ms vs 60.8 s on that problem).
  - Agreement with R is still tested, against stored R outputs
    (`tests/mvnmle/references/*_reference.json`) and stored results of the
    removed optimizer (`pre_base_rewrite_fits.json`).

- **mvnmle: NumPy implementation of the forward-Cholesky direct estimator; it
  is what a PyTorch-less install runs.** New
  `pystatistics/mvnmle/_objectives/_batched_cholesky_np.py` (kernel),
  `_objectives/numpy_fp64.py` (`NumpyObjectiveFP64`) and
  `backends/numpy_cholesky.py` (`NumpyCholeskyMLEBackend`, `backend_name`
  `'cpu_cholesky_numpy_fp64'`). Same estimator as the PyTorch FP64 CPU path:
  log-Cholesky parameters `[mu; log diag(L); offdiag(L)]`, per-pattern
  principal submatrices padded and factorised with one batched
  `np.linalg.cholesky` per chunk, closed-form gradient, the shared
  `run_direct_solve` optimizer path with the same defaults (BFGS, per-observation
  `gtol` 1e-5, 100 iterations) and the same `info` keys
  (`parameterization='cholesky'`, `device='cpu'`, `precision='fp64'`).
  - `solvers._fast_cpu_backend()` returns `NumpyCholeskyMLEBackend` when
    `import torch` fails (previously the inverse-Cholesky reference). The
    `UserWarning` text changed accordingly; its `stacklevel` was corrected so
    it points at the caller's `mlest(...)` line instead of inside `solvers.py`.
  - User impact: without PyTorch, `mlest(X)` / `backend='cpu'` / `'auto'` gives
    the same estimates as before to optimizer tolerance, far faster, and
    `result.backend_name` changes from `'cpu_bfgs_fp64'` to
    `'cpu_cholesky_numpy_fp64'`. With PyTorch installed nothing changes.
  - Measured (Apple M5 Pro, numpy 2.4.4, torch 2.11.0, scipy 1.17.1):
    objective identical and gradient within 6e-16 relative of the PyTorch FP64
    objective on seeded problems (p = 5, 10, 30). End-to-end, n=2000, p=10, 15%
    MCAR (237 patterns): NumPy 25 ms, PyTorch FP64 CPU 37 ms; |dloglik| = 0,
    max|dmu| = 6e-12, max|dSigma| = 1e-10, identical iteration counts.
    n=5000, p=30, 10% MCAR (3236 patterns): NumPy 2.2 s, PyTorch 1.7 s.

- **mvnmle: shared data preparation rewritten from the definitions.**
  `pystatistics/mvnmle/_objectives/base.py` (`MLEObjectiveBase`, used by the
  forward-Cholesky objectives, EM and the monotone solver) was replaced.
  - Pattern grouping: new `_objectives/_pattern_grouping.py`
    (`group_rows_by_pattern`). Rows are grouped by exact equality of their
    missingness mask (packed bytes), for any number of variables; there are no
    integer pattern codes. Pattern order: fewest missing variables first, ties
    lexicographic on the missing-indicator vector; rows keep their original
    order within a pattern. `PatternData` gained `row_indices` and lost
    `pattern_start` / `pattern_end`.
  - Starting values: new `_objectives/_start_values.py`. Available-case means
    and pairwise-complete covariances (ML divisor) — numerically the same
    matrix as before. The positive-definite repair changed: instead of adding
    an absolute ridge (>= 1e-4) to the diagonal when the smallest covariance
    eigenvalue is below 1e-6, the correlation matrix is shrunk towards the
    identity just enough for its smallest eigenvalue to reach 1e-4
    (`MIN_CORRELATION_EIGENVALUE`); variances are untouched, so the rule is
    unit-invariant. When the available-case covariance is already positive
    definite the start is unchanged. The repair only affects where iteration
    starts. `MLEObjectiveBase.compute_gradient` no longer has a
    finite-difference default.
  - `patterns.py::identify_missingness_patterns` (behind `analyze_patterns`
    and `little_mcar_test`) is built on the same grouping. Its output is
    unchanged — same patterns, order, ids, data and percentages, and
    bit-identical MCAR statistic and p-value on `apple`, `missvals`, seeded
    problems and a 70-variable case.
  - User impact: the objective function is unchanged (value at a fixed
    parameter vector within 3e-14 relative of the previous value), so converged
    estimates are unchanged to optimizer tolerance. When the available-case
    covariance is positive definite (the usual case), iteration counts are
    identical on every path. When it is not, the start differs and iteration
    counts change: `missvals` forward-Cholesky FP64 570 -> 152 iterations
    (NumPy; PyTorch 563 -> 152), EM 78 -> 85; a seeded n=500, p=5 problem with
    an indefinite available-case covariance 100 (not converged) -> 52
    (converged). FP32/MPS fits that stop at the loose FP32 tolerance land at a
    different point in those cases (`missvals` log-likelihood -87.94 -> -90.12;
    FP64 optimum -86.98).

- **mvnmle: `datasets` docstring.** `apple` and `missvals` are described as the
  example datasets also distributed with the R package `mvnmle` (same values);
  the data are unchanged.
