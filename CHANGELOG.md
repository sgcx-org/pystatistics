# Changelog

## 1.3.0

### Summary

Linux/NVIDIA validation on RTX 5070 Ti revealed that GPU backends for Monte
Carlo methods were stubs raising `NotImplementedError`. This release implements
working GPU acceleration for permutation tests and bootstrap resampling.

### Added

- Vectorized GPU backend for `permutation_test(backend='gpu')` using
  mean-difference statistic. Generates random permutations directly on GPU via
  `torch.rand` + `argsort` (random-key sorting), avoiding the CPU bottleneck of
  sequential `rng.permutation()` calls.
- Vectorized GPU backend for `boot(backend='gpu')` on simple mean statistic
  with 1-D data. Generates bootstrap index sets via `torch.randint` on GPU and
  computes all R means in a single vectorized pass.
- Chunked processing for permutation test GPU backend keeps VRAM usage under
  ~1 GB regardless of problem size.
- Auto-detection for both backends: permutation test detects mean-difference
  statistic, bootstrap detects simple mean. Non-vectorizable statistics fall
  back transparently to CPU.

### Changed

- `backend='auto'` now selects GPU when CUDA is available for both
  `permutation_test()` and `boot()` (was CPU-only before). Bootstrap
  auto-selection additionally requires the statistic to be vectorizable.
- GPU RNG (PyTorch) differs from CPU RNG (NumPy). P-values and bootstrap
  replicates are statistically equivalent but not bitwise identical across
  backends. Observed statistics (`t0`, `observed_stat`) remain identical since
  they are computed on the original data.

### Performance

- Permutation test GPU benchmarks (RTX 5070 Ti, R=50,000 permutations):
  - n=1,000: 5x speedup (CPU 1.4s, GPU 0.28s)
  - n=10,000: 23x speedup (CPU 6.7s, GPU 0.29s)
  - n=50,000: 23x speedup (CPU 33s, GPU 1.4s)

### Tests

- Updated backend name assertions to check for `'bootstrap'` / `'permutation'`
  rather than hardcoding `'cpu'`, reflecting GPU auto-selection behavior.

## 1.2.1

### Summary

Full codebase audit and refactor to enforce strict adherence to the project's
seven coding rules. Silent model switches are now hard errors, module files are
split to stay under 500 lines, and numerical guard comments document every
stability operation.

### Added

- `seed` parameter for `chisq_test()` and `fisher_test()` enabling reproducible
  Monte Carlo hypothesis tests when `simulate_p_value=True`.
- `# NUMERICAL GUARD:` comments on ~30 numerical stability operations (clipping,
  clamping, floors) documenting why each exists.

### Changed

- Split `regression/solution.py` into `_linear.py`, `_glm.py`, and
  `_formatting.py` (backward-compatible re-export shim maintained).
- Split `hypothesis/design.py` factory methods into `_design_factories.py`
  (classmethods still work via thin wrappers).
- All files now under 500 code lines.
- Removed dead code (`signaltonoise` try/except in Wilcoxon test).

### Breaking

- Silent model switches are now errors. Functions that previously fell back
  silently to ridge regularization, LSTSQ, or CPU backends now raise
  `NumericalError` or `RuntimeError` with descriptive messages suggesting
  alternatives. Affected call sites:
  - `mvnmle.mcar_test.regularized_inverse()` -- raises on ill-conditioned matrices
  - `mvnmle.mlest()` -- raises if EM encounters non-PD covariance
  - `mvnmle` parameter extraction -- raises instead of returning identity covariance
  - `regression.fit(backend='gpu')` -- raises on Cholesky failure (use `force=True`
    to proceed with LSTSQ, or `backend='cpu'` for QR)
  - `mixed.lmm()` / `mixed.glmm()` -- raises on singular random effects covariance
- `backend='gpu'` now errors when GPU is unavailable. Previously fell back to
  CPU silently. `backend='auto'` still falls back silently (it means "best
  available").
- GPU bootstrap/permutation raises `NotImplementedError`. These were silently
  running on CPU while reporting a GPU backend name. Now they honestly report
  that GPU acceleration is not yet implemented for these operations.

### Tests

- `tests/test_code_quality.py` -- LOC limit enforcement.
- `tests/regression/test_module_split.py` -- split integrity + named coefficients.
- `tests/hypothesis/test_design_split.py` -- factory split + seed reproducibility.
- `tests/mvnmle/test_no_silent_fallback.py` -- hard stop verification.

## 1.1.0

### Summary

Named coefficients bring R-style labeled output to regression and survival
models. Summary output for OLS and Cox PH now matches R's formatting, and a
22-analysis PBC clinical trial test suite validates end-to-end correctness.

### Added

- `names=` parameter for `fit()`, `coxph()`, and `discrete_time()` enabling
  labeled output matching R's style.
- `result.coef` dict property for accessing coefficients by variable name
  (e.g., `result.coef["albumin"]`) on `LinearSolution`, `GLMSolution`,
  `CoxSolution`, and `DiscreteTimeSolution`.
- `result.hr` dict property for accessing hazard ratios by name on
  `CoxSolution` and `DiscreteTimeSolution`.
- Intercept auto-detection: when `names` has one fewer element than columns in
  X, `"(Intercept)"` is prepended automatically.
- OLS summary now includes residual quantiles (Min, 1Q, Median, 3Q, Max) and
  overall F-statistic with p-value, matching R's `summary(lm())`.
- Cox PH summary now includes hazard ratio confidence intervals table
  (`exp(coef)`, `exp(-coef)`, `lower .95`, `upper .95`), matching R's
  `summary(coxph())`.

### Fixed

- Kaplan-Meier `summary()` printing literal `{ci_pct}` instead of the actual
  confidence level percentage (e.g., `95`).

### Tests

- PBC clinical trial integration test suite (`tests/test_pbc_analysis.py`) with
  22 end-to-end analyses on the Mayo Clinic PBC dataset.
- R cross-validation test (`tests/test_pbc_vs_r.py`) confirming all 15
  comparable analyses match R to `rtol=1e-10`.

## 1.0.2

### Summary

Initial stable release with all eight statistical modules complete and validated
against R.

### Added

- All 8 modules: regression, descriptive, hypothesis, montecarlo, survival,
  anova, mixed, mvnmle.
- CPU backends validated against R to `rtol=1e-10`.
- GPU backends validated against CPU per documented tolerance tiers.
