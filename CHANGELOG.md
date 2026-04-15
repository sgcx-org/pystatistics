# Changelog

## 1.3.0 (unreleased)

### GPU Backends — Permutation Test and Bootstrap

Linux/NVIDIA validation on RTX 5070 Ti exposed that the GPU backends for
Monte Carlo methods were stubs raising `NotImplementedError`. This release
implements working GPU acceleration for the two most common workloads.

#### Permutation Test GPU (`permutation_test(backend='gpu')`)

- **Implemented** vectorized GPU backend for mean-difference statistic.
- Generates random permutations directly on GPU via `torch.rand` + `argsort`
  (random-key sorting), avoiding the CPU bottleneck of sequential
  `rng.permutation()` calls.
- Chunked processing keeps VRAM usage under ~1 GB regardless of problem size.
- Auto-detects whether the user's statistic is mean-difference; falls back
  transparently to CPU for non-vectorizable statistics.
- **Benchmarks** (RTX 5070 Ti, R=50,000 permutations):
  - n=1,000: 5x speedup (CPU 1.4s, GPU 0.28s)
  - n=10,000: 23x speedup (CPU 6.7s, GPU 0.29s)
  - n=50,000: 23x speedup (CPU 33s, GPU 1.4s)
- `backend='auto'` now selects GPU when CUDA is available (was CPU-only before).

#### Bootstrap GPU (`boot(backend='gpu')`)

- **Implemented** vectorized GPU backend for simple mean statistic on 1-D data.
- Generates bootstrap index sets via `torch.randint` on GPU; computes all R
  means in a single vectorized pass.
- Auto-detects if the user's statistic computes a simple mean; falls back to
  CPU for multivariate statistics, balanced/parametric sim, or non-mean functions.
- `backend='auto'` now selects GPU when CUDA is available and statistic is
  vectorizable (was CPU-only before).

#### Notes

- GPU RNG (PyTorch) differs from CPU RNG (NumPy). P-values and bootstrap
  replicates are statistically equivalent but not bitwise identical across
  backends. Observed statistics (t0, observed_stat) remain identical since
  they are computed on the original data.
- Tests updated to reflect GPU auto-selection behavior. Backend name assertions
  now check for `'bootstrap'` / `'permutation'` rather than hardcoding `'cpu'`.

## 1.2.1

### Code Quality Audit

Full codebase audit and refactor to enforce strict adherence to the
project's seven coding rules.

### Breaking Changes (by design)

- **Silent model switches are now errors.** Functions that previously fell back
  silently to ridge regularization, LSTSQ, or CPU backends now raise
  `NumericalError` or `RuntimeError` with descriptive messages suggesting
  alternatives. This affects:
  - `mvnmle.mcar_test.regularized_inverse()` — raises on ill-conditioned matrices
  - `mvnmle.mlest()` — raises if EM encounters non-PD covariance
  - `mvnmle` parameter extraction — raises instead of returning identity covariance
  - `regression.fit(backend='gpu')` — raises on Cholesky failure (use `force=True`
    to proceed with LSTSQ, or `backend='cpu'` for QR)
  - `mixed.lmm()` / `mixed.glmm()` — raises on singular random effects covariance
- **`backend='gpu'` now errors when GPU is unavailable.** Previously fell back
  to CPU silently. `backend='auto'` still falls back silently (it means
  "best available").
- **GPU bootstrap/permutation raises `NotImplementedError`.** These were silently
  running on CPU while reporting a GPU backend name. Now they honestly report
  that GPU acceleration is not yet implemented for these operations.

### New Features

- **Reproducible Monte Carlo hypothesis tests**: `chisq_test()` and
  `fisher_test()` now accept `seed` parameter for deterministic results
  when `simulate_p_value=True`

### Module Structure

- Split `regression/solution.py` into `_linear.py`, `_glm.py`, and
  `_formatting.py` (backward-compatible re-export shim maintained)
- Split `hypothesis/design.py` factory methods into `_design_factories.py`
  (classmethods still work via thin wrappers)
- All files now under 500 code lines

### Code Quality

- Added `# NUMERICAL GUARD:` comments to ~30 numerical stability operations
  (clipping, clamping, floors) documenting why each exists
- Removed dead code (`signaltonoise` try/except in Wilcoxon test)
- Added per-module compliance tests:
  - `tests/test_code_quality.py` — LOC limit enforcement
  - `tests/regression/test_module_split.py` — split integrity + named coefficients
  - `tests/hypothesis/test_design_split.py` — factory split + seed reproducibility
  - `tests/mvnmle/test_no_silent_fallback.py` — hard stop verification

## 1.1.0

### New Features

- **Named coefficients**: `fit()`, `coxph()`, and `discrete_time()` accept a
  `names=` parameter for labeled output matching R's style
- **`result.coef` dict property**: Access coefficients by variable name
  (`result.coef["albumin"]`) on `LinearSolution`, `GLMSolution`,
  `CoxSolution`, and `DiscreteTimeSolution`
- **`result.hr` dict property**: Access hazard ratios by name on `CoxSolution`
  and `DiscreteTimeSolution`
- **Intercept auto-detection**: When `names` has one fewer element than columns
  in X, `"(Intercept)"` is prepended automatically

### Summary Output Improvements

- **OLS**: Added residual quantiles (Min, 1Q, Median, 3Q, Max) and overall
  F-statistic with p-value, matching R's `summary(lm())`
- **Cox PH**: Added hazard ratio confidence intervals table (`exp(coef)`,
  `exp(-coef)`, `lower .95`, `upper .95`), matching R's `summary(coxph())`

### Bug Fixes

- Fixed Kaplan-Meier `summary()` printing literal `{ci_pct}` instead of the
  actual confidence level percentage (e.g., `95`)

### Real-Data Validation

- Added PBC clinical trial integration test suite (`tests/test_pbc_analysis.py`)
  with 22 end-to-end analyses on the Mayo Clinic PBC dataset
- Added R cross-validation test (`tests/test_pbc_vs_r.py`) confirming all 15
  comparable analyses match R to `rtol=1e-10`

## 1.0.2

- Initial stable release
- All 8 modules complete: regression, descriptive, hypothesis, montecarlo,
  survival, anova, mixed, mvnmle
- CPU backends validated against R to rtol=1e-10
- GPU backends validated against CPU per documented tolerance tiers
