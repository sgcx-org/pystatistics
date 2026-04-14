# Changelog

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
