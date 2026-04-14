# Changelog

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
