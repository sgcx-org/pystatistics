# Changelog

## 1.6.2

### Re-release of 1.6.1 fixes

**Why 1.6.2 exists:** the 1.6.1 release commit (`Release v1.6.1`) was
created after the source fixes were staged but before they were actually
committed to the branch. The CI `publish.yml` workflow then built the
PyPI package from the `v1.6.1` tag, which pointed at the version-bump
commit only — **the compiled wheel lacked the ARIMA / Gamma / var /
scipy fixes it was supposed to ship**. PyPI does not allow re-uploading
the same version number, so a patch version was the only clean path.
Users who installed `pystatistics==1.6.1` should upgrade to `1.6.2`.

The release script flow was adjusted in this cycle to ensure the release
commit carries all staged fixes; see Historical Notes in
`.release/CHECKLIST.md`.

### Fixed — content is the same as the 1.6.1 changelog entry

All fixes listed under 1.6.1 in `CHANGELOG.md` are now actually present
in the shipped wheel:

- **`timeseries.arima(method='CSS-ML')` silent fallback removed.** Raises
  `ConvergenceError` instead of silently returning CSS estimates labeled
  as CSS-ML.
- **`timeseries.arima` zero-parameter case.** Closed-form MLE for
  ARIMA(0,d,0); bypasses scipy's `nit=0 "ABNORMAL"` degenerate path.
- **`regression.GammaFamily.log_likelihood`** on non-positive dispersion
  returns explicit NaN instead of emitting `RuntimeWarning` and silently
  returning NaN from `np.log(negative)`.
- **`descriptive.var(n=1)`** short-circuits to NaN without triggering
  numpy's `Degrees of freedom <= 0` warning.
- **scipy 1.18 forward-compat**: removed deprecated `disp` option from
  `scipy.optimize.minimize` in mvnmle CPU and GPU backends.
- **mvnmle test suite** updated: `TestMissvalsDataset` uses EM
  explicitly; `TestDirectNonConvergence` codifies the fail-loud contract
  on the missvals pathological dataset.


## 1.6.1

### Fixed — Coding Bible Rule 1 violations (silent failures / degraded paths)

- **`timeseries.arima(method='CSS-ML')` silent fallback removed.** When ML
  refinement failed, the previous code emitted a `UserWarning` and silently
  returned CSS estimates while labeling the result "CSS" — despite the
  user having requested "CSS-ML". Now raises `ConvergenceError` with
  actionable guidance (use `method='CSS'`, adjust `tol`/`max_iter`).
  `pystatistics/timeseries/_arima_fit.py`.

- **`timeseries.arima` zero-parameter case.** For ARIMA(0,d,0) (and any
  configuration with p_eff = q_eff = 0), the code was calling scipy's
  `minimize` with a near-MLE start, which causes L-BFGS-B to exit with
  `nit=0, "ABNORMAL"` and trip the silent fallback path. The MLE is
  closed-form here (sample mean of the differenced series, or a constant
  if no mean) — no optimization is needed. Added an explicit closed-form
  branch that bypasses scipy. `pystatistics/timeseries/_arima_fit.py`.

- **`regression.GammaFamily.log_likelihood` on non-positive dispersion.**
  When the Gamma GLM fit perfectly (e.g. constant y), dispersion = dev/df
  came out as ≈ 0 or slightly negative, causing `np.log(rate)` to emit a
  `RuntimeWarning: invalid value encountered in log` and silently return
  NaN. Now validates dispersion > 0 explicitly and returns `nan` without
  triggering numpy's warning. `pystatistics/regression/families.py`.

- **`descriptive.var` of single-observation input.** For n=1 the sample
  variance is undefined. numpy correctly returns NaN but emits
  `RuntimeWarning: Degrees of freedom <= 0 for slice`. Added a short-circuit
  that returns NaN explicitly (matching R `var()`) without triggering the
  internal numpy warning. `pystatistics/descriptive/backends/cpu.py`.

### Fixed — scipy 1.18 forward-compatibility

- **Removed deprecated `disp` option from `scipy.optimize.minimize`** in
  mvnmle CPU and GPU backends. scipy 1.18 emits `DeprecationWarning` for
  `disp`/`iprint` on L-BFGS-B; the option is removed entirely (we do not
  print optimizer progress, so the default is fine).
  `pystatistics/mvnmle/backends/{cpu,gpu}.py`.

### Fixed — mvnmle test suite reflects code contract

- **`TestMissvalsDataset` now uses EM explicitly.** The `missvals` dataset
  (n=13, p=5, high missingness) is pathological for L-BFGS-B direct
  optimization: the likelihood surface is near-flat at this sample size,
  and direct does not converge. R's `mvnmle` uses an EM-equivalent
  algorithm — so the R-comparison tests must also use EM in pystatistics.
  EM converges to machine precision on this dataset and matches R exactly.

- **`TestEMMatchesDirect::test_missvals_*` removed.** These tests asserted
  that EM and direct estimates agree on missvals, but direct genuinely
  cannot converge on that dataset. Replaced with a single test that
  verifies EM matches R on missvals — the contract that actually holds.

- **Added `TestDirectNonConvergence`.** Codifies the explicit fail-loud
  contract: on pathological datasets like missvals, the direct optimizer
  must return `converged=False` rather than silently returning a
  meaningless answer. A future change which "fixes" direct non-convergence
  (e.g. by switching optimizers) must update this test deliberately.

### Test impact

- `pytest tests/` passes clean under
  `-W error::UserWarning -W error::RuntimeWarning -W error::DeprecationWarning`
  (was 20 warning-induced failures, now 0). Normal run: 2,301 passing,
  0 failing, 19 skipped.


## 1.6.0

### Summary

Major expansion of classical statistics coverage. Five new top-level modules
(`ordinal`, `multinomial`, `multivariate`, `timeseries`, `gam`), two new GLM
families (`Gamma`, `NegativeBinomial`), and reinforced "fail loud" numerical
policy. Adds ~650 new tests across all modules. Estimated coverage of standard
applied frequentist statistics goes from ~85% to ~95%.

### Added

#### GLM Families: Gamma and Negative Binomial

- **`GammaFamily`** — Gamma regression for positive continuous data (cost data,
  survival times, insurance claims). V(μ) = μ². Supports inverse (default), log,
  and identity links. Dispersion (1/shape) estimated from Pearson chi-squared /
  df_residual. Validates against R `stats::Gamma()`.

- **`NegativeBinomial`** — Negative binomial regression for overdispersed count
  data. V(μ) = μ + μ²/θ. Default link: log. Two usage modes:
  (1) `NegativeBinomial(theta=5)` for fixed θ via standard IRLS;
  (2) `fit(X, y, family='negative.binomial')` for automatic θ estimation via
  alternating profile likelihood, matching R `MASS::glm.nb()`.

#### Ordinal Regression Module

- **`polr(y, X, method='logistic')`** — Proportional odds (cumulative link)
  model matching R `MASS::polr()`. Supports logistic, probit, and complementary
  log-log links. Threshold ordering enforced via unconstrained parameterization
  (incremental exp-transform). L-BFGS-B with analytical gradient.

#### Multinomial Regression Module

- **`multinom(y, X)`** — Multinomial logit (softmax) regression matching R
  `nnet::multinom()`. Estimates (J-1) × p coefficient matrix with last class as
  reference. Log-sum-exp trick for numerical stability, L-BFGS-B with analytical
  gradient.

#### Multivariate Analysis Module

- **`pca(X, center=True, scale=False)`** — PCA via SVD matching R
  `stats::prcomp()`. Enforces R sign convention.

- **`factor_analysis(X, n_factors, rotation='varimax')`** — Maximum likelihood
  factor analysis matching R `stats::factanal()`. Varimax and promax rotations.

#### Time Series Module (Complete)

Full time series analysis framework. Validates against R packages `stats`,
`tseries`, and `forecast`.

- **ACF / PACF** — `acf(x)` and `pacf(x)` matching R `stats::acf()` / `stats::pacf()`.
- **Stationarity tests** — `adf_test(x)` and `kpss_test(x)` matching R
  `tseries::adf.test()` / `tseries::kpss.test()`.
- **Differencing** — `diff(x)` and `ndiffs(x)` matching R `base::diff()` /
  `forecast::ndiffs()`.
- **ETS** — `ets(y, model='ANN')` fitting 12 ETS model types matching R
  `forecast::ets()`. `forecast_ets()` with prediction intervals.
- **ARIMA / SARIMA** — `arima(y, order=(p,d,q), seasonal=(P,D,Q,m))` with CSS,
  ML, and CSS-ML methods matching R `stats::arima()`. `forecast_arima()` with
  MA(∞) psi weights. `auto_arima(y)` with stepwise or grid search matching R
  `forecast::auto.arima()`.
- **Decomposition** — `decompose(x, period)` and `stl(x, period)` matching R
  `stats::decompose()` / `stats::stl()`.

#### Generalized Additive Models Module

- **`gam(y, smooths=[s('x1')], smooth_data={...})`** — Penalized regression
  spline GAMs via P-IRLS matching R `mgcv::gam()`. Cubic regression splines and
  thin plate splines. GCV and REML smoothing parameter selection.
- **`s(var_name, k=10, bs='cr')`** — Smooth term specification matching `mgcv::s()`.

### Changed

- **GPU behavior enforces "fail loud" policy** — Explicit `backend='gpu'` calls
  on unsupported operations now raise `NotImplementedError` instead of silently
  falling back to CPU. Users who want automatic fallback should use
  `backend='auto'`.
- **GPU GLM tests require CUDA** — Skip condition narrowed from "any GPU" to
  "CUDA available" (MPS does not support `torch.linalg.lstsq`).

### Fixed

- **5 stale GPU hypothesis tests** — Tests expecting silent CPU fallback updated
  to expect `NotImplementedError`, matching v1.2.1 "fail loud" behavior.

### Tests

~650 new tests. Total: 2,275 fast + 13 slow = 2,288.

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
