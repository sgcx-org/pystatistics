# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Changes

- **New GLM family: `GammaFamily`** — Gamma regression for positive continuous data
  with variance proportional to mean². Supports inverse (default), log, and identity
  links. Validated against R's `stats::Gamma()`. Dispersion (1/shape) estimated from
  Pearson chi-squared / df_residual. Files: `regression/families.py`.

- **New GLM family: `NegativeBinomial`** — Negative binomial regression for
  overdispersed count data with V(μ) = μ + μ²/θ. Default link: log. Two usage modes:
  (1) `NegativeBinomial(theta=5)` for fixed θ, fits via standard IRLS;
  (2) `fit(X, y, family='negative.binomial')` for automatic θ estimation via
  alternating profile likelihood (matches R's `MASS::glm.nb()` algorithm).
  Files: `regression/families.py`, `regression/_nb_theta.py`, `regression/solvers.py`.

- **53 new tests** covering Gamma and NB families: unit tests for variance functions,
  link defaults, family resolution; integration tests for coefficient recovery,
  convergence, AIC/deviance computation; theta estimation accuracy; edge cases
  (constant y, all-zero counts, single predictor).
  File: `tests/regression/test_gamma_nb.py`.

- **New module: `ordinal`** — Proportional odds model (cumulative link model) matching
  R's `MASS::polr()`. Supports logistic (proportional odds), probit, and complementary
  log-log links. Entry point: `polr(y, X, method='logistic')`. Threshold ordering
  enforced via unconstrained parameterization (incremental exp-transform). Optimization
  via L-BFGS-B with analytical gradient. Solution provides `coef`, `thresholds`,
  `standard_errors`, `z_values`, `p_values`, `summary()` with R-style formatting.
  Files: `ordinal/__init__.py`, `ordinal/_common.py`, `ordinal/_likelihood.py`,
  `ordinal/_solver.py`, `ordinal/solution.py`. 41 tests in `tests/ordinal/test_ordinal.py`.

- **New module: `multinomial`** — Multinomial logit (softmax) regression matching R's
  `nnet::multinom()`. Estimates (J-1) × p coefficient matrix with the last class as
  reference. Entry point: `multinom(y, X)`. Uses log-sum-exp trick for numerical
  stability, L-BFGS-B with analytical gradient. Solution provides `coefficient_matrix`,
  `coef` (nested dict), `standard_errors`, `predicted_class`, `pseudo_r_squared`,
  `summary()` with per-class coefficient tables. Files: `multinomial/__init__.py`,
  `multinomial/_common.py`, `multinomial/_likelihood.py`, `multinomial/_solver.py`,
  `multinomial/solution.py`. 41 tests in `tests/multinomial/test_multinom.py`.

- **New module: `multivariate`** — PCA and maximum likelihood factor analysis.

  - `pca(X, center=True, scale=False, n_components=None)` — Principal component analysis
    via SVD matching R's `stats::prcomp()`. Supports centering, scaling (correlation-based
    PCA), component truncation, and enforces R's sign convention (largest absolute loading
    positive). Returns `PCAResult` with sdev, rotation (loadings), scores, explained
    variance ratios, and R-style `summary()`.

  - `factor_analysis(X, n_factors, rotation='varimax')` — Maximum likelihood factor
    analysis matching R's `stats::factanal()`. L-BFGS-B optimization over log-uniquenesses.
    Chi-squared goodness-of-fit test with Bartlett correction. Supports varimax (orthogonal,
    Kaiser-normalized) and promax (oblique, power target) rotations.

  Files: `multivariate/__init__.py`, `multivariate/_common.py`, `multivariate/_pca.py`,
  `multivariate/_factor.py`, `multivariate/_rotation.py`.
  46 tests in `tests/multivariate/test_multivariate.py`.

- **New module: `timeseries` (Phase 7A)** — Autocorrelation, differencing, and
  stationarity tests — the foundation for ARIMA and ETS (coming in later phases).

  - `acf(x)` — Autocorrelation function with biased (1/n) normalization matching
    R's `stats::acf()`. Default max_lag = `floor(10*log10(n))`. Bartlett CI bands.

  - `pacf(x)` — Partial autocorrelation via Durbin-Levinson recursion matching
    R's `stats::pacf()`. Lags start at 1 (no lag 0), matching R convention.

  - `diff(x, differences, lag)` — Time series differencing matching R's `base::diff()`.
    Supports seasonal differencing (lag > 1) and repeated differencing.

  - `ndiffs(x, test='adf')` — Automatic differencing order estimation matching
    R's `forecast::ndiffs()`. Supports ADF and KPSS tests.

  - `adf_test(x, regression='c')` — Augmented Dickey-Fuller unit root test matching
    R's `tseries::adf.test()`. Regression types: 'nc', 'c', 'ct'. P-values from
    MacKinnon (1996) finite-sample response surface with interpolation.

  - `kpss_test(x, regression='c')` — KPSS stationarity test matching R's
    `tseries::kpss.test()`. Bartlett kernel long-run variance. P-values interpolated
    from KPSS critical value tables, clamped to [0.01, 0.10].

  Files: `timeseries/_common.py`, `timeseries/_acf.py`,
  `timeseries/_differencing.py`, `timeseries/_stationarity.py`.
  78 tests in `tests/timeseries/test_acf_stationarity.py`.

- **`timeseries` module (Phase 7B)** — Exponential smoothing (ETS) models.

  - `ets(y, model='ANN', period=1)` — Fit ETS state space models matching R's
    `forecast::ets()`. Supports 12 model types: Error (A/M) × Trend (N/A/Ad) ×
    Season (N/A/M). Parameter estimation via L-BFGS-B on logit-transformed
    smoothing parameters with box constraints. Supports fixing individual
    parameters. Returns `ETSResult` with fitted values, residuals, states,
    AIC/AICc/BIC, MSE/MAE, and R-style `summary()`.

  - `forecast_ets(fitted, h=10, levels=[80, 95])` — Point forecasts and
    prediction intervals from fitted ETS models. Analytical PI formulas for
    additive non-seasonal models, σ√h fallback for others. Handles damped
    trend phi-sum correctly.

  Files: `timeseries/_ets_models.py`, `timeseries/_ets_fit.py`,
  `timeseries/_ets_forecast.py`.
  72 tests in `tests/timeseries/test_ets.py`.

- **`timeseries` module (Phase 7C)** — ARIMA and seasonal ARIMA models.

  - `arima(y, order=(p,d,q), seasonal=(P,D,Q,m))` — Fit ARIMA and SARIMA models
    matching R's `stats::arima()`. Three methods: `'CSS'` (conditional sum of squares,
    fast), `'ML'` (exact MLE via innovations algorithm), `'CSS-ML'` (CSS initialization
    then ML refinement, default). Seasonal models multiply out AR/MA polynomials.
    Yule-Walker starting values for AR. Returns `ARIMAResult` with coefficients,
    residuals, fitted values, sigma², vcov, AIC/AICc/BIC, R-style `summary()`.

  - `forecast_arima(fitted, y_original, h=10)` — Point forecasts and prediction
    intervals on the original (un-differenced) scale. Uses MA(∞) psi weights for
    forecast-error variance. Handles seasonal and non-seasonal un-differencing.

  - `auto_arima(y, max_p=5, max_q=5, ic='aicc')` — Automatic ARIMA order selection
    matching R's `forecast::auto.arima()`. Stepwise search (Hyndman-Khandakar 2008,
    default) or exhaustive grid search. Determines d via ADF test, seasonal D via
    variance ratio heuristic.

  Files: `timeseries/_arima_likelihood.py`, `timeseries/_arima_fit.py`,
  `timeseries/_arima_forecast.py`, `timeseries/_arima_order.py`.
  59 tests (46 fast + 13 slow) in `tests/timeseries/test_arima.py`.

- **`timeseries` module (Phase 7D)** — Time series decomposition.

  - `decompose(x, period, type='additive')` — Classical decomposition matching
    R's `stats::decompose()`. Centered moving average for trend (2×m MA for even
    periods), seasonal averaging with zero-sum/unit-mean centering. Supports
    additive and multiplicative types.

  - `stl(x, period)` — STL decomposition (Cleveland et al., 1990) matching R's
    `stats::stl()`. LOESS smoothing with tricube weights for seasonal and trend
    extraction. Configurable seasonal/trend windows, inner/outer iterations,
    and robustness weights.

  File: `timeseries/_decomposition.py`.
  34 tests in `tests/timeseries/test_decomposition.py`.

- **Fixed 5 stale GPU tests** — Tests expecting silent CPU fallback updated to expect
  `NotImplementedError`, matching the v1.2.1 "fail loud" behavior change.
  `test_glm_gpu.py` skip condition narrowed to require CUDA (MPS doesn't support lstsq).
  Files: `tests/hypothesis/test_gpu.py`, `tests/regression/test_glm_gpu.py`.
