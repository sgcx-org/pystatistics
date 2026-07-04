# PyStatistics

GPU-accelerated statistical computing for Python.

## Design Philosophy

PyStatistics maintains two parallel computational paths with distinct goals:

- **CPU implementations aim for R-level reproducibility.** CPU backends are validated against R reference implementations to near machine precision (rtol = 1e-10). When a CPU result disagrees with R, PyStatistics has a bug.

- **GPU implementations prioritize modern numerical performance and scalability.** GPU backends use FP32 arithmetic and algorithms optimized for throughput. They are validated against CPU backends, not directly against R.

- **Divergence between CPU and GPU outputs may occur due to floating-point precision, algorithmic differences, or both.** This is by design, not a defect. The section below specifies exactly how much divergence is acceptable.

### Operating Principles

1. **Correctness > Fidelity > Performance > Convenience**
2. **Fail fast, fail loud** — no silent fallbacks or "helpful" defaults
3. **Explicit over implicit** — require parameters, don't assume intent
4. **Two-tier validation** — CPU vs R, then GPU vs CPU

---

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `regression/` LM | Complete | Linear models (OLS) with CPU QR and GPU Cholesky |
| `regression/` GLM | Complete | Generalized linear models (Gaussian, Binomial, Poisson, Gamma, Negative Binomial) via IRLS |
| `mvnmle/` | Complete | Multivariate normal MLE with missing data (Direct + EM) |
| `descriptive/` | Complete | Descriptive statistics, correlation, quantiles, skewness, kurtosis |
| `hypothesis/` | Complete | t-test, chi-squared, Fisher exact, Wilcoxon, KS, proportions, F-test, p.adjust |
| `montecarlo/` | Complete | Bootstrap (ordinary, balanced, parametric), permutation tests, 5 CI methods, batched GPU solver |
| `survival/` | Complete | Survival analysis: Kaplan-Meier, log-rank test, Cox PH (CPU), discrete-time (GPU) |
| `anova/` | Complete | ANOVA: one-way, factorial, ANCOVA, repeated measures, Type I/II/III SS, Tukey/Bonferroni/Dunnett, Levene's test |
| `mixed/` LMM/GLMM | Complete | Linear and generalized linear mixed models (random intercepts/slopes, nested/crossed, REML/ML, Satterthwaite df, GLMM Laplace) |
| `ordinal/` | Complete | Proportional odds (cumulative link) models matching R MASS::polr |
| `multinomial/` | Complete | Multinomial logit (softmax) regression matching R nnet::multinom |
| `multivariate/` | Complete | PCA and maximum likelihood factor analysis with varimax/promax rotation |
| `timeseries/` | Complete | ACF, PACF, ADF, KPSS, ETS, ARIMA, SARIMA, auto_arima, decompose, STL |
| `gam/` | Complete | Generalized additive models with penalized regression splines matching R mgcv::gam |
| `mice/` | Complete | Multiple imputation by chained equations: numeric (PMM, Bayesian regression) and categorical (logistic, multinomial, proportional-odds), Rubin's-rules pooling, validated against R mice; CUDA and Apple Silicon (MPS) GPU backend for numeric |

See [docs/ROADMAP.md](docs/ROADMAP.md) for detailed scope, GPU applicability, and implementation priority for each module.

## Architecture

Every module follows the same pattern:

```
DataSource -> Design -> fit() -> Backend.solve() -> Result[Params] -> Solution
```

- **CPU backends** are the gold standard, validated against R to rtol = 1e-10.
- **GPU backends** are validated against CPU backends per the tolerances below.
- **Two-tier validation** ensures correctness at any scale: Python-CPU vs R, then Python-GPU vs Python-CPU.

---

## Statistical Equivalence: GPU vs CPU

GPU backends produce results in FP32 (single precision) while CPU backends use FP64 (double precision). This section defines exactly what "statistically equivalent" means and when it breaks down.

All tolerances below are relative (`rtol`) unless stated otherwise. They apply to **well-conditioned problems** (condition number < 10^6) at **moderate scale** (n < 1M, p < 1000). Degradation at larger scale or worse conditioning is documented below.

### Tier 1: Parameter Estimates

| Quantity | Tolerance | Notes |
|----------|-----------|-------|
| Coefficients / means | rtol <= 1e-3 | Tightest at ~1e-4 for simple LM |
| Fitted values | rtol <= 1e-3 | Directly derived from coefficients |
| GPU-CPU correlation | > 0.9999 | Binding constraint at all scales |

### Tier 2: Uncertainty Estimates

| Quantity | Tolerance | Notes |
|----------|-----------|-------|
| Standard errors | rtol <= 1e-2 | Computed from (X'WX)^-1 which amplifies FP32 rounding |
| Covariance matrices (MLE) | rtol <= 5e-2 | Hessian inversion is sensitive to precision |

Standard errors are the weakest link in the GPU pipeline. They depend on the inverse of X'WX (or X'X for LM), which squares the condition number. A well-conditioned problem at FP64 can become a poorly-conditioned inversion at FP32.

### Tier 3: Model Fit Statistics

| Quantity | Tolerance | Notes |
|----------|-----------|-------|
| Deviance | rtol <= 1e-4 | Scalar reduction — tightest GPU metric |
| Log-likelihood | abs <= 1.0 | Absolute, not relative (log scale) |
| AIC / BIC values | rtol <= 1e-3 | Derived from log-likelihood + rank |
| R-squared (LM) | rtol <= 1e-3 | Ratio of reductions |

### Tier 4: Inference Decisions

| Quantity | Guarantee | Notes |
|----------|-----------|-------|
| Model ranking under AIC/BIC | Identical | For models with AIC/BIC gap > 2 |
| Rejection at alpha = 0.05 | Identical | For p-values outside [0.01, 0.10] |
| Rejection at alpha = 0.05 | Not guaranteed | For p-values in [0.01, 0.10] ("boundary zone") |

The boundary zone exists because a ~1% relative difference in a test statistic near the critical value can flip a rejection decision. This is inherent to FP32, not a software defect. If a p-value falls in the boundary zone, use the CPU backend for the definitive answer.

### When Guarantees Degrade

**Large scale (n > 1M):** FP32 accumulation over millions of rows introduces drift. Element-wise tolerance relaxes to rtol = 1e-2, but correlation remains > 0.9999. This means GPU coefficients track CPU coefficients nearly perfectly in direction, with small magnitude drift from accumulated rounding.

**Ill-conditioned problems (condition number > 10^6):** The GPU backend refuses by default and raises `NumericalError`. Passing `force=True` overrides this, but no numerical guarantees apply. Use the CPU backend for ill-conditioned problems.

**Pathological missing data patterns (MLE):** FP32 L-BFGS-B optimization can stall in near-flat regions of the likelihood surface. Means may deviate by up to rtol = 0.5 in extreme cases. The GPU backend will issue a convergence warning. Use the CPU backend for complex missingness patterns.

### Why FP32?

Consumer GPUs (NVIDIA RTX series) execute FP32 at 5-10x the throughput of FP64. Apple Silicon GPUs (MPS) do not support FP64 at all. FP32 is the only path to practical GPU acceleration on hardware that researchers actually have. The tolerances above are the honest cost of that acceleration.

### CUDA vs MPS: Not All GPU Backends Are Equal

Certain operations (notably `scatter_add_` with sparse targets) are 1000x slower on Apple MPS than on NVIDIA CUDA due to Metal's weaker atomic memory support. PyStatistics detects these cases and either fails fast or routes to CPU. See [docs/GPU_BACKEND_NOTES.md](docs/GPU_BACKEND_NOTES.md) for detailed benchmarks and guidance on when GPU helps vs hurts.

---

## Quick Start

```python
import numpy as np

# --- Descriptive statistics ---
from pystatistics.descriptive import describe, cor, quantile

data = np.random.randn(1000, 5)
result = describe(data)
print(result.mean, result.sd, result.skewness, result.kurtosis)

# Correlation (Pearson, Spearman, Kendall)
r = cor(data, method='spearman')
print(r.correlation_matrix)

# Quantiles (all 9 R types supported)
q = quantile(data, type=7)
print(q.quantiles)

# --- Hypothesis testing ---
from pystatistics.hypothesis import t_test, chisq_test, p_adjust

result = t_test([1,2,3,4,5], [3,4,5,6,7])
print(result.statistic, result.p_value, result.conf_int)
print(result.summary())  # R-style print.htest output

# Multiple testing correction
p_adjusted = p_adjust([0.01, 0.04, 0.03, 0.005], method='BH')

# --- Linear regression ---
from pystatistics.regression import fit

X = np.random.randn(1000, 5)
y = X @ [1, 2, 3, -1, 0.5] + np.random.randn(1000) * 0.1
result = fit(X, y, names=['x1', 'x2', 'x3', 'x4', 'x5'])
print(result.summary())          # R-style output with variable names
print(result.coef)                # {'x1': 1.00, 'x2': 2.00, ...}
print(result.coef['x3'])          # 3.00

# Logistic regression
y_binary = (X @ [1, -1, 0.5, 0, 0] + np.random.randn(1000) > 0).astype(float)
result = fit(X, y_binary, family='binomial')
print(result.summary())

# Prior weights and offsets (like R's lm/glm weights= and offset=)
result = fit(X, y, weights=w)                        # weighted least squares
result = fit(X, counts, family='poisson',            # Poisson rate model:
             offset=np.log(exposure))                #   log(exposure) offset

# --- Categorical predictors & interactions ---
# Describe a model as a list of terms (no R-style formula strings):
#   "name"          -> numeric main effect
#   C(name, ref=…)  -> categorical, treatment-coded with a chosen baseline
#   (a, b)          -> interaction (numeric and/or categorical)
from pystatistics import DataSource
from pystatistics.regression import Design, fit, C

ds = DataSource.from_dataframe(df)   # df has age, sex, treatment, response
design = Design.from_datasource(
    ds, y='response',
    terms=['age', C('sex', ref='F'), C('treatment', ref='A'),
           (C('treatment', ref='A'), C('sex', ref='F'))],
)
result = fit(design)                       # also works with family=… for GLMs
print(result.coef['treatment[B]:sex[M]'])  # interaction coefficient

# Cox PH takes the same spec (no intercept):
from pystatistics.survival import coxph
cox = coxph(time, event, ds, terms=['age', C('sex', ref='F')])

# GPU acceleration (any model)
result = fit(X, y, backend='gpu')

# --- Monte Carlo methods ---
from pystatistics.montecarlo import boot, boot_ci, permutation_test

# Bootstrap for the mean
data = np.random.randn(100)
def mean_stat(data, indices):
    return np.array([np.mean(data[indices])])

result = boot(data, mean_stat, R=2000, seed=42)
print(result.t0, result.bias, result.se)

# Bootstrap confidence intervals (all 5 types)
ci_result = boot_ci(result, type='all')
print(ci_result.ci['perc'])  # percentile CI
print(ci_result.ci['bca'])   # BCa CI

# Permutation test
x = np.random.randn(30)
y = np.random.randn(30) + 1.0
def mean_diff(x, y): return np.mean(x) - np.mean(y)
result = permutation_test(x, y, mean_diff, R=9999, seed=42)
print(result.p_value, result.summary())

# --- Survival analysis ---
from pystatistics.survival import kaplan_meier, survdiff, coxph, discrete_time

time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
event = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1])

# Kaplan-Meier survival curve
km = kaplan_meier(time, event)
print(km.survival, km.se, km.ci_lower, km.ci_upper)

# Log-rank test (compare groups)
group = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
lr = survdiff(time, event, group)
print(lr.statistic, lr.p_value, lr.summary())

# Cox proportional hazards (CPU only)
X = np.column_stack([np.random.randn(10)])
cox = coxph(time, event, X)
print(cox.coefficients, cox.hazard_ratios, cox.summary())

# Discrete-time survival (GPU-accelerated)
dt = discrete_time(time, event, X, backend='auto')
print(dt.coefficients, dt.hazard_ratios, dt.baseline_hazard)

# --- ANOVA ---
from pystatistics.anova import anova_oneway, anova, anova_rm, anova_posthoc, levene_test

# One-way ANOVA
y = np.concatenate([np.random.randn(20) + mu for mu in [0, 1, 3]])
group = np.array(['A']*20 + ['B']*20 + ['C']*20)
result = anova_oneway(y, group)
print(result.summary())          # R-style ANOVA table
print(result.eta_squared)        # effect sizes

# Post-hoc: Tukey HSD
posthoc = anova_posthoc(result, method='tukey')
print(posthoc.summary())         # pairwise comparisons with adjusted p-values

# Factorial ANOVA (Type II SS, matches R's car::Anova)
result = anova(y, {'treatment': tx, 'dose': dose}, ss_type=2)

# ANCOVA (continuous covariate)
result = anova(y, {'group': group}, covariates={'age': age}, ss_type=2)

# Repeated measures with sphericity correction
result = anova_rm(y, subject=subj, within={'condition': cond}, correction='auto')
print(result.sphericity[0].gg_epsilon)  # Greenhouse-Geisser correction

# Levene's test for homogeneity of variances
lev = levene_test(y, group, center='median')  # Brown-Forsythe variant
print(lev.f_value, lev.p_value)

# --- Mixed models ---
from pystatistics.mixed import lmm, glmm

# Random intercept model (matches R lme4::lmer + lmerTest)
result = lmm(y, X, groups={'subject': subject_ids})
print(result.summary())         # lmerTest-style output with Satterthwaite df
print(result.icc)               # intraclass correlation coefficient
print(result.ranef['subject'])  # BLUPs (conditional modes) per subject

# Random intercept + slope
result = lmm(y, X, groups={'subject': subject_ids},
             random_effects={'subject': ['1', 'time']},
             random_data={'time': time_array})

# Crossed random effects (subjects x items)
result = lmm(y, X, groups={'subject': subj_ids, 'item': item_ids})

# Model comparison via LRT (requires ML, not REML)
m1 = lmm(y, X_reduced, groups={'subject': subj_ids}, reml=False)
m2 = lmm(y, X_full, groups={'subject': subj_ids}, reml=False)
print(m1.compare(m2))  # LRT chi-squared, df, p-value

# GLMM — logistic with random intercept
result = glmm(y_binary, X, groups={'subject': subject_ids},
              family='binomial')
print(result.summary())

# GLMM — Poisson with random intercept
result = glmm(y_count, X, groups={'subject': subject_ids},
              family='poisson')

# --- Gamma GLM ---
from pystatistics.regression import fit

y_positive = np.abs(np.random.randn(200)) + 0.1
X = np.random.randn(200, 3)
result = fit(X, y_positive, family='gamma')
print(result.summary())

# --- Ordinal regression ---
from pystatistics.ordinal import polr

y_ordinal = np.random.choice([1, 2, 3, 4, 5], size=200)
X = np.random.randn(200, 3)
result = polr(y_ordinal, X)
print(result.coefficients, result.thresholds)
print(result.summary())

# --- Time series (ARIMA) ---
from pystatistics.timeseries import arima, auto_arima, acf

ts = np.cumsum(np.random.randn(200))  # random walk
acf_result = acf(ts, nlags=20)
result = arima(ts, order=(1, 1, 1))
print(result.coefficients, result.aic)
best = auto_arima(ts)
print(best.order, best.aic)

# --- GAM ---
from pystatistics.gam import gam, s

x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(x) + np.random.randn(200) * 0.3
result = gam(y, smooths=[s('x1')], smooth_data={'x1': x})
print(result.edf, result.gcv)
print(result.summary())

# --- Multiple imputation (MICE) ---
from pystatistics.mice import mice, pool

# data is an (n, p) array with np.nan marking missing values.
# Predictive mean matching (R default) for numeric columns; seed is required
# so the imputation is fully reproducible.
imp = mice(data, n_imputations=5, max_iter=5, method='pmm', seed=0)
completed = imp.completed_datasets()        # list of 5 completed (n, p) arrays

# Fit your analysis on each completed dataset, then combine with Rubin's rules:
estimates, variances = [], []
for d in completed:
    X = np.column_stack([np.ones(len(d)), d[:, 1]])
    beta, *_ = np.linalg.lstsq(X, d[:, 0], rcond=None)
    resid = d[:, 0] - X @ beta
    cov = (resid @ resid / (len(d) - 2)) * np.linalg.inv(X.T @ X)
    estimates.append(beta[1]); variances.append(cov[1, 1])

pooled = pool(estimates, variances, dfcom=len(data) - 2)
print(pooled.estimate, pooled.se, pooled.ci_low, pooled.ci_high, pooled.fmi)
```

## Installation

```bash
pip install pystatistics

# With GPU support (requires PyTorch)
pip install pystatistics[gpu]

# Development
pip install pystatistics[dev]
```

---

## What's New

### 4.6.3 — damped-trend ETS fits no longer stall

Damped-trend ETS models (`Ad`) could stop short of their optimum and report
`converged=False` (the damping parameter started nearly pinned to its upper
bound, and large seasonal fits could exhaust their evaluation budget).
Damped models are now optimised from two starting points with the better
result kept; damped fits improve or stay identical, non-damped fits are
unchanged, and automatic `"ZZZ"` selection picks the same models on every
reference dataset (elsewhere an improved damped fit can now win a selection
it previously lost — the intended effect). Where that
selection differs from R's `forecast::ets`, the choice is now verified by
evaluating the fitted parameters under R's own likelihood code — our pick
scores a strictly lower AICc there on every divergent reference dataset —
and regression tests enforce that this stays true.

### 4.6.2 — STL matches R exactly and runs as fast; ETS automatic model selection

`stl()` was rewritten after validation showed it leaking trend into the
seasonal component on trending series (on the monthly CO2 benchmark the old
"seasonal" had a range of ~83 where the truth is ~6). It is now an exact
implementation of the STL procedure and reproduces R's `stats::stl`
component-for-component (differences below 1e-10 across a reference suite
including robust fits), with the full STL parameter set exposed
(`seasonal_degree`, `trend_degree`, `lowpass_window`, jumps, and a
`"periodic"` default seasonal window). The smoother is compiled, so it now
runs as fast as R's Fortran implementation — faster for non-robust
decompositions, on par for robust ones. `ets()` now auto-selects its model
like `forecast::ets`: the new default `model="ZZZ"` tries every admissible
error/trend/season combination and returns the best by AICc, with the full
candidate table disclosed on the result; explicit component requests that
cannot be honoured raise instead of being silently adjusted. ETS parameter
estimation now matches R's parameter region as well, so fitted parameters
and information criteria (AIC/AICc/BIC) line up with `forecast::ets`.
`ndiffs()` now defaults to the KPSS test, matching `forecast::ndiffs`.

### 4.6.1 — faster multi-smooth Gaussian GAMs

`gam()` with the default Gaussian family now selects its smoothing parameters
with an exact analytic gradient of the GCV/REML criterion instead of finite
differences, so fits with several smooth terms no longer slow down as terms are
added — about 2× faster with four smooths and 3× with six on a typical fit.
Estimates are unchanged, and single-smooth and generalized (Poisson, binomial,
Gamma) additive models are unaffected.

### 4.6.0 — generalized additive models rewritten to match `mgcv`

`gam()` received a full numerical rewrite. Automatic smoothing-parameter
selection is fixed (the previous version could over- or under-smooth badly and
report impossible diagnostics), thin-plate smooths now represent linear trends
correctly, standard errors and REML are real, and results agree with R's
`mgcv::gam`. New: fix the smoothing parameters with `gam(..., sp=[...])`, and
read `.lambdas`, `.se`, and `.covariance` off the result. The experimental GPU
backend for `gam()` was removed — GAM fitting is CPU-only.

### 4.5.7 — faster random-slope LMMs

`lmm()` now optimizes the variance parameters with an exact analytic gradient
instead of finite differences, so correlated random-slope models fit about 1.35×
faster (random-intercept and crossed designs are unaffected). Estimates are
unchanged.

### 4.5.6 — GLMM variance estimation robustness

`glmm()` no longer settles at a zero random-effect variance on flat-likelihood
data: a fit that lands at zero variance is verified with a derivative-free search
and kept only if it genuinely improves, so real random-effect variance is
recovered (matching `lme4::glmer`) while genuinely singular fits still return zero.

### 4.5.5 — GLMM scales to many groups

`glmm()` now uses a structure-exploiting solver (batched/sparse) so fitting time
grows roughly linearly with the number of groups instead of cubically — a
many-group binomial fit that was several times slower than `lme4::glmer` is now
close to it. Estimates are unchanged.

### 4.5.4 — more robust GLMM fitting

`glmm()` Poisson models now fit reliably without occasional numerical failures
during optimization, and the optimizer no longer settles at a suboptimal fit with
a collapsed (near-zero) random-effect variance — a derivative-free fallback
recovers the correct optimum. Well-converged fits are unchanged.

### 4.5.3 — clearer errors for unsupported GLMM families

`glmm()` now raises a clear error when given a Gaussian or Gamma family (which
have a free dispersion parameter its Laplace fit does not estimate) instead of
returning misleading fit statistics. Use `lmm()` for Gaussian mixed models.

### 4.5.2 — accurate generalized linear mixed models

`glmm()` now fits the Laplace approximation (nAGQ = 1), matching the default fit of
`lme4::glmer`: the fixed effects are optimized together with the variance
components rather than in the inner loop, so estimates and log-likelihood agree
with `glmer` to about 1e-3 (previously they matched only the cruder `nAGQ = 0`
fit). A standard-error bug that made `glmm()` fixed-effect standard errors — and
the z-statistics, p-values and confidence intervals derived from them — wrong for
models with correlated predictors is also fixed.

### 4.5.1 — robust convergence for near-perfect-ICC mixed models

`lmm()` now converges to the correct optimum in the extreme variance-ratio regime
(intraclass correlation approaching 1), where it could previously fail to converge
or report a biased variance component. Well-converged fits are unchanged.

### 4.5.0 — mixed models scale up, report singular fits, and gain a GPU low-rank / GRM model

- `lmm()` now scales to large and crossed designs — a structure-exploiting
  batched/sparse solver drops a 2000-group fit from ~197 s to ~0.05 s, and crossed
  designs that used to run out of memory now fit — with fixed-effect standard
  errors computed in O(p³) instead of O(n³). Estimates are unchanged and match
  `lme4`.
- `lmm()` flags boundary (singular) fits via `LMMSolution.is_singular` and a
  warning, like `lme4`'s `isSingular()`.
- New `grm_lmm()` — a low-rank / GRM mixed model (a genomic relatedness matrix or
  any reduced-rank random effect) with REML/ML, narrow-sense heritability and
  genetic-value BLUPs, and a `backend=` (`cpu`/`gpu`/`gpu_fp64`/`auto`) whose
  float32 GPU path refuses loudly rather than return a biased fit.

### 4.4.1 — factor analysis matches R on multi-factor and Heywood cases

- `factor_analysis(X, n_factors=2)` with the default varimax rotation no longer
  raises a spurious "did not converge" error on clean, well-fitting multi-factor
  data; the rotation uses a relative convergence test matching R's `stats::varimax`
  and its rotated loadings now agree with R.
- New `lower=` parameter (default `0.005`, matching R `factanal`) floors the
  uniquenesses during fitting, preventing degenerate Heywood solutions; a
  one-factor iris fit now matches R's constrained optimum. Must satisfy
  `0 < lower < 1`.

### 4.4.0 — PCA on Apple Silicon GPUs

- `pca(..., backend='gpu')` now runs on Apple Silicon (Metal/MPS), where it used
  to raise "not supported." A new randomized truncated SVD solver keeps the whole
  computation on the Metal GPU; the default GPU path on a Mac selects it
  automatically, so no code change is needed. On the large tall/wide data PCA is
  used on, it is several times faster than the same computation on the CPU (e.g. a
  500k×200 fit drops from ~3.6 s to ~0.2 s on an M2 Max). Small inputs (under
  ~10k rows) remain faster on the CPU.
- New `solver='randomized'` (selectable on CUDA too) with `oversample` / `n_iter`
  accuracy knobs and a reproducible `seed`. It refuses near-singular data at
  single precision unless `force=True`. On Apple Silicon an explicit
  `solver='svd'` or `'gram'` now fails with a clear message instead of silently
  running on the CPU; CUDA behavior is unchanged.

### 4.3.3 — convergence signal for discrete-time survival

- `survival.discrete_time(...)` results now expose `.converged` and `.n_iter`,
  reporting whether the underlying person-period logistic (IRLS) fit converged
  or stopped at the iteration cap — the same accessors `coxph` already provides.
  `summary()` shows a matching "Converged: … (N iterations)" line. Additive; no
  existing attribute changed.

### 4.3.2 — correctness fixes for fit statistics

- GPU OLS standard errors no longer understate on ill-conditioned (near-collinear)
  designs — a regression since 4.3.0. `fit(..., backend='gpu')` now computes the
  coefficient covariance in double precision, so its standard errors match the
  CPU backend and R. Coefficients are unchanged.
- Gamma and Gaussian `.bic` now count the estimated dispersion parameter with the
  `log(n)` penalty, matching `BIC(glm(...))`.
- Binomial `.deviance` (and the AIC/BIC derived from it) now matches `glm()` for
  models with very extreme fitted probabilities.

### 4.3.0 — prior weights and offsets

- `fit()` now takes `weights=` and `offset=`, matching R's `lm(..., weights=)`
  and `glm(..., weights=, offset=)`. `weights` are per-observation prior
  (precision) weights — weighted least squares for an OLS fit, IRLS prior
  weights for every GLM family. `offset` is a fixed term in the linear predictor
  (`η = Xβ + offset`), e.g. `log(exposure)` for a Poisson rate model. Both are
  validated against R to round-off, and the GPU backends support them. Not
  available together with a ridge penalty (`l2 > 0`).
- Gamma GLM AIC now matches R's `glm.fit()` (a different dispersion convention
  had made it disagree); coefficients, standard errors, and deviance are
  unchanged.

### 4.0.0 — the consistency release (breaking)

A library-wide pass so every module names parameters, selects its backend,
returns results, and raises errors the same way. **Breaking:** several
parameters, option values, and result classes were renamed and the old
spellings removed (no alias). **No statistical or numerical behavior changed** —
every number is exactly as before. The full rules live in
`pystatistics/CONVENTIONS.md`; the [changelog](CHANGELOG.md) has the complete
rename table. Highlights:

- `backend=` now means only *where and at what precision* a fit runs: `'cpu'`
  (double), `'gpu'` (single), `'gpu_fp64'` (CUDA double), `'auto'`. The separate
  `use_fp64=` flag is gone — pass `backend='gpu_fp64'` for double-precision GPU.
  On linear regression, choose the numerical routine with the new `solver=`
  (`'qr'`/`'svd'`) instead of an algorithm-encoding backend string.
- Parameters unified under one naming law: e.g. `mice(m=, maxit=)` →
  `mice(n_imputations=, max_iter=)`, `polr(method=, ridge=)` → `polr(link=,
  l2=)`, `t_test(mu=, var_equal=)` → `t_test(pop_mean=, equal_var=)`,
  `boot(R=, sim=, stype=)` → `boot(n_resamples=, method=, statistic_type=)`.
  Option values lose their dots too: `alternative="two-sided"` (was
  `"two.sided"`).
- Every fit returns a `…Solution` with uniform accessors (`.coefficients`,
  `.standard_errors`, `.z_values`/`.t_values`, `.p_values`, `.converged`,
  `.n_iter`, `.backend_name`) and a Jupyter HTML summary.
- Invalid input raises `ValidationError` (which is also a `ValueError`, so
  existing `except ValueError` code keeps working); non-convergence raises
  `ConvergenceError`; an unavailable GPU raises `RuntimeError`.

### 3.20.0 — Ridge regression, and double-precision GPU fits

- Ridge (L2-penalized) regression: `fit(..., l2=lambda)` or `ridge(X, y, lam=lambda,
  family=...)`, for linear models and all GLM families. Predictors are standardized
  and the intercept unpenalized (scale-invariant penalty, matching `MASS::lm.ridge`
  / `glmnet`). Ridge fits report `NA` standard errors, since penalized estimates
  don't have valid frequentist inference.
- Ridge runs fast and stably on the GPU at very large scale — where a plain GLM can
  be ill-conditioned in single precision, the ridge penalty fixes the conditioning,
  so `ridge(..., backend='gpu')` fits a regularized GLM the unpenalized fit can't.
- New `backend='gpu_fp64'`: double-precision GPU fits, numerically equivalent to the
  CPU reference. CUDA only (Apple Silicon has no float64). Backend strings now name
  device and precision: `cpu` (double), `gpu` (single), `gpu_fp64` (CUDA double).

### 3.19.1 — Correctness fix for large GPU GLM fits

- A GLM fit with `backend='gpu'` for a log-link family (Poisson/Gamma) at large
  sample size could fail to converge in float32 (especially on Apple Silicon) and
  return non-converged coefficients. It now raises a clear error pointing you to
  `backend='cpu'` for a correct fit, instead of returning unreliable numbers.
  Small/medium GLM fits and large fits on CUDA are unaffected.

### 3.19.0 — GLM on Apple Silicon GPUs, faster GPU OLS and count models

- Generalized linear models (`fit(..., family=...)` with `backend='gpu'`) now run
  on Apple Silicon GPUs. Logistic, Poisson, Gamma, and negative-binomial fits
  previously errored on Metal; the IRLS inner step now uses a Metal-supported
  Cholesky solve, matching the CPU result to single-precision rounding and running
  several times faster than the CPU on large problems. CUDA GLM fits are faster too.
- GPU ordinary least squares no longer decomposes the full design matrix on every
  fit to check conditioning — it uses the small cross-product matrix instead, with
  identical results. Small and medium OLS fits on Apple Silicon are up to ~3x faster.
- Negative-binomial and Gamma fits are about 2x faster (vectorized deviance
  residuals); a negative-binomial fit is now faster than R's `MASS::glm.nb`.
  Results are unchanged.

### 3.18.0 — Faster GPU MVN MLE on Apple Silicon

- `mlest(..., algorithm='direct', backend='gpu')` now evaluates the objective and
  its gradient in a single GPU pass per optimizer step, with one device-to-host
  transfer instead of two. On Apple Silicon (MPS) this roughly halves the
  per-evaluation cost on wide data — about 1.9x faster at 100 variables over
  50,000 rows — and also speeds up NVIDIA CUDA fits. Estimates are unchanged.

### 3.17.0 — Much faster CPU MLE by default

- `mlest(..., algorithm='direct')` with `backend='cpu'` (the default) now uses a
  PyTorch forward-Cholesky estimator in double precision. It matches R's `mvnmle`
  to about 1e-9 and is far faster than before — a 10-variable fit over 2,000 rows
  with 15% missingness runs in roughly 0.1s instead of about 100s, with identical
  estimates. The fast path uses PyTorch (the optional `pystatistics[gpu]` extra).
- New `backend='cpu-reference'` selects the original numpy reference optimizer:
  it matches R, needs no PyTorch, and is handy as a dependency-free reference. It
  is valid only with `algorithm='direct'`.
- Without PyTorch installed, the default CPU path falls back to the numpy
  reference automatically (with a warning), so results stay correct everywhere.

### 3.16.4 — Correct convergence reporting for direct MLE on large datasets

- `mlest(..., algorithm='direct')` no longer reports spurious non-convergence on
  large datasets. The optimizer now judges convergence on a per-observation
  scale, so a good fit on tens of thousands of rows is correctly flagged as
  converged instead of warning "Optimization did not converge". The estimates and
  log-likelihood are unchanged; the fix applies to the CPU and GPU backends.
- `little_mcar_test` no longer raises a spurious non-convergence error on large,
  well-conditioned data whose underlying fit had actually converged.

### 3.16.3 — Faster GPU ordered-factor imputation on Apple Silicon

- GPU `mice` imputation of ordered factors (`polr`) is faster on Apple Silicon
  (MPS): the line search that stabilizes the fit under near-separation now reaches
  its step by quadratic interpolation instead of repeated halving, using far fewer
  objective evaluations. Purely a performance change — the imputation results and
  stability guarantees are unchanged; CUDA is unaffected.

### 3.16.2 — Reliable GPU categorical & ordered imputation on imbalanced data

- GPU `mice` imputation of binary, unordered-categorical, and ordered columns no
  longer collapses a column onto a single category on imbalanced mixed data
  (a binary column nearly separated by a covariate, or an ordered column with a
  very sparse middle category). The GPU fits are numerically stabilized and run
  in double precision internally on CUDA and CPU, so imputed category
  proportions track the CPU backend and R's `mice`. (Apple Silicon / MPS, which
  has no double precision, is unchanged.)
- A genuinely degenerate GPU categorical/ordered fit now raises a clear error
  rather than silently imputing every missing cell as category 0.

### 3.16.1 — Reliable GPU ordered-factor imputation under near-separation

- GPU `mice` imputation of ordered factors (`polr`, on CUDA and Apple Silicon)
  no longer collapses a column onto a single category when chained equations push
  it into near-separation (a sparse extreme category ordered almost perfectly by
  a continuous predictor). The GPU fit now matches the CPU `polr` method,
  recovering the sparse extreme category instead of assigning nearly every
  imputed cell the same value.

### 3.16.0 — Reliable ordered-factor imputation under near-separation

- `mice` imputation of ordered factors (the `polr` method) no longer degrades to
  a predictor-blind marginal draw when chained equations push a column into
  near-separation (a sparse extreme category ordered almost perfectly by a
  continuous predictor). A small ridge on the slopes keeps the fit finite and
  fast, so it stays a proper predictor-aware imputer on exactly the columns where
  it used to fall back.
- `polr` gains an optional `ridge` parameter — an L2 penalty on the slope
  coefficients that keeps the fit finite and well-conditioned under
  (quasi-)complete separation. The default `ridge=0.0` is the exact
  maximum-likelihood fit.

### 3.15.2 — GPU ordered-factor imputation matches R's threshold variability

- GPU `mice` imputation of ordered factors (`polr`, on CUDA and Apple Silicon)
  now draws the proportional-odds thresholds on the natural scale, matching the
  CPU `polr` method and R's `MASS::polr`. Imputed category proportions were
  already correct, but the between-imputation variability — and the pooled
  (Rubin's rules) variances and intervals that depend on it — is now correct too.

### 3.15.1 — Faster, more reliable ordered-factor imputation

- Imputing ordered factors with `mice` (the `polr` method) is now several times
  faster — and faster than R's `mice` — and no longer falls back to a cruder
  draw on realistic correlated data. Proportional-odds threshold standard errors
  now match R's `MASS::polr`.

### 3.15.0 — MICE GPU imputes categorical data

- `mice(..., backend='gpu')` now imputes categorical columns on CUDA and Apple
  Silicon (MPS), using the same models as R's `mice`: `logreg` for binary
  columns, `polyreg` for unordered factors, and `polr` for ordered factors. The
  fits run batched across all imputations; imputed category distributions match
  the CPU reference and R within Monte-Carlo tolerance. Categorical *predictors*
  are supported too (treatment-dummy-encoded) when imputing numeric columns.
  Together these let a full mixed-type dataset be imputed on the GPU — incomplete
  categorical columns were previously refused.

### 3.14.0 — Faster MICE GPU imputation on Apple Silicon

- MICE GPU imputation runs about 1.7–2x faster on Apple Silicon (MPS), with no
  change to results. The per-iteration step no longer synchronizes the GPU with
  the CPU on every step, and for larger problems it replaces `solve_triangular`
  (slow on MPS) with a short sequence of matrix multiplications. Measured ~1.8x
  at n=2000, ~2.0x at n=8000, ~1.7x at n=20000 (20 variables, 100 imputations).
  Imputations are unchanged within the GPU/FP32 tolerance; the CUDA path is
  unaffected.

### 3.13.0 — MICE GPU acceleration on Apple Silicon, faster on every GPU

- `mice(..., backend='gpu')` now runs on Apple Silicon (MPS), not only CUDA. The
  batched imputation sweep runs on the Mac GPU in FP32 — about 12x faster than
  the CPU backend on a large problem (n=20000, p=20, m=100: 3.3 s vs 42 s) —
  validated against the CPU reference for both `pmm` and `norm`. `backend='auto'`
  stays on CPU on a Mac; request the GPU explicitly with `backend='gpu'`.
  `use_fp64=True` is rejected on MPS (no double precision there). The GPU
  posterior draw and donor search were also reworked to run faster on CUDA too.

### 3.12.0 — MVN MLE rejects rank-deficient input

- `mlest` now raises `SingularMatrixError` on (near-)collinear input instead of
  returning a meaningless "converged" fit with a near-singular covariance — such
  input has no interior maximum-likelihood estimate. Pass `force=True` to return
  the degenerate result anyway (with `converged=False` and a warning), or
  `collinearity_tol` to tune the detection threshold. Collinear columns are never
  dropped automatically. Full-rank problems are unaffected. This is a behaviour
  change: collinear input that previously returned a result now raises by default.

### 3.11.0 — Portable inverse path and selectable inverse algorithm in the GPU objective

- The GPU objective's triangular-solve inverse path now runs on every device
  (it previously relied on `cholesky_inverse`, unavailable on Apple Metal).
- The batched GPU kernel functions accept a `method` argument (`"auto"`,
  `"solve"`, `"blocked"`) to select the per-pattern inverse algorithm; `"auto"`
  keeps the existing device-aware default. Results are identical regardless of
  `method`.

### 3.10.0 — Closed-form GPU gradient: fast, practical wide-data fits on Apple Silicon

- `mlest(backend='gpu')` now uses a closed-form gradient instead of automatic
  differentiation, which previously backpropagated through `cholesky` —
  pathologically slow on Apple Metal. A 100-variable survey fit on Apple Silicon
  goes from a >30-minute timeout to roughly 3 minutes (converged), and the
  per-gradient cost falls about 20-fold. Results are unchanged; CUDA and CPU
  benefit too.

### 3.9.0 — GPU MLE scales to wide data within bounded memory

- GPU `mlest` now evaluates the missing-data objective and gradient in chunks
  of missingness patterns, so GPU memory stays bounded no matter how many
  distinct patterns the data has. Wide data (100+ variables, tens of thousands
  of patterns) that previously hit CUDA out-of-memory now fits. The chunk size
  is auto-tuned (override via `chunk_size`); results are unchanged.

### 3.8.1 — Correct MLE for missing data with >62 variables

- `mlest` now groups missingness patterns correctly when a dataset has more
  than 62 variables. An integer-overflow bug in the pattern code previously
  merged distinct patterns and produced NaN estimates on wide data (e.g.
  survey instruments with 100+ items). Results are unchanged at ≤62 variables.

### 3.8.0 — Survival results expose warnings

- All survival results (`kaplan_meier`, `survdiff`, `coxph`, `discrete_time`)
  now expose a `.warnings` attribute, consistent with every other analysis type.
  Non-fatal issues found during fitting — such as a non-converged Cox model — are
  now reachable instead of silently dropped.
- The log-rank test (`survdiff`) now warns when its chi-square approximation may
  be unreliable: when any group's expected event count is below 5, or when a
  group has no observed events.

### 3.7.1 — Correct covariance from the double-precision GPU estimator

- Fixed an incorrect covariance matrix returned by `mlest(backend='gpu')` in
  double precision (FP64, NVIDIA/CUDA) when fitting 3 or more variables: the
  optimiser and the reported result referred to mismatched covariances. The FP64
  and FP32 GPU paths now share one validated reconstruction that matches the CPU
  result to floating-point precision. FP32 GPU and CPU fits were unaffected.

### 3.7.0 — Much faster GPU MLE on Apple Silicon

- `mlest(backend='gpu')` (direct / BFGS) on Apple Silicon (MPS) now computes
  the per-pattern trace term with a matmul-only blocked matrix inversion,
  sidestepping Metal's slow triangular-solve kernels. For data with many
  distinct missingness patterns (survey scale), this makes Apple-GPU fits
  dramatically faster, with results identical to before. CUDA is unchanged.

### 3.6.0 — Faster GPU MLE for missing-data multivariate normal

- `mlest(backend='gpu')` (direct / BFGS) now evaluates the per-pattern
  log-likelihood with a single batched Cholesky across all missingness
  patterns instead of looping over them one at a time. On data with many
  distinct patterns — common at survey scale — this is substantially
  faster. Results are unchanged.
- More numerically stable FP32 covariance computation on the GPU path.

### 3.5.1 — GPU MICE scales to large datasets

- The GPU predictive-mean-matching donor search now uses the same memory-light
  windowed approach as the CPU backend, batched across imputation chains. This
  removes out-of-memory failures on large problems and makes the GPU backend
  much faster at scale — on an RTX 5070 Ti, GPU PMM is roughly 30–50× faster
  than the CPU backend at n=20000, and imputes n=100000 in under a second.

### 3.5.0 — Categorical imputation for MICE

- `mice` now imputes categorical columns, not only numeric ones. Declare each
  column's kind via `column_kinds` (`'binary'`, `'categorical'`, `'ordered'`)
  and it is imputed with logistic, multinomial, or proportional-odds regression
  respectively — mirroring R `mice`'s `logreg`/`polyreg`/`polr`. Categorical
  columns are integer category codes.
- `method='auto'` (the new default) selects the right method per column kind;
  mixed numeric/categorical datasets impute coherently (categorical predictors
  are dummy-encoded). Imputed category proportions are validated against R
  `mice`.
- GPU acceleration stays numeric-only; categorical imputation runs on the CPU.

### 3.4.1 — Faster CPU predictive mean matching

- CPU PMM in `mice` now scales to large datasets: the donor search sorts the
  observed predictions and scans a small window per missing value (as R's
  `mice` does) instead of forming a full distance matrix, cutting time and
  memory from quadratic to roughly `n log n`. Large problems that were
  effectively unusable on the CPU now finish in seconds. Results are
  statistically unchanged.

### 3.4.0 — GPU acceleration for MICE

- `mice(..., backend='gpu')` runs the imputation chains on a CUDA GPU, batching
  the per-variable solves and the predictive-mean-matching donor search across
  chains. `backend='auto'` uses a CUDA GPU when available, else the CPU.
- The GPU advantage grows with sample size (the donor search batches well across
  chains); see 3.5.1 for current benchmark figures. GPU results match the CPU
  backend at the GPU/FP32 tolerance; pass `use_fp64=True` for double precision.
- Requires a CUDA GPU; Apple Silicon (MPS) is not yet supported for MICE.

### 3.3.0 — Multiple imputation (MICE)

- New `mice` module: multiple imputation by chained equations for numeric data
  with missing values. `mice(data, m=5, method='pmm', seed=...)` returns `m`
  completed datasets, using predictive mean matching (the R default) or
  Bayesian linear regression (`method='norm'`). Defaults follow R's `mice`.
- Imputation is fully reproducible — `seed` is required, and each chain uses an
  independent random stream.
- `pool(estimates, variances)` combines per-dataset analyses with Rubin's rules
  (Barnard–Rubin degrees of freedom, confidence intervals, fraction of missing
  information).
- Numeric columns on the CPU in this release; validated against R's `mice`.

### 3.2.0 — Apple Silicon (MPS) GPU support

- `multinom`, `polr`, `gam`, and `arima` / `arima_batch` (Whittle) now run
  on Apple Silicon GPUs with `backend='gpu'`, in FP32 and entirely on
  native Metal kernels (no hidden CPU fallback). Results match the CPU
  backend at the GPU/FP32 tolerance tier.
- `DataSource.to('mps')` transfers data to the Apple GPU (float64 →
  float32), so you can pay the host→device copy once and reuse it across
  fits.
- `backend='auto'` uses the CPU on Apple Silicon; the Apple GPU is opt-in
  via an explicit `backend='gpu'`. CUDA is still auto-selected.
- `pca` and MVN MLE `em` GPU paths remain CUDA-only and now raise a clear
  error on Apple Silicon rather than silently running on the CPU — PCA's
  SVD/eigendecomposition and the EM scatter/iteration pattern have no
  efficient Metal equivalent. Use `backend='cpu'` or `'auto'` on a Mac.
  (MVN MLE *direct* GPU fitting works on MPS.)
- Whittle ARIMA GPU fits no longer raise a spurious convergence error when
  the FP32 line search stalls at an already-converged optimum.

### 3.1.0 — Categorical predictors & interaction terms

- Regression now supports categorical predictors and interactions via a
  `terms=` spec on `Design.from_datasource`: bare names are numeric main
  effects, `C(name, ref=...)` marks a categorical predictor with a selectable
  baseline level, and tuples express interactions (numeric and/or
  categorical). Works for OLS, all GLM families, and Cox PH (no intercept).
- Expanded columns are labeled `sex[M]`, `treatment[B]:sex[M]`, with `coef`
  and inference outputs aligned to those labels. Design matrices match R's
  `model.matrix` for factors and interactions.
- `DataSource.from_dataframe` now keeps non-numeric columns as-is (previously
  force-cast to float), so categorical columns can feed `C(...)`.
- New public symbol: `pystatistics.regression.C`.

### 3.0.1 — Metadata and documentation polish

- Development Status classifier bumped from Alpha to Production/Stable.
- Stale `[nonparametric_mcar]` optional-dependency extra removed from
  `pyproject.toml` (the subpackage itself was removed in 3.0.0).
- README restructured to lead with library identity and module overview
  rather than changelog.

No API changes.

### 3.0.0 — MCAR helpers removed (breaking)

**Removed (breaking):**
  - `pystatistics.mvnmle.mom_mcar_test` and its helpers.
  - `pystatistics.nonparametric_mcar` subpackage in its entirety
    (`propensity_mcar_test`, `hsic_mcar_test`, `missmech_mcar_test`,
    `NonparametricMCARResult`).
  - The `[nonparametric_mcar]` optional-dependency extra.

If you were using these tests, `little_mcar_test` (the canonical Little
1988 MLE-plug-in test) remains and is unchanged. The removed tests were
project-specific feature-extraction utilities rather than textbook
methods.

**Retained (unchanged):** `little_mcar_test`, `MCARTestResult`, `mlest`,
`analyze_patterns`, `PatternInfo`, and every EM / SQUAREM /
monotone-closed-form path.

**Bug fixes:**
  - GAM GPU smooth-term chi-squared no longer diverges from CPU on
    ill-conditioned penalised normal matrices. The GPU backend now
    canonicalises the final coefficients via Cholesky-with-LU-fallback
    to match CPU bit-for-bit.
  - GAM GPU FP64 `total_edf` test tolerance widened to `rel=5e-3` on
    that quantity only, reflecting its linear sensitivity to λ near the
    GCV optimum.

### 2.3.0 — Nonparametric MCAR tests (introduced, removed in 3.0.0)

Shipped three distribution-free MCAR tests in a new `nonparametric_mcar`
subpackage. Removed in 3.0.0 — see above.

### 2.2.0 — Real-data robustness

Four classes of numerical failure on realistic tabular data fixed —
Cholesky fast-path crash on GPU FP32 roundoff, bare-`RuntimeError`
wrapping that broke `PyStatisticsError` catch patterns, M-step sigma
PD-check false negatives from FP64 roundoff, and per-pattern Cholesky on
indefinite sub-blocks — with a unified `regularize=True` opt-out-to-strict
convention across `mlest`, `little_mcar_test`, and the batched E-step.

### 2.1.0 — EM speedup + monotone closed-form MLE

`little_mcar_test` on realistic tabular data sped up 1.6–2.1× via batched
per-pattern E-step, SQUAREM acceleration, and fully batched
log-likelihood. Fully-batched device-resident EM on GPU added: 14.6× at
n=569, v=30. New `mvnmle.is_monotone`, `mvnmle.monotone_permutation`,
and `mlest(data, algorithm='monotone')` — Anderson (1957) closed-form
MLE for monotone missingness, bit-equivalent to R `mvnmle` on canonical
datasets and orders of magnitude faster than EM on larger-v longitudinal
data.

### 2.0.1 — GPU-backend exposure gaps closed

`little_mcar_test` and `auto_arima` gained `backend=` and
`algorithm=`/`method=` parameters that had been missing, so GPU paths are
now reachable from both entry points.

### 2.0.0 — CPU is the default backend everywhere (breaking)

Every public solver that previously defaulted to `backend='auto'` now
defaults to **CPU** — the R-reference, validated-for-regulated-industries
path. GPU is never selected implicitly. Affected: `regression.fit`,
`mvnmle.mlest`, `survival.discrete_time`, `montecarlo.boot`,
`montecarlo.permutation_test`, `descriptive.*`, `hypothesis.*`.

The GPU path is opt-in:

```python
result = fit(X, y, backend='gpu')    # require GPU; fail loud if absent
result = fit(X, y, backend='auto')   # prefer GPU, fall back to CPU
```

Migration: if you relied on implicit GPU selection on a GPU-equipped box,
add `backend='auto'` or `backend='gpu'` to the affected calls.

### 1.9.0 — Device-resident PCA results and batched ARMA fits

- **GPU-resident `PCAResult`** (`pca(..., device_resident=True)`).
  Numeric fields stay as `torch.Tensor` on the fit's device. 3.4× speedup
  on 1M × 100 FP32 PCA by skipping the D2H score copy.
- **`arima_batch(Y, order=(p, d, q), method='Whittle')`.** Fits K
  independent ARMA models on the rows of a `(K, n)` matrix simultaneously.
  Crossover at K ≈ 100; 13× at K=1000.

### 1.8.0 — GPU backends for the 1.6.x modules

GPU backends added across PCA, multinomial logit, ordinal polr, GAM, and
ARIMA Whittle. Typical speedups: 3–4× (PCA SVD), up to 100× (PCA Gram on
tall-skinny), 49–183× (multinomial), **448× at n=100k** (ordinal polr),
10–29× (GAM with 3 smooths), **36× at n=1M** (ARIMA Whittle). New
`DataSource.to(device)` API for amortised-transfer workflows. Whittle
ARIMA (`method='Whittle'`) added as a FFT-based approximate MLE alongside
CSS / ML / CSS-ML. CPU multinomial `vcov` now uses the analytical block
Hessian (29–33× CPU speedup on that step).

### Previous Releases

**1.7.0** — Performance parity with R on OLS first-call (578 ms → 5 ms),
polr (277 ms → 23 ms), and SARIMA airline-model fit (2,100 ms → 14 ms via
numba-JIT'd Kalman state-space path).

**1.6.2** — Re-shipped 1.6.1 fixes left out of the PyPI wheel. Fail-loud
fixes in ARIMA CSS-ML, ARIMA(0,d,0) closed-form MLE, Gamma GLM dispersion,
`descriptive.var(n=1)`, scipy 1.18 forward-compat.

**1.6.0** — Five new modules (`ordinal`, `multinomial`, `multivariate`,
`timeseries`, `gam`), two new GLM families (`Gamma`, `NegativeBinomial`).

**1.2.1** — No silent model switches; `backend='gpu'` is honest;
reproducible Monte Carlo via `seed=`; module structure refactoring.

**1.1** — Named coefficients via `names=`; `result.coef` dict; OLS/Cox
summary improvements matching R output.

---

## License

MIT

## Author

Hai-Shuo (contact@sgcx.org)
