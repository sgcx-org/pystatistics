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
