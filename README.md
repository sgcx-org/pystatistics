# PyStatistics

GPU-accelerated statistical computing for Python.

## What's New

### 2.0.1 — GPU-backend exposure gaps and a convention rule

Two public functions had GPU-capable inner calls but no `backend=`
parameter, so there was no way to route them through the GPU path —
exactly the regression the 2.0.0 CPU-default sweep was trying *not*
to create. Both fixed:

- **`little_mcar_test`** now accepts `backend=` and `algorithm=`,
  forwarded to `mlest`. The per-pattern test-statistic accumulation
  still runs on CPU (O(P × v³) for tiny v — never the bottleneck).
  GPU results match CPU within FP32 tolerance (Δ stat ≈ 1.4e-4 on
  the apple dataset).
- **`auto_arima`** now accepts `backend=` and `method=`, threaded
  through `_stepwise_search` / `_grid_search` / `_try_fit` so every
  candidate fit honours the same backend. Pass
  `method='Whittle', backend='gpu'` to run each candidate on GPU.

Also codified the "when to add a GPU backend, and when not to" rule
as Section 0 of `pystatistics/GPU_BACKEND_CONVENTION.md` — the
absence of `backends/gpu*.py` in a module (`anova`, `ets`, `coxph`,
`factor_analysis`, acf / stationarity) is a deliberate statement,
not an oversight. GPU backends belong on workloads that actually
map to GPU hardware (large dense linear algebra, big-N likelihoods,
batched fits, frequency-domain transforms), not on everything.

### 2.0.0 — CPU is now the default backend everywhere (breaking)

Every public solver that previously defaulted to `backend='auto'` now
defaults to **CPU — the R-reference, validated-for-regulated-industries
path**. GPU is never selected implicitly. Affected entry points:

- `regression.fit` (OLS and all GLM families)
- `mvnmle.mlest`
- `survival.discrete_time` and `discrete_time_fit`
- `montecarlo.boot`, `montecarlo.permutation_test`
- `descriptive.describe`, `.cor`, `.cov`, `.var`, `.quantile`, `.summary`
- `hypothesis.*` (signatures normalized — behaviour unchanged; CPU was
  already the effective default)

The GPU path is opt-in:

```python
result = fit(X, y, backend='gpu')    # require GPU; fail loud if absent
result = fit(X, y, backend='auto')   # prefer GPU, fall back to CPU
```

Rationale: GPU behaviour is not guaranteed across installs, and
regulated-industry users need "unspecified backend" to mean the
validated path. This formalises the convention already documented in
`pystatistics/GPU_BACKEND_CONVENTION.md` and followed by the
`multivariate.pca`, `multinomial`, `ordinal`, `timeseries.arima`, and
`gam` modules since 1.6.0.

Migration: if you were relying on implicit GPU selection on a
GPU-equipped box, add `backend='auto'` (best-effort GPU) or
`backend='gpu'` (require GPU) to the affected calls.

### 1.9.0 — Device-resident PCA results and batched ARMA fits

Two follow-ons to the 1.8.0 GPU sweep, focused on removing remaining
PCIe transfer bottlenecks and on making many-series workflows fast.

- **GPU-resident `PCAResult`** (`pca(..., device_resident=True)`).
  Numeric fields (`sdev`, `rotation`, `center`, `scale`, `x`) stay
  as `torch.Tensor` on the fit's device instead of being copied
  back to numpy. On a 1M × 100 FP32 PCA via a GPU `DataSource`,
  skipping the scores D2H copy cuts per-fit wall time from 202.9 ms
  to 59.3 ms — **3.4×** — and removes what was otherwise the
  dominant cost of any downstream GPU computation that consumes PCA
  output. Explicit `PCAResult.to_numpy()` / `.to(device)` materialise
  a numpy-backed copy; `.device` reports where the fields live.
  Default `device_resident=False` preserves 1.8.0 behaviour.
- **`arima_batch(Y, order=(p, d, q), method='Whittle')`.** Fits K
  independent ARMA models on the rows of a `(K, n)` matrix
  simultaneously. One batched `torch.fft.rfft` computes the full
  `(K, m)` periodogram; batched Adam runs K independent
  optimizations on a shared `(K, p+q)` parameter tensor with
  per-row gradient-norm convergence freezing. Non-seasonal,
  Whittle-method only; CPU path is a Python loop over the
  single-series `arima(method='Whittle')` (no batch speedup).
  Crossover at **K ≈ 100**; measured 6.9× at K=500, **13× at
  K=1000**, 10.7× at K=500/n=10000.

Validation: 2,371 pystatistics tests, 117 R-vs-Python cross-validation.

### 1.8.0 — GPU backends for the 1.6.x modules

Major release adding GPU backends across the five modules introduced
in 1.6.0 plus GEE in pystatsbio, a new `DataSource.to(device)` API
for amortised-transfer workflows, and a frequency-domain ARIMA method.
Also includes a CPU-only perf win for the multinomial vcov step that
fell out of the GPU work.

**New GPU backends** (measured on an RTX 5070 Ti; see CHANGELOG for
shape-by-shape tables):

| Module | Approach | Typical speedup vs CPU |
|---|---|---|
| PCA | SVD or Gram-matrix eigh, cond-gated | 3–4× (SVD), up to 100× (Gram, tall-skinny) |
| Multinomial logit | Analytical block-Hessian `X'·diag(Wⱼₖ)·X` | 49–183× |
| Ordinal polr | Autograd NLL + Hessian-via-autograd vcov | **448× at n=100k** |
| GAM (P-IRLS) | Batched penalty-sum + LU, hat-trace via numpy LAPACK | 10–29× with 3 smooths |
| GEE (pystatsbio 1.6.0) | Cluster-size grouped batched `torch.linalg.solve` | 13–67× at K=500–5000 |
| ARIMA Whittle | FFT-based approximate MLE, all on device | **36× at n=1M** |

Two-tier validation convention is documented in
`GPU_BACKEND_CONVENTION.md`: CPU is validated against R; GPU is
validated against CPU at the `GPU_FP32` tolerance tier for FP32
runs and to machine precision on CUDA FP64.

**`DataSource.to(device)`** — pay the host→device transfer once up
front, reach the compute ceiling on every subsequent fit. Rule-1
safe (explicit device mismatches raise). Underpins the amortised
numbers above.

**Whittle ARIMA** (`method='Whittle'`) — FFT-based approximate MLE
alongside CSS / ML / CSS-ML. Non-seasonal only in 1.8.0; `vcov`
returned as NaN (use ML/CSS-ML for SEs). CPU-only Whittle still
wins 1.4–17.5× over CSS-ML at n ≥ 2000 via precomputed cos/sin
tables; GPU Whittle hits 36× at n=1M.

**CPU multinomial analytical Hessian (backport).** The CPU vcov
step now uses the same block `X'·diag(Wⱼₖ)·X` formula the GPU
backend does, replacing the central-difference Hessian. **29–33×
CPU speedup** on the vcov step alone with no new dependencies.

Validation: 2,353 pystatistics tests, 117 R-vs-Python cross-validation.

### Previous Releases

**1.7.0** — Performance parity with R on OLS first-call (578 ms →
5 ms via lazy torch provenance probe), polr (277 ms → 23 ms via
vectorised `_cumulative_probs_vectorized`), and SARIMA airline-model
fit (2,100 ms → 14 ms via a numba-JIT'd Kalman state-space path +
factored-parameter optimisation + MA sign-convention fix). Added
`numba>=0.59` as a required dependency.

**1.6.2** — Re-shipped the 1.6.1 fixes after a release-process bug
left them out of the PyPI wheel. Closes five Rule 1 silent-failure
violations: ARIMA CSS-ML fails loud on refinement failure;
ARIMA(0,d,0) uses closed-form MLE; Gamma GLM returns explicit NaN
on non-positive dispersion; `descriptive.var(n=1)` returns NaN
without numpy warnings; scipy 1.18 forward-compat.

**1.6.0** — Five new modules (`ordinal`, `multinomial`, `multivariate`, `timeseries`, `gam`), two new GLM families (`Gamma`, `NegativeBinomial`), ~650 new tests.

**1.2.1** — No silent model switches; `backend='gpu'` is honest; reproducible Monte Carlo via `seed=`; module structure refactoring.

**1.1** — Named coefficients via `names=`; `result.coef` dict; OLS/Cox summary improvements matching R output.

---

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
- **GPU backends** are validated against CPU backends per the tolerances above.
- **Two-tier validation** ensures correctness at any scale: Python-CPU vs R, then Python-GPU vs Python-CPU.

## Installation

```bash
pip install pystatistics

# With GPU support (requires PyTorch)
pip install pystatistics[gpu]

# Development
pip install pystatistics[dev]
```

## License

MIT

## Author

Hai-Shuo (contact@sgcx.org)
