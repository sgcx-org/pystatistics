# PyStatistics

GPU-accelerated statistical computing for Python.

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
result = fit(X, y)
print(result.summary())

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
```

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `regression/` LM | Complete | Linear models (OLS) with CPU QR and GPU Cholesky |
| `regression/` GLM | Complete | Generalized linear models (Gaussian, Binomial, Poisson) via IRLS |
| `mvnmle/` | Complete | Multivariate normal MLE with missing data (Direct + EM) |
| `descriptive/` | Complete | Descriptive statistics, correlation, quantiles, skewness, kurtosis |
| `hypothesis/` | Complete | t-test, chi-squared, Fisher exact, Wilcoxon, KS, proportions, F-test, p.adjust |
| `montecarlo/` | Complete | Bootstrap (ordinary, balanced, parametric), permutation tests, 5 CI methods, batched GPU solver |
| `survival/` | Complete | Survival analysis: Kaplan-Meier, log-rank test, Cox PH (CPU), discrete-time (GPU) |
| `anova/` | Complete | ANOVA: one-way, factorial, ANCOVA, repeated measures, Type I/II/III SS, Tukey/Bonferroni/Dunnett, Levene's test |
| `regression/` LMM/GLMM | Planned | Linear and generalized linear mixed models |

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
