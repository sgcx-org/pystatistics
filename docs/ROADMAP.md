# PyStatistics Roadmap

**Last Updated:** February 2025

---

## Mission

PyStatistics aims to change two narratives:

1. **"Real statistics means R, SAS, SPSS, or Stata."** Python is dismissed as "ML stuff" — not for real statisticians, not for regulatory submissions. PyStatistics' CPU backends match R to machine precision (validated via R fixtures to rtol=1e-10), making Python a credible choice for rigorous applied statistics.

2. **"Too much data? Distribute it."** The reason "Big Data" spawned Hadoop and Spark is that standard tools choke at scale. But what if you could run regular statistics — not approximations, not sampling — on millions of observations in seconds? PyStatistics' GPU backends use PyTorch FP32 to scale exact statistical methods to datasets that would overwhelm R/SAS/SPSS/Stata.

---

## Architecture

Every module follows the same pattern:

```
DataSource → Domain Design → fit()/solve() → Backend.solve() → Result[Params] → Solution
```

- **CPU backends** are the gold standard, validated against R.
- **GPU backends** are validated against CPU backends.
- This two-tier validation ensures correctness at any scale.

Shared infrastructure lives in `core/`: DataSource, Result[P], device detection, timing, and linear algebra kernels (QR, Cholesky, SVD) in `core/compute/linalg/`.

---

## Module Status

### Completed

#### `regression/` — Linear Models (LM)
- **CPU backend**: QR with column pivoting, matches R `lm()` to 1e-12
- **GPU backend**: Cholesky normal equations on PyTorch FP32
- **Validated**: 10 fixture scenarios against R (well-conditioned, ill-conditioned, collinear, different scales, etc.)
- **API**: `fit(X, y) → LinearSolution`

#### `mvnmle/` — Multivariate Normal MLE with Missing Data
- **CPU backends**: Direct BFGS optimization + EM algorithm
- **GPU backends**: PyTorch for both direct and EM
- **Validated**: Against R `mvnmle` + `norm` packages
- **Includes**: Little's MCAR test, missingness pattern analysis
- **API**: `mlest(data) → MVNSolution`, `mlest(data, algorithm='em') → MVNSolution`

#### `regression/` — Generalized Linear Models (GLM)
- **Families**: Gaussian (identity), Binomial (logit), Poisson (log)
- **Algorithm**: IRLS (Iteratively Reweighted Least Squares / Fisher scoring)
- **CPU backend**: IRLS with QR inner solve, matches R `glm.fit()` to rtol=1e-7
- **GPU backend**: IRLS with torch WLS step (FP32), validated against CPU per README spec
- **Validated**: 9 fixture scenarios against R (basic, balanced, separated, zeros, large counts, scale tests)
- **API**: `fit(X, y, family='binomial') → GLMSolution`

### Planned

#### `descriptive/` — Descriptive Statistics
- **Priority**: HIGH (foundational — used by other modules and by users directly)
- **Scope**: Mean, variance, standard deviation, covariance matrices, correlation (Pearson, Spearman, Kendall), quantiles, skewness, kurtosis, summary tables
- **GPU applicability**: MODERATE — reduction operations (sum, sum-of-squares) parallelize well for large n; rank-based statistics (Spearman, Kendall) are harder
- **R validation**: `cor()`, `cov()`, `var()`, `summary()`, `quantile()`
- **Note**: Some of this overlaps with numpy. The value-add is (a) exact R matching for edge cases (e.g., Bessel correction, quantile types) and (b) GPU acceleration for large datasets

#### `hypothesis/` — Hypothesis Testing
- **Priority**: MEDIUM
- **Scope**: t-tests (one-sample, two-sample, paired), chi-squared tests, F-tests, Mann-Whitney U, Wilcoxon signed-rank, Kolmogorov-Smirnov, proportion tests, multiple testing corrections (Bonferroni, Holm, FDR/BH)
- **GPU applicability**: LOW for individual tests; HIGH for permutation tests and multiple testing corrections across thousands of features
- **R validation**: `t.test()`, `chisq.test()`, `wilcox.test()`, `ks.test()`, `p.adjust()`

#### `anova/` — Analysis of Variance
- **Priority**: MEDIUM (depends on `regression/`)
- **Scope**: One-way ANOVA, two-way ANOVA, repeated measures, ANCOVA, MANOVA, post-hoc tests (Tukey HSD, Bonferroni)
- **GPU applicability**: MODERATE — sums of squares computation on large datasets
- **R validation**: `aov()`, `anova()`, `TukeyHSD()`
- **Note**: ANOVA is fundamentally linear models. May be implemented as a thin wrapper on `regression/` with specialized output formatting and Type I/II/III SS

#### `survival/` — Survival Analysis
- **Priority**: MEDIUM-HIGH
- **Scope**:
  - Kaplan-Meier estimation and plotting
  - Cox proportional hazards (CPU only)
  - Discrete-time survival models (GPU-friendly)
  - Log-rank test, Schoenfeld residuals
- **GPU applicability**: Cox PH — **NO** (partial likelihood involves sorting and sequential risk-set updates that do not parallelize). Discrete-time survival — **YES** (reduces to logistic regression on person-period data, directly leverages `regression/` GLM infrastructure)
- **R validation**: `survival::coxph()`, `survival::survfit()`, `survival::Surv()`
- **Note**: This is a key reason discrete-time survival matters — it brings GPU acceleration to survival analysis, where Cox PH fundamentally cannot

#### `longitudinal/` — Longitudinal and Mixed Models
- **Priority**: LOW (complex, depends on `regression/` GLM)
- **Scope**:
  - Linear mixed models (LMM)
  - Generalized linear mixed models (GLMM)
  - GEE (Generalized Estimating Equations)
- **GPU applicability**: HIGH — iterative estimation with large cross-products at each step; the inner WLS/penalized LS loop in GLMM is where GPU wins
- **R validation**: `lme4::lmer()`, `lme4::glmer()`
- **Note**: Most complex module. LMM/GLMM estimation involves iterative optimization over variance components with a WLS or penalized least squares inner loop

#### `montecarlo/` — Monte Carlo Methods
- **Priority**: LOW
- **Scope**: Bootstrap (parametric, nonparametric), permutation tests, Monte Carlo simulation infrastructure, confidence intervals via resampling
- **GPU applicability**: VERY HIGH — embarrassingly parallel. Running 10,000 bootstrap replicates in parallel on GPU can be 100x+ faster than sequential CPU
- **R validation**: `boot` package
- **Note**: This is where GPU shines most. The entire selling point is massive parallelism

#### `timeseries/` — Time Series Analysis
- **Priority**: LOW
- **Scope**: TBD — likely ARIMA, state space models, spectral analysis, autocorrelation/partial autocorrelation
- **GPU applicability**: MODERATE — depends on specific methods; Kalman filtering has sequential dependencies but spectral analysis parallelizes well
- **R validation**: `arima()`, `forecast` package

---

## Implementation Priority

| Order | Module | Rationale |
|-------|--------|-----------|
| 1 | `regression/` GLM | Extends existing module; unlocks discrete-time survival and ANOVA |
| 2 | `descriptive/` | Foundational; every analysis starts with descriptive stats |
| 3 | `hypothesis/` | Natural companion to descriptive stats |
| 4 | `survival/` | Independent, high demand in biostatistics and clinical trials |
| 5 | `anova/` | Thin wrapper on `regression/`; straightforward once GLM exists |
| 6 | `longitudinal/` | Complex; depends on GLM being solid |
| 7 | `montecarlo/` | GPU showcase; useful but not a blocker |
| 8 | `timeseries/` | Lowest priority for v1 |

---

## Validation Strategy

Every module follows the same pattern:

1. **Generate test fixtures** — Python script creates CSV data + metadata JSON
2. **Run R reference** — R script computes results with 17-digit precision, saves as JSON
3. **Parametrized pytest** — Auto-discovers fixture pairs, runs against Python implementation
4. **CPU backend**: must match R to `rtol=1e-10` (near machine epsilon)
5. **GPU backend**: must match CPU per the tiered tolerances defined in `README.md`
6. **GPU ≡ CPU cross-validation** for every test case

This two-tier validation (Python-CPU vs R, then Python-GPU vs Python-CPU) ensures that GPU results are statistically valid even when FP32 precision differs from FP64.

GPU tolerances are defined centrally in the project README and enforced in test suites. See the "Statistical Equivalence: GPU vs CPU" section for the complete specification.

---

## Contributing

This is a living document. As modules are built, their status moves from **Planned → In Progress → Completed**. Each module's scope may evolve based on implementation experience and user feedback.
