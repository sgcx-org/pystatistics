# PyStatistics Roadmap

**Last Updated:** February 2026

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

#### `descriptive/` — Descriptive Statistics
- **CPU backend**: Mean, variance (Bessel-corrected), SD, covariance matrices, correlation (Pearson, Spearman via ranked Pearson, Kendall tau-b), all 9 R quantile types (Hyndman & Fan 1996), bias-adjusted skewness and kurtosis (e1071 type 2), six-number summary
- **GPU backend**: PyTorch FP32 for mean/var/SD/covariance/Pearson/skewness/kurtosis; CPU fallback for Spearman (scipy rankdata), Kendall (scipy kendalltau), quantiles (exact R matching)
- **Missing data**: R-compatible modes — `use='everything'` (propagate NaN), `'complete.obs'` (listwise deletion), `'pairwise.complete.obs'` (pairwise deletion for cov/cor)
- **Validated**: 10 fixture scenarios against R (basic, large-scale, scattered NaN, columnwise NaN, perfect correlation, ties, single column, constant column, extreme values, negative correlation) — 190 parametrized R validation tests at rtol=1e-10
- **API**: `describe(data)`, `cor(x)`, `cov(x)`, `var(x)`, `quantile(x, type=7)`, `summary(x)`

### Planned

#### `hypothesis/` — Hypothesis Testing
- **Priority**: MEDIUM
- **Scope**: t-tests (one-sample, two-sample, paired), chi-squared tests (Pearson, likelihood ratio), Fisher's exact test (2×2 and r×c), F-tests, Mann-Whitney U, Wilcoxon signed-rank, Kolmogorov-Smirnov, proportion tests, multiple testing corrections (Bonferroni, Holm, FDR/BH)
- **GPU applicability**: LOW for individual asymptotic tests (chi-squared, t); **HIGH for exact tests on larger tables** — Fisher's exact test for r×c tables requires enumerating or simulating tables with fixed marginals, which is combinatorially expensive on CPU but parallelizes naturally on GPU. Monte Carlo exact p-values (sampling random tables under the null) are embarrassingly parallel, same pattern as permutation tests.
- **R validation**: `t.test()`, `chisq.test()`, `fisher.test()`, `wilcox.test()`, `ks.test()`, `p.adjust()`
- **Note**: For 2×2 tables, Fisher's exact test is trivial (direct hypergeometric computation, no GPU needed). The GPU value appears for r×c tables where the network algorithm or Monte Carlo simulation of exact p-values becomes expensive. Chi-squared remains important as the asymptotic baseline for very large tables where even GPU exact tests become infeasible.

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

#### `regression/` — Linear and Generalized Linear Mixed Models (LMM / GLMM)
- **Priority**: LOW (most complex extension of regression; depends on GLM being solid)
- **Scope**: LMM (random intercepts, random slopes, nested/crossed designs), GLMM (Binomial, Poisson families with random effects)
- **Algorithm**: Outer optimization over variance components (profiled deviance for LMM, penalized quasi-likelihood or adaptive Gauss-Hermite quadrature for GLMM), with an inner penalized IRLS loop reusing the existing family/link/IRLS infrastructure from GLM
- **GPU applicability**: HIGH — the inner penalized WLS step at each iteration is dense linear algebra on (X|Z) augmented design matrices, same GPU pattern as GLM
- **R validation**: `lme4::lmer()`, `lme4::glmer()`
- **Integration point**: `regression/backends/cpu_lmm.py`, `gpu_lmm.py`, `cpu_glmm.py`, `gpu_glmm.py`. Shares `families.py`, links, and IRLS convergence machinery from GLM
- **Note**: Most complex planned extension of `regression/`. LMM is the natural entry point (profiled deviance is well-understood); GLMM adds the family/link layer on top

#### `regression/` — Generalized Estimating Equations (GEE)
- **Priority**: LOW (depends on GLM; simpler than LMM/GLMM but less commonly requested)
- **Scope**: Marginal models for correlated data with working correlation structures (independence, exchangeable, AR(1), unstructured)
- **Algorithm**: Modified IRLS — same score equations as GLM but with a working correlation matrix in the weight step, plus sandwich (robust) covariance estimation
- **GPU applicability**: MODERATE — IRLS inner loop same as GLM; the sandwich covariance computation involves cluster-level sums that parallelize on GPU
- **R validation**: `geepack::geeglm()`
- **Note**: GEE estimates population-averaged (marginal) parameters, vs LMM/GLMM which estimate subject-specific (conditional) parameters. Different inferential targets, same `regression/` home. Not inherently longitudinal — handles any correlated data (clustered, spatial, repeated measures)

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
| ~~1~~ | ~~`regression/` GLM~~ | ~~Extends existing module; unlocks discrete-time survival and ANOVA~~ ✅ |
| ~~2~~ | ~~`descriptive/`~~ | ~~Foundational; every analysis starts with descriptive stats~~ ✅ |
| 3 | `hypothesis/` | Natural companion to descriptive stats |
| 4 | `survival/` | Independent, high demand in biostatistics and clinical trials |
| 5 | `anova/` | Thin wrapper on `regression/`; straightforward once GLM exists |
| 6 | `regression/` LMM/GLMM | Complex; extends GLM with random effects. Reuses IRLS infrastructure |
| 7 | `montecarlo/` | GPU showcase; useful but not a blocker |
| 8 | `timeseries/` | Lowest priority for v1 |

---

## GPU Solver Improvements

The current GPU backends use direct solvers: Cholesky normal equations for LM, `torch.linalg.lstsq()` for GLM. These work well for current scales but have known limitations that become important as problems grow. The two improvements below are future work — documented here so the design is captured when the need arises.

### Iterative Solvers (CGLS / LSMR)

- **Priority**: LOW (current Cholesky + lstsq fallback handles all existing test cases)
- **What**: Add CGLS (Conjugate Gradient Least Squares) and LSMR as alternative GPU solvers in `core/compute/linalg/iterative.py`
- **Why**: Cholesky requires forming X'X explicitly, which squares the condition number and uses O(p²) memory for the Gram matrix. Iterative methods solve the normal equations *implicitly* — working only with matrix-vector products X·v and X'·w — without ever materializing X'X. This is both more memory-efficient and numerically better-conditioned.
- **Key insight**: "Exact Estimator, Different Solver." CGLS and LSMR converge to the same OLS solution as Cholesky. The statistical estimator (β = (X'X)⁻¹X'y) doesn't change; only the numerical path to computing it changes. All existing validation and tolerance specifications apply unchanged.
- **Tradeoffs**: Iterative solvers require a convergence tolerance (e.g., `tol=1e-7`) and may need preconditioning for fast convergence. For well-conditioned, moderate-sized problems, Cholesky is simpler and faster. Iterative methods become compelling when n > 10M, when p is large enough that X'X is expensive, or as a foundation for future sparse regression support.
- **Integration point**: GPU backends (`regression/backends/gpu.py`, `gpu_glm.py`) would gain a `solver='iterative'` option alongside the existing `'cholesky'` and `'lstsq'` paths

### Batched Multi-Problem Regression

- **Priority**: Implement when `montecarlo/` module begins
- **What**: Solve X·B = Y where Y is n×k (k = thousands of resampled response vectors) in a single batched GPU call
- **Why**: Bootstrap and permutation tests require thousands of regressions sharing the same design matrix X but with different response vectors. Currently each would require a separate `fit()` call. Batched solve runs all replicates in a single kernel launch.
- **Key insight**: X'X and its Cholesky factorization are computed once. Only X'y changes per replicate — the marginal cost per additional replicate is one matrix-vector product (X'yᵢ) plus one triangular solve. 10,000 bootstrap replicates run in approximately the time of 2 sequential regressions.
- **Integration point**: New batched solve primitives in `core/compute/linalg/`, consumed by the `montecarlo/` module's bootstrap and permutation infrastructure
- **Note**: This is the GPU "killer feature" for Monte Carlo methods. The embarrassingly parallel structure of resampling-based inference maps perfectly onto GPU batch parallelism. This is a stronger argument for GPU than any single regression speedup.

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
