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

#### `hypothesis/` — Hypothesis Testing
- **CPU backend**: t-test (one-sample, two-sample Welch/pooled, paired), Pearson's chi-squared (independence with Yates, GOF), Fisher's exact test (2×2 with conditional MLE OR + CI, r×c Monte Carlo), Wilcoxon signed-rank and rank-sum (exact + normal approximation, Hodges-Lehmann CI), Kolmogorov-Smirnov (one-sample against distributions, two-sample), proportion test (chi-squared based with Wilson score CI), F-test for variances, p.adjust (8 methods: none, bonferroni, holm, hochberg, hommel, BH, BY, fdr)
- **GPU backend**: Monte Carlo simulation only (chi-squared independence/GOF, Fisher r×c). Hybrid approach: Patefield's algorithm on CPU for table generation, batched statistic computation on GPU. All scalar tests (t, Wilcoxon, KS, prop, var) are CPU-only by design — GPU overhead would dominate O(n) operations.
- **R `htest` structure**: All tests return `HTestParams` matching R's `htest` class (statistic, parameter, p.value, conf.int, estimate, null.value, alternative, method, data.name)
- **Validated**: 18 fixture scenarios against R 4.5.2 — 71 parametrized R validation tests; t-test/chi-squared/prop/KS/var at rtol=1e-10, Fisher OR/CI at rtol=2e-2 (Brent solver vs R exact), Wilcoxon CI algorithmically different (Walsh averages vs R's uniroot)
- **API**: `t_test(x, y)`, `chisq_test(x)`, `fisher_test(x)`, `wilcox_test(x, y)`, `ks_test(x, y)`, `prop_test(x, n)`, `var_test(x, y)`, `p_adjust(p)`

#### `montecarlo/` — Monte Carlo Methods
- **CPU backend**: Bootstrap resampling (ordinary, balanced, parametric with ran_gen/mle), permutation testing (two-sample, Phipson-Smyth corrected p-values), five bootstrap CI methods (normal, basic, percentile, BCa, studentized), jackknife influence values for BCa acceleration
- **GPU backend**: Falls back to CPU for arbitrary user statistics (Python functions cannot run on GPU). GPU acceleration is via the batched multi-problem OLS solver in `core/compute/linalg/batched.py` — one Cholesky factorization for X'X, then k triangular solves for k bootstrap replicates. 10,000 bootstrap regression replicates ≈ time of 2 sequential regressions.
- **Batched solver**: `batched_ols_solve(X, Y, device)` in `core/compute/linalg/batched.py`. CPU: Cholesky + triangular solve via scipy. GPU: PyTorch Cholesky + torch.linalg.solve_triangular in FP32. Validated against individual `np.linalg.lstsq` to rtol=1e-10 (CPU) and rtol=1e-4 (GPU FP32).
- **R `boot` API match**: `boot(data, statistic, R)` where statistic receives `(data, indices)` and returns a 1D array. Three stype modes: "i" (indices), "f" (frequencies), "w" (weights). Stratified resampling. `boot_ci()` with all 5 CI types matching R's `boot.ci()`.
- **Validated**: 10 fixture scenarios against R's `boot` package — 37 parametrized R validation tests; t0 (observed statistic) at rtol=1e-10 (deterministic), bias/SE at moderate tolerance (stochastic — different RNGs), CI endpoints at abs=0.5 (stochastic), permutation p-values at abs=0.05 (stochastic). 133 total tests including unit tests.
- **API**: `boot(data, statistic, R)`, `boot_ci(boot_out, type='all')`, `permutation_test(x, y, statistic, R)`

#### `survival/` — Survival Analysis
- **Kaplan-Meier estimator**: Product-limit S(t) = ∏(1 - d_j/n_j), Greenwood standard errors, three CI transformations (plain, log, log-log). Matches R `survival::survfit()` to rtol=1e-10.
- **Log-rank test (G-rho family)**: Standard log-rank (rho=0, Mantel-Haenszel), Peto & Peto (rho=1, Gehan-Wilcoxon modification). Supports K-group comparisons with chi-squared statistic via variance-covariance matrix. Matches R `survival::survdiff()`.
- **Cox PH**: CPU only — no `backend=` parameter. Partial likelihood with Newton-Raphson, Efron's (default) and Breslow's approximations for tied event times. Numerical stability via eta centering and step-size limiting. Harrell's C-statistic. Matches R `survival::coxph()` to rtol=1e-4.
- **Discrete-time survival**: GPU-accelerated. Converts time-to-event data into person-period format, fits logistic regression via `regression.fit(X, y, family='binomial', backend=backend)`. This is the GPU pathway for survival analysis — Cox PH's sequential risk-set updates do not parallelize, but discrete-time reduces to standard GLM.
- **Validated**: 13 fixture scenarios against R's `survival` package — 59 parametrized R validation tests; KM S(t)/SE/CI at rtol=1e-10/1e-6, log-rank statistic/p-value at rtol=1e-4/1e-3, Cox coef/SE/HR/loglik at rtol=1e-4/1e-3. 183 total tests including unit tests.
- **API**: `kaplan_meier(time, event)`, `survdiff(time, event, group)`, `coxph(time, event, X)`, `discrete_time(time, event, X, backend='auto')`

#### `anova/` — Analysis of Variance
- **One-way ANOVA**: `anova_oneway(y, group)` with Type I/II/III sums of squares, all giving identical results for one-way designs. Matches R `aov()` and `car::Anova()` to rtol=1e-10.
- **Factorial ANOVA**: `anova(y, factors, ss_type=2)` for multi-factor designs with interactions. Type I (sequential), Type II (marginal, respects marginality), Type III (each term last, deviation coding). Matches R `anova(lm(...))`, `car::Anova(..., type="II")`, and `car::Anova(..., type="III")` with `contr.sum` to rtol=1e-10.
- **ANCOVA**: Continuous covariates via `anova(y, factors, covariates={'age': age})`. Matches R `car::Anova(lm(y ~ group + x), type="II")`.
- **Repeated measures**: `anova_rm(y, subject, within)` with long-format input. Mauchly's sphericity test, Greenhouse-Geisser and Huynh-Feldt epsilon corrections, mixed designs (between + within). Matches R `aov(y ~ cond + Error(subj/cond))`.
- **Post-hoc tests**: `anova_posthoc(result, method='tukey')` — Tukey HSD (studentized range), Bonferroni pairwise (corrected t-tests), Dunnett (many-to-one vs control). Tukey matches R `TukeyHSD()` to rtol=1e-4.
- **Effect sizes**: eta-squared and partial eta-squared for all terms. Matches R `effectsize::eta_squared()`.
- **Levene's test**: `levene_test(y, group)` with `center='median'` (Brown-Forsythe) or `center='mean'` (original Levene). Matches R `car::leveneTest()` to rtol=1e-8.
- **Architecture**: Thin wrapper on `regression/` — all SS computation delegates to `regression.fit()` and compares RSS values. No new solver math, no GPU backend (ANOVA designs are small).
- **Validated**: 11 fixture scenarios against R (one-way balanced/unbalanced, factorial balanced/unbalanced, ANCOVA, Levene, Tukey HSD, Bonferroni, repeated measures within/mixed, eta-squared) — 46 parametrized R validation tests plus 135 unit tests. Total: 181 ANOVA tests.
- **API**: `anova_oneway(y, group)`, `anova(y, factors)`, `anova_rm(y, subject, within)`, `anova_posthoc(result)`, `levene_test(y, group)`

### Planned

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
| ~~3~~ | ~~`hypothesis/`~~ | ~~Natural companion to descriptive stats~~ ✅ |
| ~~4~~ | ~~`montecarlo/`~~ | ~~GPU showcase; general resampling inference~~ ✅ |
| ~~5~~ | ~~`survival/`~~ | ~~Independent, high demand in biostatistics and clinical trials~~ ✅ |
| ~~6~~ | ~~`anova/`~~ | ~~Thin wrapper on `regression/`; straightforward once GLM exists~~ ✅ |
| 7 | `regression/` LMM/GLMM | Complex; extends GLM with random effects. Reuses IRLS infrastructure |
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

### Batched Multi-Problem Regression ✅

- **Status**: Implemented in `core/compute/linalg/batched.py`
- **What**: Solve X·B = Y where Y is n×k (k = thousands of resampled response vectors) in a single batched call
- **Implementation**: `batched_ols_solve(X, Y, device='auto')` computes X'X and Cholesky factorization once, then solves k right-hand sides via triangular solve. CPU path uses scipy `solve_triangular`; GPU path uses `torch.linalg.cholesky_ex` + `torch.linalg.solve_triangular` in FP32.
- **Key insight**: X'X and its Cholesky factorization are computed once. Only X'y changes per replicate — the marginal cost per additional replicate is one matrix-vector product (X'yᵢ) plus one triangular solve. 10,000 bootstrap replicates run in approximately the time of 2 sequential regressions.
- **Validated**: CPU matches `np.linalg.lstsq` to rtol=1e-10 for all k columns. GPU matches CPU to rtol=1e-4. 11 tests including residual bootstrap use case with R=1000 replicates.
- **Consumed by**: `montecarlo/` module's bootstrap and permutation infrastructure for regression bootstrap (residual bootstrap with shared X, k resampled response vectors)

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
