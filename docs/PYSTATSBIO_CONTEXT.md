# Context Document for PyStatsBio (and SGC-Bio)

**Purpose:** This document gives a future Claude Code session everything it needs to build `pystatsbio` — a Python package for biotech/pharma statistical computing. It describes what PyStatsBio is, how it relates to SGC-Bio, and exactly what it can import from `pystatistics`.

---

## What Is PyStatsBio?

PyStatsBio is a Python package for **biotech and pharmaceutical statistical computing** — the methods used across the drug development pipeline, from preclinical assay analysis through clinical trial design.

It is a separate package (separate repo, separate `pip install`) that depends on `pystatistics` for its general statistical computing layer.

Think of it as the domain-specific layer:
- `pystatistics` = general statistical computing (regression, survival, mixed models, tests)
- `pystatsbio` = biotech/pharma-specific methods that *use* those tools

The relationship is analogous to R's ecosystem: R ships `survival`, `lme4`, `stats` — and then domain packages like `drc`, `pROC`, `gsDesign`, `rpact` build pharma-specific functionality on top.

### What PyStatsBio Is NOT

- Not a GUI or dashboard (that's SGC-Bio's job)
- Not a clinical data management system (no EDC, no CDISC-SDTM conversion)
- Not a reporting tool (no RTF/PDF table generation — that's SGC-Bio's job)
- Not a regulatory submission builder
- Not a bioinformatics/genomics pipeline (no sequence alignment, no variant calling)
- Not a second engine — **PyStatsBio v1 should be smaller than PyStatistics** (~4-6k lines vs ~10k). If it grows larger, you're building a second engine and something went wrong.

### Target Users

Scientists and biostatisticians at 5-20 person Kendall Square biotechs who daily:
- Fit dose-response curves
- Run PK summaries
- Calculate sample sizes / power
- Evaluate biomarkers
- Run survival models

They do NOT typically:
- Design Lan-DeMets adaptive boundaries from scratch
- Implement Bretz graphical multiplicity procedures by hand
- Run population PK with NONMEM-grade NLME

Those are outsourced to CRO statisticians. The early modules must align with real daily workflows.

---

## What Is SGC-Bio?

SGC-Bio is a **web application** (built on the SGC platform) that provides a user-facing interface for biotech/pharma statistical computing. It sits on top of PyStatsBio:

```
┌─────────────────────────────┐
│  SGC-Bio (Web App)          │   ← user-facing UI, tables, reports, GPU infra
├─────────────────────────────┤
│  PyStatsBio (Package)       │   ← biotech/pharma statistical methods
├─────────────────────────────┤
│  PyStatistics (Package)     │   ← general statistical computing
└─────────────────────────────┘
```

This is the same pattern as:
- NumPy → SciPy → scikit-learn
- torch → transformers → HuggingFace Hub
- TensorFlow → Keras → enterprise ML stack

SGC-Bio matters for PyStatsBio's design because:
1. PyStatsBio's API must be clean enough for SGC-Bio to call programmatically
2. PyStatsBio should return structured results (not just print output) so SGC-Bio can render them in tables/reports
3. PyStatsBio should NOT assume interactive use — no plots, no print-to-console
4. **SGC-Bio provides GPU infrastructure** — cloud GPUs for compute-intensive modules (high-throughput screening, population PK). PyStatsBio should expose `backend=` parameters where GPU acceleration is meaningful.

The SGC-Bio layer will handle:
- Table formatting (regulatory-grade output, CDISC-style TLFs)
- PDF/RTF report generation
- Interactive parameter exploration (UI sliders for sample size, power curves)
- Study protocol templates
- Cloud GPU provisioning for heavy compute
- Integration with SGC's broader platform

---

## Release Phases

### CRITICAL: Build Like a Founder, Not a PhD

PyStatsBio must ship in phases. Not all modules are created equal. Some are bounded and high-leverage; others are multi-month research projects that could sink the project.

---

### Phase 1: Wedge Release (Build This First)

These four modules cover trial planning, preclinical HTS, biomarker validation, and PK summary. That alone serves the core daily workflow of a Kendall Square biotech.

**Target: ~4,000 lines. Ship this before touching anything else.**

#### `power/` — Sample Size and Power Calculations

The bread and butter of clinical trial planning. Every trial starts with "how many subjects do we need?"

**Scope:**
- **Two-sample tests**: t-test (equal/unequal variance), proportion test (chi-squared, Fisher), rate comparison (Poisson)
- **Paired tests**: paired t-test, McNemar's test
- **ANOVA**: one-way, factorial
- **Survival**: log-rank test (Schoenfeld formula, Freedman formula, Lachin-Foulkes)
- **Non-inferiority / equivalence / superiority**: all three framings for means and proportions
- **Crossover designs**: 2x2 crossover (bioequivalence)
- **Cluster randomized trials**: design effect, ICC-adjusted sample size
- **Multi-arm trials**: Dunnett-style many-to-one comparisons

**R packages to match:** `pwr`, `TrialSize`, `gsDesign` (power functions), `samplesize`, `PowerTOST`

**CPU-only.** Solving one equation — microseconds on CPU.

**Key design principle:** Each function should support "solve for any one parameter given the others":
```python
# Solve for n (given effect size, alpha, power)
result = power_t_test(d=0.5, alpha=0.05, power=0.80)
print(result.n)  # 64 per group

# Solve for power (given n, effect size, alpha)
result = power_t_test(n=50, d=0.5, alpha=0.05)
print(result.power)  # 0.697

# Solve for detectable effect size (given n, alpha, power)
result = power_t_test(n=50, alpha=0.05, power=0.80)
print(result.d)  # 0.569
```

#### `doseresponse/` — Dose-Response Modeling

The workhorse of preclinical pharmacology. Every in vitro assay, every toxicology study. **There is no good modern Python equivalent to R's `drc` — this is the killer wedge.**

**Scope:**
- **4-parameter logistic (4PL)**: the standard sigmoidal dose-response curve. `Bottom + (Top - Bottom) / (1 + (EC50/x)^Hill)`
- **5-parameter logistic (5PL)**: asymmetric 4PL with extra shape parameter
- **Log-logistic**: `drc`-style LL.4, LL.5 models
- **Weibull models**: W1.4, W2.4 (for asymmetric dose-response)
- **Brain-Cousens hormesis models**: biphasic dose-response with low-dose stimulation
- **EC50/IC50 estimation**: with delta method confidence intervals
- **Relative potency**: ratio of EC50s with Fieller's CI
- **BMD (benchmark dose)**: BMDL/BMDU computation for toxicology
- **Model comparison**: AIC/BIC/lack-of-fit F-test for model selection
- **High-throughput screening (HTS)**: fit thousands of dose-response curves simultaneously (**GPU**: batch 4PL fitting across compounds)

**R packages to match:** `drc`, `nplr`, `BMDS` (EPA benchmark dose software)

**GPU: Yes.** This is the primary GPU showcase. HTS campaigns generate thousands of compounds x multiple doses. Fitting a 4PL to each is independent — perfect for GPU batching. The inner solver is Levenberg-Marquardt nonlinear least squares, which itself can be GPU-accelerated (batched Jacobian computation, batched normal equations).

```python
# Single curve (CPU)
result = fit_4pl(dose, response)
print(result.ec50, result.hill, result.top, result.bottom)

# High-throughput: fit 10,000 curves at once (GPU)
results = fit_4pl_batch(dose_matrix, response_matrix, backend='gpu')
print(results.ec50)  # array of 10,000 EC50 values
```

**The real GPU killer feature of SGC-Bio is batched nonlinear curve fitting.** R is not GPU-native. Torch is. If you make HTS 4PL fitting absurdly fast, that's flashy *and* practical.

#### `diagnostic/` — Diagnostic Accuracy

Evaluating biomarkers, screening tests, and diagnostic tools. Python's ROC ecosystem is weak. DeLong test especially.

**Scope:**
- **ROC analysis**: empirical ROC curve, AUC with DeLong confidence intervals, optimal cutoff (Youden index, closest-to-corner)
- **ROC comparison**: DeLong test for comparing two correlated ROC curves
- **Sensitivity / specificity**: point estimates with exact (Clopper-Pearson) confidence intervals
- **Predictive values**: PPV, NPV with prevalence adjustment
- **Likelihood ratios**: LR+, LR- with confidence intervals
- **Diagnostic odds ratio**: with confidence interval
- **Multi-class**: extension to >2 categories (multi-class AUC)
- **High-throughput panel**: evaluate hundreds/thousands of biomarker candidates simultaneously (**GPU**: batch AUC computation across markers)

**R packages to match:** `pROC`, `OptimalCutpoints`, `epiR`

**GPU: Optional.** Single ROC curves are CPU-fine. Batch AUC over thousands of biomarkers in HTS benefits from GPU.

#### `pk/` (NCA only) — Non-Compartmental Pharmacokinetic Analysis

NCA is required for every PK study. Self-contained, well-defined, formulaic calculations. **Do NCA in Phase 1. Do NOT jump to compartmental/PopPK yet.**

**Scope (NCA only):**
- **AUC**: linear trapezoidal, log-linear trapezoidal, linear-up/log-down
- **Cmax, Tmax**: peak concentration and time to peak
- **Half-life**: terminal elimination rate constant via log-linear regression
- **Clearance**: CL = Dose / AUC
- **Volume of distribution**: Vz = Dose / (lambda_z * AUC)
- **PK summary statistics**: geometric means and CVs (standard for PK data), confidence intervals on log-scale parameters
- **Bioequivalence PK**: Cmax and AUC ratio analysis (links to Phase 2 `equivalence/`)

**R packages to match:** `PKNCA`, `NonCompart`

**CPU-only for NCA.** Always small data.

```python
# NCA (CPU - always small data)
result = nca(time, concentration, dose=100, route='ev')
print(result.auc_inf, result.cmax, result.half_life, result.clearance)
```

---

### Phase 2: Additive Modules (Clean, Bounded)

These are safe, bounded, and add real value. Build after Phase 1 ships.

#### `agreement/` — Inter-Rater Agreement and Method Comparison

Critical for assay validation and analytical method bridging.

**Scope:**
- **Cohen's kappa**: unweighted, linear-weighted, quadratic-weighted; with SE and CI
- **Fleiss' kappa**: multi-rater extension
- **ICC**: intraclass correlation wrapper with all 6 Shrout-Fleiss forms: ICC(1,1), ICC(2,1), ICC(3,1), ICC(1,k), ICC(2,k), ICC(3,k). Delegates to `pystatistics.mixed.lmm()` internally.
- **Bland-Altman**: bias, limits of agreement (+/-1.96 SD), with confidence intervals; proportional bias detection; repeated measures extension
- **Concordance correlation coefficient (CCC)**: Lin's CCC with CI
- **Total deviation index (TDI)**: and coverage probability (CP)

**R packages to match:** `irr`, `BlandAltmanLeh`, `psych::ICC`, `DescTools::CCC`

**CPU-only.** Small-n rater studies.

#### `equivalence/` — Bioequivalence and Non-Inferiority

Specialized inference for showing treatments are "close enough" rather than "different." Critical for generic drug approval.

**Scope:**
- **TOST (Two One-Sided Tests)**: for means (bioequivalence)
- **Schuirmann's test**: standard bioequivalence test
- **2x2 crossover analysis**: period effects, carryover effects, sequence effects
- **Non-inferiority margins**: for means, proportions, survival (hazard ratios)
- **Equivalence of proportions**: Farrington-Manning, Miettinen-Nurminen
- **Ratio tests**: for geometric means (log-scale analysis, common in PK studies)

**R packages to match:** `PowerTOST`, `equivalence`, `TOSTER`

**CPU-only.** Small-n crossover studies.

#### `epi/` — Epidemiological Measures

Common in safety analyses and observational components of clinical programs.

**Scope:**
- **Relative risk (RR)**: with Wald and score CIs
- **Odds ratio (OR)**: Woolf, Gart, conditional MLE (already in `pystatistics.hypothesis.fisher_test`, but clinical wrapper)
- **Risk difference (RD)**: with Newcombe CI, Miettinen-Nurminen CI
- **Number needed to treat (NNT)**: with CI (from RD)
- **Incidence rate ratio**: with exact CI
- **Stratified analysis**: Mantel-Haenszel OR/RR, Breslow-Day test for homogeneity
- **Matched pair analysis**: McNemar's test (already in hypothesis), conditional logistic regression wrapper

**R packages to match:** `epiR`, `epitools`, `Epi`

**CPU-only.** Contingency tables, small-n.

**NOTE: Propensity score methods (matching, IPTW) are explicitly OUT of scope for v1.** Once you add matching + IPTW + stabilized weights + balance diagnostics, you're in causal inference land — that's a whole ecosystem. Keep it minimal.

---

### Phase 3: Extensions (Build After Revenue Exists)

These modules are valuable but either depend on Phase 1/2 predecessors or require significant effort.

#### `assay/` — Assay Validation and Analytical Methods

Bioanalytical method validation for regulated environments. Pulls from `doseresponse/` and `pystatistics.mixed.lmm()`.

**Scope:**
- **Linearity assessment**: weighted regression, residual analysis, lack-of-fit test
- **Precision**: repeatability (within-run), intermediate precision (between-run), reproducibility. Uses nested ANOVA or mixed models from `pystatistics.mixed.lmm()`
- **Accuracy**: bias, recovery, percent relative error
- **Limit of detection (LOD)** and **limit of quantitation (LOQ)**: signal-to-noise, calibration curve, blank-based methods
- **Parallelism testing**: for immunoassays (PLA — parallel line analysis)
- **Stability**: real-time and accelerated (Arrhenius), shelf-life estimation via regression
- **Standard curve fitting**: 4PL/5PL (shared with `doseresponse/`), back-calculation of concentrations from standard curves

**R packages to match:** No single R package dominates here — scattered across `drc`, custom code, and commercial tools (SoftMax Pro, Watson LIMS). This is a real gap.

**GPU: Optional** for high-throughput plate processing (batch curve fits).

#### `pd/` — Pharmacodynamic Modeling (Direct Effect Only in v1)

PD is "what the drug does to the body." **v1: Emax and sigmoid Emax only.** Indirect response models and PK/PD link models are Phase 4.

**Scope (v1):**
- **Emax model**: direct effect. `E0 + Emax * C / (EC50 + C)`
- **Sigmoid Emax**: with Hill coefficient
- **Exposure-response (E-R)**: logistic regression with exposure metrics as predictors (uses `pystatistics.regression.fit(family='binomial')`)

**R packages to match:** `mrgsolve` (simulation), `RxODE`

**GPU: Deferred.** Direct Emax is just nonlinear regression — CPU is fine. GPU matters for simulation-based analyses which are Phase 4.

#### `multiplicity/` — Multiple Testing in Clinical Trials (Simple Procedures Only in v1)

Standard `p.adjust` isn't enough for regulatory work. But the full graphical approach is complex.

**Scope (v1 — simple only):**
- **Hierarchical (fixed-sequence)**: test in pre-specified order
- **Bonferroni-Holm with weights**: weighted versions for unequal importance
- **Fallback procedures**: for primary/secondary endpoint hierarchies

**Explicitly deferred to v2:** Bretz-Maurer-Brannath graphical approach, gatekeeping, closed testing. These are algorithmically complex (graph-based weight propagation, dynamic alpha redistribution, edge-case heavy) and need extreme validation care.

**R packages to match:** `gMCP` (v2), `multcomp`

**CPU-only.**

---

### Phase 4: Danger Zones (DO NOT BUILD IN v1)

These are explicitly out of scope for initial releases. Each is a multi-month research-grade project.

#### `pk/` (Population PK — NLME / SAEM) — DO NOT BUILD EARLY

**Why this is dangerous:** This is NONMEM territory. You are now in:
- Nonlinear mixed effects models
- ODE systems + random effects on parameters
- Stochastic EM (SAEM) algorithm
- Likelihood approximations (Laplace, adaptive Gaussian quadrature)
- High-dimensional integration
- Boundary constraints on variance components
- Convergence pathologies

This is a research-grade problem. R's `saemix` is ~8,000 lines. NONMEM has been developed for 40+ years. Monolix is a commercial product by a funded team.

**Do not put this in v1. This is v2 or v3 after revenue exists.**

When you do build it:
- **GPU is the killer feature**: each subject's PK profile = independent ODE solve. 500-5,000 subjects = embarrassingly parallel. SAEM simulation step also parallel.
- **CPU**: `scipy.integrate.solve_ivp` with RK45 or LSODA
- **GPU**: `torchdiffeq` for batched forward solves
- **ODE solver is an implementation detail** — user specifies the PK model, not the integration method. Do NOT expose solver knobs (RK45 vs BDF vs adjoint sensitivity). The user says `pk_model("2cmt_oral")`, not `solve_ivp(method='RK45')`.

#### `adaptive/` — Group Sequential and Adaptive Designs — DO NOT BUILD EARLY

**Why this is dangerous:** On paper it's "just alpha spending." In practice:
- Recursive boundary computation with numerical integration
- Simulation validation for type I error guarantees
- Subtle regulatory requirements
- Heavy statistical literature (Lan-DeMets, Hwang-Shih-DeCani)

This requires extreme validation care and deep domain expertise.

**Scope (when built):**
- Group sequential boundaries: O'Brien-Fleming, Pocock, alpha spending functions (Lan-DeMets)
- Alpha spending: Hwang-Shih-DeCani family, custom spending functions
- Futility boundaries: binding and non-binding
- Information fractions: equal and unequal spacing
- Sample size re-estimation: Chen-DeMets-Lan (promising zone), conditional power

**R packages to match:** `gsDesign`, `rpact`, `GroupSeq`

#### `pd/` (Indirect Response, PK/PD Link) — DO NOT BUILD EARLY

Jusko's 4 indirect response models, turnover models, effect compartment, hysteresis correction. These require ODE infrastructure from PopPK. Build after PopPK exists.

#### Full Graphical Multiplicity (Bretz-Maurer-Brannath) — DO NOT BUILD EARLY

The graph-based weight propagation approach is the regulatory standard for multi-endpoint trials, but it's algorithmically complex with many edge cases. Defer to v2.

---

## GPU Strategy Summary

| Module | GPU? | Phase | Rationale |
|--------|------|-------|-----------|
| `power/` | No | 1 | Solving one equation — microseconds on CPU |
| `doseresponse/` | **Yes** | 1 | Batch 4PL fitting: thousands of compounds x multiple doses. **The GPU showcase.** |
| `diagnostic/` | Optional | 1 | Batch AUC over thousands of biomarkers in HTS |
| `pk/` (NCA) | No | 1 | Formulaic, small data |
| `agreement/` | No | 2 | Small-n rater studies |
| `equivalence/` | No | 2 | Small-n crossover studies |
| `epi/` | No | 2 | Contingency tables, small-n |
| `assay/` | Optional | 3 | Batch curve fitting for high-throughput plate processing |
| `pd/` (Emax) | No | 3 | Direct Emax is just nonlinear regression |
| `multiplicity/` | No | 3 | Graph algorithms, small-n |
| `pk/` (PopPK) | **Yes** | 4+ | Parallel ODE solves across subjects. Killer use case. |
| `pd/` (indirect) | **Yes** | 4+ | PK/PD simulation: parallel ODE + likelihood |
| `adaptive/` | No | 4+ | Boundaries are recursive but small |

Modules with `backend=` parameter in v1: `doseresponse/`, and optionally `diagnostic/`.

All other v1 modules are CPU-only — no `backend=` parameter.

---

## What PyStatsBio Can Import From PyStatistics

This is the critical section. PyStatsBio should treat `pystatistics` as a dependency and import freely. Here is the complete public API available:

### `pystatistics.regression`

```python
from pystatistics.regression import fit, Design, Family, Gaussian, Binomial, Poisson

# fit(X, y, *, family=None, backend='auto', force=False, tol=1e-8, max_iter=25)
#   family=None -> OLS
#   family='binomial' -> logistic regression
#   family='poisson' -> Poisson regression
#   Returns: LinearSolution | GLMSolution
#
# LinearSolution properties:
#   .coefficients, .residuals, .fitted_values, .rss, .tss,
#   .r_squared, .adjusted_r_squared, .residual_std_error,
#   .standard_errors, .t_statistics, .p_values, .rank, .df_residual
#   .summary() -> str
#
# GLMSolution properties:
#   .coefficients, .fitted_values, .linear_predictor,
#   .residuals_deviance, .residuals_pearson, .residuals_working, .residuals_response,
#   .deviance, .null_deviance, .aic, .bic, .dispersion,
#   .standard_errors, .test_statistics, .p_values,
#   .rank, .df_residual, .df_null, .converged, .n_iter,
#   .family_name, .link_name
#   .summary() -> str
```

### `pystatistics.descriptive`

```python
from pystatistics.descriptive import describe, cor, cov, var, quantile, summary

# describe(data, *, use='everything', quantile_type=7, backend='auto')
# cor(x, y=None, *, method='pearson', use='everything', backend='auto')
#   method: 'pearson' | 'spearman' | 'kendall'
# cov(x, y=None, *, use='everything', backend='auto')
# var(x, *, use='everything', backend='auto')
# quantile(x, probs=None, *, type=7, use='everything', backend='auto')
#   type: 1-9 (R quantile types)
# summary(x, *, use='everything', backend='auto')
#
# use: 'everything' | 'complete.obs' | 'pairwise.complete.obs'
```

### `pystatistics.hypothesis`

```python
from pystatistics.hypothesis import (
    t_test, chisq_test, fisher_test, wilcox_test,
    ks_test, prop_test, var_test, p_adjust,
)

# t_test(x, y=None, *, alternative='two.sided', mu=0.0,
#        paired=False, var_equal=False, conf_level=0.95)
#
# chisq_test(x, y=None, *, correct=True, p=None, rescale_p=False,
#            simulate_p_value=False, B=2000)
#
# fisher_test(x, y=None, *, alternative='two.sided', conf_int=True,
#             conf_level=0.95, simulate_p_value=False, B=2000)
#   Returns HTestSolution with:
#     .statistic, .p_value, .parameter (df), .conf_int, .estimate,
#     .null_value, .alternative, .method, .data_name
#     .summary() -> str (R-style print.htest output)
#
# prop_test(x, n=None, *, p=None, alternative='two.sided',
#           conf_level=0.95, correct=True)
#
# wilcox_test(x, y=None, *, alternative='two.sided', mu=0.0,
#             paired=False, exact=None, correct=True,
#             conf_int=True, conf_level=0.95)
#
# ks_test(x, y=None, *, alternative='two.sided', distribution=None, **dist_params)
#
# var_test(x, y=None, *, ratio=1.0, alternative='two.sided', conf_level=0.95)
#
# p_adjust(p, method='holm', n=None)
#   method: 'holm' | 'hochberg' | 'hommel' | 'bonferroni' | 'BH' | 'BY' | 'fdr' | 'none'
#   Returns: NDArray
```

### `pystatistics.montecarlo`

```python
from pystatistics.montecarlo import boot, boot_ci, permutation_test

# boot(data, statistic, R=999, *, sim='ordinary', stype='i',
#      strata=None, ran_gen=None, mle=None, seed=None, backend='auto')
#   sim: 'ordinary' | 'parametric' | 'balanced'
#   stype: 'i' (index) | 'f' (frequency) | 'w' (weight)
#
# boot_ci(boot_out, *, conf=0.95, type='all', index=0,
#         var_t0=None, var_t=None)
#   type: 'normal' | 'basic' | 'perc' | 'bca' | 'stud' | 'all'
#   Returns BootstrapSolution with .ci dict
#
# permutation_test(x, y, statistic, R=9999, *,
#                  alternative='two.sided', seed=None, backend='auto')
```

### `pystatistics.survival`

```python
from pystatistics.survival import kaplan_meier, survdiff, coxph, discrete_time

# kaplan_meier(time, event, *, conf_level=0.95, conf_type='log')
#   conf_type: 'log' | 'plain' | 'log-log'
#   Returns KMSolution with:
#     .time, .survival, .se, .ci_lower, .ci_upper,
#     .n_risk, .n_event, .n_censor, .median_survival
#     .summary() -> str
#
# survdiff(time, event, group, *, rho=0.0)
#   rho=0 -> log-rank, rho=1 -> Gehan-Wilcoxon (Peto & Peto)
#   Returns LogRankSolution with:
#     .statistic (chi-squared), .p_value, .df,
#     .observed, .expected (per group)
#     .summary() -> str
#
# coxph(time, event, X, *, ties='efron', tol=1e-9, max_iter=20)
#   ties: 'efron' | 'breslow'
#   Returns CoxSolution with:
#     .coefficients, .hazard_ratios, .standard_errors,
#     .z_statistics, .p_values, .conf_int_hr,
#     .log_likelihood (null, full), .concordance,
#     .residuals_martingale, .residuals_deviance
#     .summary() -> str
#
# discrete_time(time, event, X, *, intervals=None, backend='auto')
#   Returns DiscreteTimeSolution
```

### `pystatistics.anova`

```python
from pystatistics.anova import (
    anova_oneway, anova, anova_rm, anova_posthoc, levene_test,
)

# anova_oneway(y, group, *, ss_type=1)
# anova(y, factors, *, covariates=None, ss_type=2, interactions=True)
# anova_rm(y, subject, within, *, between=None, correction='auto')
#   correction: 'none' | 'gg' | 'hf' | 'auto'
# anova_posthoc(anova_result, *, method='tukey', factor=None,
#               control=None, conf_level=0.95)
#   method: 'tukey' | 'bonferroni' | 'dunnett'
# levene_test(y, group, *, center='median')
#   center: 'median' (Brown-Forsythe) | 'mean' (original Levene)
```

### `pystatistics.mixed`

```python
from pystatistics.mixed import lmm, glmm, LMMSolution, GLMMSolution

# lmm(y, X, groups, *, random_effects=None, random_data=None,
#     reml=True, tol=1e-8, max_iter=200, compute_satterthwaite=True)
#
#   groups: dict[str, ArrayLike] -- e.g., {'subject': subject_ids}
#   random_effects: dict[str, list[str]] | None
#     e.g., {'subject': ['1', 'time']} for random intercept + slope
#   random_data: dict[str, ArrayLike] | None
#     e.g., {'time': time_array} -- data for slope variables
#
#   Returns LMMSolution with:
#     .coefficients, .se, .df_satterthwaite, .t_values, .p_values
#     .var_components (tuple of VarCompSummary)
#     .residual_variance, .residual_std
#     .log_likelihood, .aic, .bic
#     .ranef -> dict[str, NDArray] (BLUPs per group)
#     .fixef -> dict[str, float]
#     .icc -> dict[str, float]
#     .fitted_values, .residuals
#     .converged, .n_iter
#     .compare(other) -> str (LRT for nested models, requires ML)
#     .summary() -> str (lmerTest-style)
#
# glmm(y, X, groups, *, family='binomial', random_effects=None,
#      random_data=None, tol=1e-8, max_iter=200)
#
#   Returns GLMMSolution with:
#     .coefficients, .se, .z_values, .p_values
#     .var_components, .deviance, .log_likelihood, .aic, .bic
#     .ranef, .fixef
#     .family_name, .link_name
#     .summary() -> str
```

### `pystatistics.mvnmle`

```python
from pystatistics.mvnmle import mlest

# mlest(data, *, algorithm='direct', backend='auto', tol=None, max_iter=None)
#   algorithm: 'direct' | 'em'
#   Returns MVNSolution with:
#     .mean, .sigma (covariance matrix)
#     .converged, .n_iter
```

### Shared Infrastructure

```python
from pystatistics import DataSource
from pystatistics.regression import Family, Gaussian, Binomial, Poisson

# Family classes for GLMM and GLM:
#   Family.variance(mu) -> NDArray
#   Family.deviance(y, mu, wt) -> float
#   Family.initialize(y) -> NDArray
#   Family.log_likelihood(y, mu, wt, dispersion) -> float
```

---

## Design Principles for PyStatsBio

### 1. Return Structured Results, Not Strings

Every function returns a frozen dataclass or Solution object. SGC-Bio needs to extract numbers, not parse text.

```python
# Good:
@dataclass(frozen=True)
class PowerResult:
    n: int | None
    power: float | None
    effect_size: float | None
    alpha: float
    method: str
    note: str

# Bad:
def power_t_test(...) -> str:
    return "n = 64 per group"
```

### 2. Follow PyStatistics Patterns

- Use `Result[Params]` wrapper pattern for consistency (or define a simpler equivalent)
- Provide `.summary()` for human-readable output
- Use numpy arrays, not lists
- Validate inputs early with clear error messages

### 3. Match R Reference Implementations

Same validation strategy as pystatistics:
1. Generate fixture data in Python
2. Compute reference results in R (with 17-digit precision)
3. Parametrized pytest against the JSON fixtures
4. Document which R package + function each result is validated against

### 4. "Solve For Any One" Pattern (power/ module)

Power/sample size functions should accept all-but-one parameter and solve for the missing one. Use `None` as the signal for "solve for this."

```python
def power_t_test(
    n: int | None = None,
    d: float | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    alternative: str = 'two.sided',
    type: str = 'two.sample',
) -> PowerResult:
    """Exactly one of n, d, power must be None."""
```

### 5. GPU Only Where It Matters

- Phase 1: `doseresponse/` gets `backend='auto'`, optionally `diagnostic/`
- Phase 4+: `pk/` (PopPK), `pd/` (indirect) get `backend='auto'`
- All clinical trial modules: CPU-only, no `backend=` parameter

GPU backends follow the same two-tier validation as pystatistics: CPU matches R, GPU matches CPU.

### 6. Separate Computation From Presentation

PyStatsBio computes. SGC-Bio formats. Don't put table rendering, LaTeX output, or plot generation in PyStatsBio.

### 7. Hide Implementation Details

For PK/PD and dose-response: do NOT expose ODE solver knobs. The user says `pk_model("2cmt_oral")`, not `solve_ivp(method='RK45')`. Solver selection is an implementation detail.

### 8. Stay Small

PyStatsBio v1 (Phase 1) should be ~4,000 lines. If it exceeds PyStatistics (~10k lines), you're building a second engine. Every module should be a thin domain layer over pystatistics primitives, not a reimplementation.

---

## Suggested File Tree (Phase 1 Only)

```
pystatsbio/
    __init__.py
    power/
        __init__.py
        _means.py           # t-test power, paired t-test power
        _proportions.py      # proportion test power, Fisher power
        _survival.py         # log-rank power (Schoenfeld, Freedman, Lachin-Foulkes)
        _anova.py            # one-way and factorial ANOVA power
        _noninferiority.py   # NI/equivalence/superiority power
        _crossover.py        # 2x2 crossover, bioequivalence
        _cluster.py          # cluster randomized trial power
        _common.py           # PowerResult, shared utilities
    doseresponse/
        __init__.py
        _models.py           # 4PL, 5PL, log-logistic, Weibull, hormesis
        _fit.py              # single curve fitting (Levenberg-Marquardt)
        _batch.py            # batch fitting for HTS (GPU)
        _potency.py          # EC50, IC50, relative potency, Fieller CI
        _bmd.py              # benchmark dose (BMDL/BMDU)
        _common.py           # DoseResponseResult, CurveParams
        backends/
            __init__.py
            cpu.py
            gpu.py
    diagnostic/
        __init__.py
        _roc.py              # ROC curve, AUC, DeLong CI, DeLong test
        _accuracy.py         # sensitivity, specificity, PPV, NPV, LR
        _cutoff.py           # optimal cutoff methods
        _batch.py            # batch AUC for biomarker panels (GPU)
        _common.py           # ROCResult, DiagnosticResult
    pk/
        __init__.py
        _nca.py              # non-compartmental analysis (Phase 1)
        _common.py           # NCAResult
tests/
    power/
    doseresponse/
    diagnostic/
    pk/
    fixtures/
        generate_*.py        # Python fixture generators
        run_r_*.R            # R validation scripts
        *.json               # R reference results
```

**Phase 2 adds:** `agreement/`, `equivalence/`, `epi/`

**Phase 3 adds:** `assay/`, `pd/` (Emax only), `multiplicity/` (simple only)

**Phase 4+ adds:** `pk/_compartmental.py`, `pk/_population.py`, `pk/backends/`, `pd/_indirect.py`, `pd/_link.py`, `pd/backends/`, `adaptive/`, full `multiplicity/`

---

## R Packages to Validate Against

| Module | Phase | R Packages | Key Functions |
|--------|-------|-----------|---------------|
| `power/` | 1 | `pwr`, `TrialSize`, `gsDesign`, `PowerTOST` | `pwr.t.test()`, `pwr.2p.test()`, `ssizeEpiCont()` |
| `doseresponse/` | 1 | `drc`, `nplr` | `drm()`, `ED()`, `compParm()` |
| `diagnostic/` | 1 | `pROC`, `OptimalCutpoints`, `epiR` | `roc()`, `auc()`, `roc.test()`, `epi.tests()` |
| `pk/` (NCA) | 1 | `PKNCA`, `NonCompart` | `pk.nca()`, `AUC()` |
| `agreement/` | 2 | `irr`, `psych`, `BlandAltmanLeh`, `DescTools` | `kappa2()`, `ICC()`, `bland.altman.stats()`, `CCC()` |
| `equivalence/` | 2 | `PowerTOST`, `equivalence`, `TOSTER` | `power.TOST()`, `TOSTtwo()` |
| `epi/` | 2 | `epiR`, `epitools`, `Epi` | `epi.2by2()`, `oddsratio()`, `riskratio()` |
| `assay/` | 3 | `drc`, custom | `drm()` for calibration curves |
| `pd/` (Emax) | 3 | `mrgsolve`, `RxODE` | `mrgsolve::mrgsim()` |
| `multiplicity/` | 3 | `multcomp` | `simConfint()` |
| `pk/` (PopPK) | 4+ | `nlme`, `saemix` | `nlme()`, `saemix()` |
| `adaptive/` | 4+ | `gsDesign`, `rpact`, `GroupSeq` | `gsDesign()`, `getDesignGroupSequential()` |
| `multiplicity/` (graphical) | 4+ | `gMCP` | `graphTest()` |

---

## Dependencies

```toml
# pyproject.toml
[project]
name = "pystatsbio"
requires-python = ">=3.11"
dependencies = [
    "pystatistics>=0.1.0",
    "numpy>=1.24",
    "scipy>=1.10",
]

[project.optional-dependencies]
gpu = [
    "torch>=2.0",
]
dev = [
    "pytest>=7.0",
    "pytest-cov",
]
```

Note: `pystatistics` must be installed first (`pip install -e /path/to/pystatistics`). It brings numpy and scipy as transitive dependencies. The `gpu` extra adds PyTorch for GPU-accelerated dose-response batch fitting.

**Phase 4+ will add `torchdiffeq>=0.2` to the gpu extra** for ODE-based PK/PD. Don't add it until PopPK is actually being built.
