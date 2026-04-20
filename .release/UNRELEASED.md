# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py <version>`
> That script uses this file to build the CHANGELOG entry, bumps versions
> everywhere, and resets this file for the next cycle.

## Changes

- **New subpackage `pystatistics.nonparametric_mcar`** for distribution-
  free MCAR tests, motivated by the Lacuna ablation finding that the
  MVN-based Little's MCAR feature (MLE or MoM plug-in) does not help
  mechanism classification on heavy-tailed / categorical tabular data.

  - Added `propensity_mcar_test(data, *, model='rf'|'gbm', cv_folds=5,
    n_permutations=199, seed=0, alpha=0.05)`. Fits a sklearn
    `RandomForestClassifier` or `GradientBoostingClassifier` to predict
    each column's missingness indicator from the other columns
    (mean-imputed + per-column missing-indicator features), computes
    out-of-fold AUC, and calibrates against a permutation null. Returns
    a `NonparametricMCARResult` with `statistic = mean_auc - 0.5` and
    permutation-smoothed `p_value`. scikit-learn is an optional extra:
    `pip install pystatistics[nonparametric_mcar]`.

  - Added `NonparametricMCARResult` dataclass (statistic, p_value,
    rejected, alpha, method, n_observations, n_variables,
    n_missing_cells, extra). Intentionally narrower than the MVN-based
    `MCARTestResult` — no df / ml_mean / ml_cov / patterns, because
    nonparametric tests don't produce those.

  - Tests: 11 in `tests/nonparametric_mcar/test_propensity.py` covering
    normal cases (MCAR non-rejection, MAR rejection, reproducibility
    under fixed seed, GBM option), edge cases (fully-observed column
    ignored), and failure cases (1D input, too few rows/columns, no
    missingness, invalid hyperparameters).

  - Added `hsic_mcar_test(data, *, alpha=0.05, n_permutations=199,
    seed=0)`. Hilbert-Schmidt Independence Criterion (Gretton et al.
    2005/2008) between stochastically-imputed observed values and the
    missingness-indicator matrix, with Gaussian RBF kernel and
    median-heuristic bandwidth. Biased HSIC estimator, permutation null
    for calibration. Uses **stochastic** (column-mean + column-std
    noise) imputation rather than plain mean imputation — pure
    mean-imputation pulls heavy-missing rows toward the column
    centroid, which creates a systematic X-R coupling and rejects MCAR
    spuriously on MCAR-generated data. Pure numpy; no sklearn dep.

  - Added `missmech_mcar_test(data, *, alpha=0.05, n_permutations=199,
    n_neighbors=5, min_pattern_size=6, seed=0)`. Jamshidian-Jalal-style
    test of homogeneity of means across missingness-pattern groups,
    after k-NN imputation (via `sklearn.impute.KNNImputer`). Statistic
    is the between-pattern weighted sum of squared mean differences
    (Σ_p n_p ||μ_p − μ||²), calibrated against a pattern-label
    permutation null — equivalent in hypothesis to Jamshidian & Jalal
    (2010)'s bootstrap, but faster for the cached-scalar use case.
    Requires the same `nonparametric_mcar` extra for sklearn.

  - Tests: 9 in `test_hsic.py`, 10 in `test_missmech.py` — both
    covering MCAR non-rejection, MAR rejection, reproducibility under
    seed, and the same failure-case matrix as propensity.

  - Suite: 156/156 across `tests/nonparametric_mcar/` and `tests/mvnmle/`
    pass; no mvnmle regressions from the new subpackage.

- **Split `pystatistics/mvnmle/backends/_em_batched.py`** (501 SLOC →
  over the Rule 4 hard limit of 500) into three focused files plus a
  compatibility shim:
    - `_em_batched_patterns.py` (63 SLOC) — `_BatchedPatternIndex`
      dataclass, `_pattern_n`, `build_pattern_index`.
    - `_em_batched_np.py` (203 SLOC) — NumPy CPU backend
      (`compute_conditional_parameters_np`, `e_step_full_batched_np`,
      `compute_loglik_batched_np`, `chi_square_mcar_batched_np`).
    - `_em_batched_torch.py` (239 SLOC) — Torch GPU backend
      (`_e_step_full_torch`, `_loglik_full_torch`,
      `chi_square_mcar_batched_torch`, `compute_conditional_parameters_torch`).
    - `_em_batched.py` (30 SLOC) — thin shim re-exporting every
      symbol so existing importers in `em.py`, `solvers.py`, and
      `mcar_test.py` need no changes.
  The 157-test suite across `tests/test_code_quality.py`,
  `tests/mvnmle/`, and `tests/nonparametric_mcar/` passes after the
  split; `test_no_file_exceeds_500_code_lines` now passes.
