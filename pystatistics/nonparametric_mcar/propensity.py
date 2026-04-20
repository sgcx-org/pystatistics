"""Propensity-score MCAR test.

Idea
----

Under MCAR the missingness indicator ``R_j`` for column ``j`` is
independent of the observed values. Therefore, a classifier trying to
predict ``R_j`` from the OTHER columns should perform no better than
chance (out-of-fold AUC = 0.5). If a flexible, mixed-type-aware
classifier (random forest or gradient boosting) achieves AUC > 0.5
with a permutation-calibrated p-value, MCAR is rejected.

Multiple columns with missingness are aggregated by the mean
per-column AUC; the permutation null is obtained by shuffling the
column's missingness indicator vector and recomputing.

Why RF / GBM
------------

Both natively handle mixed types (integer-encoded categoricals coexist
with continuous columns), capture nonlinear dependencies, and have
deterministic fits given a seed. sklearn's
``RandomForestClassifier`` / ``GradientBoostingClassifier`` are the
canonical reference implementations.

Dependencies
------------

Requires scikit-learn (optional extra: ``pip install pystatistics[nonparametric_mcar]``).
The import failure message at module load time tells the user exactly
what to install.
"""

from typing import List, Optional, Tuple
import warnings

import numpy as np

from pystatistics.nonparametric_mcar.result import NonparametricMCARResult


# -------------------------------------------------------------------------
# sklearn is an optional dependency — fail loud at call time if missing.
# -------------------------------------------------------------------------

def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "propensity_mcar_test requires scikit-learn. Install with "
            "`pip install pystatistics[nonparametric_mcar]` or "
            "`pip install scikit-learn`."
        ) from e


def _validate_inputs(data: np.ndarray, alpha: float, n_permutations: int,
                     cv_folds: int, model: str) -> np.ndarray:
    """Validate and normalise the data matrix; raise on any issue."""
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_observations, n_variables); got shape {data.shape}"
        )
    if data.shape[0] < 10:
        raise ValueError(
            f"propensity_mcar_test needs at least 10 rows; got {data.shape[0]}"
        )
    if data.shape[1] < 2:
        raise ValueError(
            f"propensity_mcar_test needs at least 2 columns (one to predict, "
            f"one or more to predict from); got {data.shape[1]}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}")
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2; got {cv_folds}")
    if model not in {"rf", "gbm"}:
        raise ValueError(f"model must be 'rf' or 'gbm'; got {model!r}")
    return data


def _build_classifier(model: str, seed: int):
    """Construct a seeded sklearn classifier matching the Lacuna bakeoff spec."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    if model == "rf":
        # Small, deterministic, fast. n_estimators=100 is sklearn default;
        # max_depth=None lets trees grow but the data is finite so bounded
        # in practice. n_jobs=1 to keep determinism given seed.
        return RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=1,
        )
    return GradientBoostingClassifier(
        n_estimators=100, random_state=seed,
    )


def _imputed_features(data: np.ndarray, predictor_cols: np.ndarray) -> np.ndarray:
    """Column-mean impute predictor columns and append per-column missing
    indicators as additional features. This lets the classifier use
    cross-column missingness (MAR signal) as well as observed values.
    """
    X = data[:, predictor_cols].copy()
    col_means = np.nanmean(X, axis=0)
    # Guard: if a predictor column is entirely NaN, use 0 for its mean so
    # imputation is defined but the column carries no signal.
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    miss_ind = np.isnan(X).astype(float)
    np.copyto(X, np.broadcast_to(col_means, X.shape), where=np.isnan(X))
    return np.hstack([X, miss_ind])


def _oof_auc(
    X: np.ndarray, y: np.ndarray, model: str, seed: int, cv_folds: int,
) -> float:
    """Out-of-fold AUC for predicting ``y`` from ``X``.

    Returns 0.5 if ``y`` has only one class (AUC undefined — treated as
    chance-level).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return 0.5

    # Need at least cv_folds of the minority class for StratifiedKFold.
    min_class_count = int(min(np.sum(y == 0), np.sum(y == 1)))
    k = min(cv_folds, min_class_count)
    if k < 2:
        return 0.5

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof_proba = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X, y):
        clf = _build_classifier(model, seed)
        clf.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return float(roc_auc_score(y, oof_proba))


def _columns_with_missingness(data: np.ndarray) -> np.ndarray:
    """Indices of columns with at least one missing cell AND at least one
    observed cell — the only columns where the propensity test is
    defined."""
    miss = np.isnan(data)
    col_any_missing = miss.any(axis=0)
    col_any_observed = (~miss).any(axis=0)
    return np.where(col_any_missing & col_any_observed)[0]


def propensity_mcar_test(
    data,
    *,
    alpha: float = 0.05,
    model: str = "rf",
    cv_folds: int = 5,
    n_permutations: int = 199,
    seed: int = 0,
    verbose: bool = False,
) -> NonparametricMCARResult:
    """Propensity-score MCAR test.

    For each column with missingness, fit a classifier to predict that
    column's observed-vs-missing indicator from the other columns
    (column-mean imputed + per-column missing-indicator features).
    Compute out-of-fold AUC. Aggregate across columns by taking the
    mean observed AUC. Calibrate against a permutation null produced by
    shuffling the missingness indicator before re-scoring.

    Parameters
    ----------
    data : array-like, shape (n_observations, n_variables)
        Data matrix with ``np.nan`` marking missing entries.
    alpha : float, default 0.05
        Significance level.
    model : {'rf', 'gbm'}, default 'rf'
        Classifier family. 'rf' = ``RandomForestClassifier``; 'gbm' =
        ``GradientBoostingClassifier``.
    cv_folds : int, default 5
        Number of stratified folds for out-of-fold AUC.
    n_permutations : int, default 199
        Number of permutation draws for the null distribution. The
        p-value is ``(1 + #{perm >= observed}) / (1 + n_permutations)``
        (add-one smoothing, two-sided not applicable — AUC is directed).
    seed : int, default 0
        Seed for classifier fits and permutation draws. Determines
        reproducibility.
    verbose : bool, default False
        If True, print per-column AUCs as they complete.

    Returns
    -------
    NonparametricMCARResult
        ``statistic`` = mean observed AUC minus 0.5, clipped to [0, 0.5];
        ``extra`` contains:

        - ``per_column_auc``: list of (column_index, observed_auc) tuples.
        - ``mean_observed_auc``: raw mean AUC before the −0.5 shift.
        - ``permutation_null_mean_auc``: mean of the permuted-AUC null,
          for sanity checks; should centre on 0.5 if the permutation is
          valid.
        - ``model``, ``cv_folds``, ``n_permutations``, ``seed``:
          echoed hyperparameters.

    Raises
    ------
    ValueError
        If ``data`` is not 2D, has <10 rows, has <2 columns, or has no
        columns with both missing and observed values (the test is
        undefined in that case).
    ImportError
        If scikit-learn is not installed.
    """
    _require_sklearn()
    data = _validate_inputs(data, alpha, n_permutations, cv_folds, model)

    n_obs, n_vars = data.shape
    n_missing = int(np.isnan(data).sum())

    target_cols = _columns_with_missingness(data)
    if target_cols.size == 0:
        raise ValueError(
            "propensity_mcar_test requires at least one column with BOTH "
            "missing and observed values; got none."
        )

    per_col_auc: List[Tuple[int, float]] = []
    # Permutation null needs to be built per-column too, then aggregated.
    # We use the SAME seed sequence across columns' permutations so the
    # null is reproducible.
    rng = np.random.default_rng(seed)
    perm_seeds = rng.integers(0, 2**31 - 1, size=n_permutations)

    null_sum = np.zeros(n_permutations, dtype=float)

    for col in target_cols:
        predictor_cols = np.array([c for c in range(n_vars) if c != col])
        X = _imputed_features(data, predictor_cols)
        y = np.isnan(data[:, col]).astype(int)

        observed_auc = _oof_auc(X, y, model, seed, cv_folds)
        per_col_auc.append((int(col), observed_auc))
        if verbose:
            print(f"  col {col}: observed AUC = {observed_auc:.4f}")

        for p, ps in enumerate(perm_seeds):
            perm_rng = np.random.default_rng(ps)
            y_perm = perm_rng.permutation(y)
            null_sum[p] += _oof_auc(X, y_perm, model, int(ps), cv_folds)

    mean_observed_auc = float(np.mean([a for _, a in per_col_auc]))
    null_mean_auc = null_sum / len(target_cols)

    # One-sided p-value: how often does the null match or exceed the
    # observed statistic? Add-one smoothing keeps the p-value from
    # collapsing to zero under finite permutations (Phipson & Smyth 2010).
    n_ge = int(np.sum(null_mean_auc >= mean_observed_auc))
    p_value = (1 + n_ge) / (1 + n_permutations)

    statistic = float(np.clip(mean_observed_auc - 0.5, 0.0, 0.5))

    return NonparametricMCARResult(
        statistic=statistic,
        p_value=p_value,
        rejected=p_value < alpha,
        alpha=alpha,
        method=f"Propensity-AUC ({model.upper()}, cv={cv_folds}, perm={n_permutations})",
        n_observations=n_obs,
        n_variables=n_vars,
        n_missing_cells=n_missing,
        extra={
            "per_column_auc": per_col_auc,
            "mean_observed_auc": mean_observed_auc,
            "permutation_null_mean_auc": float(np.mean(null_mean_auc)),
            "model": model,
            "cv_folds": cv_folds,
            "n_permutations": n_permutations,
            "seed": seed,
        },
    )
