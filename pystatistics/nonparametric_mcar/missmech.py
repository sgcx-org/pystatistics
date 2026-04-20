"""MissMech-style MCAR test (Jamshidian & Jalal 2010).

Idea
----

Under MCAR the conditional distributions of the observed values given
the missingness pattern are identical across patterns. Equivalently:
group the rows by missingness pattern and the per-pattern means of the
observed values should be equal (up to sampling noise).

Jamshidian & Jalal (2010) implement this as a test of homogeneity of
means across missingness-pattern groups, using k-nearest-neighbor
imputation to fill missing values and a bootstrap null to calibrate
the test statistic (avoiding any MVN assumption). This module
implements a permutation-based close variant suitable for a
deterministic, cached MCAR-feature scalar:

  1. Identify unique missingness patterns; keep only patterns with
     enough rows (``min_pattern_size``).
  2. Fit a k-nearest-neighbor imputer on the full matrix and impute.
  3. On the kept rows, compute per-pattern mean vectors ``μ_p`` and
     the grand mean ``μ``. Statistic:
         T = Σ_p n_p ||μ_p − μ||²
     (between-pattern weighted sum of squared mean differences).
  4. Permutation null: shuffle the pattern label across rows,
     recompute T. Add-one-smoothed tail p-value.

Under MCAR, pattern label is independent of X, so the permuted T has
the same distribution as the observed T. Under MAR/MNAR, observed
values differ across patterns and T is inflated.

Design notes
------------

- The Jamshidian-Jalal paper uses a BOOTSTRAP of the imputed data
  under the null (resample rows with replacement and re-impute). That
  is strictly correct but expensive. A permutation null on the
  pattern labels is a simpler, faster, and still valid test of
  independence between pattern and observed values — it answers the
  same hypothesis with less computational cost, which matters for the
  cached-scalar use case.
- k-NN imputation requires scikit-learn (already an optional extra
  for ``propensity_mcar_test``).

References
----------

Jamshidian, M., & Jalal, S. (2010). Tests of homogeneity of means
and covariance matrices for multivariate incomplete data.
Psychometrika, 75(4), 649-674.
"""

from typing import List, Tuple
import numpy as np

from pystatistics.nonparametric_mcar.result import NonparametricMCARResult


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "missmech_mcar_test requires scikit-learn. Install with "
            "`pip install pystatistics[nonparametric_mcar]` or "
            "`pip install scikit-learn`."
        ) from e


def _validate_inputs(
    data: np.ndarray, alpha: float, n_permutations: int,
    n_neighbors: int, min_pattern_size: int,
) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_observations, n_variables); got shape {data.shape}"
        )
    if data.shape[0] < 10:
        raise ValueError(
            f"missmech_mcar_test needs at least 10 rows; got {data.shape[0]}"
        )
    if data.shape[1] < 2:
        raise ValueError(
            f"missmech_mcar_test needs at least 2 columns; got {data.shape[1]}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}")
    if n_neighbors < 1:
        raise ValueError(f"n_neighbors must be >= 1; got {n_neighbors}")
    if min_pattern_size < 2:
        raise ValueError(
            f"min_pattern_size must be >= 2 (need >=2 rows per pattern to "
            f"estimate a mean); got {min_pattern_size}"
        )
    return data


def _identify_patterns(miss_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Assign each row an integer pattern label based on its missingness
    indicator row. Returns (labels, n_patterns)."""
    n = miss_mask.shape[0]
    # Encode each row's binary pattern as a tuple for hashing.
    labels = np.empty(n, dtype=int)
    key_to_label = {}
    for i in range(n):
        key = miss_mask[i].tobytes()
        if key not in key_to_label:
            key_to_label[key] = len(key_to_label)
        labels[i] = key_to_label[key]
    return labels, len(key_to_label)


def _between_pattern_ss(
    X_imputed: np.ndarray, labels: np.ndarray, pattern_ids: List[int],
) -> float:
    """Σ_p n_p ||μ_p - μ||² on the rows belonging to `pattern_ids`."""
    keep_mask = np.isin(labels, pattern_ids)
    X_keep = X_imputed[keep_mask]
    lbl_keep = labels[keep_mask]
    grand_mean = X_keep.mean(axis=0)
    ss = 0.0
    for p in pattern_ids:
        rows = X_keep[lbl_keep == p]
        diff = rows.mean(axis=0) - grand_mean
        ss += rows.shape[0] * float(np.dot(diff, diff))
    return ss


def missmech_mcar_test(
    data,
    *,
    alpha: float = 0.05,
    n_permutations: int = 199,
    n_neighbors: int = 5,
    min_pattern_size: int = 6,
    seed: int = 0,
) -> NonparametricMCARResult:
    """Jamshidian-Jalal-style nonparametric MCAR test with permutation null.

    Tests equality of mean vectors across missingness patterns after
    k-NN imputation; a between-pattern sum-of-squares statistic is
    calibrated against permutation of the pattern labels.

    Parameters
    ----------
    data : array-like, shape (n_observations, n_variables)
        Data matrix with ``np.nan`` marking missing entries.
    alpha : float, default 0.05
        Significance level.
    n_permutations : int, default 199
        Number of pattern-label permutations for the null distribution.
    n_neighbors : int, default 5
        k for the k-NN imputer. Clamped internally to
        ``min(n_neighbors, n_observations - 1)``.
    min_pattern_size : int, default 6
        Minimum number of rows a missingness pattern must contain to be
        included in the test. Small patterns give noisy mean estimates
        and inflate the between-group SS under the null.
    seed : int, default 0
        Seed for the permutation draws.

    Returns
    -------
    NonparametricMCARResult
        ``statistic`` = between-pattern weighted sum of squared mean
        differences on the kept rows. ``extra`` contains n_patterns_used,
        n_patterns_total, n_rows_used, bandwidth-equivalent hyperparameters.

    Raises
    ------
    ValueError
        On malformed input, fully-observed matrices, or fewer than two
        patterns meeting the ``min_pattern_size`` threshold (the test
        is undefined with only one group).
    ImportError
        If scikit-learn is not installed.
    """
    _require_sklearn()
    data = _validate_inputs(
        data, alpha, n_permutations, n_neighbors, min_pattern_size
    )

    miss_mask = np.isnan(data)
    n_missing = int(miss_mask.sum())
    if n_missing == 0:
        raise ValueError(
            "missmech_mcar_test requires at least one missing cell; got a "
            "fully-observed matrix."
        )

    n, d = data.shape
    labels, n_patterns_total = _identify_patterns(miss_mask)
    counts = np.bincount(labels)
    kept_patterns = [int(p) for p, c in enumerate(counts) if c >= min_pattern_size]
    if len(kept_patterns) < 2:
        raise ValueError(
            f"Need at least 2 missingness patterns with >= {min_pattern_size} "
            f"rows each; got {len(kept_patterns)} (of {n_patterns_total} "
            f"total). Increase n_observations, reduce min_pattern_size, or "
            f"check that the matrix has structured missingness."
        )

    from sklearn.impute import KNNImputer

    k = min(n_neighbors, n - 1)
    imputer = KNNImputer(n_neighbors=k)
    X_imputed = imputer.fit_transform(data)

    observed_stat = _between_pattern_ss(X_imputed, labels, kept_patterns)

    # Permutation null: shuffle labels on the kept rows only.
    keep_idx = np.where(np.isin(labels, kept_patterns))[0]
    kept_labels = labels[keep_idx]
    rng = np.random.default_rng(seed)
    n_ge = 0
    null_sum = 0.0
    for _ in range(n_permutations):
        perm = rng.permutation(kept_labels)
        perm_full = labels.copy()
        perm_full[keep_idx] = perm
        stat_perm = _between_pattern_ss(X_imputed, perm_full, kept_patterns)
        null_sum += stat_perm
        if stat_perm >= observed_stat:
            n_ge += 1
    p_value = (1 + n_ge) / (1 + n_permutations)

    return NonparametricMCARResult(
        statistic=float(observed_stat),
        p_value=float(p_value),
        rejected=p_value < alpha,
        alpha=alpha,
        method=(
            f"Jamshidian-Jalal-style (k-NN impute k={k}, "
            f"min_pattern_size={min_pattern_size}, perm={n_permutations})"
        ),
        n_observations=n,
        n_variables=d,
        n_missing_cells=n_missing,
        extra={
            "observed_statistic": float(observed_stat),
            "permutation_null_mean": float(null_sum / n_permutations),
            "n_patterns_total": int(n_patterns_total),
            "n_patterns_used": int(len(kept_patterns)),
            "n_rows_used": int(keep_idx.size),
            "n_neighbors": int(k),
            "min_pattern_size": int(min_pattern_size),
            "n_permutations": n_permutations,
            "seed": seed,
        },
    )
