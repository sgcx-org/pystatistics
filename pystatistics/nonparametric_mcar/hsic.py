"""HSIC-based MCAR test.

Idea
----

Under MCAR, the observed values ``X`` are independent of the
missingness indicator matrix ``R``. The Hilbert-Schmidt Independence
Criterion (Gretton et al. 2005) is a kernel-based measure of
dependence that equals zero iff two random variables are independent
(for characteristic kernels like the Gaussian RBF) and is strictly
positive otherwise. We use HSIC(X_mean_imputed, R) with a Gaussian
RBF kernel and median-heuristic bandwidth, calibrated against a
permutation null (rows of R shuffled).

Test statistic
--------------

Biased HSIC estimator (Gretton et al. 2008):

    HSIC_b(X, R) = (1 / n^2) * tr(K_c L_c)

where ``K_c = H K H``, ``L_c = H L H``, ``H = I - (1/n) 1 1^T`` is the
centring matrix, ``K`` is the RBF kernel on ``X``, and ``L`` is the
RBF kernel on ``R``. Larger values indicate stronger dependence.

Null distribution
-----------------

Permutation: shuffle rows of ``R`` (which destroys any dependence on
``X`` while preserving the marginal distribution of missingness
patterns), recompute HSIC, and use the add-one-smoothed tail
probability as the p-value. Permutation is always valid; we avoid the
Gretton gamma approximation here to keep the implementation auditable
and numerically robust on small / degenerate inputs.

References
----------

Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005).
Measuring statistical dependence with Hilbert-Schmidt norms. ALT.

Gretton, A., Fukumizu, K., Teo, C.H., Song, L., Schölkopf, B., &
Smola, A.J. (2008). A kernel statistical test of independence. NIPS.
"""

import numpy as np

from pystatistics.nonparametric_mcar.result import NonparametricMCARResult


def _validate_inputs(data: np.ndarray, alpha: float, n_permutations: int) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2D (n_observations, n_variables); got shape {data.shape}"
        )
    if data.shape[0] < 10:
        raise ValueError(
            f"hsic_mcar_test needs at least 10 rows; got {data.shape[0]}"
        )
    if data.shape[1] < 2:
        raise ValueError(
            f"hsic_mcar_test needs at least 2 columns; got {data.shape[1]}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}")
    return data


def _stochastic_impute(data: np.ndarray, seed: int) -> np.ndarray:
    """Stochastic single imputation: replace NaNs with draws from
    ``N(col_mean, col_std)`` using a seeded RNG.

    Plain column-mean imputation introduces a systematic artefact that
    breaks the HSIC null: rows with many missings get pulled toward the
    column means, which clusters them in ``X``-space AND correlates them
    with their rows in the missingness matrix ``R``. That spurious
    coupling rejects MCAR even when MCAR holds. Stochastic imputation
    with column std preserves the marginal column distribution and
    removes the centroid-clustering artefact; the seed keeps the test
    deterministic (Rule 6).
    """
    X = data.copy()
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    col_stds = np.nanstd(X, axis=0, ddof=0)
    col_stds = np.where(np.isnan(col_stds) | (col_stds == 0.0), 1.0, col_stds)
    nan_mask = np.isnan(X)
    rng = np.random.default_rng(seed)
    n, d = X.shape
    noise = rng.standard_normal((n, d)) * col_stds + col_means
    X[nan_mask] = noise[nan_mask]
    return X


def _pairwise_sq_distances(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean pairwise distance matrix [n, n]."""
    sq_norms = np.sum(X * X, axis=1)
    # (a-b)^2 = a^2 - 2ab + b^2
    D = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)  # clip tiny negative numerical noise
    return D


def _median_bandwidth(D_sq: np.ndarray, eps: float = 1e-12) -> float:
    """Median-heuristic bandwidth: sigma = sqrt(median of non-zero pairwise
    squared distances / 2). The /2 convention makes the RBF kernel
    exp(-d^2 / (2 sigma^2)) match the standard form in Gretton's
    papers."""
    tri = D_sq[np.triu_indices_from(D_sq, k=1)]
    tri = tri[tri > 0]
    if tri.size == 0:
        # Degenerate: all points coincide. Fall back to a tiny positive
        # bandwidth so the kernel is defined (will produce HSIC ~= 0).
        return eps
    med = float(np.median(tri))
    return float(np.sqrt(max(med, eps) / 2.0))


def _rbf_kernel(X: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF kernel matrix. sigma is the bandwidth (not sigma^2)."""
    D_sq = _pairwise_sq_distances(X)
    return np.exp(-D_sq / (2.0 * sigma * sigma))


def _hsic_biased(K: np.ndarray, L: np.ndarray) -> float:
    """Biased HSIC estimator: (1/n^2) tr(K_c L_c), with K_c = H K H.

    Using the identity tr(K_c L_c) = tr(K H L H) and properties of the
    centring matrix, we compute this without materialising H as a dense
    n×n matrix.
    """
    n = K.shape[0]
    Kc = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
    Lc = L - L.mean(axis=0, keepdims=True) - L.mean(axis=1, keepdims=True) + L.mean()
    return float(np.sum(Kc * Lc) / (n * n))


def hsic_mcar_test(
    data,
    *,
    alpha: float = 0.05,
    n_permutations: int = 199,
    seed: int = 0,
) -> NonparametricMCARResult:
    """Kernel-based (HSIC) MCAR test.

    Parameters
    ----------
    data : array-like, shape (n_observations, n_variables)
        Data matrix with ``np.nan`` marking missing entries.
    alpha : float, default 0.05
        Significance level.
    n_permutations : int, default 199
        Number of row-permutations of the missingness matrix for the
        null distribution. The p-value uses add-one smoothing.
    seed : int, default 0
        Seed for the permutation draws.

    Returns
    -------
    NonparametricMCARResult
        ``statistic`` = biased HSIC estimator between mean-imputed
        observed values and the missingness-indicator matrix.
        ``extra`` contains the observed HSIC, the bandwidths used for
        X and R kernels, the permutation null mean, and the n_permutations
        / seed used.

    Raises
    ------
    ValueError
        On malformed inputs (non-2D, too few rows, too few columns,
        invalid alpha / n_permutations), or if ``data`` has no
        missingness (test is undefined).
    """
    data = _validate_inputs(data, alpha, n_permutations)

    miss_mask = np.isnan(data)
    n_missing = int(miss_mask.sum())
    if n_missing == 0:
        raise ValueError(
            "hsic_mcar_test requires at least one missing cell in data; "
            "got a fully-observed matrix."
        )

    n, d = data.shape

    X = _stochastic_impute(data, seed)
    R = miss_mask.astype(float)

    D_sq_X = _pairwise_sq_distances(X)
    D_sq_R = _pairwise_sq_distances(R)
    sigma_X = _median_bandwidth(D_sq_X)
    sigma_R = _median_bandwidth(D_sq_R)

    K = np.exp(-D_sq_X / (2.0 * sigma_X * sigma_X))
    L = np.exp(-D_sq_R / (2.0 * sigma_R * sigma_R))

    observed_hsic = _hsic_biased(K, L)

    # Permutation null: shuffle rows (equivalently columns) of L. Under
    # H0 (independence), the HSIC is invariant in expectation under row
    # permutation of one factor; under dependence, permutation destroys
    # the coupling.
    rng = np.random.default_rng(seed)
    n_ge = 0
    null_sum = 0.0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        L_perm = L[np.ix_(perm, perm)]
        h_null = _hsic_biased(K, L_perm)
        null_sum += h_null
        if h_null >= observed_hsic:
            n_ge += 1
    p_value = (1 + n_ge) / (1 + n_permutations)

    return NonparametricMCARResult(
        statistic=float(observed_hsic),
        p_value=float(p_value),
        rejected=p_value < alpha,
        alpha=alpha,
        method=f"HSIC (Gaussian RBF, median bandwidth, perm={n_permutations})",
        n_observations=n,
        n_variables=d,
        n_missing_cells=n_missing,
        extra={
            "observed_hsic": float(observed_hsic),
            "bandwidth_X": float(sigma_X),
            "bandwidth_R": float(sigma_R),
            "permutation_null_mean_hsic": float(null_sum / n_permutations),
            "n_permutations": n_permutations,
            "seed": seed,
        },
    )
