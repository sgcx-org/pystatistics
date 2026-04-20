"""
Little's MCAR Test.

Implementation of Little's (1988) test for Missing Completely at Random (MCAR).

Reference:
    Little, R.J.A. (1988). A test of missing completely at random for
    multivariate data with missing values. JASA, 83(404), 1198-1202.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings

from pystatistics.core.exceptions import PyStatisticsError
from pystatistics.mvnmle.patterns import PatternInfo, identify_missingness_patterns


@dataclass
class MCARTestResult:
    """Result of an MCAR test.

    Shared result type for ``little_mcar_test`` (Little 1988, MLE
    plug-in) and ``mom_mcar_test`` (method-of-moments pairwise-
    deletion plug-in). The ``method`` field identifies which test
    produced the result so downstream code can disambiguate without
    having to track the calling function.

    Notes
    -----
    ``ml_mean`` / ``ml_cov`` carry the plug-in estimates used in the
    chi-square statistic. For ``method='Little (MLE plug-in)'`` these
    are the observed-data MLE; for ``method='Method-of-moments
    (pairwise-deletion plug-in)'`` they are the pairwise-deletion
    sample moments. The field names are retained for backward
    compatibility; consult ``method`` to know what's inside.
    """
    statistic: float
    df: int
    p_value: float
    rejected: bool
    alpha: float
    patterns: List[PatternInfo]
    n_patterns: int
    n_patterns_used: int
    ml_mean: np.ndarray
    ml_cov: np.ndarray
    convergence_warnings: List[str]
    method: str = "Little (MLE plug-in)"

    def summary(self) -> str:
        """Generate human-readable summary of test results."""
        summary_lines = [
            "MCAR Test Results",
            "=" * 40,
            f"Method: {self.method}",
            f"Test statistic (chi-sq): {self.statistic:.4f}",
            f"Degrees of freedom: {self.df}",
            f"P-value: {self.p_value:.4f}",
            f"",
            f"Decision at alpha={self.alpha}: {'Reject MCAR' if self.rejected else 'Fail to reject MCAR'}",
            f"",
            f"Number of patterns: {self.n_patterns}",
        ]

        if self.convergence_warnings:
            summary_lines.append("\nWarnings:")
            for warning in self.convergence_warnings:
                summary_lines.append(f"  - {warning}")

        return "\n".join(summary_lines)


def regularized_inverse(matrix: np.ndarray,
                       condition_threshold: float = 1e12,
                       regularize: bool = True) -> Tuple[np.ndarray, bool]:
    """
    Compute inverse of a covariance matrix, pseudo-inverting when
    ill-conditioned.

    Real-world tabular data routinely produces ill-conditioned per-pattern
    covariance matrices — the canonical sklearn demo datasets (iris,
    wine, breast_cancer) all cross the default 1e12 condition threshold
    under Little's MCAR. R's ``BaylorEdPsych::LittleMCAR`` and
    ``misty::na.test`` silently fall back to a Moore-Penrose
    pseudo-inverse here; pystatistics does the same by default but
    emits a warning so the degradation is visible rather than silent.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to invert.
    condition_threshold : float
        Maximum condition number before the matrix is treated as
        ill-conditioned. Default 1e12.
    regularize : bool
        When True (default), ill-conditioned matrices are handled via
        ``np.linalg.pinv`` with a ``UserWarning``. When False, raises
        ``NumericalError`` — strict mode for callers that want the
        fail-fast behaviour (e.g. regulated pipelines where
        pseudo-inverse precision loss is unacceptable).

    Returns
    -------
    inv_matrix : np.ndarray
        Inverted (or pseudo-inverted) matrix.
    was_regularized : bool
        True iff the pseudo-inverse path was taken.
    """
    import warnings

    from pystatistics.core.exceptions import NumericalError

    cond = float(np.linalg.cond(matrix))

    if cond < condition_threshold:
        try:
            return np.linalg.inv(matrix), False
        except np.linalg.LinAlgError:
            # Matrix is exactly singular despite a finite condition
            # number — fall through to the ill-conditioned branch below.
            pass

    if regularize:
        warnings.warn(
            f"Covariance matrix for a missingness pattern is "
            f"ill-conditioned (cond={cond:.2e} > threshold="
            f"{condition_threshold:.0e}). Using Moore-Penrose "
            f"pseudo-inverse; chi-square contribution for this pattern "
            f"may have reduced precision. Pass regularize=False to "
            f"raise instead.",
            UserWarning,
            stacklevel=3,
        )
        return np.linalg.pinv(matrix), True

    raise NumericalError(
        f"Covariance matrix is ill-conditioned (condition number: "
        f"{cond:.2e}, threshold: {condition_threshold:.0e}). Cannot "
        f"reliably invert.\n"
        f"Options:\n"
        f"  - Pass regularize=True to use Moore-Penrose pseudo-inverse "
        f"(matches R's BaylorEdPsych::LittleMCAR default).\n"
        f"  - Pass a larger condition_threshold to loosen the gate.\n"
        f"  - Remove near-collinear variables from the input."
    )


def little_mcar_test(data,
                     alpha: float = 0.05,
                     backend: str | None = None,
                     algorithm: str = 'em',
                     regularize: bool = True,
                     condition_threshold: float = 1e12,
                     drop_all_missing_rows: bool = True,
                     verbose: bool = False) -> MCARTestResult:
    """
    Little's test for Missing Completely at Random (MCAR).

    Parameters
    ----------
    data : array-like, shape (n_observations, n_variables)
        Data matrix with missing values as np.nan.
    alpha : float, default=0.05
        Significance level.
    backend : str or None, default None
        Backend for the ML estimation step (the dominant cost).
        Default None → 'cpu' (R-reference path). Explicit: 'cpu',
        'gpu', or 'auto' to prefer GPU when available. The per-pattern
        test-statistic accumulation runs on CPU regardless — it's
        O(P × v³) for tiny v and is not the bottleneck.
    algorithm : str, default 'em'
        ML algorithm forwarded to mlest: 'em' (EM, default) or
        'direct' (BFGS). EM is the default because BFGS convergence
        scales poorly with the number of missingness patterns — on
        realistic tabular data (e.g. 13 vars × 107 patterns) BFGS
        can take 400+ seconds while EM finishes in under a second.
        Little's statistic depends only on the final ML estimates,
        not on which algorithm produced them.
    regularize : bool, default True
        When an observed-variable covariance submatrix is
        ill-conditioned (common on real tabular data — iris, wine,
        breast_cancer all trip the default threshold), fall back to
        the Moore-Penrose pseudo-inverse with a warning rather than
        raising. Matches R's ``BaylorEdPsych::LittleMCAR`` and
        ``misty::na.test`` defaults. Pass False for strict
        fail-fast behaviour.
    condition_threshold : float, default 1e12
        Maximum condition number before a per-pattern covariance
        submatrix is considered ill-conditioned.
    drop_all_missing_rows : bool, default True
        Drop rows with no observed values before fitting. Such rows
        contribute nothing to either the MLE or the chi-square
        statistic; at realistic missingness rates on low-dimensional
        data they show up routinely and shouldn't block the test.
        A warning reports how many rows were dropped.
    verbose : bool, default False
        Print detailed progress.

    Returns
    -------
    MCARTestResult
    """
    import warnings

    # Import mlest here to avoid circular imports
    from pystatistics.mvnmle.solvers import mlest

    # Input conversion
    if hasattr(data, 'values'):
        data_array = np.asarray(data.values, dtype=float)
    else:
        data_array = np.asarray(data, dtype=float)

    if data_array.ndim != 2:
        raise ValueError("Data must be 2-dimensional")

    # Drop all-missing rows (they contribute nothing) before validation.
    # The underlying MVNDesign.from_array would otherwise reject them
    # outright, which is correct at the mlest layer but user-hostile
    # here: an MCAR diagnostic should not fall over just because the
    # dataset happens to contain one fully-missing record.
    if drop_all_missing_rows:
        all_nan_mask = np.all(np.isnan(data_array), axis=1)
        n_dropped = int(np.sum(all_nan_mask))
        if n_dropped > 0:
            warnings.warn(
                f"Dropping {n_dropped} row(s) with all values missing. "
                f"Such rows carry no information for Little's MCAR test. "
                f"Pass drop_all_missing_rows=False to opt out.",
                UserWarning,
                stacklevel=2,
            )
            data_array = data_array[~all_nan_mask]

    n_obs, n_vars = data_array.shape

    # Step 1: Get ML estimates
    if verbose:
        print("Step 1: Computing ML estimates...")

    try:
        ml_result = mlest(
            data_array,
            backend=backend,
            algorithm=algorithm,
            regularize=regularize,
            verbose=False,
        )
        mu_ml = ml_result.muhat
        sigma_ml = ml_result.sigmahat
    except PyStatisticsError:
        # Preserve pystatistics exception type so callers using a
        # `except PyStatisticsError:` catch (the documented pattern)
        # actually catch MLE failures here.
        raise
    except Exception as e:
        raise RuntimeError(f"ML estimation failed: {e}") from e

    # Rule 1: do not quietly hand the caller a statistic built on top
    # of unconverged ML estimates. If BFGS ran out of iterations (the
    # common failure mode for `algorithm='direct'` on data with many
    # missingness patterns — see mcar_test release notes), the muhat /
    # sigmahat returned are whatever the optimizer's last iterate
    # happened to be, and the chi-square contribution computed against
    # them is not the Little's statistic — it is noise.
    if not ml_result.converged:
        n_patterns_hint = len(identify_missingness_patterns(data_array))
        raise RuntimeError(
            f"ML estimation did not converge (algorithm={algorithm!r}, "
            f"n_iter={ml_result.n_iter}, loglik={ml_result.loglik:.4f}). "
            f"The chi-square statistic built on non-MLE estimates is "
            f"not Little's statistic.\n"
            f"Data shape: {data_array.shape[0]} rows x "
            f"{data_array.shape[1]} cols, {n_patterns_hint} missingness "
            f"patterns.\n"
            f"Options:\n"
            f"  - Use algorithm='em' (the default for this function — "
            f"robust on data with many patterns).\n"
            f"  - Raise max_iter if you specifically need BFGS "
            f"(algorithm='direct').\n"
            f"  - Inspect ml_result manually by calling mlest(...) "
            f"directly to see intermediate state."
        )

    # Step 2: Identify missingness patterns
    if verbose:
        print("Step 2: Identifying missingness patterns...")

    patterns = identify_missingness_patterns(data_array)

    # Step 3: Compute test statistic
    test_statistic = 0.0
    convergence_warnings = []
    n_patterns_used = 0

    for pattern in patterns:
        if pattern.n_observed == 0:
            continue

        obs_idx = pattern.observed_indices
        n_k = pattern.n_cases

        y_bar_k = np.mean(pattern.data, axis=0)
        mu_obs_k = mu_ml[obs_idx]
        sigma_obs_k = sigma_ml[np.ix_(obs_idx, obs_idx)]

        # regularize=True routes ill-conditioned submatrices through
        # pseudo-inverse with a warning (R's default); regularize=False
        # raises NumericalError.
        sigma_inv_k, _ = regularized_inverse(
            sigma_obs_k,
            condition_threshold=condition_threshold,
            regularize=regularize,
        )

        diff = y_bar_k - mu_obs_k
        contribution = n_k * (diff @ sigma_inv_k @ diff)
        test_statistic += contribution
        n_patterns_used += 1

    # Step 4: Degrees of freedom
    df = sum(p.n_observed for p in patterns) - n_vars

    # Handle edge cases
    if len(patterns) == 1 and patterns[0].n_observed == n_vars:
        return MCARTestResult(
            statistic=0.0,
            df=0,
            p_value=1.0,
            rejected=False,
            alpha=alpha,
            patterns=patterns,
            n_patterns=1,
            n_patterns_used=0,
            ml_mean=mu_ml,
            ml_cov=sigma_ml,
            convergence_warnings=["No missing data - MCAR test not applicable"]
        )

    if df <= 0:
        raise ValueError(f"Invalid degrees of freedom: {df}")

    # Step 5: P-value
    p_value = 1 - stats.chi2.cdf(test_statistic, df)
    rejected = p_value < alpha

    return MCARTestResult(
        statistic=test_statistic,
        df=df,
        p_value=p_value,
        rejected=rejected,
        alpha=alpha,
        patterns=patterns,
        n_patterns=len(patterns),
        n_patterns_used=n_patterns_used,
        ml_mean=mu_ml,
        ml_cov=sigma_ml,
        convergence_warnings=convergence_warnings,
        method="Little (MLE plug-in)",
    )


def _resolve_mom_backend(backend: 'str | None', gpu_worth: bool) -> str:
    """Dispatch helper for mom_mcar_test. Mirrors the EM path's
    Rule-1 visibility: every non-obvious choice emits a warning.

    Returns 'cpu' or 'gpu'.
    """
    import warnings

    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    if backend is None:
        # Default: CPU. No warning — user made no choice, and the
        # default is the conservative safe one per 2.0.0 convention.
        return 'cpu'

    if backend == 'cpu':
        return 'cpu'

    if backend == 'gpu':
        if not _cuda_available():
            raise RuntimeError(
                "backend='gpu' requested but no CUDA device is "
                "available. Pass backend='cpu' or 'auto'."
            )
        if not gpu_worth:
            warnings.warn(
                f"backend='gpu': proceeding on GPU as requested, but "
                f"data size (n*v) is below the MoM GPU-worth-it "
                f"threshold of {_MOM_GPU_WORTH_IT_THRESHOLD}. MoM's "
                f"chi-square assembly is one-shot (no iteration) so "
                f"CPU overhead is already small; GPU is usually a "
                f"wash or slower on this shape. Pass backend='cpu' "
                f"or 'auto' to skip GPU.",
                UserWarning, stacklevel=3,
            )
        return 'gpu'

    if backend == 'auto':
        if gpu_worth and _cuda_available():
            # Silent: user asked for auto, this is the normal choice.
            return 'gpu'
        if _cuda_available() and not gpu_worth:
            warnings.warn(
                f"backend='auto': dispatching MoM MCAR test to CPU "
                f"(data size below GPU-worth-it threshold of "
                f"{_MOM_GPU_WORTH_IT_THRESHOLD}). GPU is available "
                f"but would likely not win here — MoM is one-shot "
                f"and CPU completes in milliseconds. Pass "
                f"backend='gpu' to force GPU anyway.",
                UserWarning, stacklevel=3,
            )
        return 'cpu'

    raise ValueError(
        f"Unknown backend: {backend!r}. Use 'cpu', 'gpu', 'auto', or None."
    )


def _pairwise_deletion_moments(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise-deletion sample mean and covariance.

    The mean of column j uses all rows where column j is observed.
    The covariance of columns (i, j) uses all rows where both
    columns are observed — i.e. different cell counts for different
    cells of the covariance matrix. This is the method-of-moments
    estimator under MCAR.

    Implementation: single matmul + single division, so the whole
    thing is O(n v^2) with no Python loop over columns.
    """
    mask = ~np.isnan(data)
    n_per_col = mask.sum(axis=0).astype(np.float64)
    if np.any(n_per_col < 1):
        raise ValueError("At least one column is fully missing")
    means = np.where(
        mask, data, 0.0,
    ).sum(axis=0) / n_per_col

    # Pairwise covariance: replace NaN with 0 after centering, then
    # X_centered.T @ X_centered sums product only where both
    # observations are present; divide elementwise by per-pair count.
    X_centered = np.where(mask, data - means, 0.0)
    pair_counts = mask.astype(np.float64).T @ mask.astype(np.float64)
    # Minimum 1 to avoid div-by-zero on cells with no co-observations.
    # Such cells shouldn't appear in a well-formed dataset (would
    # mean two columns never observed together); guard just in case.
    pair_counts_safe = np.maximum(pair_counts, 1.0)
    cov = (X_centered.T @ X_centered) / pair_counts_safe
    # Symmetrise to wash out any FP asymmetry from the division.
    cov = 0.5 * (cov + cov.T)
    return means, cov


_MOM_GPU_WORTH_IT_THRESHOLD = 10_000  # n*v empirical crossover; see solvers._em_gpu_worth_it


def mom_mcar_test(
    data,
    alpha: float = 0.05,
    regularize: bool = True,
    condition_threshold: float = 1e12,
    drop_all_missing_rows: bool = True,
    backend: 'str | None' = None,
    verbose: bool = False,
) -> MCARTestResult:
    """Method-of-moments MCAR test.

    **This is not Little's test.** Little (1988) defines the MCAR
    chi-square statistic with the observed-data MLE plug-in, and the
    asymptotic chi-square distribution theory leans on MLE's
    efficiency. This function computes a statistic of the same shape
    but with *pairwise-deletion sample moments* substituted for the
    MLE:

        - :math:`\\hat\\mu_j` = sample mean of column j over all
          observations where j is observed.
        - :math:`\\hat\\Sigma_{ij}` = sample covariance of columns
          (i, j) over all observations where both are observed.

    Under the MCAR null these moment estimators are consistent, so
    the statistic is approximately chi-square with the same degrees
    of freedom as Little's test at moderate to large n. It is not
    asymptotically efficient, and the finite-sample distribution
    deviates from chi-square more than Little's does. In exchange
    it is *dramatically* faster because there is no EM / BFGS
    iteration — the computation is O(n v^2) + O(P v^3) one-shot.

    When to prefer this over ``little_mcar_test``:

    - You are sweeping MCAR over many (thousands+) datasets for
      diagnostic screening rather than a single rigorous hypothesis
      test (e.g. filtering which datasets to include in a study).
    - You need a p-value that is order-of-magnitude correct, not
      exact to 3 decimals.

    When **not** to use this:

    - Regulated submissions, published papers that cite Little 1988,
      or anywhere the precise asymptotic distribution matters. Use
      ``little_mcar_test`` — it matches Little's specification and
      the R ``BaylorEdPsych::LittleMCAR`` reference exactly.

    Parameters and return value mirror ``little_mcar_test``. The
    returned ``MCARTestResult.method`` is set to ``"Method-of-moments
    (pairwise-deletion plug-in)"`` so downstream code can tell which
    test produced a given result.

    References
    ----------
    Little, R. J. A. (1988). A test of missing completely at random
    for multivariate data with missing values. JASA, 83(404),
    1198-1202. (Original MLE-plug-in formulation.)

    Park, T. & Davis, C. S. (1993). A test of the missing data
    mechanism for repeated categorical data. Biometrics, 49(2),
    631-638. (Moment-based variants in a related family.)
    """
    import warnings

    if hasattr(data, 'values'):
        data_array = np.asarray(data.values, dtype=float)
    else:
        data_array = np.asarray(data, dtype=float)

    if data_array.ndim != 2:
        raise ValueError("Data must be 2-dimensional")

    if drop_all_missing_rows:
        all_nan_mask = np.all(np.isnan(data_array), axis=1)
        n_dropped = int(np.sum(all_nan_mask))
        if n_dropped > 0:
            warnings.warn(
                f"Dropping {n_dropped} row(s) with all values missing. "
                f"Such rows carry no information for the MCAR test. "
                f"Pass drop_all_missing_rows=False to opt out.",
                UserWarning,
                stacklevel=2,
            )
            data_array = data_array[~all_nan_mask]

    n_obs, n_vars = data_array.shape

    if verbose:
        print("Step 1: Computing pairwise-deletion moments...")

    mu_mom, sigma_mom = _pairwise_deletion_moments(data_array)

    if verbose:
        print("Step 2: Identifying missingness patterns...")

    patterns = identify_missingness_patterns(data_array)

    # Backend dispatch with the same size-heuristic + visible-warning
    # discipline used by the EM path. For GPU the per-pattern SVD+solve
    # is fully batched; for CPU we use the same batched numpy path. The
    # threshold for "GPU worth it" is higher for MoM than for EM
    # because MoM is one-shot (no EM iterations to amortise GPU launch
    # overhead over).
    gpu_worth = (n_obs * n_vars) >= _MOM_GPU_WORTH_IT_THRESHOLD
    chosen_backend = _resolve_mom_backend(backend, gpu_worth)
    convergence_warnings: List[str] = []

    from pystatistics.mvnmle.backends._em_batched import (
        build_pattern_index,
        chi_square_mcar_batched_np,
        chi_square_mcar_batched_torch,
    )

    index = build_pattern_index(patterns, n_vars)

    if chosen_backend == 'gpu':
        import torch
        device = torch.device('cuda')
        dtype = torch.float64

        mu_t = torch.as_tensor(mu_mom, device=device, dtype=dtype)
        sigma_t = torch.as_tensor(sigma_mom, device=device, dtype=dtype)
        n_per_pattern_t = torch.as_tensor(
            index.n_per_pattern, device=device, dtype=dtype,
        )
        obs_idx_t = torch.as_tensor(index.obs_idx, device=device, dtype=torch.long)
        obs_mask_t = torch.as_tensor(index.obs_mask, device=device, dtype=torch.bool)
        eye_oo = torch.eye(
            index.v_obs_max, device=device, dtype=dtype,
        ).expand(index.n_patterns, -1, -1)

        # Pre-compute per-pattern observed means on-device — still
        # a small Python loop over patterns because n_k varies, but
        # the values transfer to device once.
        y_bars_np = np.zeros((index.n_patterns, index.v_obs_max),
                             dtype=np.float64)
        for k, pattern in enumerate(patterns):
            if pattern.n_observed == 0:
                continue
            y_bars_np[k, :pattern.n_observed] = np.mean(pattern.data, axis=0)
        y_bars_t = torch.as_tensor(y_bars_np, device=device, dtype=dtype)

        test_statistic, n_patterns_used, n_regularized = (
            chi_square_mcar_batched_torch(
                mu_t, sigma_t, index, n_per_pattern_t, obs_idx_t, obs_mask_t,
                y_bars_t, eye_oo, torch, device, dtype,
                condition_threshold=condition_threshold,
                regularize=regularize,
            )
        )
    else:
        test_statistic, n_patterns_used, n_regularized = (
            chi_square_mcar_batched_np(
                mu_mom, sigma_mom, patterns, index,
                condition_threshold=condition_threshold,
                regularize=regularize,
            )
        )

    df = sum(p.n_observed for p in patterns) - n_vars

    if len(patterns) == 1 and patterns[0].n_observed == n_vars:
        return MCARTestResult(
            statistic=0.0,
            df=0,
            p_value=1.0,
            rejected=False,
            alpha=alpha,
            patterns=patterns,
            n_patterns=1,
            n_patterns_used=0,
            ml_mean=mu_mom,
            ml_cov=sigma_mom,
            convergence_warnings=["No missing data - MCAR test not applicable"],
            method="Method-of-moments (pairwise-deletion plug-in)",
        )

    if df <= 0:
        raise ValueError(f"Invalid degrees of freedom: {df}")

    p_value = 1 - stats.chi2.cdf(test_statistic, df)
    rejected = p_value < alpha

    return MCARTestResult(
        statistic=test_statistic,
        df=df,
        p_value=p_value,
        rejected=rejected,
        alpha=alpha,
        patterns=patterns,
        n_patterns=len(patterns),
        n_patterns_used=n_patterns_used,
        ml_mean=mu_mom,
        ml_cov=sigma_mom,
        convergence_warnings=convergence_warnings,
        method="Method-of-moments (pairwise-deletion plug-in)",
    )
