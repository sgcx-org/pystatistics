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

    Canonical result type for ``little_mcar_test`` (Little 1988, MLE
    plug-in). The dataclass is also reused by downstream packages that
    ship their own MCAR-test variants (e.g. Lacuna's MoM plug-in at
    ``lacuna.analysis.mcar.mom_mcar_test``) — the ``method`` field
    identifies which estimator produced the result so a caller can
    disambiguate without tracking the calling function.

    Notes
    -----
    ``ml_mean`` / ``ml_cov`` carry the plug-in estimates used in the
    chi-square statistic.
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

