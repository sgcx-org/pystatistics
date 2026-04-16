"""
Proportional odds model solver.

Fits cumulative link models via L-BFGS-B optimization with analytical
gradients. Matches R's MASS::polr() for logistic, probit, and cloglog
link functions.

The entry point is polr(), which validates inputs, computes starting
values, runs the optimizer, and returns an OrdinalSolution.

References
----------
    Venables, W. N. & Ripley, B. D. (2002). Modern Applied Statistics with S.
    R package MASS, function polr().
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize, approx_fprime

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.core.validation import (
    check_array,
    check_1d,
    check_2d,
    check_finite,
)
from pystatistics.core.result import Result
from pystatistics.regression.families import LogitLink, ProbitLink, Link
from pystatistics.ordinal._common import OrdinalParams
from pystatistics.ordinal._likelihood import (
    CLogLogLink,
    cumulative_negloglik,
    cumulative_gradient,
    raw_to_thresholds,
    thresholds_to_raw,
)
from pystatistics.ordinal.solution import OrdinalSolution


_LINK_MAP: dict[str, type[Link]] = {
    'logistic': LogitLink,
    'probit': ProbitLink,
    'cloglog': CLogLogLink,
}


def _resolve_method_link(method: str) -> Link:
    """
    Resolve a method name to a Link instance.

    Args:
        method: One of 'logistic', 'probit', 'cloglog'.

    Returns:
        Corresponding Link instance.

    Raises:
        ValidationError: If method is not recognized.
    """
    cls = _LINK_MAP.get(method.lower())
    if cls is None:
        valid = ', '.join(sorted(_LINK_MAP.keys()))
        raise ValidationError(
            f"Unknown method: {method!r}. Valid methods: {valid}"
        )
    return cls()


def _compute_starting_values(
    y_codes: NDArray[np.integer[Any]],
    n_levels: int,
    link: Link,
    p: int,
) -> NDArray[np.floating[Any]]:
    """
    Compute starting values for the optimizer.

    Threshold starting values are derived from empirical cumulative
    proportions transformed through the link function. Beta starting
    values are zeros.

    Args:
        y_codes: Integer response codes 0, ..., K-1 of length n.
        n_levels: Number of categories K.
        link: Link function instance.
        p: Number of predictor variables.

    Returns:
        Starting parameter vector [raw_thresholds, beta] of length K-1+p.
    """
    n = len(y_codes)
    n_thresh = n_levels - 1

    # Empirical cumulative proportions
    counts = np.bincount(y_codes, minlength=n_levels)
    cum_props = np.cumsum(counts[:n_thresh]) / n

    # Clamp away from 0 and 1 for link function stability
    cum_props = np.clip(cum_props, 0.01, 0.99)

    # Transform through link function to get initial thresholds
    alpha_init = link.link(cum_props)

    # Ensure strict ordering (in case link function produces ties)
    for j in range(1, n_thresh):
        if alpha_init[j] <= alpha_init[j - 1]:
            alpha_init[j] = alpha_init[j - 1] + 0.1

    # Convert to unconstrained parameterization
    raw_init = thresholds_to_raw(alpha_init)

    # Beta starts at zero
    beta_init = np.zeros(p)

    return np.concatenate([raw_init, beta_init])


def _validate_inputs(
    y: ArrayLike,
    X: ArrayLike,
    names: list[str] | None,
    level_names: list[str] | None,
) -> tuple[
    NDArray[np.integer[Any]],
    NDArray[np.floating[Any]],
    list[str],
    list[str],
    int,
]:
    """
    Validate and prepare inputs for polr fitting.

    Args:
        y: Response variable (integer codes or convertible).
        X: Design matrix (no intercept).
        names: Column names for X.
        level_names: Category labels.

    Returns:
        Tuple of (y_codes, X_array, names, level_names, n_levels).

    Raises:
        ValidationError: On invalid inputs.
        DimensionError: On shape mismatches.
    """
    # Validate X
    X_arr = check_array(X, 'X')
    check_finite(X_arr, 'X')
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    check_2d(X_arr, 'X')
    n, p = X_arr.shape

    # Validate y
    y_arr = check_array(y, 'y')
    check_finite(y_arr, 'y')
    check_1d(y_arr, 'y')

    if len(y_arr) != n:
        raise ValidationError(
            f"y length ({len(y_arr)}) must match X rows ({n})"
        )

    # Convert y to integer codes
    y_codes = y_arr.astype(np.intp)
    if not np.allclose(y_arr, y_codes):
        raise ValidationError(
            "y must contain integer values (category codes 0, 1, ..., K-1)"
        )

    unique_codes = np.unique(y_codes)
    min_code = unique_codes[0]
    max_code = unique_codes[-1]
    n_levels = max_code - min_code + 1

    if min_code != 0:
        raise ValidationError(
            f"y codes must start at 0, got minimum code {min_code}"
        )

    # Check that all levels are present
    expected = np.arange(n_levels)
    missing = np.setdiff1d(expected, unique_codes)
    if len(missing) > 0:
        raise ValidationError(
            f"y has gaps in category codes. Missing levels: {missing.tolist()}. "
            f"All codes 0 through {n_levels - 1} must be present."
        )

    if n_levels < 2:
        raise ValidationError(
            f"y must have at least 2 distinct levels, got {n_levels}"
        )

    # Validate names
    if names is not None:
        if len(names) != p:
            raise ValidationError(
                f"names length ({len(names)}) must match X columns ({p})"
            )
    else:
        names = [f"x{i + 1}" for i in range(p)]

    # Validate level_names
    if level_names is not None:
        if len(level_names) != n_levels:
            raise ValidationError(
                f"level_names length ({len(level_names)}) must match "
                f"number of levels ({n_levels})"
            )
    else:
        level_names = [str(i) for i in range(n_levels)]

    return y_codes, X_arr, names, level_names, n_levels


def _fit_polr(
    y_codes: NDArray[np.integer[Any]],
    X: NDArray[np.floating[Any]],
    link: Link,
    n_levels: int,
    tol: float,
    max_iter: int,
) -> tuple[NDArray, int, bool, float]:
    """
    Fit proportional odds model via L-BFGS-B with analytical gradient.

    Args:
        y_codes: Integer codes 0, ..., K-1, length n.
        X: Design matrix (n, p), no intercept.
        link: Link function instance.
        n_levels: Number of categories K.
        tol: Convergence tolerance for L-BFGS-B (gtol).
        max_iter: Maximum optimizer iterations.

    Returns:
        Tuple of (optimal_params, n_iter, converged, neg_loglik).

    Raises:
        ConvergenceError: If optimizer fails to converge and result
            is not usable.
    """
    p = X.shape[1]
    x0 = _compute_starting_values(y_codes, n_levels, link, p)

    result = minimize(
        fun=cumulative_negloglik,
        x0=x0,
        args=(y_codes, X, link, n_levels),
        method='L-BFGS-B',
        jac=cumulative_gradient,
        options={
            'maxiter': max_iter,
            'gtol': tol,
            'ftol': 1e-15,
        },
    )

    converged = result.success
    n_iter = result.nit
    neg_loglik = result.fun

    if not converged:
        raise ConvergenceError(
            f"polr optimizer did not converge: {result.message}. "
            f"Completed {n_iter} iterations. Try increasing max_iter "
            f"or check data for separation.",
            iterations=n_iter,
            reason=str(result.message),
        )

    return result.x, n_iter, converged, neg_loglik


def _compute_vcov(
    params: NDArray[np.floating[Any]],
    y_codes: NDArray[np.integer[Any]],
    X: NDArray[np.floating[Any]],
    link: Link,
    n_levels: int,
) -> NDArray[np.floating[Any]]:
    """
    Compute variance-covariance matrix via numerical Hessian.

    Uses scipy.optimize.approx_fprime to compute the Hessian of the
    negative log-likelihood, then inverts it to get vcov.

    Args:
        params: Fitted parameters [raw_thresholds, beta].
        y_codes: Integer response codes.
        X: Design matrix.
        link: Link function instance.
        n_levels: Number of categories.

    Returns:
        Variance-covariance matrix, shape (K-1+p, K-1+p).
    """
    n_params = len(params)
    eps = np.sqrt(np.finfo(float).eps)

    # Compute Hessian row by row using approx_fprime on the gradient
    hessian = np.empty((n_params, n_params))
    for i in range(n_params):
        def grad_i(x: NDArray) -> float:
            """Extract i-th component of the gradient."""
            return cumulative_gradient(x, y_codes, X, link, n_levels)[i]
        hessian[i, :] = approx_fprime(params, grad_i, eps)

    # Symmetrize (numerical approximation may not be perfectly symmetric)
    hessian = 0.5 * (hessian + hessian.T)

    # Invert to get vcov. The Hessian of negative log-likelihood should
    # be positive definite at the MLE.
    try:
        vcov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        # If Hessian is singular, return NaN matrix as warning
        vcov = np.full((n_params, n_params), np.nan)

    return vcov


def polr(
    y: ArrayLike,
    X: ArrayLike,
    *,
    method: str = 'logistic',
    names: list[str] | None = None,
    level_names: list[str] | None = None,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> OrdinalSolution:
    """
    Fit a proportional odds (cumulative link) model.

    Fits the model P(Y <= j | x) = g^{-1}(alpha_j - x'beta), matching
    R's MASS::polr(). The proportional odds assumption means all
    categories share the same slope coefficients beta; only the
    threshold (intercept) parameters alpha_j differ.

    Args:
        y: Ordered categorical response as integer codes 0, 1, ..., K-1.
            Must contain all levels from 0 to max(y) with no gaps.
        X: Design matrix of shape (n, p). Must NOT include an intercept
            column (thresholds serve as category-specific intercepts).
        method: Link function. One of 'logistic' (default, proportional
            odds), 'probit', or 'cloglog'.
        names: Labels for the p predictor columns. If None, defaults to
            'x1', 'x2', etc.
        level_names: Labels for the K ordered categories. If None,
            defaults to '0', '1', etc.
        tol: Convergence tolerance (gradient norm) for L-BFGS-B.
        max_iter: Maximum number of optimizer iterations.

    Returns:
        OrdinalSolution wrapping the fitted model parameters, with
        properties for coefficients, thresholds, standard errors,
        z-values, p-values, and a summary() method matching R's
        print.summary.polr output.

    Raises:
        ValidationError: If inputs fail validation (wrong shapes,
            non-integer y, missing levels, etc.).
        ConvergenceError: If the optimizer fails to converge.

    Examples:
        >>> import numpy as np
        >>> from pystatistics.ordinal import polr
        >>> rng = np.random.default_rng(42)
        >>> n = 500
        >>> X = rng.standard_normal((n, 2))
        >>> eta = 1.0 * X[:, 0] - 0.5 * X[:, 1]
        >>> # Generate ordinal y with 3 levels
        >>> cum_p1 = 1 / (1 + np.exp(-(-1 - eta)))
        >>> cum_p2 = 1 / (1 + np.exp(-(1 - eta)))
        >>> u = rng.uniform(size=n)
        >>> y = np.where(u < cum_p1, 0, np.where(u < cum_p2, 1, 2))
        >>> sol = polr(y, X, method='logistic', names=['x1', 'x2'])
        >>> print(sol.summary())
    """
    # Validate inputs
    y_codes, X_arr, col_names, lvl_names, n_levels = _validate_inputs(
        y, X, names, level_names,
    )

    # Resolve link
    link = _resolve_method_link(method)

    # Fit the model
    n, p = X_arr.shape
    opt_params, n_iter, converged, neg_loglik = _fit_polr(
        y_codes, X_arr, link, n_levels, tol, max_iter,
    )

    # Extract parameters
    n_thresh = n_levels - 1
    raw_thresh = opt_params[:n_thresh]
    beta = opt_params[n_thresh:]
    alpha = raw_to_thresholds(raw_thresh)

    log_lik = -neg_loglik
    n_total_params = n_thresh + p
    deviance = -2.0 * log_lik
    aic = deviance + 2.0 * n_total_params

    # Compute vcov on the raw (unconstrained) parameterization
    vcov = _compute_vcov(opt_params, y_codes, X_arr, link, n_levels)

    # Build result payload
    params = OrdinalParams(
        coefficients=beta,
        thresholds=alpha,
        vcov=vcov,
        log_likelihood=log_lik,
        deviance=deviance,
        aic=aic,
        n_obs=n,
        n_levels=n_levels,
        level_names=tuple(lvl_names),
        n_iter=n_iter,
        converged=converged,
        method=method.lower(),
    )

    result = Result(
        params=params,
        info={
            'method': method.lower(),
            'link': link.name,
            'converged': converged,
            'iterations': n_iter,
        },
        timing=None,
        backend_name='cpu_lbfgsb',
    )

    return OrdinalSolution(
        _result=result,
        _names=col_names,
        _level_names=lvl_names,
    )
