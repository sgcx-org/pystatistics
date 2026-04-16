"""
Solver for multinomial logistic regression.

Fits the multinomial logit model via L-BFGS-B optimization with
analytical gradient. Computes the variance-covariance matrix from
the numerical Hessian at convergence.

Public API:
    multinom(y, X, ...) -> MultinomialSolution
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize, approx_fprime

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.core.result import Result
from pystatistics.core.validation import (
    check_array,
    check_1d,
    check_2d,
    check_finite,
    check_consistent_length,
)
from pystatistics.multinomial._common import MultinomialParams
from pystatistics.multinomial._likelihood import (
    multinomial_negloglik,
    multinomial_gradient,
    compute_probs,
)
from pystatistics.multinomial.solution import MultinomialSolution


def _validate_inputs(
    y: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.int64], NDArray[np.floating[Any]], int]:
    """Validate and prepare inputs for multinomial fitting.

    Args:
        y: Response vector (will be converted to integer codes).
        X: Design matrix.

    Returns:
        Tuple of (y_codes as int64, X as float64, n_classes).

    Raises:
        ValidationError: If inputs are invalid.
        DimensionError: If dimensions are inconsistent.
    """
    check_1d(y, "y")
    check_2d(X, "X")
    check_finite(y, "y")
    check_finite(X, "X")
    check_consistent_length(y, X, names=("y", "X"))

    # Convert y to integer codes
    y_codes = y.astype(np.int64)
    if not np.allclose(y, y_codes):
        raise ValidationError(
            "y: must contain integer class codes (0, 1, ..., J-1), "
            f"but found non-integer values"
        )

    unique_classes = np.unique(y_codes)
    n_classes = len(unique_classes)

    if n_classes < 2:
        raise ValidationError(
            f"y: requires at least 2 distinct classes, got {n_classes}"
        )

    # Verify codes are 0, 1, ..., J-1
    expected = np.arange(n_classes)
    if not np.array_equal(unique_classes, expected):
        raise ValidationError(
            f"y: class codes must be consecutive integers starting from 0. "
            f"Got unique values {unique_classes.tolist()}, "
            f"expected {expected.tolist()}"
        )

    n, p = X.shape
    if n <= p * (n_classes - 1):
        raise ValidationError(
            f"Insufficient observations: n={n} but model has "
            f"{p * (n_classes - 1)} parameters. "
            f"Need n > (J-1)*p = {p * (n_classes - 1)}."
        )

    return y_codes, X, n_classes


def _compute_null_loglik(y_codes: NDArray[np.int64], n_classes: int) -> float:
    """Compute log-likelihood of the null (intercept-only) model.

    The null model predicts each class with its marginal frequency:
        P(Y = j) = n_j / n

    Args:
        y_codes: Integer class codes of shape (n,).
        n_classes: Number of classes J.

    Returns:
        Null model log-likelihood.
    """
    n = len(y_codes)
    counts = np.bincount(y_codes, minlength=n_classes)
    proportions = counts / n

    # Log-likelihood: sum over all observations of log(p_j)
    # Equivalent to sum_j n_j * log(n_j / n)
    loglik = 0.0
    for j in range(n_classes):
        if counts[j] > 0:
            loglik += counts[j] * np.log(proportions[j])

    return float(loglik)


def _fit_multinom(
    y_codes: NDArray[np.int64],
    X: NDArray[np.floating[Any]],
    n_classes: int,
    tol: float,
    max_iter: int,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], int, bool]:
    """Fit multinomial logit model via L-BFGS-B.

    Args:
        y_codes: Integer class codes of shape (n,).
        X: Design matrix of shape (n, p).
        n_classes: Number of classes J.
        tol: Convergence tolerance for the optimizer.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple of (params_flat, vcov, n_iter, converged).

    Raises:
        ConvergenceError: If optimization fails to converge.
    """
    n, p = X.shape
    n_nonref = n_classes - 1
    n_params = n_nonref * p

    # One-hot encode y
    y_onehot = np.zeros((n, n_classes), dtype=np.float64)
    y_onehot[np.arange(n), y_codes] = 1.0

    # Starting values: all zeros
    x0 = np.zeros(n_params, dtype=np.float64)

    # Optimize with L-BFGS-B
    result = minimize(
        fun=multinomial_negloglik,
        x0=x0,
        args=(y_onehot, X, n_classes),
        method="L-BFGS-B",
        jac=multinomial_gradient,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )

    converged = result.success
    n_iter = int(result.nit)
    params_flat = result.x

    if not converged:
        raise ConvergenceError(
            f"Multinomial logit optimization did not converge after "
            f"{n_iter} iterations. Optimizer message: {result.message}",
            iterations=n_iter,
            reason=str(result.message),
            threshold=tol,
        )

    # Compute variance-covariance matrix via numerical Hessian
    vcov = _compute_vcov(params_flat, y_onehot, X, n_classes)

    return params_flat, vcov, n_iter, converged


def _compute_vcov(
    params_flat: NDArray[np.floating[Any]],
    y_onehot: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    n_classes: int,
) -> NDArray[np.floating[Any]]:
    """Compute variance-covariance matrix from numerical Hessian.

    Uses finite differences to approximate the Hessian of the negative
    log-likelihood, then inverts it to get the vcov matrix.

    Args:
        params_flat: Fitted parameter vector.
        y_onehot: One-hot encoded responses.
        X: Design matrix.
        n_classes: Number of classes.

    Returns:
        Variance-covariance matrix of shape (n_params, n_params).
    """
    n_params = len(params_flat)
    eps = np.sqrt(np.finfo(np.float64).eps)

    # Numerical Hessian via finite differences of the gradient
    hessian = np.zeros((n_params, n_params), dtype=np.float64)
    for i in range(n_params):
        ei = np.zeros(n_params, dtype=np.float64)
        ei[i] = eps

        grad_plus = multinomial_gradient(
            params_flat + ei, y_onehot, X, n_classes
        )
        grad_minus = multinomial_gradient(
            params_flat - ei, y_onehot, X, n_classes
        )
        hessian[i, :] = (grad_plus - grad_minus) / (2 * eps)

    # Symmetrize
    hessian = (hessian + hessian.T) / 2

    # Invert to get vcov
    try:
        vcov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        # If Hessian is singular, use pseudo-inverse
        vcov = np.linalg.pinv(hessian)

    return vcov


def multinom(
    y: ArrayLike,
    X: ArrayLike,
    *,
    names: list[str] | None = None,
    class_names: list[str] | None = None,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> MultinomialSolution:
    """Fit a multinomial logistic regression model.

    Estimates the multinomial logit (softmax regression) model via
    maximum likelihood using L-BFGS-B optimization. The last class
    (highest code) is the reference category, matching the convention
    used by R's nnet::multinom().

    The model assumes:
        P(Y = j | x) = exp(x' beta_j) / sum_k exp(x' beta_k)
    with beta_J = 0 for identifiability.

    Args:
        y: Response vector of integer class codes 0, 1, ..., J-1.
            Shape (n,).
        X: Design matrix of shape (n, p). The user is responsible for
            including an intercept column if desired, matching R's
            behavior where formula syntax handles intercept addition.
        names: Optional list of predictor names matching columns of X.
            If None, defaults to ["x0", "x1", ...].
        class_names: Optional list of class labels in order 0, 1, ..., J-1.
            If None, defaults to ["0", "1", ...].
        tol: Convergence tolerance for L-BFGS-B optimizer. Both
            function tolerance (ftol) and gradient tolerance (gtol)
            are set to this value.
        max_iter: Maximum number of optimizer iterations.

    Returns:
        MultinomialSolution wrapping the fitted model.

    Raises:
        ValidationError: If inputs fail validation.
        ConvergenceError: If optimization does not converge.

    Examples:
        >>> import numpy as np
        >>> from pystatistics.multinomial import multinom
        >>> rng = np.random.default_rng(42)
        >>> n = 200
        >>> X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
        >>> # True model: 3 classes, last is reference
        >>> y = rng.choice(3, size=n)
        >>> result = multinom(y, X)
        >>> result.summary()
    """
    t0 = time.perf_counter()

    # Validate and convert inputs
    y_arr = check_array(y, "y")
    X_arr = check_array(X, "X")
    y_codes, X_arr, n_classes = _validate_inputs(y_arr, X_arr)

    n, p = X_arr.shape

    # Set default names
    if names is not None:
        if len(names) != p:
            raise ValidationError(
                f"names: length {len(names)} does not match number of "
                f"predictors {p}"
            )
        feature_names = tuple(names)
    else:
        feature_names = tuple(f"x{i}" for i in range(p))

    if class_names is not None:
        if len(class_names) != n_classes:
            raise ValidationError(
                f"class_names: length {len(class_names)} does not match "
                f"number of classes {n_classes}"
            )
        cls_names = tuple(class_names)
    else:
        cls_names = tuple(str(i) for i in range(n_classes))

    # Fit the model
    params_flat, vcov, n_iter, converged = _fit_multinom(
        y_codes, X_arr, n_classes, tol, max_iter
    )

    # Compute fitted values
    fitted_probs = compute_probs(params_flat, X_arr, n_classes)

    # Compute log-likelihoods
    y_onehot = np.zeros((n, n_classes), dtype=np.float64)
    y_onehot[np.arange(n), y_codes] = 1.0
    log_lik = -multinomial_negloglik(params_flat, y_onehot, X_arr, n_classes)

    null_loglik = _compute_null_loglik(y_codes, n_classes)

    # Model quantities
    n_nonref = n_classes - 1
    n_estimated_params = n_nonref * p
    deviance = -2.0 * log_lik
    null_deviance = -2.0 * null_loglik
    aic = deviance + 2.0 * n_estimated_params

    coef_matrix = params_flat.reshape(n_nonref, p)

    elapsed = time.perf_counter() - t0

    # Build result
    params = MultinomialParams(
        coefficient_matrix=coef_matrix,
        vcov=vcov,
        fitted_probs=fitted_probs,
        log_likelihood=log_lik,
        deviance=deviance,
        null_deviance=null_deviance,
        aic=aic,
        n_obs=n,
        n_classes=n_classes,
        class_names=cls_names,
        feature_names=feature_names,
        n_iter=n_iter,
        converged=converged,
    )

    result = Result(
        params=params,
        info={
            "method": "L-BFGS-B",
            "converged": converged,
            "iterations": n_iter,
            "n_parameters": n_estimated_params,
        },
        timing={"total_seconds": elapsed},
        backend_name="cpu_lbfgsb",
    )

    return MultinomialSolution(result)
