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
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.core.compute.backend import (
    resolve_backend, valid_backends, unknown_backend_message,
    FP64_REQUIRES_CUDA_MSG,
)
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
from pystatistics.ordinal._information import (
    observed_information,
    raw_to_natural_jacobian,
)
from pystatistics.ordinal.solution import OrdinalSolution


# L-BFGS-B's CPU function tolerance. The historical 1e-15 sits below the
# line-search's achievable precision once the iterate reaches the optimum,
# so L-BFGS-B routinely reported ABNORMAL_TERMINATION_IN_LNSRCH on fits that
# had in fact converged (the score was ~0). R's optim() BFGS uses
# reltol = sqrt(eps) ~ 1.5e-8 and never sees this. We relax to a comparable
# value and delegate the convergence *decision* to a score-based guard
# (_accept_fit_vcov) rather than the optimizer's success flag.
_CPU_FTOL = 1e-10

# Convergence acceptance threshold on the half-Newton-decrement
# 0.5 * grad^T H^{-1} grad, which estimates the remaining negative-log-
# likelihood improvement to the MLE (in deviance/2 units) and is invariant to
# the parameterization. The log-likelihood is concave, so wherever the
# observed information is positive definite a Newton step moves toward the
# unique maximum; we therefore polish the optimizer's iterate with a few
# safeguarded Newton steps until the decrement falls below this threshold.
# Only genuine data separation (a non-positive-definite / runaway information)
# fails to converge and is rejected.
_DECREMENT_TOL = 1e-8

# Maximum safeguarded Newton polishing steps after the quasi-Newton optimizer.
# L-BFGS-B leaves the iterate within the Newton basin (small gradient), so a
# handful of quadratically-convergent steps reach the MLE; the cap is a guard
# against a non-converging (separated) problem, which the decrement test then
# rejects.
_MAX_POLISH_STEPS = 12


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
    ridge: float = 0.0,
) -> tuple[NDArray, int, bool, float, NDArray]:
    """
    Fit proportional odds model via L-BFGS-B with analytical gradient.

    Convergence is decided by a score-based guard (see ``_accept_fit_vcov``)
    rather than by L-BFGS-B's ``success`` flag: the flag spuriously reports
    failure at the optimum under the tight tolerances the model needs, and a
    loose tolerance could spuriously report success short of it. The guard
    instead verifies that the maximum-likelihood conditions hold at the
    returned point.

    Args:
        y_codes: Integer codes 0, ..., K-1, length n.
        X: Design matrix (n, p), no intercept.
        link: Link function instance.
        n_levels: Number of categories K.
        tol: Convergence tolerance for L-BFGS-B (gtol).
        max_iter: Maximum optimizer iterations.
        ridge: Optional L2 (ridge) penalty on the slopes beta (see ``polr``).
            With ``ridge=0.0`` (default) this is the exact maximum-likelihood
            fit. A positive value makes the penalized objective strongly convex,
            so a (quasi-)separated problem converges in a handful of iterations
            with a positive-definite observed information instead of running
            L-BFGS-B to ``max_iter`` and being rejected by the guard.

    Returns:
        Tuple of (optimal_params, n_iter, converged, neg_loglik, vcov_raw),
        where vcov_raw is the raw-coordinate variance-covariance matrix.

    Raises:
        ConvergenceError: If the score-based guard rejects the fit (the
            observed information is not positive definite, or the iterate is
            not at the maximum — typically data separation with ``ridge=0``).
    """
    p = X.shape[1]
    x0 = _compute_starting_values(y_codes, n_levels, link, p)

    result = minimize(
        fun=cumulative_negloglik,
        x0=x0,
        args=(y_codes, X, link, n_levels, ridge),
        method='L-BFGS-B',
        jac=cumulative_gradient,
        options={
            'maxiter': max_iter,
            'gtol': tol,
            'ftol': _CPU_FTOL,
        },
    )

    params_mle, neg_loglik, vcov_raw = _polish_to_mle(
        result.x, y_codes, X, link, n_levels, result.nit, result.message,
        ridge=ridge,
    )
    return params_mle, result.nit, True, neg_loglik, vcov_raw


def _polish_to_mle(
    params: NDArray[np.floating[Any]],
    y_codes: NDArray[np.integer[Any]],
    X: NDArray[np.floating[Any]],
    link: Link,
    n_levels: int,
    n_iter: int,
    message: Any,
    ridge: float = 0.0,
) -> tuple[NDArray[np.floating[Any]], float, NDArray[np.floating[Any]]]:
    """
    Safeguarded Newton polish to the MLE, with the score-based convergence
    guard; returns the polished estimate, its negative log-likelihood, and the
    raw-coordinate vcov.

    At each step the analytic observed information ``H`` is built (reusing a
    single Cholesky factor for the decrement test, the Newton step, and the
    final ``H^{-1}``). The fit is accepted once the half-Newton-decrement
    ``0.5 * grad^T H^{-1} grad`` — the estimated remaining log-likelihood gain
    to the MLE — falls below ``_DECREMENT_TOL``. Because the log-likelihood is
    concave, a backtracking-safeguarded Newton step always reduces it while the
    information is positive definite; if the information ceases to be positive
    definite the data are separated and the fit is rejected.

    This replaces trusting the optimizer's ``success`` flag (see ``_fit_polr``).

    Args:
        params: Raw-parameterization estimate [raw_thresholds, beta] from the
            quasi-Newton optimizer.
        y_codes: Integer response codes.
        X: Design matrix.
        link: Link function instance.
        n_levels: Number of categories.
        n_iter: Optimizer iteration count (for error context).
        message: Optimizer termination message (for error context).
        ridge: Optional L2 (ridge) penalty on the slopes beta (see ``polr``).
            The polish optimizes the same penalized objective the optimizer did,
            so the decrement test, Newton step, and returned ``vcov_raw`` are all
            for the penalized fit. Default 0.0 is the unpenalized MLE polish.

    Returns:
        Tuple (params_mle, neg_loglik, vcov_raw): the polished estimate, its
        negative log-likelihood, and the raw-coordinate variance-covariance.

    Raises:
        ConvergenceError: If the observed information is not positive definite
            (data separation) or the decrement stays above threshold after the
            Newton-step budget is exhausted.
    """
    x = np.array(params, dtype=np.float64, copy=True)

    for _ in range(_MAX_POLISH_STEPS + 1):
        grad = cumulative_gradient(x, y_codes, X, link, n_levels, ridge)
        hess = observed_information(
            x, y_codes, X, link, n_levels, grad0=grad, ridge=ridge,
        )

        try:
            chol = cho_factor(hess, lower=True)
        except np.linalg.LinAlgError as exc:
            raise ConvergenceError(
                "polr observed information is not positive definite "
                f"(likely data separation): {exc}. Optimizer ran {n_iter} "
                f"iterations ({message}).",
                iterations=n_iter,
                reason="non-PD observed information",
            ) from exc

        step = cho_solve(chol, grad)
        half_decrement = 0.5 * float(grad @ step)
        if np.isfinite(half_decrement) and half_decrement <= _DECREMENT_TOL:
            # Report the genuine (unpenalized) log-likelihood at the estimate so
            # deviance/AIC describe the model fit, not the penalized objective.
            neg_loglik = float(
                cumulative_negloglik(x, y_codes, X, link, n_levels)
            )
            vcov_raw = cho_solve(chol, np.eye(len(grad)))
            return x, neg_loglik, vcov_raw

        # Backtracking Newton step (concave objective => the full step reduces
        # the negative log-likelihood near the optimum; backtrack otherwise).
        f0 = cumulative_negloglik(x, y_codes, X, link, n_levels, ridge)
        t = 1.0
        while t >= 1e-4:
            x_new = x - t * step
            if cumulative_negloglik(x_new, y_codes, X, link, n_levels, ridge) < f0:
                break
            t *= 0.5
        else:
            break  # no productive step found; reject below
        x = x_new

    raise ConvergenceError(
        "polr did not reach the maximum likelihood estimate within the Newton "
        f"polishing budget (likely data separation). Optimizer ran {n_iter} "
        f"iterations ({message}).",
        iterations=n_iter,
        reason="score test failed",
    )


def _fit_polr_gpu(
    y_codes: NDArray[np.integer[Any]],
    X: NDArray[np.floating[Any]] | Any,   # numpy OR torch.Tensor
    link: Link,
    n_levels: int,
    tol: float,
    max_iter: int,
    device: str,
    use_fp64: bool,
) -> tuple[NDArray, int, bool, float, Any]:
    """GPU counterpart of :func:`_fit_polr`.

    Builds a :class:`PolrGPULikelihood` that keeps X and y_codes on
    device across all L-BFGS-B evaluations, returns the optimizer
    solution, and also returns the likelihood object so the caller can
    reuse it for the analytical-autograd Hessian (vcov) without
    rebuilding the GPU tensors.
    """
    from pystatistics.ordinal.backends.gpu_likelihood import PolrGPULikelihood

    p = X.shape[1]
    x0 = _compute_starting_values(y_codes, n_levels, link, p)

    like = PolrGPULikelihood(
        X, y_codes, n_levels, link_name=link.name,
        device=device, use_fp64=use_fp64,
    )

    # On FP32 the CPU ftol=1e-15 is well below gradient precision
    # (~1e-7) and routinely triggers L-BFGS-B "ABNORMAL" line-search
    # stalls. Use tol for both gtol and ftol on the GPU FP32 path; on
    # FP64 keep the tight CPU-matching ftol.
    ftol = 1e-15 if use_fp64 else tol
    result = minimize(
        fun=like.fun,
        x0=x0,
        method="L-BFGS-B",
        jac=like.jac,
        options={"maxiter": max_iter, "gtol": tol, "ftol": ftol},
    )
    converged = bool(result.success)
    n_iter = int(result.nit)
    neg_loglik = float(result.fun)

    if not converged:
        raise ConvergenceError(
            f"polr optimizer did not converge: {result.message}. "
            f"Completed {n_iter} iterations. Try increasing max_iter "
            f"or check data for separation.",
            iterations=n_iter,
            reason=str(result.message),
        )

    return result.x, n_iter, converged, neg_loglik, like


def polr(
    y: ArrayLike,
    X: ArrayLike,
    *,
    link: str = 'logistic',
    names: list[str] | None = None,
    category_names: list[str] | None = None,
    tol: float = 1e-8,
    max_iter: int = 200,
    l2: float = 0.0,
    backend: str | None = None,
    conf_level: float = 0.95,
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
        link: Link function. One of 'logistic' (default, proportional
            odds), 'probit', or 'cloglog'.
        names: Labels for the p predictor columns. If None, defaults to
            'x1', 'x2', etc.
        category_names: Labels for the K ordered categories. If None,
            defaults to '0', '1', etc.
        tol: Convergence tolerance (gradient norm) for L-BFGS-B.
        max_iter: Maximum number of optimizer iterations.
        l2: L2 (ridge) penalty coefficient on the slope coefficients beta
            (the threshold/intercept parameters are never penalized). The
            penalty ``0.5 * l2 * ||beta||^2`` is added to the negative
            log-likelihood. The default 0.0 fits the exact, R-validated
            maximum-likelihood model. A small positive value makes the
            objective strongly convex, which keeps the fit finite and
            well-conditioned under (quasi-)complete separation — where the
            unpenalized slopes would diverge, L-BFGS-B would run to ``max_iter``,
            and the fit would be rejected. Reported coefficients, standard
            errors, and vcov are then those of the penalized fit. CPU backend
            only; the GPU backend applies its own ridge stabilization.
        backend: Compute backend = (device, precision). ``'cpu'`` (default):
            the R-validated reference path. ``'gpu'``: float32 (raises if no
            GPU). ``'gpu_fp64'``: float64, CUDA only (raises on MPS).
            ``'auto'``: GPU-float32 if CUDA is present, else CPU.

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
        >>> sol = polr(y, X, link='logistic', names=['x1', 'x2'])
        >>> print(sol.summary())
    """
    if conf_level <= 0 or conf_level >= 1:
        raise ValidationError(f"conf_level must be in (0, 1), got {conf_level}")

    # Convention (see CONVENTIONS.md): numpy input defaults
    # to CPU, GPU torch.Tensor input defaults to GPU. Explicit
    # backend='cpu' with a GPU tensor raises (Rule 1: no silent
    # migration).
    import sys as _sys
    _is_X_tensor = (
        "torch" in _sys.modules
        and isinstance(X, _sys.modules["torch"].Tensor)
    )

    if _is_X_tensor:
        import torch
        if X.ndim != 2:
            raise ValidationError(f"X: expected 2-D tensor, got {X.ndim}-D")
        if not torch.isfinite(X).all():
            raise ValidationError("X contains non-finite values")
        if backend in (None, "auto"):
            backend = "gpu" if X.device.type != "cpu" else "cpu"
        if backend == "cpu":
            raise ValidationError(
                "backend='cpu' was specified but X is a torch.Tensor "
                f"on device {X.device}. Either pass a numpy array / "
                "CPU DataSource to the CPU backend, or call `.to('cpu')` "
                "on the DataSource explicitly to move it back."
            )
        if backend not in ("gpu", "gpu_fp64"):
            raise ValidationError(
                unknown_backend_message(backend, valid_backends(True))
            )
        # Precision lives in the backend string; guard fp64-needs-CUDA against
        # the tensor's own device.
        use_fp64 = backend == "gpu_fp64"
        if use_fp64 and X.device.type != "cuda":
            raise RuntimeError(FP64_REQUIRES_CUDA_MSG)
        gpu_device_type = X.device.type
        # Pull y (tiny integer vector) to CPU for the shared validator.
        y_host = (
            y.detach().cpu().numpy()
            if isinstance(y, torch.Tensor) else y
        )
        y_codes, _discard_X, col_names, lvl_names, n_levels = _validate_inputs(
            y_host, np.zeros((X.shape[0], X.shape[1]), dtype=np.float64),
            names, category_names,
        )
        # Replace the placeholder numpy X with the real device tensor.
        X_arr = None
        X_for_gpu = X
        n, p = X.shape
    else:
        target = resolve_backend(backend, supports_fp64=True)
        backend = target.backend
        use_fp64 = target.use_fp64
        gpu_device_type = target.device_type
        y_codes, X_arr, col_names, lvl_names, n_levels = _validate_inputs(
            y, X, names, category_names,
        )
        X_for_gpu = None
        n, p = X_arr.shape

    if not np.isfinite(l2) or l2 < 0.0:
        raise ValidationError(
            f"l2: must be a finite non-negative float, got {l2!r}"
        )
    # The L2 penalty is wired through the CPU L-BFGS-B/Newton path only; the
    # GPU path applies its own ridge stabilization. Fail loud rather than
    # silently ignore a requested penalty (Rule 1).
    if l2 > 0.0 and backend != "cpu":
        raise ValidationError(
            "l2 > 0 is only supported on the CPU backend (the GPU backend "
            "applies its own ridge stabilization). Use backend='cpu' or "
            "l2=0.0."
        )

    # Resolve the link function (``link`` is the user's string choice).
    link_fn = _resolve_method_link(link)

    # FP32 gradient precision is ~1e-7 — floor tol on the GPU FP32 path
    # for the same reason as multinomial (L-BFGS-B ABNORMAL otherwise).
    gpu_fp32_min_tol = 1e-5
    effective_tol = tol
    if backend != "cpu" and not use_fp64 and tol < gpu_fp32_min_tol:
        effective_tol = gpu_fp32_min_tol

    gpu_like = None
    vcov_raw = None

    if backend == "cpu":
        opt_params, n_iter, converged, neg_loglik, vcov_raw = _fit_polr(
            y_codes, X_arr, link_fn, n_levels, tol, max_iter, ridge=l2,
        )
        backend_name = "cpu_lbfgsb"
    else:
        # GPU path: device + precision resolved above. ``X_for_gpu`` is the
        # device tensor (tensor input); ``X_arr`` the numpy design (numpy input).
        X_gpu_in = X_for_gpu if X_arr is None else X_arr
        opt_params, n_iter, converged, neg_loglik, gpu_like = _fit_polr_gpu(
            y_codes, X_gpu_in, link_fn, n_levels, effective_tol, max_iter,
            device=gpu_device_type, use_fp64=use_fp64,
        )
        backend_name = (
            f"gpu_lbfgsb ({gpu_device_type}, {'fp64' if use_fp64 else 'fp32'})"
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

    # Both backends produce the variance-covariance in raw (unconstrained
    # log-gap) coordinates; the GPU path computes it from its autograd
    # Hessian here, the CPU path returned it from the score guard above.
    # Map to natural threshold coordinates (delta method) so the reported
    # threshold SEs and the MICE posterior draw over [alpha, beta] are on the
    # same scale as MASS::polr.
    if gpu_like is not None:
        vcov_raw = gpu_like.compute_vcov(opt_params)
    jac = raw_to_natural_jacobian(raw_thresh, vcov_raw.shape[0])
    vcov = jac @ vcov_raw @ jac.T

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
        link=link.lower(),
    )

    result = Result(
        params=params,
        info={
            'method': link.lower(),
            'link': link_fn.name,
            'converged': converged,
            'iterations': n_iter,
        },
        timing=None,
        backend_name=backend_name,
    )

    return OrdinalSolution(
        _result=result,
        _names=col_names,
        _level_names=lvl_names,
        _conf_level=conf_level,
    )
