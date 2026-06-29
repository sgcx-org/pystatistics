"""
Solver dispatch for regression.

Handles both Linear Models (LM) and Generalized Linear Models (GLM).

When family is None (default), the existing LM path is used → LinearSolution.
When family is provided, the IRLS path is used → GLMSolution.
"""

from typing import Literal, Union
import numpy as np
from numpy.typing import ArrayLike

from pystatistics.core.compute.backend import resolve_backend
from pystatistics.core.exceptions import ValidationError
from pystatistics.regression.design import Design
from pystatistics.regression.solution import (
    LinearSolution, LinearParams, GLMSolution, GLMParams,
)
from pystatistics.regression.backends.cpu import CPUQRBackend
from pystatistics.regression._inputs import resolve_weights, resolve_offset


# backend = (device, precision); solver = the numerical routine (LM only).
# See pystatistics/CONVENTIONS.md.
BackendChoice = Literal['auto', 'cpu', 'gpu', 'gpu_fp64']
SolverChoice = Literal['qr', 'svd']


def fit(
    X_or_design: ArrayLike | Design,
    y: ArrayLike | None = None,
    *,
    family: 'str | Family | None' = None,
    backend: BackendChoice | None = None,
    solver: SolverChoice | None = None,
    force: bool = False,
    tol: float = 1e-8,
    max_iter: int = 25,
    names: list[str] | None = None,
    l2: float = 0.0,
    weights: ArrayLike | None = None,
    offset: ArrayLike | None = None,
    conf_level: float = 0.95,
) -> Union[LinearSolution, GLMSolution]:
    """
    Fit a linear or generalized linear model.

    When family is None (default), fits ordinary least squares (LM) via
    QR decomposition or GPU Cholesky. When family is specified, fits a
    GLM via IRLS (Iteratively Reweighted Least Squares).

    Accepts EITHER:
        1. A Design object (from DataSource or arrays)
        2. Raw X and y arrays (convenience)

    Args:
        X_or_design: Design object or X matrix
        y: Response vector (required if X_or_design is array)
        family: GLM family specification. None for OLS, or a string
            ('gaussian', 'binomial', 'poisson') or Family instance.
        backend: Compute backend = (device, precision). Default None → 'cpu'
            (the R-reference path, validated for regulated-industry use).
            Values: 'cpu' (float64), 'gpu' (float32, CUDA or MPS), 'gpu_fp64'
            (float64, CUDA only), or 'auto' (GPU-fp32 if CUDA present, else CPU).
        solver: Numerical routine for the linear-model fit (family=None only):
            'qr' (default) or 'svd'. Not configurable on the GPU backend (which
            uses Cholesky on the normal equations) and not applicable to GLMs.
        force: If True, proceed with the GPU float32 Cholesky path even when it
            is unreliable — for OLS, on ill-conditioned designs; for GLM, when
            IRLS does not converge in float32 (returns the possibly-inaccurate
            fit instead of raising). Has no effect on CPU backends.
        tol: Convergence tolerance for IRLS (GLM only). Default 1e-8
            matches R's glm.control().
        max_iter: Maximum IRLS iterations (GLM only). Default 25
            matches R's glm.control().
        names: Optional list of predictor names for output labeling.
            If len(names) == p - 1 (one fewer than columns in X),
            "(Intercept)" is prepended automatically.
            If len(names) == p, used as-is.
        l2: L2 (ridge) penalty strength. Default 0.0 (unpenalized). When > 0,
            fits a ridge-penalized model: predictors are standardized, the
            intercept is left unpenalized, and ``l2`` is the penalty added on the
            standardized scale (matching ``MASS::lm.ridge``'s lambda). A penalized
            fit does not report standard errors / t / p values (not valid for a
            biased estimator). See the ``ridge()`` convenience wrapper.
        weights: Per-observation prior weights (n,), matching R's
            ``lm(..., weights=)`` / ``glm(..., weights=)``. For OLS this is
            weighted least squares; for a GLM these are the IRLS prior weights.
            Must be non-negative and not all zero. ``None`` ⇒ unit weights.
            Not supported together with ``l2 > 0`` (raises).
        offset: Additive term in the linear predictor, η = Xβ + offset (n,),
            matching R's ``glm(..., offset=)``. Used as-is, never estimated —
            e.g. ``log(exposure)`` for a Poisson rate model. ``None`` ⇒ no
            offset. Not supported together with ``l2 > 0`` (raises).

    Returns:
        LinearSolution when family is None.
        GLMSolution when family is specified.

    Examples:
        # OLS with named output
        >>> result = fit(X, y, names=['albumin', 'copper', 'protime'])
        >>> result.coef['copper']
        0.005255

        # Logistic regression
        >>> result = fit(X, y, family='binomial')

        # Poisson regression
        >>> result = fit(X, y, family='poisson')
    """
    # Unspecified backend → CPU (R-reference path). GPU is never the
    # default; callers must opt in explicitly or request 'auto'.
    if backend is None:
        backend = 'cpu'

    # Get or build Design
    if isinstance(X_or_design, Design):
        design = X_or_design
    else:
        if y is None:
            raise ValidationError("y required when passing arrays")
        design = Design.from_arrays(np.asarray(X_or_design), np.asarray(y))

    # Resolve names: a Design built from a term spec carries its own column
    # labels; an explicit names= argument overrides them.
    if names is None and design.names is not None:
        resolved_names = design.names
    else:
        resolved_names = _resolve_names(names, design.p)

    if conf_level <= 0 or conf_level >= 1:
        raise ValidationError(f"conf_level must be in (0, 1), got {conf_level}")

    # Validate prior weights / offset at the boundary (Rule 2). Returns
    # float64 arrays or None (the unit-weight / no-offset fast path).
    wt = resolve_weights(weights, design.n)
    off = resolve_offset(offset, design.n)
    if l2 > 0 and (wt is not None or off is not None):
        raise NotImplementedError(
            "weights= / offset= are not yet supported with a ridge penalty "
            "(l2 > 0): MASS::lm.ridge takes neither, so there is no R reference "
            "to validate against. Use them on the unpenalized fit (l2=0)."
        )

    # Dispatch: GLM path if family specified, otherwise LM path
    if family is not None:
        if solver is not None:
            raise ValidationError(
                "solver= applies only to linear models (family=None); GLMs are "
                "fit by IRLS. Remove solver=, or drop family= for an OLS fit."
            )
        sol = _fit_glm(design, family, backend, tol, max_iter, resolved_names,
                       force, l2, wt, off)
    else:
        sol = _fit_lm(design, backend, force, resolved_names, l2, solver, wt, off)
    sol._conf_level = conf_level   # uniform .conf_int level (internal field; A5)
    return sol


def ridge(
    X_or_design: ArrayLike | Design,
    y: ArrayLike | None = None,
    *,
    lam: float,
    family: 'str | Family | None' = None,
    backend: BackendChoice | None = None,
    tol: float = 1e-8,
    max_iter: int = 25,
    names: list[str] | None = None,
    weights: ArrayLike | None = None,
    offset: ArrayLike | None = None,
) -> Union[LinearSolution, GLMSolution]:
    """Ridge (L2-penalized) regression — a thin wrapper over ``fit(..., l2=lam)``.

    ``ridge(X, y, lam=λ)`` fits an L2-penalized linear model; pass ``family=`` for
    a penalized GLM (e.g. logistic ridge). Predictors are standardized and the
    intercept is unpenalized; ``lam`` is the penalty on the standardized scale,
    matching ``MASS::lm.ridge``. Penalized fits do not report standard errors.

    Equivalent to ``fit(X_or_design, y, family=family, l2=lam, ...)``; provided
    for discoverability. (Future: ``lasso`` / ``elastic_net`` wrappers will sit
    alongside this once the coordinate-descent solver exists.)
    """
    if lam < 0:
        raise ValidationError(f"ridge penalty lam must be non-negative, got {lam}")
    return fit(X_or_design, y, family=family, backend=backend, tol=tol,
               max_iter=max_iter, names=names, l2=lam,
               weights=weights, offset=offset)


def _resolve_names(
    names: list[str] | None,
    p: int,
) -> tuple[str, ...] | None:
    """Resolve user-provided names into a tuple matching the number of columns.

    If names has p-1 elements, prepend '(Intercept)'.
    If names has p elements, use as-is.
    If None, return None (Solutions will fall back to generic labels).
    """
    if names is None:
        return None
    if len(names) == p:
        return tuple(names)
    if len(names) == p - 1:
        return ("(Intercept)",) + tuple(names)
    raise ValidationError(
        f"names must have {p} or {p - 1} elements to match X with "
        f"{p} columns, got {len(names)}"
    )


def _fit_lm(
    design: Design,
    backend: BackendChoice,
    force: bool,
    names: tuple[str, ...] | None,
    l2: float = 0.0,
    solver: SolverChoice | None = None,
    weights: 'np.ndarray | None' = None,
    offset: 'np.ndarray | None' = None,
) -> LinearSolution:
    """Fit ordinary least squares, or ridge when l2 > 0."""
    if l2 > 0:
        if solver is not None:
            raise ValidationError(
                "solver= is not applicable to ridge fits (l2 > 0), which use a "
                "backward-stable augmented solve. Remove solver=."
            )
        return _fit_lm_ridge(design, l2, names)

    backend_impl = _get_lm_backend(backend, solver, design)
    result = _solve_with_inputs(backend_impl, (design,), weights, offset, force)
    return LinearSolution(_result=result, _design=design, _names=names)


def _solve_with_inputs(backend_impl, solve_args, weights, offset, force,
                       extra_kwargs=None):
    """Call a backend ``solve`` forwarding only the kwargs it accepts.

    Backends opt in to ``weights`` / ``offset`` / ``force`` by declaring them.
    If prior weights or an offset were supplied but the chosen backend cannot
    honor them, fail loud (Rule 1) rather than silently dropping them.
    ``extra_kwargs`` carries always-passed parameters (e.g. tol, max_iter).
    """
    varnames = backend_impl.solve.__code__.co_varnames
    kwargs = dict(extra_kwargs) if extra_kwargs else {}
    if 'weights' in varnames:
        kwargs['weights'] = weights
    elif weights is not None:
        raise NotImplementedError(
            f"{backend_impl.name} does not support weights= yet"
        )
    if 'offset' in varnames:
        kwargs['offset'] = offset
    elif offset is not None:
        raise NotImplementedError(
            f"{backend_impl.name} does not support offset= yet"
        )
    if 'force' in varnames:
        kwargs['force'] = force
    return backend_impl.solve(*solve_args, **kwargs)


def _fit_lm_ridge(
    design: Design,
    l2: float,
    names: tuple[str, ...] | None,
) -> LinearSolution:
    """Fit an L2-penalized linear model (ridge), matching MASS::lm.ridge.

    Predictors are standardized and the intercept left unpenalized; ``l2`` is
    added on the standardized scale via a backward-stable augmented solve, then
    coefficients are mapped back to the original units. Penalized fits do not
    carry valid frequentist standard errors (the solution marks itself
    ``penalized`` so they read as NA).
    """
    from pystatistics.core.result import Result
    from pystatistics.regression._penalty import (
        standardize, back_transform, augmented_ridge_solve,
    )

    X, y = np.asarray(design.X, dtype=np.float64), np.asarray(design.y, dtype=np.float64)
    n, p = design.n, design.p

    Z, y_c, info = standardize(X, y)
    beta_z = augmented_ridge_solve(Z, y_c, l2)
    coefficients = back_transform(beta_z, info, p)

    fitted = X @ coefficients
    residuals = y - fitted
    rss = float(residuals @ residuals)
    tss = float(np.sum((y - np.mean(y)) ** 2))

    params = LinearParams(
        coefficients=coefficients,
        residuals=residuals,
        fitted_values=fitted,
        rss=rss,
        tss=tss,
        rank=p,
        df_residual=n - p,
    )
    result = Result(
        params=params,
        info={'method': 'ridge', 'penalized': True, 'l2': float(l2)},
        timing=None,
        backend_name='cpu_ridge',
        warnings=(),
    )
    return LinearSolution(_result=result, _design=design, _names=names)


def _fit_glm(
    design: Design,
    family: 'str | Family',
    backend: BackendChoice,
    tol: float,
    max_iter: int,
    names: tuple[str, ...] | None,
    force: bool = False,
    l2: float = 0.0,
    weights: 'np.ndarray | None' = None,
    offset: 'np.ndarray | None' = None,
) -> GLMSolution:
    """Fit GLM via IRLS, or ridge-penalized IRLS when l2 > 0.

    ``force`` is passed to GPU backends only: when True it returns a
    non-converged float32 GPU fit instead of raising (the CPU backend, which is
    always stable, ignores it).

    For NegativeBinomial with unknown theta (theta=None), runs the
    alternating estimation loop matching R's MASS::glm.nb():
        1. Fit Poisson GLM for initial μ
        2. Estimate θ via profile likelihood
        3. Refit NB GLM with new θ
        4. Repeat until θ converges
    """
    from pystatistics.regression.families import (
        Family, NegativeBinomial, Poisson, resolve_family,
    )

    family_obj = resolve_family(family) if not isinstance(family, Family) else family

    if l2 > 0:
        if isinstance(family_obj, NegativeBinomial) and family_obj.theta is None:
            raise NotImplementedError(
                "Ridge (l2 > 0) is not supported for negative-binomial with "
                "auto-estimated theta. Fix theta (NegativeBinomial(theta=...)) to "
                "use a penalized fit.")
        return _fit_glm_ridge(design, family_obj, backend, tol, max_iter, names, l2)

    backend_impl = _get_glm_backend(backend)

    # NB with unknown theta: alternating estimation loop
    if isinstance(family_obj, NegativeBinomial) and family_obj.theta is None:
        return _fit_nb(design, family_obj, backend_impl, tol, max_iter, names,
                       force=force, weights=weights, offset=offset)

    result = _solve_with_inputs(
        backend_impl, (design, family_obj), weights, offset, force,
        extra_kwargs={'tol': tol, 'max_iter': max_iter},
    )
    return GLMSolution(_result=result, _design=design, _names=names)


def _fit_glm_ridge(
    design: Design,
    family: 'Family',
    backend: BackendChoice,
    tol: float,
    max_iter: int,
    names: tuple[str, ...] | None,
    l2: float,
) -> GLMSolution:
    """Fit an L2-penalized GLM via ridge-penalized IRLS.

    On the GPU this is the headline path: the ridge penalty makes the float32
    Cholesky on XᵀWX well-conditioned, so a GLM that is unstable/ill-conditioned
    in plain float32 fits *fast and stably* on the GPU at very large scale. The
    CPU path (fp64) is the correctness reference. Both standardize the design
    (intercept unpenalized) and back-transform; penalized fits carry no SEs.
    """
    backend_impl = _get_glm_backend(backend)
    if 'penalty' in backend_impl.solve.__code__.co_varnames:
        return _fit_glm_ridge_via_backend(
            design, family, backend_impl, tol, max_iter, names, l2)

    from pystatistics.core.result import Result
    from pystatistics.regression._penalty import (
        standardized_design, weighted_augmented_solve, back_transform_in_design,
    )
    from pystatistics.regression.backends.cpu_glm import CPUIRLSBackend

    X = np.asarray(design.X, dtype=np.float64)
    y = np.asarray(design.y, dtype=np.float64)
    n, p = design.n, design.p
    link = family.link
    wt = np.ones(n)

    A, center, scale, icol = standardized_design(X)
    pen = np.full(p, float(l2))
    if icol is not None:
        pen[icol] = 0.0

    mu = family.initialize(y)
    eta = link.link(mu)
    dev_old = family.deviance(y, mu, wt)
    converged = False
    iteration = 0
    beta_A = np.zeros(p)
    for iteration in range(1, max_iter + 1):
        mu_eta = link.mu_eta(eta)
        var_mu = family.variance(mu)
        z = eta + (y - mu) / mu_eta
        w = np.maximum(wt * (mu_eta ** 2) / var_mu, 1e-30)
        beta_A = weighted_augmented_solve(A, z, pen, w)
        eta = A @ beta_A
        mu = link.linkinv(eta)
        dev_new = family.deviance(y, mu, wt)
        if abs(dev_new - dev_old) / (abs(dev_old) + 0.1) < tol:
            converged = True
            break
        dev_old = dev_new

    coefficients = back_transform_in_design(beta_A, center, scale, icol)
    eta_final = X @ coefficients
    mu_final = link.linkinv(eta_final)
    dev = family.deviance(y, mu_final, wt)

    resid_response = y - mu_final
    resid_pearson = resid_response / np.sqrt(family.variance(mu_final))
    resid_deviance = CPUIRLSBackend._deviance_residuals(y, mu_final, wt, family)
    mu_eta_final = link.mu_eta(eta_final)
    resid_working = (y - mu_final) / mu_eta_final
    null_deviance = CPUIRLSBackend._null_deviance(y, wt, family)

    df_residual = n - p
    dispersion = 1.0 if family.dispersion_is_fixed else (
        dev / df_residual if df_residual > 0 else float('nan'))
    aic = family.aic(y, mu_final, wt, p, dispersion)

    params = GLMParams(
        coefficients=coefficients, fitted_values=mu_final,
        linear_predictor=eta_final, residuals_working=resid_working,
        residuals_deviance=resid_deviance, residuals_pearson=resid_pearson,
        residuals_response=resid_response, deviance=dev,
        null_deviance=null_deviance, aic=aic, dispersion=dispersion,
        rank=p, df_residual=df_residual, df_null=n - 1,
        n_iter=iteration, converged=converged,
        family_name=family.name, link_name=link.name,
    )
    result = Result(
        params=params,
        info={'method': 'ridge_irls', 'penalized': True, 'l2': float(l2)},
        timing=None, backend_name='cpu_ridge_irls', warnings=(),
    )
    return GLMSolution(_result=result, _design=design, _names=names)


def _fit_glm_ridge_via_backend(
    design: Design,
    family: 'Family',
    backend_impl: object,
    tol: float,
    max_iter: int,
    names: tuple[str, ...] | None,
    l2: float,
) -> GLMSolution:
    """Ridge GLM on a penalty-aware backend (the GPU path).

    Standardizes the design, runs the backend's penalized IRLS on the standardized
    design (so the float32 XᵀWX + λI Cholesky is well-conditioned), then maps the
    coefficients back to the original scale. Everything else the backend computes
    (fitted/residuals/deviance) is η-based and so already correct, since
    η = A·β_std = X·β_raw.
    """
    from dataclasses import replace
    from pystatistics.core.result import Result
    from pystatistics.regression._penalty import (
        standardized_design, back_transform_in_design,
    )

    X = np.asarray(design.X, dtype=np.float64)
    p = design.p
    A, center, scale, icol = standardized_design(X)
    pen = np.full(p, float(l2))
    if icol is not None:
        pen[icol] = 0.0

    std_design = Design.from_arrays(A, np.asarray(design.y, dtype=np.float64))
    result = backend_impl.solve(std_design, family, tol=tol, max_iter=max_iter,
                                penalty=pen)

    beta_raw = back_transform_in_design(
        result.params.coefficients, center, scale, icol)
    new_params = replace(result.params, coefficients=beta_raw)
    new_info = {**result.info, 'penalized': True, 'l2': float(l2),
                'method': 'ridge_irls_gpu'}
    new_result = Result(
        params=new_params, info=new_info, timing=result.timing,
        backend_name=result.backend_name, warnings=result.warnings,
    )
    return GLMSolution(_result=new_result, _design=design, _names=names)


def _fit_nb(
    design: Design,
    family: 'NegativeBinomial',
    backend_impl: object,
    tol: float,
    max_iter: int,
    names: tuple[str, ...] | None,
    theta_max_iter: int = 25,
    theta_tol: float = 1e-6,
    force: bool = False,
    weights: 'np.ndarray | None' = None,
    offset: 'np.ndarray | None' = None,
) -> GLMSolution:
    """Fit negative binomial GLM with theta estimation.

    Alternates between GLM fitting (given theta) and theta estimation
    (given mu), matching R's MASS::glm.nb() algorithm. ``force`` is forwarded
    to each inner GLM solve (GPU backends only); prior ``weights`` enter both
    the inner GLM fits and the θ profile likelihood, and ``offset`` enters each
    inner linear predictor.
    """
    from pystatistics.core.exceptions import ConvergenceError
    from pystatistics.regression.families import NegativeBinomial, Poisson
    from pystatistics.regression._nb_theta import theta_ml

    def _solve(fam):
        return _solve_with_inputs(
            backend_impl, (design, fam), weights, offset, force,
            extra_kwargs={'tol': tol, 'max_iter': max_iter},
        )

    y = design.y
    wt = np.ones(design.n) if weights is None else weights

    # Step 1: Initial Poisson fit for starting mu
    poisson_result = _solve(Poisson())
    mu = poisson_result.params.fitted_values

    # Step 2: Initial theta from Poisson mu
    theta = theta_ml(y, mu, wt)

    # Step 3: Iterate: refit NB with new theta → re-estimate theta
    for iteration in range(theta_max_iter):
        nb_family = NegativeBinomial(theta=theta, link=family._link)
        result = _solve(nb_family)
        mu = result.params.fitted_values
        theta_new = theta_ml(y, mu, wt)

        if abs(theta_new - theta) / (theta + 1e-10) < theta_tol:
            # Converged — final result uses the converged theta
            nb_final = NegativeBinomial(theta=theta_new, link=family._link)
            result = _solve(nb_final)
            return GLMSolution(_result=result, _design=design, _names=names)

        theta = theta_new

    raise ConvergenceError(
        f"NB theta estimation did not converge after {theta_max_iter} "
        f"outer iterations. Last theta = {theta:.4f}.",
        details={'theta': theta, 'n_outer_iter': theta_max_iter},
    )


# =====================================================================
# Backend selection
# =====================================================================

def _get_lm_backend(choice: BackendChoice, solver: SolverChoice | None, design: Design):
    """Select the LM backend from the resolved (device, precision) target.

    ``backend`` decides the device + precision (via the canonical resolver);
    ``solver`` decides the numerical routine on the CPU path. The GPU path uses
    Cholesky on the normal equations and is not solver-configurable.
    """
    target = resolve_backend(choice, supports_fp64=True)

    if target.device_type == 'cpu':
        if solver in (None, 'qr'):
            return CPUQRBackend()
        if solver == 'svd':
            raise NotImplementedError("CPU SVD solver not yet implemented")
        raise ValidationError(
            f"Unknown solver {solver!r}. Valid options: 'qr', 'svd'."
        )

    if solver is not None:
        raise ValidationError(
            "solver= is not configurable on the GPU backend, which uses "
            "Cholesky on the normal equations. Omit solver=, or use "
            "backend='cpu' for QR/SVD."
        )
    from pystatistics.regression.backends.gpu import GPUQRBackend
    return GPUQRBackend(device=target.device_type, use_fp64=target.use_fp64)


def _get_glm_backend(choice: BackendChoice):
    """Select the GLM (IRLS) backend from the resolved (device, precision) target."""
    from pystatistics.regression.backends.cpu_glm import CPUIRLSBackend

    target = resolve_backend(choice, supports_fp64=True)
    if target.device_type == 'cpu':
        return CPUIRLSBackend()

    from pystatistics.regression.backends.gpu_glm import GPUIRLSBackend
    return GPUIRLSBackend(device=target.device_type, use_fp64=target.use_fp64)
