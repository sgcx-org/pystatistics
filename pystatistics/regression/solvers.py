"""
Solver dispatch for regression.

Handles both Linear Models (LM) and Generalized Linear Models (GLM).

When family is None (default), the existing LM path is used → LinearSolution.
When family is provided, the IRLS path is used → GLMSolution.
"""

import warnings
from typing import Literal, Union
import numpy as np
from numpy.typing import ArrayLike

from pystatistics.core.compute.device import select_device
from pystatistics.regression.design import Design
from pystatistics.regression.solution import LinearSolution, GLMSolution
from pystatistics.regression.backends.cpu import CPUQRBackend


BackendChoice = Literal['auto', 'cpu', 'gpu', 'cpu_qr', 'cpu_svd', 'gpu_qr']


def fit(
    X_or_design: ArrayLike | Design,
    y: ArrayLike | None = None,
    *,
    family: 'str | Family | None' = None,
    backend: BackendChoice | None = None,
    force: bool = False,
    tol: float = 1e-8,
    max_iter: int = 25,
    names: list[str] | None = None,
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
        backend: Compute backend. Default None → 'cpu' (the R-reference
            path, validated for regulated-industry use). Explicit values:
            'cpu', 'gpu', 'cpu_qr', 'cpu_svd', 'gpu_qr', or 'auto' to
            prefer GPU when available and fall back to CPU.
        force: If True, proceed with GPU Cholesky even on ill-conditioned
            matrices. Has no effect on CPU backends.
        tol: Convergence tolerance for IRLS (GLM only). Default 1e-8
            matches R's glm.control().
        max_iter: Maximum IRLS iterations (GLM only). Default 25
            matches R's glm.control().
        names: Optional list of predictor names for output labeling.
            If len(names) == p - 1 (one fewer than columns in X),
            "(Intercept)" is prepended automatically.
            If len(names) == p, used as-is.

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
            raise ValueError("y required when passing arrays")
        design = Design.from_arrays(np.asarray(X_or_design), np.asarray(y))

    # Resolve names: auto-prepend "(Intercept)" if needed
    resolved_names = _resolve_names(names, design.p)

    # Dispatch: GLM path if family specified, otherwise LM path
    if family is not None:
        return _fit_glm(design, family, backend, tol, max_iter, resolved_names)
    else:
        return _fit_lm(design, backend, force, resolved_names)


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
    raise ValueError(
        f"names must have {p} or {p - 1} elements to match X with "
        f"{p} columns, got {len(names)}"
    )


def _fit_lm(
    design: Design,
    backend: BackendChoice,
    force: bool,
    names: tuple[str, ...] | None,
) -> LinearSolution:
    """Fit ordinary least squares (existing LM path, unchanged)."""
    backend_impl = _get_lm_backend(backend, design)

    # Solve — pass force to GPU backends, CPU ignores it
    if hasattr(backend_impl, 'solve') and 'force' in backend_impl.solve.__code__.co_varnames:
        result = backend_impl.solve(design, force=force)
    else:
        result = backend_impl.solve(design)

    return LinearSolution(_result=result, _design=design, _names=names)


def _fit_glm(
    design: Design,
    family: 'str | Family',
    backend: BackendChoice,
    tol: float,
    max_iter: int,
    names: tuple[str, ...] | None,
) -> GLMSolution:
    """Fit GLM via IRLS.

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
    backend_impl = _get_glm_backend(backend)

    # NB with unknown theta: alternating estimation loop
    if isinstance(family_obj, NegativeBinomial) and family_obj.theta is None:
        return _fit_nb(design, family_obj, backend_impl, tol, max_iter, names)

    result = backend_impl.solve(design, family_obj, tol=tol, max_iter=max_iter)

    return GLMSolution(_result=result, _design=design, _names=names)


def _fit_nb(
    design: Design,
    family: 'NegativeBinomial',
    backend_impl: object,
    tol: float,
    max_iter: int,
    names: tuple[str, ...] | None,
    theta_max_iter: int = 25,
    theta_tol: float = 1e-6,
) -> GLMSolution:
    """Fit negative binomial GLM with theta estimation.

    Alternates between GLM fitting (given theta) and theta estimation
    (given mu), matching R's MASS::glm.nb() algorithm.
    """
    from pystatistics.core.exceptions import ConvergenceError
    from pystatistics.regression.families import NegativeBinomial, Poisson
    from pystatistics.regression._nb_theta import theta_ml

    y = design.y
    wt = np.ones(design.n)

    # Step 1: Initial Poisson fit for starting mu
    poisson_result = backend_impl.solve(
        design, Poisson(), tol=tol, max_iter=max_iter,
    )
    mu = poisson_result.params.fitted_values

    # Step 2: Initial theta from Poisson mu
    theta = theta_ml(y, mu, wt)

    # Step 3: Iterate: refit NB with new theta → re-estimate theta
    for iteration in range(theta_max_iter):
        nb_family = NegativeBinomial(theta=theta, link=family._link)
        result = backend_impl.solve(
            design, nb_family, tol=tol, max_iter=max_iter,
        )
        mu = result.params.fitted_values
        theta_new = theta_ml(y, mu, wt)

        if abs(theta_new - theta) / (theta + 1e-10) < theta_tol:
            # Converged — final result uses the converged theta
            nb_final = NegativeBinomial(theta=theta_new, link=family._link)
            result = backend_impl.solve(
                design, nb_final, tol=tol, max_iter=max_iter,
            )
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

def _get_lm_backend(choice: BackendChoice, design: Design):
    """Select LM backend based on user choice and hardware availability."""
    if choice == 'auto':
        device = select_device('auto')
        if device.device_type == 'cuda':
            try:
                from pystatistics.regression.backends.gpu import GPUQRBackend
                return GPUQRBackend(device='cuda')
            except ImportError:
                warnings.warn(
                    "CUDA detected but PyTorch not available. "
                    "Falling back to CPU backend.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return CPUQRBackend()
        # auto + MPS -> CPU (MPS is FP32-only, not reliable for auto)
        # auto + CPU -> CPU
        return CPUQRBackend()

    elif choice in ('cpu', 'cpu_qr'):
        return CPUQRBackend()

    elif choice == 'cpu_svd':
        raise NotImplementedError("CPU SVD backend not yet implemented")

    elif choice in ('gpu', 'gpu_qr'):
        device = select_device('gpu')  # raises RuntimeError if no GPU
        from pystatistics.regression.backends.gpu import GPUQRBackend
        return GPUQRBackend(device=device.device_type)

    else:
        raise ValueError(f"Unknown backend: {choice!r}")


def _get_glm_backend(choice: BackendChoice):
    """Select GLM backend based on user choice and hardware availability."""
    from pystatistics.regression.backends.cpu_glm import CPUIRLSBackend

    if choice == 'auto':
        device = select_device('auto')
        if device.device_type == 'cuda':
            try:
                from pystatistics.regression.backends.gpu_glm import GPUIRLSBackend
                return GPUIRLSBackend(device='cuda')
            except ImportError:
                warnings.warn(
                    "CUDA detected but GPU GLM backend not available. "
                    "Falling back to CPU IRLS.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return CPUIRLSBackend()
        # auto + MPS -> CPU (MPS not auto-selected)
        # auto + CPU -> CPU
        return CPUIRLSBackend()

    elif choice in ('cpu', 'cpu_qr'):
        return CPUIRLSBackend()

    elif choice in ('gpu', 'gpu_qr'):
        device = select_device('gpu')  # raises RuntimeError if no GPU
        try:
            from pystatistics.regression.backends.gpu_glm import GPUIRLSBackend
            return GPUIRLSBackend(device=device.device_type)
        except ImportError:
            raise ImportError(
                "GPU GLM backend requires PyTorch. "
                "Install with: pip install torch"
            )

    else:
        raise ValueError(f"Unknown backend: {choice!r}")
