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


def _validate_y_codes(
    y: NDArray[np.floating[Any]],
    expected_n: int,
) -> NDArray[np.int64]:
    """Validate the response vector alone.

    Used by the GPU-tensor entry path, where X is already a device-
    resident ``torch.Tensor`` (so it bypasses this module's numpy-only
    ``check_array`` helpers) but y is still small enough to live on
    CPU as int codes. The GPU likelihood machinery reads y as numpy
    because the one-hot encoding is a one-shot build, not a per-
    iteration cost.

    Same checks as the y side of :func:`_validate_inputs`.
    """
    check_1d(y, "y")
    check_finite(y, "y")
    y_codes = y.astype(np.int64)
    if not np.allclose(y, y_codes):
        raise ValidationError(
            "y: must contain integer class codes (0, 1, ..., J-1), "
            "but found non-integer values"
        )
    if len(y_codes) != expected_n:
        raise ValidationError(
            f"y length {len(y_codes)} does not match X rows {expected_n}"
        )

    unique = np.unique(y_codes)
    if len(unique) < 2:
        raise ValidationError(
            f"y: requires at least 2 distinct classes, got {len(unique)}"
        )
    expected = np.arange(len(unique))
    if not np.array_equal(unique, expected):
        raise ValidationError(
            f"y: class codes must be consecutive integers starting from 0. "
            f"Got unique values {unique.tolist()}, "
            f"expected {expected.tolist()}"
        )
    return y_codes


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


def _fit_multinom_gpu(
    y_codes: NDArray[np.int64],
    X: NDArray[np.floating[Any]] | Any,   # numpy array OR torch.Tensor
    n_classes: int,
    tol: float,
    max_iter: int,
    device: str,
    use_fp64: bool,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], int, bool, Any]:
    """GPU version of _fit_multinom.

    Accepts ``X`` as either a numpy array (per-call H2D transfer) or an
    already-on-device torch.Tensor (zero extra transfer — the whole
    point of the device-resident DataSource API). The
    :class:`MultinomialGPULikelihood` helper handles both input shapes
    internally.

    Returns (params_flat, vcov, n_iter, converged, like) — the last
    element is the likelihood object so the caller can reuse it for
    the final fitted-probs and log-likelihood computations without
    rebuilding the GPU tensors.
    """
    from pystatistics.multinomial.backends.gpu_likelihood import (
        MultinomialGPULikelihood,
    )

    n, p = X.shape
    n_nonref = n_classes - 1
    n_params = n_nonref * p

    like = MultinomialGPULikelihood(
        X, y_codes, n_classes, device=device, use_fp64=use_fp64,
    )

    x0 = np.zeros(n_params, dtype=np.float64)
    result = minimize(
        fun=like.fun,
        x0=x0,
        method="L-BFGS-B",
        jac=like.jac,
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )
    converged = bool(result.success)
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

    # Analytical Hessian on GPU: X' · W · X per class-pair block.
    # This replaces the CPU path's 300-iteration numerical Hessian
    # (each iteration costing ~10 ms for n=50k), which dominated total
    # fit time — ~3 seconds of vcov on the CPU side vs the 18 ms the
    # GPU optimizer itself was spending.
    vcov = like.compute_vcov(params_flat)

    return params_flat, vcov, n_iter, converged, like


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
    """Analytical multinomial-logit Hessian → inverse → vcov.

    The Hessian of the multinomial softmax NLL decomposes as a block
    matrix over non-reference class pairs:

        H[j, k] = X' · diag(W_{jk}) · X

    with ``W_{jj} = diag(p_j · (1 − p_j))`` and
    ``W_{jk} = diag(−p_j · p_k)`` for ``j != k``. Each block is a
    single weighted-GEMM; the full (J-1)p × (J-1)p Hessian is assembled
    by slotting the p × p blocks into position.

    This replaces the original central-difference Hessian, which ran
    ``multinomial_gradient`` ``2 · n_params`` times per vcov call. For
    J=10, p=30 that was 600 gradient sweeps — each scanning the full
    sample — dominating total fit time past modest scale. The
    analytical form is one p × p GEMM per (j, k) block, is bit-
    identical to the true Hessian, and removes the finite-difference
    step-size trade-off. This is the numpy counterpart to the
    ``MultinomialGPULikelihood.compute_vcov`` formula that the GPU
    backend uses.
    """
    n, p = X.shape
    n_nonref = n_classes - 1

    # Predicted probabilities for non-reference classes. We recompute
    # from params to avoid coupling with the optimizer's cached state.
    probs = compute_probs(params_flat, X, n_classes)[:, :n_nonref]   # (n, J-1)

    n_params_h = n_nonref * p
    hessian = np.empty((n_params_h, n_params_h), dtype=np.float64)
    for j in range(n_nonref):
        p_j = probs[:, j]
        for k in range(j, n_nonref):
            if j == k:
                w = p_j * (1.0 - p_j)
            else:
                w = -p_j * probs[:, k]
            # (p, p) = (X * w[:, None])' · X
            block = (X * w[:, np.newaxis]).T @ X
            rs = slice(j * p, (j + 1) * p)
            cs = slice(k * p, (k + 1) * p)
            hessian[rs, cs] = block
            if k != j:
                hessian[cs, rs] = block.T

    # Invert to get vcov. The Hessian of the multinomial NLL at the
    # MLE is positive-definite for well-identified models; fall back
    # to the pseudoinverse on numerical-singular edge cases.
    try:
        vcov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
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
    backend: str | None = None,
    use_fp64: bool = False,
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
        backend: Compute backend: ``'cpu'`` (default), ``'auto'``, or
            ``'gpu'``. ``'cpu'`` uses the numpy reference path,
            validated against R ``nnet::multinom``. ``'auto'`` prefers
            GPU when available, falls back to CPU. ``'gpu'`` is
            explicit — raises if no GPU is available (Rule 1: no
            silent fallback). GPU keeps ``X`` and the one-hot ``y`` on
            the device for the full optimization; only the parameter
            vector and (scalar NLL, gradient vector) cross the bus per
            L-BFGS-B iteration.
        use_fp64: Only relevant when actually running on GPU. Default
            False (FP32): matches CPU at the project's ``GPU_FP32``
            tolerance tier (rtol = 1e-4). Set True on CUDA to get
            CPU-matching precision; note the consumer-NVIDIA FP64
            throughput penalty applies.

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

    # Convention shared with pca() and applied to all GPU-capable
    # backends in pystatistics (see README "Design Philosophy"):
    #   numpy input → default backend is 'cpu' (R-reference path)
    #   GPU torch.Tensor input → default backend is 'gpu'
    # The reasoning: creating a GPU DataSource via ``ds.to('cuda')`` is
    # itself the opt-in to GPU. Demanding the user also say
    # ``backend='gpu'`` after that is redundant friction. Explicit
    # ``backend='cpu'`` with a GPU tensor still raises — no silent
    # device migration.
    import sys as _sys
    _is_X_tensor = (
        "torch" in _sys.modules
        and isinstance(X, _sys.modules["torch"].Tensor)
    )
    if _is_X_tensor:
        import torch
        if X.ndim != 2:
            raise ValidationError(
                f"X: expected 2-D tensor, got {X.ndim}-D"
            )
        if not torch.isfinite(X).all():
            raise ValidationError("X contains non-finite values")
        if backend is None:
            backend = "gpu" if X.device.type != "cpu" else "cpu"
        if backend == "cpu":
            raise ValidationError(
                "backend='cpu' was specified but X is a torch.Tensor "
                f"on device {X.device}. Either pass a numpy array / "
                "CPU DataSource to the CPU backend, or call `.to('cpu')` "
                "on the DataSource explicitly to move it back."
            )
        # y may also be a tensor (from the same DataSource). It's
        # tiny (n labels) so we pull to CPU numpy once — no benefit
        # keeping it on-device for the one-shot one-hot build.
        if "torch" in _sys.modules and isinstance(
            y, _sys.modules["torch"].Tensor,
        ):
            y_arr = y.detach().cpu().numpy()
        else:
            y_arr = check_array(y, "y")
        # X validation happens inside _validate_inputs via
        # check_array — but check_array is numpy-only. The tensor is
        # already shape/finite-checked above; build a shim so the
        # validation routine's numpy-only helpers work on the label
        # side while the design matrix keeps device residency.
        n, p = X.shape
        y_codes = _validate_y_codes(y_arr, expected_n=n)
        n_classes = int(np.max(y_codes)) + 1
        X_arr = None
        X_for_gpu = X
    else:
        if backend is None:
            backend = "cpu"
        y_arr = check_array(y, "y")
        X_arr = check_array(X, "X")
        y_codes, X_arr, n_classes = _validate_inputs(y_arr, X_arr)
        n, p = X_arr.shape
        X_for_gpu = None

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

    # --- Backend dispatch (see README "Design Philosophy") ---
    # 'cpu' (default): R-reference numpy path, validated against nnet::multinom.
    # 'auto': use GPU if available, else CPU — silent CPU fallback is the
    #   definition of 'auto'.
    # 'gpu': explicit — raise if no GPU is available (Rule 1: no silent
    #   substitution when the caller made an explicit choice).
    if backend not in ("cpu", "auto", "gpu"):
        raise ValidationError(
            f"backend: must be 'cpu', 'auto', or 'gpu', got {backend!r}"
        )

    # FP32 gradient precision is ~1e-7, so L-BFGS-B's default ``gtol`` of
    # 1e-8 routinely reports ABNORMAL (line search stall below noise
    # floor). The tol we pass down must be consistent with the arithmetic
    # precision — a user who asks for FP32 can't reasonably demand 1e-8
    # gradient convergence. When running FP32, floor tol at 1e-5.
    gpu_fp32_min_tol = 1e-5
    effective_tol = tol
    if backend != "cpu" and not use_fp64 and tol < gpu_fp32_min_tol:
        effective_tol = gpu_fp32_min_tol

    # ``gpu_like`` is the MultinomialGPULikelihood object when a GPU
    # fit runs; None otherwise. Lets the post-fit compute_probs / log-
    # likelihood calls below reuse the on-device state instead of
    # round-tripping through numpy.
    gpu_like = None

    if X_arr is None:
        # GPU-tensor entry path: X_for_gpu already on device. We know
        # the device from the tensor; no device selection needed.
        gpu_device = X_for_gpu.device.type
        params_flat, vcov, n_iter, converged, gpu_like = _fit_multinom_gpu(
            y_codes, X_for_gpu, n_classes, effective_tol, max_iter,
            device=gpu_device,
            use_fp64=use_fp64,
        )
        backend_name = f"gpu_lbfgsb ({gpu_device}, {'fp64' if use_fp64 else 'fp32'})"
    elif backend == "cpu":
        backend_name = "cpu_lbfgsb"
        params_flat, vcov, n_iter, converged = _fit_multinom(
            y_codes, X_arr, n_classes, tol, max_iter,
        )
    else:
        from pystatistics.core.compute.device import select_device
        dev = select_device("gpu" if backend == "gpu" else "auto")
        if dev.is_gpu:
            params_flat, vcov, n_iter, converged, gpu_like = _fit_multinom_gpu(
                y_codes, X_arr, n_classes, effective_tol, max_iter,
                device=dev.device_type,
                use_fp64=use_fp64,
            )
            backend_name = f"gpu_lbfgsb ({dev.device_type}, {'fp64' if use_fp64 else 'fp32'})"
        elif backend == "gpu":
            raise RuntimeError(
                "backend='gpu' requested but no GPU is available. "
                "Install PyTorch with CUDA/MPS support or use backend='cpu'."
            )
        else:
            # backend='auto' and no GPU: fall through to CPU.
            backend_name = "cpu_lbfgsb"
            params_flat, vcov, n_iter, converged = _fit_multinom(
                y_codes, X_arr, n_classes, tol, max_iter,
            )

    # Post-fit quantities. When we ran on GPU, reuse the likelihood
    # object — it already holds the on-device X and one-hot y, so
    # ``compute_probs`` and the NLL recompute avoid repeating the
    # ~X.nbytes per-call host→device transfer that would otherwise
    # dominate these steps on 1M+ row data.
    if gpu_like is not None:
        fitted_probs = gpu_like.compute_probs(params_flat)
        log_lik = -float(gpu_like.fun(params_flat))
    else:
        fitted_probs = compute_probs(params_flat, X_arr, n_classes)
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
        backend_name=backend_name,
    )

    return MultinomialSolution(result)
