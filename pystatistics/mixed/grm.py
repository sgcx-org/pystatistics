"""Low-rank / GRM mixed model — public ``grm_lmm`` entry point.

A mixed model with a *single* variance component whose covariance is low-rank,
K = WW'/M (a genomic relatedness matrix from M standardized markers, or any
reduced-rank random effect). Unlike the general ``lmm`` (a CPU-only,
lme4-equivalent sparse-design model), this model's deviance reduces to a dense
M×M Gram + n×M GEMMs — the regime where a GPU genuinely earns its keep — so it
exposes a ``backend=``. It is a *separate, honestly-named model*: it is **not**
"lmm on a GPU", and the general ``lmm`` deliberately has no GPU backend.

Backends (per CONVENTIONS.md):
    'cpu'       CPU float64 — the reference path (matches R/GCTA-style REML).
    'gpu'       GPU float32 — the speed path. Forms the M×M Gram; the
                conditioning of W'W is gated up front in float64 and a design
                past the float32-safe boundary is REFUSED loudly (no silent
                wrong answer). ~15–70× over CPU on a consumer card.
    'gpu_fp64'  GPU float64 (CUDA only) — numerically exact vs CPU; ~1.8–1.9×
                on a consumer card (more on a datacenter card).
    'auto'      CUDA-float32 if present, else CPU.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.validation import check_finite
from pystatistics.core.compute.backend import resolve_backend
from pystatistics.core.compute.timing import Timer
from pystatistics.core.result import Result

from pystatistics.mixed._grm_cpu import grm_fit_cpu, grm_deviance_cpu, GRMFit
from pystatistics.mixed.grm_solution import GRMParams, GRMSolution


def grm_lmm(
    y: ArrayLike,
    X: ArrayLike,
    W: ArrayLike,
    *,
    backend: str | None = None,
    reml: bool = True,
    names: tuple[str, ...] | None = None,
    tol: float = 1e-8,
    max_iter: int = 200,
    conf_level: float = 0.95,
    force: bool = False,
) -> GRMSolution:
    """Fit a low-rank / GRM mixed model: y = Xβ + g + ε, g ~ N(0, σ²_g·WW'/M).

    Args:
        y: Response vector (n,).
        X: Fixed-effects design (n, p) — include an intercept column if wanted.
        W: Low-rank factor (n, M) defining K = WW'/M (e.g. a standardized
            genotype matrix). M is the rank.
        backend: Compute backend — ``'cpu'`` (float64 reference), ``'gpu'``
            (float32 speed path), ``'gpu_fp64'`` (CUDA-only exact), or
            ``'auto'``. ``None`` resolves to ``'cpu'`` for a numpy input.
        reml: REML (default) or ML.
        names: Optional fixed-effect names (length p).
        tol: Optimizer tolerance for the θ search.
        max_iter: Maximum optimizer evaluations.
        conf_level: Confidence level for ``conf_int``.
        force: Bypass the float32 Gram-conditioning gate on the GPU path. Use
            only when you know W is well-conditioned; results on an
            ill-conditioned W will be unreliable.

    Returns:
        GRMSolution with fixed effects, variance components, heritability,
        genetic-value BLUPs, and fit statistics.

    Raises:
        ValidationError: bad shapes / non-finite input / unknown backend.
        RuntimeError: a GPU backend requested but unavailable (or gpu_fp64 on
            MPS).
        NumericalError: the float32 GPU path refused an ill-conditioned W
            (raise the precision with ``backend='gpu_fp64'`` or use
            ``backend='cpu'``).
    """
    if conf_level <= 0 or conf_level >= 1:
        raise ValidationError(f"conf_level must be in (0, 1), got {conf_level}")

    y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64)).ravel()
    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    W_arr = np.ascontiguousarray(np.asarray(W, dtype=np.float64))
    if X_arr.ndim != 2:
        raise ValidationError(f"X must be 2-D (n, p), got shape {X_arr.shape}")
    if W_arr.ndim != 2:
        raise ValidationError(f"W must be 2-D (n, M), got shape {W_arr.shape}")
    n, p = X_arr.shape
    if y_arr.shape[0] != n or W_arr.shape[0] != n:
        raise ValidationError(
            f"y ({y_arr.shape[0]}), X ({n}), and W ({W_arr.shape[0]}) must "
            f"share the same number of rows.")
    check_finite(y_arr, "y")
    check_finite(X_arr, "X")
    check_finite(W_arr, "W")
    M = W_arr.shape[1]

    target = resolve_backend(backend, supports_fp64=True)

    timer = Timer()
    timer.start()
    with timer.section("fit"):
        if target.device_type == "cpu":
            fit = grm_fit_cpu(W_arr, X_arr, y_arr, reml=reml, tol=tol,
                              max_iter=max_iter)
            backend_label = "grm_cpu"
        else:
            from pystatistics.mixed.backends.grm_gpu import grm_fit_gpu
            fit = grm_fit_gpu(
                W_arr, X_arr, y_arr, reml=reml, tol=tol, max_iter=max_iter,
                device_type=target.device_type, use_fp64=target.use_fp64,
                force=force,
            )
            prec = "fp64" if target.use_fp64 else "fp32"
            backend_label = f"grm_gpu ({target.device_type}, {prec})"
    timer.stop()

    params = _assemble_params(fit, X_arr, y_arr, n, p, M, reml, names)
    result = Result(
        params=params,
        info={"method": "REML" if reml else "ML", "backend": backend_label,
              "converged": fit.converged, "theta": fit.theta},
        timing=timer.result(),
        backend_name=backend_label,
        warnings=() if fit.converged else ("Optimizer did not converge",),
    )
    return GRMSolution(_result=result, _conf_level=conf_level)


def _assemble_params(fit: GRMFit, X: NDArray, y: NDArray, n: int, p: int,
                     M: int, reml: bool, names) -> GRMParams:
    """Build the user-facing GRMParams from the raw GRMFit primitives."""
    sigma_e2 = fit.sigma_e2
    gamma = fit.theta ** 2
    sigma_g2 = gamma * sigma_e2
    heritability = gamma / (gamma + 1.0)

    # Fixed-effect SEs via form A: Var(β̂) = σ²_e (RX⁻¹)ᵀ(RX⁻¹).
    try:
        RX_inv = np.linalg.inv(fit.RX)
        vcov = sigma_e2 * (RX_inv.T @ RX_inv)
    except np.linalg.LinAlgError:
        vcov = sigma_e2 * np.linalg.pinv(fit.RX @ fit.RX.T)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    z_values = fit.beta / se
    p_values = 2.0 * stats.norm.sf(np.abs(z_values))

    # Log-likelihood (= -0.5 × profiled deviance) and AIC/BIC.
    log_det_RX = 2.0 * float(np.sum(np.log(np.maximum(np.abs(np.diag(fit.RX)), 1e-300))))
    if reml:
        df = n - p
        deviance = (fit.logdet_G + log_det_RX
                    + df * (1.0 + np.log(2.0 * np.pi * fit.pwrss / df)))
    else:
        deviance = (fit.logdet_G
                    + n * (1.0 + np.log(2.0 * np.pi * fit.pwrss / n)))
    ll = -0.5 * deviance
    n_params = p + 2  # fixed effects + (σ²_g, σ²_e)
    aic = -2.0 * ll + 2.0 * n_params
    bic = -2.0 * ll + np.log(n) * n_params

    if names is None:
        coef_names = ("(Intercept)",) + tuple(f"X{i}" for i in range(1, p))
    else:
        if len(names) != p:
            raise ValidationError(
                f"names has length {len(names)}, expected p={p}")
        coef_names = tuple(names)

    return GRMParams(
        coefficients=fit.beta,
        coefficient_names=coef_names,
        se=se,
        z_values=z_values,
        p_values=p_values,
        var_genetic=float(sigma_g2),
        var_residual=float(sigma_e2),
        variance_ratio=float(gamma),
        heritability=float(heritability),
        log_likelihood=float(ll),
        reml=reml,
        aic=float(aic),
        bic=float(bic),
        n_obs=n,
        n_covariates=p,
        rank=M,
        converged=fit.converged,
        n_iter=fit.n_iter,
        genetic_values=fit.genetic_values,
        fitted_values=fit.fitted,
        residuals=fit.residuals,
        theta=float(fit.theta),
    )
