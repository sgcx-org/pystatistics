"""
Main entry point for fitting Generalized Additive Models.

Provides the ``gam()`` function which mirrors ``mgcv::gam()`` for
common use cases: penalised regression splines with automatic
smoothing parameter selection via GCV or REML.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as sp_stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result
from pystatistics.gam._common import GAMParams, SmoothInfo
from pystatistics.gam._fit import (
    _build_model_matrix,
    _compute_edf,
    _compute_hat_matrix_trace,
    _fit_gam_fixed_lambda,
)
from pystatistics.gam._gcv import select_smoothing_parameters
from pystatistics.gam._smooth import SmoothTerm
from pystatistics.gam.solution import GAMSolution
from pystatistics.regression.families import Family, resolve_family


def gam(
    y: ArrayLike,
    X: ArrayLike | None = None,
    *,
    smooths: list[SmoothTerm] | None = None,
    smooth_data: dict[str, ArrayLike] | None = None,
    family: str | Family = "gaussian",
    method: str = "GCV",
    tol: float = 1e-8,
    max_iter: int = 200,
    names: list[str] | None = None,
) -> GAMSolution:
    """Fit a Generalized Additive Model.

    Matches R's ``mgcv::gam()`` for common cases.  A GAM models the
    response as::

        g(E[y]) = beta_0 + f_1(x_1) + f_2(x_2) + ... + X @ beta

    where each ``f_j`` is a smooth function estimated from the data
    using penalised regression splines.

    Args:
        y: Response variable, 1-D array of *n* observations.
        X: Parametric design matrix ``(n, p)``.  May include an
            intercept column.  If ``None``, an intercept-only parametric
            part is used.
        smooths: List of :class:`SmoothTerm` specifications, e.g.
            ``[s('x1'), s('x2', k=20)]``.
        smooth_data: Dict mapping variable names to arrays.  Must
            contain all variables referenced in *smooths*.
        family: GLM family -- ``'gaussian'``, ``'binomial'``,
            ``'poisson'``, ``'gamma'``, or a :class:`Family` instance.
        method: Smoothing parameter selection: ``'GCV'`` or ``'REML'``.
        tol: Convergence tolerance for P-IRLS.
        max_iter: Maximum P-IRLS iterations.
        names: Optional names for parametric coefficients.

    Returns:
        :class:`GAMSolution` with fitted model, smooth-term info,
        and diagnostics.

    Raises:
        ValidationError: On invalid or inconsistent inputs.

    Example::

        >>> from pystatistics.gam import gam, s
        >>> result = gam(y, smooths=[s('x1'), s('x2')],
        ...              smooth_data={'x1': x1, 'x2': x2})
        >>> print(result.summary())
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = y_arr.shape[0]

    if smooths is None:
        smooths = []
    if smooth_data is None:
        smooth_data = {}

    if not smooths and X is None:
        raise ValidationError(
            "At least one of 'smooths' or 'X' must be provided"
        )

    fam = resolve_family(family)

    method_upper = method.upper()
    if method_upper not in ("GCV", "REML"):
        raise ValidationError(
            f"method must be 'GCV' or 'REML', got {method!r}"
        )

    # Validate smooth_data lengths
    smooth_data_np: dict[str, NDArray] = {}
    for st in smooths:
        if st.var_name not in smooth_data:
            raise ValidationError(
                f"smooth_data missing variable {st.var_name!r}"
            )
        arr = np.asarray(smooth_data[st.var_name], dtype=np.float64).ravel()
        if arr.shape[0] != n:
            raise ValidationError(
                f"y has {n} obs but smooth variable {st.var_name!r} "
                f"has {arr.shape[0]}"
            )
        smooth_data_np[st.var_name] = arr

    # Parametric design matrix
    if X is not None:
        X_param = np.asarray(X, dtype=np.float64)
        if X_param.ndim == 1:
            X_param = X_param.reshape(-1, 1)
        if X_param.shape[0] != n:
            raise ValidationError(
                f"y has {n} obs but X has {X_param.shape[0]} rows"
            )
    else:
        # Intercept-only
        X_param = np.ones((n, 1), dtype=np.float64)

    parametric_cols = X_param.shape[1]

    # ------------------------------------------------------------------
    # Build model matrix
    # ------------------------------------------------------------------
    X_aug, S_penalties, term_indices = _build_model_matrix(
        X_param, smooth_data_np, smooths,
    )

    n_smooths = len(smooths)

    # ------------------------------------------------------------------
    # Select smoothing parameters
    # ------------------------------------------------------------------
    if n_smooths > 0:
        lambdas = select_smoothing_parameters(
            y_arr, X_aug, S_penalties, fam, parametric_cols,
            method_upper, tol, max_iter, n_smooths,
        )
    else:
        lambdas = np.array([], dtype=np.float64)

    # ------------------------------------------------------------------
    # Final fit with optimal lambdas
    # ------------------------------------------------------------------
    beta, mu, eta, W, deviance, n_iter, converged = _fit_gam_fixed_lambda(
        y_arr, X_aug, S_penalties, lambdas, fam, parametric_cols, tol, max_iter,
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    wt = np.ones(n, dtype=np.float64)

    # EDF per smooth
    if n_smooths > 0:
        edf = _compute_edf(X_aug, W, S_penalties, lambdas, term_indices)
    else:
        edf = np.array([], dtype=np.float64)

    total_edf = _compute_hat_matrix_trace(X_aug, W, S_penalties, lambdas)

    # Scale estimate
    df_resid = max(n - total_edf, 1.0)
    if fam.dispersion_is_fixed:
        scale = 1.0
    else:
        scale = deviance / df_resid

    # Null deviance
    mu_null = np.full(n, y_arr.mean(), dtype=np.float64)
    null_deviance = fam.deviance(y_arr, mu_null, wt)

    # Log-likelihood and AIC
    log_lik = fam.log_likelihood(y_arr, mu, wt, max(scale, 1e-20))
    aic = -2.0 * log_lik + 2.0 * total_edf

    # GCV
    denom = max(n - total_edf, 1.0)
    gcv = float(n) * deviance / (denom * denom)

    # UBRE
    ubre = deviance / n - scale + 2.0 * total_edf * scale / n

    # ------------------------------------------------------------------
    # Build SmoothInfo objects
    # ------------------------------------------------------------------
    smooth_infos: list[SmoothInfo] = []
    for i, (st, (s_start, s_end)) in enumerate(zip(smooths, term_indices)):
        edf_i = float(edf[i])

        # Approximate significance test (Wood, 2013)
        ref_df = min(edf_i * 1.2, float(st.k - 1))
        ref_df = max(ref_df, 1.0)

        # Chi-squared statistic: beta_j' S_j^{-1} beta_j / scale
        beta_j = beta[s_start:s_end]
        S_j = S_penalties[i][s_start:s_end, s_start:s_end]
        chi_sq = float(beta_j @ beta_j) / max(scale, 1e-20)

        # p-value from chi-squared with ref_df degrees of freedom
        p_value = float(1.0 - sp_stats.chi2.cdf(chi_sq, df=ref_df))

        smooth_infos.append(
            SmoothInfo(
                term_name=f"s({st.var_name})",
                var_name=st.var_name,
                basis_type=st.bs,
                k=st.k,
                edf=edf_i,
                ref_df=ref_df,
                chi_sq=chi_sq,
                p_value=p_value,
                coef_indices=(s_start, s_end),
            )
        )

    # ------------------------------------------------------------------
    # Assemble GAMParams and Result
    # ------------------------------------------------------------------
    residuals = y_arr - mu

    gam_params = GAMParams(
        coefficients=beta,
        fitted_values=mu,
        linear_predictor=eta,
        residuals=residuals,
        edf=edf,
        total_edf=total_edf,
        scale=scale,
        gcv=gcv,
        ubre=ubre,
        deviance=deviance,
        null_deviance=null_deviance,
        log_likelihood=log_lik,
        aic=aic,
        n_obs=n,
        family_name=fam.name,
        link_name=fam.link.name,
        converged=converged,
        n_iter=n_iter,
        method=method_upper,
    )

    result = Result(
        params=gam_params,
        info={
            "method": method_upper,
            "converged": converged,
            "n_iter": n_iter,
            "n_smooths": n_smooths,
            "lambdas": lambdas.tolist() if n_smooths > 0 else [],
        },
        timing=None,
        backend_name="cpu_pirls",
    )

    return GAMSolution(result, smooth_infos, _names=names)
