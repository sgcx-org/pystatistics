"""
Main entry point for fitting Generalized Additive Models.

Provides the ``gam()`` function which mirrors ``mgcv::gam()`` for common
use cases: penalized regression splines with automatic smoothing-parameter
selection via GCV (mgcv ``GCV.Cp`` semantics: GCV for estimated scale,
UBRE for known scale) or Laplace REML, fitted by stable augmented-QR
P-IRLS with sum-to-zero identifiability constraints — the same numerical
architecture mgcv uses (Wood 2011, 2017).

The GAM module is CPU-only: there is no ``backend=`` parameter (the
library convention is that a module with no GPU path exposes none). The
former float32 GPU path was removed after failing its no-silent-wrong
check — see the 4.6.0 changelog.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result
from pystatistics.gam._basis import BuiltSmooth, build_design
from pystatistics.gam._common import GAMParams, SmoothInfo
from pystatistics.gam._criteria import (
    estimate_scale,
    gcv_score,
    reml_score,
    select_lambdas,
    ubre_score,
)
from pystatistics.gam._edf import (
    edf_per_block,
    influence_matrix,
    posterior_covariance,
    ref_df_per_block,
    total_edf as total_edf_of,
)
from pystatistics.gam._pirls import fit_fixed_lambda, make_penalty_roots
from pystatistics.gam._smooth import SmoothTerm
from pystatistics.gam._smooth_test import smooth_term_test
from pystatistics.gam.solution import GAMSolution
from pystatistics.regression.families import Family, resolve_family

_BACKEND_NAME = "cpu_qr_pirls"


def gam(
    y: ArrayLike,
    X: ArrayLike | None = None,
    *,
    smooths: list[SmoothTerm] | None = None,
    smooth_data: dict[str, ArrayLike] | None = None,
    family: str | Family = "gaussian",
    method: str = "GCV",
    sp: ArrayLike | None = None,
    tol: float = 1e-8,
    max_iter: int = 200,
    names: list[str] | None = None,
) -> GAMSolution:
    """Fit a Generalized Additive Model.

    Matches R's ``mgcv::gam()`` for common cases. A GAM models the
    response as::

        g(E[y]) = beta_0 + f_1(x_1) + f_2(x_2) + ... + X @ beta

    where each ``f_j`` is a smooth function estimated from the data using
    penalized regression splines, with per-smooth sum-to-zero
    identifiability constraints (a smooth declared with ``k`` basis
    functions contributes ``k - 1`` coefficients, exactly as mgcv).

    Args:
        y: Response variable, 1-D array of *n* observations.
        X: Parametric design matrix ``(n, p)``. May include an intercept
            column. If ``None``, an intercept-only parametric part is used.
        smooths: Smooth term specifications from :func:`s`.
        smooth_data: Mapping from smooth variable names to ``(n,)`` arrays.
        family: Exponential family, e.g. ``'gaussian'``, ``'binomial'``,
            ``'poisson'``, ``'Gamma'``. ``method='REML'`` supports the
            fixed-dispersion families and the Gaussian-identity model.
        method: Smoothing-parameter selection criterion: ``'GCV'``
            (mgcv ``GCV.Cp`` semantics — UBRE is used automatically for
            known-scale families, as mgcv does) or ``'REML'`` (Laplace,
            Wood 2011).
        sp: Optional fixed smoothing parameters, one per smooth (mgcv's
            ``sp=``). Skips selection; use for reproducing a specific fit.
        tol: P-IRLS convergence tolerance (relative penalized deviance).
        max_iter: Maximum P-IRLS iterations.
        names: Optional display names for the parametric coefficients.

    Returns:
        A :class:`GAMSolution`.

    Raises:
        ValidationError: invalid inputs, unknown family/method/basis,
            ``k`` exceeding the unique values of a smooth variable, or a
            ``sp`` vector of the wrong length/sign.
        ConvergenceError: a genuinely divergent P-IRLS (loud, never a
            silent answer).
    """
    # ------------------------------------------------------------------
    # Input validation (boundary — Rule 2)
    # ------------------------------------------------------------------
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = y_arr.shape[0]
    if n == 0:
        raise ValidationError("y must be non-empty")
    if not np.all(np.isfinite(y_arr)):
        raise ValidationError("y contains non-finite values (NaN or Inf)")

    smooths = list(smooths) if smooths is not None else []
    smooth_data = dict(smooth_data) if smooth_data is not None else {}

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

    if X is not None:
        X_param = np.asarray(X, dtype=np.float64)
        if X_param.ndim == 1:
            X_param = X_param.reshape(-1, 1)
        if X_param.shape[0] != n:
            raise ValidationError(
                f"y has {n} obs but X has {X_param.shape[0]} rows"
            )
        if not np.all(np.isfinite(X_param)):
            raise ValidationError("X contains non-finite values")
    else:
        X_param = np.ones((n, 1), dtype=np.float64)
    parametric_cols = X_param.shape[1]

    if names is not None and len(names) != parametric_cols:
        raise ValidationError(
            f"names has {len(names)} entries but the parametric design has "
            f"{parametric_cols} column(s); they must match (smooth-term "
            f"coefficients are not named)"
        )

    n_smooths = len(smooths)
    sp_arr: NDArray | None = None
    if sp is not None:
        sp_arr = np.asarray(sp, dtype=np.float64).ravel()
        if sp_arr.shape[0] != n_smooths:
            raise ValidationError(
                f"sp has {sp_arr.shape[0]} entries but there are "
                f"{n_smooths} smooth terms"
            )
        if np.any(~np.isfinite(sp_arr)) or np.any(sp_arr <= 0.0):
            raise ValidationError("sp entries must be finite and > 0")

    # ------------------------------------------------------------------
    # Build the (full-rank, constrained) design + penalty roots
    # ------------------------------------------------------------------
    X_aug, built = build_design(X_param, smooth_data_np, smooths)
    blocks = [b.block for b in built]
    roots = make_penalty_roots([b.S_block for b in built], blocks)
    smooth_names = [f"s({st.var_name})" for st in smooths]

    # ------------------------------------------------------------------
    # Smoothing-parameter selection (or user-fixed sp)
    # ------------------------------------------------------------------
    import warnings as _warnings

    if n_smooths == 0:
        lambdas = np.array([], dtype=np.float64)
        outer_converged = True
    elif sp_arr is not None:
        lambdas = sp_arr
        outer_converged = True
    else:
        with _warnings.catch_warnings():
            # The lambda search runs ~50-150 P-IRLS fits; per-fit warnings
            # (separation, rank deficiency) would repeat once per
            # evaluation. Dedupe to one occurrence within this call.
            _warnings.simplefilter("once", UserWarning)
            lambdas, outer_converged = select_lambdas(
                y_arr, X_aug, roots, fam, method_upper, tol, max_iter,
                smooth_names=smooth_names,
            )
        if not outer_converged:
            import warnings

            warnings.warn(
                f"smoothing-parameter search did not cleanly converge "
                f"(method={method_upper}); results may sit on the search "
                f"boundary — inspect solution.params.lambdas",
                UserWarning, stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Final fit + inference quantities
    # ------------------------------------------------------------------
    fit = fit_fixed_lambda(
        y_arr, X_aug, roots, lambdas, fam, tol, max_iter,
        smooth_names=smooth_names,
    )
    H = influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank)
    tot_edf = total_edf_of(H)
    edf = edf_per_block(H, blocks)
    ref_df = ref_df_per_block(H, blocks)

    scale = estimate_scale(fit, y_arr, fam, tot_edf)
    covariance = posterior_covariance(fit.R, fit.piv, fit.rank, scale)

    gcv = gcv_score(fit.deviance, n, tot_edf)
    ubre = ubre_score(fit.deviance, n, tot_edf, scale=scale)
    reml = (
        reml_score(fit, y_arr, fam, roots, lambdas)
        if (method_upper == "REML" and n_smooths > 0) else None
    )

    # Null deviance (intercept-only fitted mean is ybar for every
    # supported family/link at the optimum).
    wt = np.ones(n, dtype=np.float64)
    mu_null = np.full(n, y_arr.mean(), dtype=np.float64)
    null_deviance = fam.deviance(y_arr, mu_null, wt)

    # Log-likelihood / AIC — classical GAM df convention (see GAMParams).
    log_lik = fam.log_likelihood(y_arr, fit.mu, wt, max(scale, 1e-300))
    aic_df = tot_edf + (0.0 if fam.dispersion_is_fixed else 1.0)
    aic = -2.0 * log_lik + 2.0 * aic_df

    # ------------------------------------------------------------------
    # Per-smooth summaries
    # ------------------------------------------------------------------
    smooth_infos: list[SmoothInfo] = []
    for i, b in enumerate(built):
        s_, e_ = b.block
        stat, p_val = smooth_term_test(
            fit.beta[s_:e_], covariance[s_:e_, s_:e_],
            edf=float(edf[i]), ref_df=float(ref_df[i]),
            scale_known=fam.dispersion_is_fixed,
            resid_df=n - tot_edf,
        )
        smooth_infos.append(SmoothInfo(
            term_name=smooth_names[i],
            var_name=b.term.var_name,
            basis_type=b.term.bs,
            k=b.term.k,
            edf=float(edf[i]),
            ref_df=float(ref_df[i]),
            chi_sq=stat,
            p_value=p_val,
            coef_indices=(s_, e_),
            lambda_=float(lambdas[i]),
            s_scale=float(b.s_scale),
        ))

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    gam_params = GAMParams(
        coefficients=fit.beta,
        fitted_values=fit.mu,
        linear_predictor=fit.eta,
        residuals=y_arr - fit.mu,
        edf=edf,
        total_edf=tot_edf,
        lambdas=lambdas,
        s_scales=np.array([b.s_scale for b in built], dtype=np.float64),
        covariance=covariance,
        scale=scale,
        gcv=gcv,
        ubre=ubre,
        reml_score=reml,
        deviance=fit.deviance,
        null_deviance=null_deviance,
        log_likelihood=log_lik,
        aic=aic,
        n_obs=n,
        family_name=fam.name,
        link_name=fam.link.name,
        dispersion_fixed=bool(fam.dispersion_is_fixed),
        converged=fit.converged,
        outer_converged=outer_converged,
        n_iter=fit.n_iter,
        method=method_upper,
        backend_name=_BACKEND_NAME,
    )

    result = Result(
        params=gam_params,
        info={
            "method": method_upper,
            "converged": fit.converged,
            "outer_converged": outer_converged,
            "n_iter": fit.n_iter,
            "n_smooths": n_smooths,
            "lambdas": lambdas.tolist(),
            "rank": fit.rank,
            "n_coefficients": int(X_aug.shape[1]),
            "parametric_cols": parametric_cols,
        },
        timing=None,
        backend_name=_BACKEND_NAME,
    )

    return GAMSolution(result, smooth_infos, _names=names)
