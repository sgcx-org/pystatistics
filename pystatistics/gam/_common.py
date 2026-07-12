"""
Parameter payloads for Generalized Additive Model results.

Each dataclass is a frozen payload describing a fitted GAM or
one of its smooth terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GAMParams:
    """Parameters from a fitted GAM.

    Carries the full numerical output of the stable (augmented-QR)
    penalized iteratively re-weighted least squares fitting procedure.

    Attributes:
        coefficients: Full coefficient vector (parametric + constrained
            basis coefficients; smooth coefficients live in the sum-to-zero
            constrained parameterisation, so a smooth declared with ``k``
            basis functions contributes ``k - 1`` coefficients — same as
            mgcv).
        fitted_values: Response-scale predictions (mu_hat).
        linear_predictor: Link-scale predictions (eta_hat).
        residuals: Working residuals (y - mu_hat).
        edf: Effective degrees of freedom per smooth term.
        total_edf: Trace of the influence matrix (parametric + smooths).
        lambdas: Selected (or user-fixed) smoothing parameters, one per
            smooth. For ``cr`` smooths these are directly comparable to
            mgcv's ``sp`` (identical penalty construction and scaling);
            for ``tp`` smooths the *fit* is function-space identical to
            mgcv's but the penalty parameterisation (and hence the sp
            units) differs by the eigenbasis convention — compare tp fits
            via EDF/fitted values, not raw sp.
        s_scales: Per-smooth ``S.scale`` penalty normalisation factors
            (divide ``lambdas`` by these to get function-space units).
        covariance: Bayesian posterior covariance of the coefficients
            (``scale * (X'WX + S_lambda)^{-1}``, mgcv's ``Vp``).
        scale: Estimated or fixed dispersion parameter.
        gcv: GCV score (scale-free families: the UBRE score is the
            criterion actually minimised; both are reported).
        ubre: UBRE / scaled-AIC score.
        reml_score: Laplace REML criterion at the selected lambdas — only
            populated when ``method='REML'``; ``None`` otherwise.
        deviance: Model deviance.
        null_deviance: Null-model deviance (intercept only).
        log_likelihood: Maximized log-likelihood.
        aic: Akaike information criterion, classical GAM convention
            ``-2 loglik + 2 (total_edf [+1 if scale estimated])``. NOTE:
            mgcv >= 1.8.x corrects the df for smoothing-parameter
            uncertainty (Wood, Pya & Saefken 2016, 'edf2'), so mgcv's
            ``AIC()`` is systematically slightly larger; documented, not
            hidden.
        n_obs: Number of observations used in fitting.
        family_name: Name of the exponential family (e.g. 'gaussian').
        link_name: Name of the link function (e.g. 'identity').
        dispersion_fixed: True when the family's dispersion is fixed at 1
            (binomial, poisson, negative.binomial) — controls z-vs-t and
            Chi.sq-vs-F conventions downstream.
        theta: Negative-binomial dispersion parameter for a ``family='nb'``
            fit — the estimated theta (mgcv's ``getTheta``) when it was fit
            automatically, or the user-supplied value for a fixed
            ``NegativeBinomial(theta=...)`` family. ``None`` for every other
            family. This is the defining output of an NB GAM; larger theta
            means less overdispersion (theta -> inf recovers Poisson).
        converged: Whether the P-IRLS algorithm converged.
        outer_converged: Whether the smoothing-parameter search converged
            away from its bounds (always True when ``sp`` was user-fixed).
        n_iter: Number of P-IRLS iterations in the final fit.
        method: Smoothing parameter selection method ('GCV' or 'REML').
        backend_name: Execution-path disclosure (always ``'cpu_qr_pirls'``;
            the GAM module is CPU-only).
    """

    coefficients: NDArray[np.floating[Any]] = field(repr=False)
    fitted_values: NDArray[np.floating[Any]] = field(repr=False)
    linear_predictor: NDArray[np.floating[Any]] = field(repr=False)
    residuals: NDArray[np.floating[Any]] = field(repr=False)
    edf: NDArray[np.floating[Any]] = field(repr=False)
    total_edf: float
    lambdas: NDArray[np.floating[Any]] = field(repr=False)
    s_scales: NDArray[np.floating[Any]] = field(repr=False)
    covariance: NDArray[np.floating[Any]] = field(repr=False)
    scale: float
    gcv: float
    ubre: float
    reml_score: float | None
    deviance: float
    null_deviance: float
    log_likelihood: float
    aic: float
    n_obs: int
    family_name: str
    link_name: str
    dispersion_fixed: bool
    theta: float | None
    converged: bool
    outer_converged: bool
    n_iter: int
    method: str
    backend_name: str


@dataclass(frozen=True)
class SmoothInfo:
    """Information about a single smooth term in a fitted GAM.

    One ``SmoothInfo`` is produced per ``s()`` term after fitting.
    It records the basis metadata and the approximate significance
    test for the smooth.

    Attributes:
        term_name: Display name, e.g. ``'s(x1)'``.
        var_name: Bare predictor name, e.g. ``'x1'``.
        basis_type: Basis identifier: ``'cr'`` (cubic regression spline)
            or ``'tp'`` (thin plate regression spline).
        k: The basis dimension as declared in ``s(x, k=...)`` (the smooth
            contributes ``k - 1`` coefficients after the sum-to-zero
            identifiability constraint, same as mgcv).
        edf: Effective degrees of freedom for this term.
        ref_df: Reference degrees of freedom, ``tr(2H - HH)`` over the
            term's block (mgcv's Ref.df convention).
        chi_sq: Approximate significance statistic. For fixed-dispersion
            families this is the chi-squared-referenced quadratic form
            ``beta' pinv(V_beta) beta``; for estimated-dispersion families
            it is that form divided by Ref.df, i.e. an F STATISTIC (the
            summary labels the column accordingly, matching mgcv). A
            simplified form of mgcv's Wood (2013) test — agrees with
            mgcv's reported statistic to ~0.2% on validated cases, not
            bit-identical.
        p_value: Approximate p-value for the term.
        coef_indices: ``(start, end)`` slice into the full coefficient
            vector identifying this term's (constrained) coefficients.
        lambdas: Selected smoothing parameter(s) for this term — a length-1
            tuple for an ordinary ``s()`` smooth, one entry per margin for a
            tensor-product ``te()``/``ti()`` smooth (each margin has its own
            penalty and smoothing parameter, as in mgcv).
        s_scales: Penalty normalisation factor(s), aligned with ``lambdas``
            (``lambda / s_scale`` is in function-space units).
    """

    term_name: str
    var_name: str
    basis_type: str
    k: int
    edf: float
    ref_df: float
    chi_sq: float
    p_value: float
    coef_indices: tuple[int, int]
    lambdas: tuple[float, ...]
    s_scales: tuple[float, ...]
