"""
Solver dispatch for mixed models.

Public API:
    lmm()  — fit a linear mixed model (REML or ML)
    glmm() — fit a generalized linear mixed model (Laplace approximation)
"""

from __future__ import annotations

import warnings
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy import stats

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer

from pystatistics.mixed._common import (
    LMMParams, GLMMParams, VarCompSummary,
)
from pystatistics.mixed._random_effects import (
    parse_random_effects, build_z_matrix, build_lambda,
    theta_lower_bounds, theta_start,
)
from pystatistics.mixed._pls import solve_pls
from pystatistics.mixed._deviance import (
    profiled_deviance_lmm, profiled_deviance_glmm,
)
from pystatistics.mixed._satterthwaite import satterthwaite_df
from pystatistics.mixed.design import MixedDesign
from pystatistics.mixed.solution import LMMSolution, GLMMSolution


def lmm(
    y: ArrayLike,
    X: ArrayLike,
    groups: dict[str, ArrayLike],
    *,
    random_effects: dict[str, list[str]] | None = None,
    random_data: dict[str, ArrayLike] | None = None,
    reml: bool = True,
    tol: float = 1e-8,
    max_iter: int = 200,
    compute_satterthwaite: bool = True,
) -> LMMSolution:
    """Fit a linear mixed model.

    Estimates fixed effects β, random effects variance components,
    and conditional modes (BLUPs) of random effects using the profiled
    REML/ML deviance approach from Bates et al. (2015).

    Args:
        y: Response vector (n,).
        X: Fixed effects design matrix (n, p). Should include an
            intercept column if desired.
        groups: Dict mapping grouping factor names to group label arrays.
            Example: {'subject': subject_ids}.
        random_effects: Optional dict mapping group names to lists of
            random effect terms. Default: random intercept per group.
            Example: {'subject': ['1', 'time']} for (1 + time | subject).
        random_data: Optional dict mapping variable names to data arrays
            for random slope variables.
            Example: {'time': time_array}.
        reml: If True (default), use REML estimation. If False, use ML.
            Use ML (reml=False) for likelihood ratio tests between models
            with different fixed effects.
        tol: Convergence tolerance for the optimizer. Default 1e-8.
        max_iter: Maximum optimizer iterations. Default 200.
        compute_satterthwaite: If True (default), compute Satterthwaite
            denominator df for fixed effects. Set to False for speed
            if p-values are not needed.

    Returns:
        LMMSolution with fixed effects, random effects, variance components,
        model fit statistics, and R-style summary().

    Examples:
        # Random intercept model
        >>> result = lmm(y, X, groups={'subject': subject_ids})

        # Random intercept + slope
        >>> result = lmm(y, X, groups={'subject': subject_ids},
        ...              random_effects={'subject': ['1', 'time']},
        ...              random_data={'time': time_array})

        # Crossed random effects
        >>> result = lmm(y, X, groups={'subject': subj, 'item': item})
    """
    timer = Timer()
    timer.start()

    # Validate inputs
    design = MixedDesign.validate(
        np.asarray(y, dtype=np.float64),
        np.asarray(X, dtype=np.float64),
        groups,
        random_effects,
        random_data,
    )

    with timer.section('setup'):
        # Parse random effects and build Z
        specs = parse_random_effects(
            design.groups, design.random_effects, design.random_data, design.n
        )
        Z = build_z_matrix(specs)

        # Starting values and bounds
        theta0 = theta_start(specs)
        lb = theta_lower_bounds(specs)
        bounds = [(lb[i], None) for i in range(len(theta0))]

    # Optimize θ — use multiple starting points for models with random slopes
    # (the profiled deviance can have local minima when q > 1)
    with timer.section('optimization'):
        has_slopes = any(s.n_terms > 1 for s in specs)

        if has_slopes:
            # Generate candidate starting values: the default [1,0,...,1]
            # plus variants with smaller diagonal values for slope terms
            starts = [theta0]
            for scale in (0.2, 0.5):
                alt = theta0.copy()
                idx = 0
                for spec in specs:
                    q = spec.n_terms
                    for row in range(q):
                        for col in range(row + 1):
                            if row == col and row > 0:
                                alt[idx] = scale
                            idx += 1
                starts.append(alt)

            best_result = None
            for start in starts:
                res = minimize(
                    profiled_deviance_lmm,
                    start,
                    args=(design.X, Z, design.y, specs, reml),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol * 10},
                )
                if best_result is None or res.fun < best_result.fun:
                    best_result = res
            opt_result = best_result
        else:
            opt_result = minimize(
                profiled_deviance_lmm,
                theta0,
                args=(design.X, Z, design.y, specs, reml),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol * 10},
            )

    converged = opt_result.success
    theta_hat = opt_result.x
    n_iter = opt_result.nit

    if not converged:
        warnings.warn(
            f"LMM optimizer did not converge after {n_iter} iterations. "
            f"Message: {opt_result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Final PLS solve at optimal θ
    with timer.section('final_solve'):
        Lambda_hat = build_lambda(theta_hat, specs)
        pls = solve_pls(design.X, Z, design.y, Lambda_hat, reml=reml)

    # Compute variance components
    with timer.section('variance_components'):
        var_comps = _extract_var_components(theta_hat, pls.sigma_sq, specs)
        n_groups_dict = {s.group_name: s.n_groups for s in specs}

    # Extract BLUPs
    with timer.section('blups'):
        random_effs = _extract_blups(pls.b, specs)

    # Compute Satterthwaite df and p-values
    with timer.section('satterthwaite'):
        se = _compute_se(pls, design.X, Z, Lambda_hat, design.X.shape[1])

        if compute_satterthwaite:
            df_satt = satterthwaite_df(
                theta_hat, design.X, Z, design.y, specs, reml=reml
            )
        else:
            # Use residual df as fallback
            df_satt = np.full(design.p, float(design.n - design.p))

        t_vals = pls.beta / se
        p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df_satt)

    # Log-likelihood, AIC, BIC
    with timer.section('model_fit'):
        ll, aic, bic = _compute_fit_stats(
            pls, theta_hat, design.n, design.p, specs, reml
        )

    # Coefficient names
    coef_names = _make_coef_names(design.p)

    timer.stop()

    # Assemble params
    params = LMMParams(
        coefficients=pls.beta,
        coefficient_names=tuple(coef_names),
        se=se,
        df_satterthwaite=df_satt,
        t_values=t_vals,
        p_values=p_vals,
        var_components=tuple(var_comps),
        residual_variance=pls.sigma_sq,
        residual_std=np.sqrt(pls.sigma_sq),
        log_likelihood=ll,
        reml=reml,
        aic=aic,
        bic=bic,
        n_obs=design.n,
        n_groups=n_groups_dict,
        converged=converged,
        n_iter=n_iter,
        random_effects=random_effs,
        fitted_values=pls.fitted,
        residuals=pls.residuals,
        theta=theta_hat,
    )

    warn_list = []
    if not converged:
        warn_list.append(f"Optimizer did not converge: {opt_result.message}")

    result = Result(
        params=params,
        info={
            'method': 'REML' if reml else 'ML',
            'optimizer': 'L-BFGS-B',
            'converged': converged,
            'n_iter': n_iter,
            'deviance': opt_result.fun,
        },
        timing=timer.result(),
        backend_name='cpu_lmm',
        warnings=tuple(warn_list),
    )

    return LMMSolution(_result=result)


def glmm(
    y: ArrayLike,
    X: ArrayLike,
    groups: dict[str, ArrayLike],
    *,
    family: 'str | Family' = 'binomial',
    random_effects: dict[str, list[str]] | None = None,
    random_data: dict[str, ArrayLike] | None = None,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> GLMMSolution:
    """Fit a generalized linear mixed model.

    Uses Laplace approximation to the marginal likelihood with
    Penalized IRLS (PIRLS) for the inner loop and L-BFGS-B for
    the outer optimization over variance components.

    Args:
        y: Response vector (n,).
        X: Fixed effects design matrix (n, p).
        groups: Dict mapping grouping factor names to group label arrays.
        family: GLM family specification. String ('binomial', 'poisson')
            or a Family instance from pystatistics.regression.families.
        random_effects: Optional random effects specification.
        random_data: Optional data for random slope variables.
        tol: Convergence tolerance.
        max_iter: Maximum optimizer iterations.

    Returns:
        GLMMSolution with fixed effects, random effects, and model fit.
    """
    from pystatistics.regression.families import resolve_family, Family

    timer = Timer()
    timer.start()

    # Resolve family
    if not isinstance(family, Family):
        family_obj = resolve_family(family)
    else:
        family_obj = family

    # Validate inputs
    design = MixedDesign.validate(
        np.asarray(y, dtype=np.float64),
        np.asarray(X, dtype=np.float64),
        groups,
        random_effects,
        random_data,
    )

    with timer.section('setup'):
        specs = parse_random_effects(
            design.groups, design.random_effects, design.random_data, design.n
        )
        Z = build_z_matrix(specs)
        theta0 = theta_start(specs)
        lb = theta_lower_bounds(specs)
        bounds = [(lb[i], None) for i in range(len(theta0))]

    # Optimize θ via Laplace-approximated deviance
    with timer.section('optimization'):
        opt_result = minimize(
            profiled_deviance_glmm,
            theta0,
            args=(design.X, Z, design.y, specs, family_obj),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol * 10},
        )

    converged = opt_result.success
    theta_hat = opt_result.x
    n_iter = opt_result.nit

    if not converged:
        warnings.warn(
            f"GLMM optimizer did not converge after {n_iter} iterations. "
            f"Message: {opt_result.message}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Final PIRLS solve at optimal θ
    with timer.section('final_solve'):
        from pystatistics.mixed._pirls import solve_pirls

        Lambda_hat = build_lambda(theta_hat, specs)
        pirls = solve_pirls(design.X, Z, design.y, Lambda_hat, family_obj)

    # Variance components (for GLMM, σ² = 1 by convention)
    with timer.section('variance_components'):
        var_comps = _extract_var_components(theta_hat, 1.0, specs)
        n_groups_dict = {s.group_name: s.n_groups for s in specs}

    # BLUPs
    with timer.section('blups'):
        random_effs = _extract_blups(pirls.pls.b, specs)

    # Fixed effect SEs and Wald z-statistics
    with timer.section('inference'):
        se = _compute_se_glmm(pirls.pls, design.X.shape[1])
        z_vals = pirls.pls.beta / se
        p_vals = 2.0 * stats.norm.sf(np.abs(z_vals))

    # Model fit
    with timer.section('model_fit'):
        wt = np.ones(design.n, dtype=np.float64)
        deviance = family_obj.deviance(design.y, pirls.mu, wt)
        n_params = len(pirls.pls.beta) + len(theta_hat)

        # Laplace-approximated marginal log-likelihood:
        # ll = conditional_loglik - 0.5 * ||u||^2 - 0.5 * log|L_theta|^2
        # where conditional_loglik = sum(f(y_i | mu_i)) is the full
        # conditional log-likelihood including normalizing constants.
        # For GLMM, dispersion = 1.
        cond_ll = family_obj.log_likelihood(design.y, pirls.mu, wt, 1.0)
        penalty = float(pirls.pls.u @ pirls.pls.u)
        log_det_L = 2.0 * np.sum(np.log(np.maximum(np.diag(pirls.pls.L), 1e-20)))
        ll = cond_ll - 0.5 * penalty - 0.5 * log_det_L
        aic = -2.0 * ll + 2.0 * n_params
        bic = -2.0 * ll + np.log(design.n) * n_params

    coef_names = _make_coef_names(design.p)

    timer.stop()

    params = GLMMParams(
        coefficients=pirls.pls.beta,
        coefficient_names=tuple(coef_names),
        se=se,
        t_values=z_vals,
        p_values=p_vals,
        var_components=tuple(var_comps),
        log_likelihood=ll,
        deviance=deviance,
        aic=aic,
        bic=bic,
        n_obs=design.n,
        n_groups=n_groups_dict,
        family_name=family_obj.name,
        link_name=family_obj.link.name,
        converged=converged,
        n_iter=n_iter,
        random_effects=random_effs,
        fitted_values=pirls.mu,
        linear_predictor=pirls.eta,
        residuals=design.y - pirls.mu,
        theta=theta_hat,
    )

    warn_list = []
    if not converged:
        warn_list.append(f"Optimizer did not converge: {opt_result.message}")
    if not pirls.converged:
        warn_list.append(f"PIRLS did not converge after {pirls.n_iter} iterations")

    result = Result(
        params=params,
        info={
            'method': 'Laplace',
            'family': family_obj.name,
            'link': family_obj.link.name,
            'optimizer': 'L-BFGS-B',
            'converged': converged,
            'pirls_converged': pirls.converged,
            'n_iter': n_iter,
            'pirls_iter': pirls.n_iter,
            'deviance': opt_result.fun,
        },
        timing=timer.result(),
        backend_name='cpu_glmm',
        warnings=tuple(warn_list),
    )

    return GLMMSolution(_result=result)


# =====================================================================
# Helpers
# =====================================================================

def _extract_var_components(
    theta: np.ndarray,
    sigma_sq: float,
    specs: list,
) -> list[VarCompSummary]:
    """Extract variance component summaries from θ and σ².

    The actual covariance of random effects is σ² × Λ Λ'.
    For each grouping factor, compute the covariance matrix and
    extract variance, std dev, and correlations.
    """
    var_comps = []
    theta_offset = 0

    for spec in specs:
        q = spec.n_terms
        n_theta = spec.theta_size

        # Reconstruct the q × q lower-triangular Cholesky factor
        theta_k = theta[theta_offset:theta_offset + n_theta]
        theta_offset += n_theta

        T = np.zeros((q, q), dtype=np.float64)
        idx = 0
        for row in range(q):
            for col in range(row + 1):
                T[row, col] = theta_k[idx]
                idx += 1

        # Covariance matrix: σ² × T T'
        cov_matrix = sigma_sq * (T @ T.T)

        # Term names
        term_names = []
        for term in spec.terms:
            if term == '1':
                term_names.append('(Intercept)')
            else:
                term_names.append(term)

        # Extract variance, std dev, correlations
        for i in range(q):
            var_i = cov_matrix[i, i]
            sd_i = np.sqrt(max(var_i, 0.0))

            # Correlation with first term (only for 2nd+ terms)
            if i > 0 and cov_matrix[0, 0] > 0 and var_i > 0:
                corr = cov_matrix[i, 0] / (np.sqrt(cov_matrix[0, 0]) * sd_i)
                corr = np.clip(corr, -1.0, 1.0)
            else:
                corr = None

            var_comps.append(VarCompSummary(
                group=spec.group_name,
                name=term_names[i],
                variance=float(var_i),
                std_dev=float(sd_i),
                corr=float(corr) if corr is not None else None,
            ))

    return var_comps


def _extract_blups(b: np.ndarray, specs: list) -> dict[str, np.ndarray]:
    """Extract BLUPs per grouping factor from the flat b vector.

    b is structured as [b_group1, b_group2, ...] where each b_groupk
    has J_k * q_k elements laid out as [term0_group0, term0_group1, ...,
    term1_group0, ...].

    Returns dict: group_name → (J_k, q_k) array.
    """
    result = {}
    offset = 0
    for spec in specs:
        block_size = spec.n_groups * spec.n_terms
        b_block = b[offset:offset + block_size]
        # Reshape: columns are terms, rows are groups
        # b_block layout: [term0_g0, term0_g1, ..., term1_g0, term1_g1, ...]
        b_matrix = np.zeros((spec.n_groups, spec.n_terms), dtype=np.float64)
        for t in range(spec.n_terms):
            start = t * spec.n_groups
            b_matrix[:, t] = b_block[start:start + spec.n_groups]
        result[spec.group_name] = b_matrix
        offset += block_size
    return result


def _compute_se(pls, X, Z, Lambda, p: int) -> np.ndarray:
    """Compute standard errors of fixed effects via V matrix.

    SE = sqrt(diag(Var(β̂))) where Var(β̂) = σ² × (X'V*⁻¹X)⁻¹
    and V* = ZΛΛ'Z' + I.

    This matches R's lme4 computation exactly, avoiding numerical
    differences from the Schur complement approach.
    """
    n = X.shape[0]
    V_star = Z @ Lambda @ Lambda.T @ Z.T + np.eye(n)
    try:
        C = np.linalg.inv(X.T @ np.linalg.solve(V_star, X))
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(X.T @ np.linalg.solve(V_star, X))

    vcov = pls.sigma_sq * C
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return se


def _compute_se_glmm(pls, p: int) -> np.ndarray:
    """Compute standard errors for GLMM (σ² = 1 by convention)."""
    try:
        RX_inv = np.linalg.inv(pls.RX)
        vcov = RX_inv @ RX_inv.T
    except np.linalg.LinAlgError:
        RtR = pls.RX @ pls.RX.T
        vcov = np.linalg.pinv(RtR)

    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return se


def _compute_fit_stats(pls, theta, n, p, specs, reml):
    """Compute log-likelihood, AIC, BIC for LMM."""
    sigma_sq = pls.sigma_sq
    pwrss = pls.pwrss

    # Number of variance parameters
    n_theta = len(theta)
    # Total parameters for AIC: fixed effects + variance components + σ²
    n_params = p + n_theta + 1

    if reml:
        df = n - p
        # REML log-likelihood
        log_det_L = 2.0 * np.sum(np.log(np.maximum(np.diag(pls.L), 1e-20)))
        log_det_RX = 2.0 * np.sum(np.log(np.maximum(np.abs(np.diag(pls.RX)), 1e-20)))

        ll = -0.5 * (
            log_det_L
            + log_det_RX
            + df * (1.0 + np.log(2.0 * np.pi * pwrss / df))
        )

        # REML AIC/BIC: R's lme4 counts all parameters (fixed + variance)
        # npar = p (fixed effects) + n_theta (RE params) + 1 (sigma)
        aic = -2.0 * ll + 2.0 * n_params
        bic = -2.0 * ll + np.log(n) * n_params
    else:
        # ML log-likelihood
        log_det_L = 2.0 * np.sum(np.log(np.maximum(np.diag(pls.L), 1e-20)))

        ll = -0.5 * (
            log_det_L
            + n * (1.0 + np.log(2.0 * np.pi * pwrss / n))
        )

        aic = -2.0 * ll + 2.0 * n_params
        bic = -2.0 * ll + np.log(n) * n_params

    return float(ll), float(aic), float(bic)


def _make_coef_names(p: int) -> list[str]:
    """Generate default coefficient names."""
    if p == 1:
        return ['(Intercept)']
    names = ['(Intercept)']
    for i in range(1, p):
        names.append(f'X{i}')
    return names
