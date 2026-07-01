"""
Profiled deviance computation for LMM and GLMM.

The profiled deviance is the objective function that the outer optimizer
minimizes over θ. For LMM, β and σ² are analytically profiled out,
leaving a function of θ only. For GLMM, the Laplace approximation
replaces the marginal likelihood integral.

References:
    Bates, D., Maechler, M., Bolker, B., & Walker, S. (2015).
    Fitting Linear Mixed-Effects Models Using lme4.
    Journal of Statistical Software, 67(1), Sections 2-3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.mixed._random_effects import RandomEffectSpec, build_lambda
from pystatistics.mixed._pls import solve_pls


def profiled_deviance_lmm(
    theta: NDArray,
    X: NDArray,
    Z: NDArray,
    y: NDArray,
    specs: list[RandomEffectSpec],
    reml: bool = True,
) -> float:
    """Compute the profiled REML (or ML) deviance for given θ.

    For LMM, the profiled deviance (up to a constant) is:

    ML:   d(θ) = log|L_θ|² + n × [1 + log(2π × pwrss/n)]

    REML: d(θ) = log|L_θ|² + log|RX|² + (n-p) × [1 + log(2π × pwrss/(n-p))]

    where:
        L_θ = cholesky(Λ'Z'ZΛ + I)        from PLS
        pwrss = penalized weighted RSS      from PLS
        RX = R factor of projected X        from PLS (Schur complement Cholesky)

    Args:
        theta: Parameter vector for Λ_θ.
        X: Fixed effects design matrix (n, p).
        Z: Random effects design matrix (n, q).
        y: Response vector (n,).
        specs: Random effect specifications.
        reml: If True, compute REML deviance; if False, ML deviance.

    Returns:
        Profiled deviance value (scalar to minimize).
    """
    n, p = X.shape

    # Build Λ from θ
    Lambda = build_lambda(theta, specs)

    # Solve PLS
    pls = solve_pls(X, Z, y, Lambda, reml=reml)

    # log|L|² = 2 × sum(log(diag(L)))
    # NUMERICAL GUARD: prevents log(0) in log-likelihood computation
    log_det_L = 2.0 * np.sum(np.log(np.maximum(np.diag(pls.L), 1e-20)))

    if reml:
        # log|RX|² = 2 × sum(log(diag(RX)))
        # NUMERICAL GUARD: prevents log(0) in log-likelihood computation
        log_det_RX = 2.0 * np.sum(np.log(np.maximum(np.abs(np.diag(pls.RX)), 1e-20)))
        df = n - p
        dev = (log_det_L
               + log_det_RX
               + df * (1.0 + np.log(2.0 * np.pi * pls.pwrss / df)))
    else:
        dev = (log_det_L
               + n * (1.0 + np.log(2.0 * np.pi * pls.pwrss / n)))

    return float(dev)
