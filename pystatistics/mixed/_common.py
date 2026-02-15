"""
Common data types for mixed models (LMM / GLMM).

Contains the frozen parameter payloads that go inside Result[P] envelopes.
Each payload is a pure data container — no methods, no computation.

References:
    Bates, D., Maechler, M., Bolker, B., & Walker, S. (2015).
    Fitting Linear Mixed-Effects Models Using lme4.
    Journal of Statistical Software, 67(1), 1-48.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VarCompSummary:
    """Variance component summary for one random effect term.

    Attributes:
        group: Grouping factor name (e.g. 'subject').
        name: Term name within the group (e.g. '(Intercept)', 'time').
        variance: Estimated variance σ²_b for this component.
        std_dev: Standard deviation (sqrt of variance).
        corr: Correlation with previous term in the same group,
              or None if this is the first (or only) term.
    """
    group: str
    name: str
    variance: float
    std_dev: float
    corr: float | None = None


@dataclass(frozen=True)
class LMMParams:
    """
    Parameter payload for a fitted linear mixed model.

    Contains all estimates needed to reconstruct the model summary,
    perform inference, and extract random effects.
    """
    # Fixed effects
    coefficients: NDArray              # β̂ (p,)
    coefficient_names: tuple[str, ...]
    se: NDArray                        # standard errors of β̂ (p,)
    df_satterthwaite: NDArray          # Satterthwaite df per fixed effect (p,)
    t_values: NDArray                  # β̂ / se (p,)
    p_values: NDArray                  # from t-distribution with Satt. df (p,)

    # Random effects
    var_components: tuple[VarCompSummary, ...]
    residual_variance: float           # σ²
    residual_std: float                # σ

    # Model fit
    log_likelihood: float
    reml: bool
    aic: float
    bic: float
    n_obs: int
    n_groups: dict[str, int]           # grouping_factor → number of unique levels

    # Convergence
    converged: bool
    n_iter: int

    # Random effects conditional modes (BLUPs)
    random_effects: dict[str, NDArray]  # group_name → (n_groups_j, n_re_terms_j)

    # Predictions
    fitted_values: NDArray             # Xβ̂ + Zb̂ (n,)
    residuals: NDArray                 # y - fitted (n,)

    # Internal
    theta: NDArray                     # converged θ parameters


@dataclass(frozen=True)
class GLMMParams:
    """
    Parameter payload for a fitted generalized linear mixed model.

    Same structure as LMMParams with additional family/link info
    and deviance instead of residual variance.
    """
    # Fixed effects
    coefficients: NDArray
    coefficient_names: tuple[str, ...]
    se: NDArray
    t_values: NDArray                  # β̂ / se (Wald z-statistics for GLMM)
    p_values: NDArray                  # from normal distribution (GLMM uses z, not t)

    # Random effects
    var_components: tuple[VarCompSummary, ...]

    # Model fit
    log_likelihood: float
    deviance: float
    aic: float
    bic: float
    n_obs: int
    n_groups: dict[str, int]

    # Family
    family_name: str
    link_name: str

    # Convergence
    converged: bool
    n_iter: int

    # Random effects conditional modes
    random_effects: dict[str, NDArray]

    # Predictions (on link scale and response scale)
    fitted_values: NDArray             # μ̂ = g⁻¹(Xβ̂ + Zb̂) (n,)
    linear_predictor: NDArray          # η̂ = Xβ̂ + Zb̂ (n,)
    residuals: NDArray                 # y - μ̂ (n,)

    # Internal
    theta: NDArray
