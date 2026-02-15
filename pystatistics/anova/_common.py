"""
Common data types for ANOVA.

Contains the frozen parameter payloads that go inside Result[P] envelopes.
Each payload is a pure data container — no methods, no computation.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AnovaTableRow:
    """One row of an ANOVA table (one term or residuals)."""
    term: str
    df: int
    sum_sq: float
    mean_sq: float
    f_value: float | None    # None for Residuals row
    p_value: float | None    # None for Residuals row


@dataclass(frozen=True)
class AnovaParams:
    """
    Parameter payload for between-subjects ANOVA.

    Used by anova_oneway() and anova() (factorial / ANCOVA).
    """
    table: tuple[AnovaTableRow, ...]
    ss_type: int                                   # 1, 2, or 3
    n_obs: int
    n_groups: dict[str, int]                       # factor -> number of levels
    grand_mean: float
    group_means: dict[str, dict[str, float]]       # factor -> {level: mean}
    residual_df: int
    residual_ss: float
    residual_ms: float
    eta_squared: dict[str, float]                  # term -> eta^2
    partial_eta_squared: dict[str, float]           # term -> partial eta^2


@dataclass(frozen=True)
class LeveneParams:
    """Parameter payload for Levene / Brown-Forsythe test."""
    f_value: float
    p_value: float
    df_between: int
    df_within: int
    center: str           # 'mean' or 'median'
    group_vars: dict[str, float]   # group -> variance


@dataclass(frozen=True)
class PostHocComparison:
    """One row of a post-hoc comparison table."""
    group1: str
    group2: str
    diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    se: float


@dataclass(frozen=True)
class PostHocParams:
    """Parameter payload for post-hoc tests."""
    method: str                               # 'tukey', 'bonferroni', 'dunnett'
    comparisons: tuple[PostHocComparison, ...]
    conf_level: float
    factor: str
    mse: float
    df_error: int


@dataclass(frozen=True)
class AnovaRMTableRow:
    """One row of a repeated-measures ANOVA table."""
    term: str
    df: float              # float because GG/HF corrections produce fractional df
    sum_sq: float
    mean_sq: float
    f_value: float | None
    p_value: float | None
    gg_p_value: float | None   # Greenhouse-Geisser corrected
    hf_p_value: float | None   # Huynh-Feldt corrected


@dataclass(frozen=True)
class SphericitySummary:
    """Results of Mauchly's sphericity test for one within-subjects factor."""
    factor: str
    mauchly_w: float
    p_value: float
    gg_epsilon: float
    hf_epsilon: float


@dataclass(frozen=True)
class AnovaRMParams:
    """
    Parameter payload for repeated-measures ANOVA.

    Includes sphericity test results and corrected p-values.
    """
    table: tuple[AnovaRMTableRow, ...]
    n_subjects: int
    n_obs: int
    within_factors: tuple[str, ...]
    between_factors: tuple[str, ...]
    sphericity: tuple[SphericitySummary, ...]
    correction: str                    # 'none', 'gg', 'hf', 'auto'
    grand_mean: float
    eta_squared: dict[str, float]
    partial_eta_squared: dict[str, float]
