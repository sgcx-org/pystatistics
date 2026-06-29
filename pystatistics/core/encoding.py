"""
Categorical encoding and design-matrix construction (shared).

Turns categorical factors into numeric design-matrix columns. This is the
single, R-validated encoder used by every domain that needs to convert group
labels into indicator columns — ANOVA (Type I/II/III SS) and regression
(categorical predictors + interactions) both consume it.

PyStatistics deliberately avoids R's "contrasts" vocabulary; the operation
here is *encoding* categorical variables, in the sklearn/pandas sense.

Key concepts:
    - Dummy encoding: k-1 indicator columns (baseline = first sorted level)
    - Deviation (sum-to-zero) encoding: k-1 columns summing to zero
    - Interaction: element-wise product of all column pairs
    - DesignMatrix: a full numeric design matrix with column/term metadata
"""

from dataclasses import dataclass
from pystatistics.core.exceptions import ValidationError
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DesignMatrix:
    """
    Encoded design matrix with metadata for ANOVA SS computation.

    Attributes:
        X: (n, p) float64 design matrix (including intercept if requested)
        term_slices: term name -> column slice in X
        term_names: ordered list of term names
        term_df: term -> number of columns (degrees of freedom for that term)
        n: number of observations
        p: total number of columns
        coding: 'treatment' (dummy) or 'deviation' (sum-to-zero)
        factor_levels: factor name -> sorted unique level strings
        has_intercept: whether column 0 is an intercept
    """
    X: NDArray[np.floating[Any]]
    term_slices: dict[str, slice]
    term_names: list[str]
    term_df: dict[str, int]
    n: int
    p: int
    coding: str
    factor_levels: dict[str, list[str]]
    has_intercept: bool


def encode_dummy(
    factor: NDArray,
    *,
    reference: str | None = None,
) -> tuple[NDArray, list[str], str]:
    """
    Dummy encoding for a single categorical factor.

    Drops the reference level (baseline) and creates k-1 indicator columns.
    (Known in R as treatment contrasts.)

    Args:
        factor: 1D array of group labels (strings or integers)
        reference: level to use as baseline. If None, the first sorted level
            is used (matches R's default factor ordering).

    Returns:
        (X_coded, level_names, baseline) where:
            X_coded: (n, k-1) float64 indicator matrix
            level_names: the k-1 non-baseline level names (column labels)
            baseline: the dropped baseline level name

    Raises:
        ValueError: if reference is given but not among the factor's levels.
    """
    levels = sorted(set(str(v) for v in factor))
    if reference is None:
        baseline = levels[0]
    else:
        baseline = str(reference)
        if baseline not in levels:
            raise ValidationError(
                f"reference level {reference!r} not found in factor levels "
                f"{levels}"
            )
    contrasts = [lvl for lvl in levels if lvl != baseline]
    factor_str = np.array([str(v) for v in factor])

    n = len(factor)
    k_minus_1 = len(contrasts)
    X = np.zeros((n, k_minus_1), dtype=np.float64)

    for j, level in enumerate(contrasts):
        X[:, j] = (factor_str == level).astype(np.float64)

    return X, contrasts, baseline


def encode_deviation(
    factor: NDArray,
) -> tuple[NDArray, list[str]]:
    """
    Deviation (sum-to-zero) encoding for a single factor.

    Each column sums to zero across levels. The last level gets -1 in all
    columns. This makes Type III SS test marginal means that are unweighted
    by cell frequencies.

    Args:
        factor: 1D array of group labels

    Returns:
        (X_coded, level_names) where:
            X_coded: (n, k-1) float64 deviation-coded matrix
            level_names: the k-1 level names (last level is the reference)
    """
    levels = sorted(set(str(v) for v in factor))
    k = len(levels)
    reference = levels[-1]
    coded_levels = levels[:-1]
    factor_str = np.array([str(v) for v in factor])

    n = len(factor)
    X = np.zeros((n, k - 1), dtype=np.float64)

    for j, level in enumerate(coded_levels):
        X[factor_str == level, j] = 1.0
        X[factor_str == reference, j] = -1.0

    return X, coded_levels


def interaction_columns(
    X_a: NDArray, X_b: NDArray,
) -> NDArray:
    """
    Compute interaction columns as the element-wise product of all
    column pairs from X_a and X_b.

    Works for any combination of encoded-factor columns and single-column
    numeric predictors: a numeric predictor is simply an (n, 1) block, so
    numeric x numeric, numeric x categorical, and categorical x categorical
    all fall out of the same pairwise product.

    Args:
        X_a: (n, p_a) columns for term A
        X_b: (n, p_b) columns for term B

    Returns:
        (n, p_a * p_b) interaction columns
    """
    n = X_a.shape[0]
    p_a = X_a.shape[1]
    p_b = X_b.shape[1]
    X_int = np.empty((n, p_a * p_b), dtype=np.float64)

    col = 0
    for i in range(p_a):
        for j in range(p_b):
            X_int[:, col] = X_a[:, i] * X_b[:, j]
            col += 1

    return X_int


def build_design_matrix(
    factors: dict[str, NDArray],
    *,
    covariates: dict[str, NDArray] | None = None,
    coding: str = 'dummy',
    include_intercept: bool = True,
    interactions: list[tuple[str, str]] | None = None,
) -> DesignMatrix:
    """
    Build a full design matrix from categorical factors and optional covariates.

    Args:
        factors: {name: 1D array of group labels}
        covariates: {name: 1D numeric array} or None
        coding: 'treatment' (dummy encoding) or 'deviation' (sum-to-zero).
            These are the ANOVA SS-coding selectors; Type I/II use 'treatment',
            Type III uses 'deviation'.
        include_intercept: whether to prepend an intercept column
        interactions: list of (factorA, factorB) pairs for interaction terms,
                     or None for all pairwise interactions when len(factors) > 1

    Returns:
        DesignMatrix with full design matrix and metadata
    """
    if coding not in ('treatment', 'deviation'):
        raise ValidationError(f"coding must be 'treatment' or 'deviation', got {coding!r}")

    n = len(next(iter(factors.values())))
    columns = []
    term_slices: dict[str, slice] = {}
    term_names: list[str] = []
    term_df: dict[str, int] = {}
    factor_levels: dict[str, list[str]] = {}
    col_start = 0

    # Intercept
    if include_intercept:
        columns.append(np.ones((n, 1), dtype=np.float64))
        term_slices['Intercept'] = slice(col_start, col_start + 1)
        term_names.append('Intercept')
        term_df['Intercept'] = 1
        col_start += 1

    # Encode each factor
    factor_columns: dict[str, NDArray] = {}

    for name in sorted(factors.keys()):
        factor = factors[name]
        levels = sorted(set(str(v) for v in factor))
        factor_levels[name] = levels

        if coding == 'treatment':
            X_coded, level_names, baseline = encode_dummy(factor)
        else:
            X_coded, level_names = encode_deviation(factor)

        factor_columns[name] = X_coded
        ncols = X_coded.shape[1]
        columns.append(X_coded)
        term_slices[name] = slice(col_start, col_start + ncols)
        term_names.append(name)
        term_df[name] = ncols
        col_start += ncols

    # Covariates (continuous predictors for ANCOVA)
    if covariates is not None:
        for name in sorted(covariates.keys()):
            cov = covariates[name].reshape(-1, 1).astype(np.float64)
            columns.append(cov)
            term_slices[name] = slice(col_start, col_start + 1)
            term_names.append(name)
            term_df[name] = 1
            col_start += 1

    # Interactions
    if interactions is None and len(factors) > 1:
        # Default: all pairwise interactions between factors
        factor_names = sorted(factors.keys())
        interactions = []
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                interactions.append((factor_names[i], factor_names[j]))

    if interactions is not None:
        for name_a, name_b in interactions:
            X_int = interaction_columns(factor_columns[name_a], factor_columns[name_b])
            int_name = f"{name_a}:{name_b}"
            ncols = X_int.shape[1]
            columns.append(X_int)
            term_slices[int_name] = slice(col_start, col_start + ncols)
            term_names.append(int_name)
            term_df[int_name] = ncols
            col_start += ncols

    X = np.hstack(columns) if columns else np.empty((n, 0), dtype=np.float64)

    return DesignMatrix(
        X=X,
        term_slices=term_slices,
        term_names=term_names,
        term_df=term_df,
        n=n,
        p=col_start,
        coding=coding,
        factor_levels=factor_levels,
        has_intercept=include_intercept,
    )
