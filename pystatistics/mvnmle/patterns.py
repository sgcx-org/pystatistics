"""
Pattern analysis utilities for MVN MLE.

Tools for analyzing missingness patterns in multivariate data.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PatternInfo:
    """Information about a single missingness pattern."""
    pattern_id: int
    observed_indices: np.ndarray
    missing_indices: np.ndarray
    n_cases: int
    data: np.ndarray
    pattern_vector: np.ndarray

    @property
    def n_observed(self) -> int:
        return len(self.observed_indices)

    @property
    def n_missing(self) -> int:
        return len(self.missing_indices)

    @property
    def percent_cases(self) -> float:
        return getattr(self, '_percent_cases', 0.0)

    @percent_cases.setter
    def percent_cases(self, value: float):
        self._percent_cases = value

    def __repr__(self) -> str:
        return (f"PatternInfo(id={self.pattern_id}, n_cases={self.n_cases}, "
                f"n_observed={self.n_observed}, n_missing={self.n_missing})")


@dataclass
class PatternSummary:
    """Summary statistics for all missingness patterns in a dataset."""
    n_patterns: int
    total_cases: int
    overall_missing_rate: float
    most_common_pattern: PatternInfo
    complete_cases: int
    complete_cases_percent: float
    variable_missing_rates: Dict[int, float]

    def __str__(self) -> str:
        lines = [
            f"Missingness Pattern Summary",
            f"=" * 40,
            f"Total patterns: {self.n_patterns}",
            f"Total cases: {self.total_cases}",
            f"Overall missing rate: {self.overall_missing_rate:.1%}",
            f"Complete cases: {self.complete_cases} ({self.complete_cases_percent:.1%})",
            f"Most common pattern: {self.most_common_pattern.n_cases} cases "
            f"({self.most_common_pattern.percent_cases:.1%})"
        ]
        return "\n".join(lines)


def identify_missingness_patterns(data: np.ndarray) -> List[PatternInfo]:
    """
    Identify and extract all unique missingness patterns in the data.

    Uses powers-of-2 pattern identification matching R's approach.

    Parameters
    ----------
    data : np.ndarray
        Data matrix with missing values as np.nan

    Returns
    -------
    List[PatternInfo]
        Patterns sorted by frequency (most common first).
    """
    n_obs, n_vars = data.shape

    # Create binary pattern matrix (1 = observed, 0 = missing)
    pattern_matrix = (~np.isnan(data)).astype(int)

    # Convert patterns to unique identifiers using powers of 2
    powers = 2 ** np.arange(n_vars - 1, -1, -1)
    pattern_ids = pattern_matrix @ powers

    # Find unique patterns
    unique_patterns, inverse_indices = np.unique(pattern_ids, return_inverse=True)

    # Build PatternInfo objects
    patterns = []
    for i, pattern_id in enumerate(unique_patterns):
        case_mask = (pattern_ids == pattern_id)
        pattern_data = data[case_mask]

        pattern_idx = np.where(pattern_ids == pattern_id)[0][0]
        pattern_vector = pattern_matrix[pattern_idx]

        observed_indices = np.where(pattern_vector == 1)[0]
        missing_indices = np.where(pattern_vector == 0)[0]

        pattern_data_observed = pattern_data[:, observed_indices]

        pattern_info = PatternInfo(
            pattern_id=i + 1,
            observed_indices=observed_indices,
            missing_indices=missing_indices,
            n_cases=int(np.sum(case_mask)),
            data=pattern_data_observed,
            pattern_vector=pattern_vector
        )

        patterns.append(pattern_info)

    # Sort by frequency (most common first)
    patterns.sort(key=lambda p: p.n_cases, reverse=True)

    # Update pattern IDs
    for i, pattern in enumerate(patterns):
        pattern.pattern_id = i + 1

    # Calculate percentages
    total_cases = n_obs
    for pattern in patterns:
        pattern.percent_cases = (pattern.n_cases / total_cases) * 100

    return patterns


def analyze_patterns(data) -> List[PatternInfo]:
    """
    Analyze missingness patterns in the data.

    Parameters
    ----------
    data : array-like
        Data matrix with missing values as np.nan.
        Can be NumPy array or pandas DataFrame.

    Returns
    -------
    List[PatternInfo]
        Patterns sorted by frequency (most common first).
    """
    if hasattr(data, 'values'):
        data_array = np.asarray(data.values, dtype=float)
    else:
        data_array = np.asarray(data, dtype=float)

    if data_array.ndim != 2:
        raise ValueError("Data must be 2-dimensional")

    if data_array.shape[0] < 1:
        raise ValueError("Data must have at least one observation")

    if data_array.shape[1] < 1:
        raise ValueError("Data must have at least one variable")

    return identify_missingness_patterns(data_array)


def pattern_summary(patterns: List[PatternInfo],
                   data_shape: Optional[Tuple[int, int]] = None) -> PatternSummary:
    """
    Generate summary statistics for missingness patterns.

    Parameters
    ----------
    patterns : List[PatternInfo]
        Pattern information from analyze_patterns()
    data_shape : Optional[Tuple[int, int]]
        Original data shape (n_obs, n_vars)

    Returns
    -------
    PatternSummary
    """
    if not patterns:
        raise ValueError("No patterns provided")

    n_patterns = len(patterns)
    total_cases = sum(p.n_cases for p in patterns)

    # Find complete cases
    complete_pattern = None
    for pattern in patterns:
        if len(pattern.missing_indices) == 0:
            complete_pattern = pattern
            break

    complete_cases = complete_pattern.n_cases if complete_pattern else 0
    complete_cases_percent = (complete_cases / total_cases * 100) if total_cases > 0 else 0

    most_common = max(patterns, key=lambda p: p.n_cases)

    # Overall missing rate
    if data_shape:
        n_obs, n_vars = data_shape
        total_possible_values = n_obs * n_vars
        total_observed_values = sum(p.n_cases * p.n_observed for p in patterns)
        overall_missing_rate = 1 - (total_observed_values / total_possible_values)

        variable_missing_rates = {}
        for var_idx in range(n_vars):
            var_observed_cases = sum(p.n_cases for p in patterns
                                   if var_idx in p.observed_indices)
            variable_missing_rates[var_idx] = 1 - (var_observed_cases / total_cases)
    else:
        overall_missing_rate = np.nan
        variable_missing_rates = {}

    return PatternSummary(
        n_patterns=n_patterns,
        total_cases=total_cases,
        overall_missing_rate=overall_missing_rate,
        most_common_pattern=most_common,
        complete_cases=complete_cases,
        complete_cases_percent=complete_cases_percent,
        variable_missing_rates=variable_missing_rates
    )
