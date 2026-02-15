"""
ANOVA design object.

Wraps validated data and metadata for ANOVA computation.
Factory methods handle different ANOVA types (one-way, factorial, RM).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.validation import (
    check_array,
    check_finite,
    check_1d,
    check_consistent_length,
    check_min_samples,
)
from pystatistics.core.exceptions import ValidationError


@dataclass(frozen=True)
class AnovaDesign:
    """
    Validated data container for ANOVA.

    Created via factory methods, not directly.
    """
    y: NDArray[np.floating[Any]]
    factors: dict[str, NDArray]
    covariates: dict[str, NDArray] | None
    subject: NDArray | None
    n: int
    design_type: str   # 'oneway', 'factorial', 'rm'

    @staticmethod
    def for_oneway(
        y: Any,
        group: Any,
    ) -> 'AnovaDesign':
        """
        Create design for one-way ANOVA.

        Args:
            y: Response variable (1D numeric)
            group: Group labels (1D, same length as y)

        Returns:
            AnovaDesign for one-way ANOVA
        """
        y_arr = check_array(y, "y")
        check_finite(y_arr, "y")
        check_1d(y_arr, "y")
        check_min_samples(y_arr, 2, "y")

        group_arr = np.asarray(group)
        if group_arr.ndim != 1:
            raise ValidationError(f"group: expected 1D, got {group_arr.ndim}D")

        check_consistent_length(y_arr, group_arr, names=("y", "group"))

        group_str = np.array([str(v) for v in group_arr])
        levels = sorted(set(group_str))
        if len(levels) < 2:
            raise ValidationError(
                f"group: need at least 2 groups, got {len(levels)}"
            )

        # Verify each group has at least 1 observation
        for level in levels:
            n_level = np.sum(group_str == level)
            if n_level < 1:
                raise ValidationError(
                    f"group: level {level!r} has 0 observations"
                )

        return AnovaDesign(
            y=y_arr,
            factors={'group': group_str},
            covariates=None,
            subject=None,
            n=len(y_arr),
            design_type='oneway',
        )

    @staticmethod
    def for_factorial(
        y: Any,
        factors: dict[str, Any],
        *,
        covariates: dict[str, Any] | None = None,
    ) -> 'AnovaDesign':
        """
        Create design for factorial ANOVA or ANCOVA.

        Args:
            y: Response variable (1D numeric)
            factors: {name: 1D array of group labels}
            covariates: {name: 1D numeric array} or None for ANCOVA

        Returns:
            AnovaDesign for factorial ANOVA
        """
        y_arr = check_array(y, "y")
        check_finite(y_arr, "y")
        check_1d(y_arr, "y")
        check_min_samples(y_arr, 2, "y")

        n = len(y_arr)
        validated_factors: dict[str, NDArray] = {}

        for name, fac in factors.items():
            fac_arr = np.asarray(fac)
            if fac_arr.ndim != 1:
                raise ValidationError(f"{name}: expected 1D, got {fac_arr.ndim}D")
            if len(fac_arr) != n:
                raise ValidationError(
                    f"{name}: length {len(fac_arr)} doesn't match y length {n}"
                )
            fac_str = np.array([str(v) for v in fac_arr])
            levels = sorted(set(fac_str))
            if len(levels) < 2:
                raise ValidationError(
                    f"{name}: need at least 2 levels, got {len(levels)}"
                )
            validated_factors[name] = fac_str

        validated_covariates: dict[str, NDArray] | None = None
        if covariates is not None:
            validated_covariates = {}
            for name, cov in covariates.items():
                cov_arr = check_array(cov, name)
                check_finite(cov_arr, name)
                check_1d(cov_arr, name)
                if len(cov_arr) != n:
                    raise ValidationError(
                        f"{name}: length {len(cov_arr)} doesn't match y length {n}"
                    )
                validated_covariates[name] = cov_arr

        return AnovaDesign(
            y=y_arr,
            factors=validated_factors,
            covariates=validated_covariates,
            subject=None,
            n=n,
            design_type='factorial',
        )

    @staticmethod
    def for_repeated_measures(
        y: Any,
        subject: Any,
        within: dict[str, Any],
        *,
        between: dict[str, Any] | None = None,
    ) -> 'AnovaDesign':
        """
        Create design for repeated-measures ANOVA.

        Args:
            y: Response variable (1D, long format)
            subject: Subject identifiers (1D)
            within: {factor_name: 1D condition labels}
            between: {factor_name: 1D group labels} or None

        Returns:
            AnovaDesign for RM ANOVA
        """
        y_arr = check_array(y, "y")
        check_finite(y_arr, "y")
        check_1d(y_arr, "y")

        n = len(y_arr)
        subject_arr = np.asarray(subject)
        if subject_arr.ndim != 1 or len(subject_arr) != n:
            raise ValidationError(
                f"subject: expected 1D with length {n}"
            )

        validated_within: dict[str, NDArray] = {}
        for name, fac in within.items():
            fac_arr = np.asarray(fac)
            if fac_arr.ndim != 1 or len(fac_arr) != n:
                raise ValidationError(
                    f"{name}: expected 1D with length {n}"
                )
            validated_within[name] = np.array([str(v) for v in fac_arr])

        # Combine within factors into factors dict
        all_factors = dict(validated_within)

        validated_between: dict[str, NDArray] | None = None
        if between is not None:
            validated_between = {}
            for name, fac in between.items():
                fac_arr = np.asarray(fac)
                if fac_arr.ndim != 1 or len(fac_arr) != n:
                    raise ValidationError(
                        f"{name}: expected 1D with length {n}"
                    )
                validated_between[name] = np.array([str(v) for v in fac_arr])
                all_factors[name] = validated_between[name]

        return AnovaDesign(
            y=y_arr,
            factors=all_factors,
            covariates=None,
            subject=np.array([str(v) for v in subject_arr]),
            n=n,
            design_type='rm',
        )
