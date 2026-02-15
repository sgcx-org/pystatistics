"""
Design classes for Monte Carlo methods.

BootstrapDesign and PermutationDesign encapsulate all inputs needed
by backends to perform resampling. Immutable, validated at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BootstrapDesign:
    """
    Frozen design for bootstrap resampling.

    Attributes:
        data: Original data array, shape (n,) or (n, p).
        statistic: User function. For nonparametric: fn(data, indices) -> (k,).
            For parametric: fn(simulated_data) -> (k,).
        R: Number of bootstrap replicates.
        sim: Simulation type — "ordinary", "balanced", or "parametric".
        stype: What the second argument to statistic represents:
            "i" (indices), "f" (frequencies), "w" (weights).
        strata: Optional stratification vector of length n.
        ran_gen: For parametric bootstrap: fn(data, mle) -> simulated data.
        mle: Parameter estimates passed to ran_gen for parametric bootstrap.
        seed: Random seed for reproducibility.
    """
    data: NDArray[np.floating[Any]]
    statistic: Callable
    R: int
    sim: str
    stype: str
    strata: NDArray | None
    ran_gen: Callable | None
    mle: Any
    seed: int | None

    @classmethod
    def for_bootstrap(
        cls,
        data,
        statistic: Callable,
        R: int = 999,
        *,
        sim: str = "ordinary",
        stype: str = "i",
        strata=None,
        ran_gen: Callable | None = None,
        mle=None,
        seed: int | None = None,
    ) -> BootstrapDesign:
        """
        Create a bootstrap design with validation.

        Args:
            data: Input data — 1D or 2D array-like.
            statistic: Function to compute the statistic of interest.
            R: Number of bootstrap replicates. Must be >= 1.
            sim: "ordinary" (default), "balanced", or "parametric".
            stype: "i" (indices), "f" (frequencies), "w" (weights).
            strata: Stratification vector (same length as data rows).
            ran_gen: Required for parametric bootstrap.
            mle: Parameter estimates for parametric bootstrap.
            seed: Random seed.

        Returns:
            Validated BootstrapDesign.

        Raises:
            ValueError: If inputs are invalid.
        """
        data_arr = np.asarray(data, dtype=np.float64)
        if data_arr.ndim == 1:
            data_arr = data_arr.copy()
        elif data_arr.ndim == 2:
            data_arr = data_arr.copy()
        else:
            raise ValueError(
                f"data must be 1D or 2D, got {data_arr.ndim}D"
            )

        n = data_arr.shape[0]
        if n < 1:
            raise ValueError("data must have at least 1 observation")

        if R < 1:
            raise ValueError(f"R must be >= 1, got {R}")

        if sim not in ("ordinary", "balanced", "parametric"):
            raise ValueError(
                f"sim must be 'ordinary', 'balanced', or 'parametric', "
                f"got {sim!r}"
            )

        if stype not in ("i", "f", "w"):
            raise ValueError(
                f"stype must be 'i', 'f', or 'w', got {stype!r}"
            )

        if sim == "parametric" and ran_gen is None:
            raise ValueError(
                "ran_gen is required for parametric bootstrap "
                "(sim='parametric')"
            )

        strata_arr = None
        if strata is not None:
            strata_arr = np.asarray(strata)
            if strata_arr.shape[0] != n:
                raise ValueError(
                    f"strata length ({strata_arr.shape[0]}) must match "
                    f"data rows ({n})"
                )

        return cls(
            data=data_arr,
            statistic=statistic,
            R=R,
            sim=sim,
            stype=stype,
            strata=strata_arr,
            ran_gen=ran_gen,
            mle=mle,
            seed=seed,
        )


@dataclass(frozen=True)
class PermutationDesign:
    """
    Frozen design for permutation testing.

    Attributes:
        x: Group 1 data, shape (n1,) or (n1, p).
        y: Group 2 data, shape (n2,) or (n2, p).
        statistic: fn(x, y) -> float.
        R: Number of permutations.
        alternative: "two.sided", "less", or "greater".
        seed: Random seed for reproducibility.
    """
    x: NDArray[np.floating[Any]]
    y: NDArray[np.floating[Any]]
    statistic: Callable
    R: int
    alternative: str
    seed: int | None

    @classmethod
    def for_permutation_test(
        cls,
        x,
        y,
        statistic: Callable,
        R: int = 9999,
        *,
        alternative: str = "two.sided",
        seed: int | None = None,
    ) -> PermutationDesign:
        """
        Create a permutation test design with validation.

        Args:
            x: Group 1 data.
            y: Group 2 data.
            statistic: fn(x, y) -> float. The test statistic.
            R: Number of permutations. Must be >= 1.
            alternative: "two.sided", "less", or "greater".
            seed: Random seed.

        Returns:
            Validated PermutationDesign.
        """
        x_arr = np.asarray(x, dtype=np.float64).copy()
        y_arr = np.asarray(y, dtype=np.float64).copy()

        if x_arr.ndim == 0 or y_arr.ndim == 0:
            raise ValueError("x and y must be arrays, not scalars")

        if len(x_arr) < 1 or len(y_arr) < 1:
            raise ValueError("x and y must each have at least 1 observation")

        if R < 1:
            raise ValueError(f"R must be >= 1, got {R}")

        if alternative not in ("two.sided", "less", "greater"):
            raise ValueError(
                f"alternative must be 'two.sided', 'less', or 'greater', "
                f"got {alternative!r}"
            )

        return cls(
            x=x_arr,
            y=y_arr,
            statistic=statistic,
            R=R,
            alternative=alternative,
            seed=seed,
        )
