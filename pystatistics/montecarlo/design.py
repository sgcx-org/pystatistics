"""
Design classes for Monte Carlo methods.

BootstrapDesign and PermutationDesign encapsulate all inputs needed
by backends to perform resampling. Immutable, validated at construction.
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError

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
        n_resamples: Number of bootstrap replicates.
        method: Simulation type — "ordinary", "balanced", or "parametric".
        statistic_type: What the second argument to statistic represents:
            "i" (indices), "f" (frequencies), "w" (weights).
        strata: Optional stratification vector of length n.
        ran_gen: For parametric bootstrap: fn(data, mle) -> simulated data.
        mle: Parameter estimates passed to ran_gen for parametric bootstrap.
        seed: Random seed for reproducibility.
        gpu_statistic: Explicit declaration that ``statistic`` is a GPU-supported
            closed form. Only ``"mean"`` is currently supported. ``None`` (the
            default) means the statistic is an arbitrary Python callable that can
            only run on CPU. The GPU backend never *infers* the statistic form —
            the caller must declare it (fail-loud opt-in).
    """
    data: NDArray[np.floating[Any]]
    statistic: Callable
    n_resamples: int
    method: str
    statistic_type: str
    strata: NDArray | None
    ran_gen: Callable | None
    mle: Any
    seed: int | None
    gpu_statistic: str | None = None

    @classmethod
    def for_bootstrap(
        cls,
        data,
        statistic: Callable,
        n_resamples: int = 999,
        *,
        method: str = "ordinary",
        statistic_type: str = "i",
        strata=None,
        ran_gen: Callable | None = None,
        mle=None,
        seed: int | None = None,
        gpu_statistic: str | None = None,
    ) -> BootstrapDesign:
        """
        Create a bootstrap design with validation.

        Args:
            data: Input data — 1D or 2D array-like.
            statistic: Function to compute the statistic of interest.
            n_resamples: Number of bootstrap replicates. Must be >= 1.
            method: "ordinary" (default), "balanced", or "parametric".
            statistic_type: "i" (indices), "f" (frequencies), "w" (weights).
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
            raise ValidationError(
                f"data must be 1D or 2D, got {data_arr.ndim}D"
            )

        n = data_arr.shape[0]
        if n < 1:
            raise ValidationError("data must have at least 1 observation")

        if n_resamples < 1:
            raise ValidationError(f"n_resamples must be >= 1, got {n_resamples}")

        if method not in ("ordinary", "balanced", "parametric"):
            raise ValidationError(
                f"method must be 'ordinary', 'balanced', or 'parametric', "
                f"got {method!r}"
            )

        if statistic_type not in ("i", "f", "w"):
            raise ValidationError(
                f"statistic_type must be 'i', 'f', or 'w', got {statistic_type!r}"
            )

        if gpu_statistic is not None and gpu_statistic != "mean":
            raise ValidationError(
                f"gpu_statistic must be 'mean' or None, got {gpu_statistic!r}"
            )

        if method == "parametric" and ran_gen is None:
            raise ValidationError(
                "ran_gen is required for parametric bootstrap "
                "(method='parametric')"
            )

        strata_arr = None
        if strata is not None:
            strata_arr = np.asarray(strata)
            if strata_arr.shape[0] != n:
                raise ValidationError(
                    f"strata length ({strata_arr.shape[0]}) must match "
                    f"data rows ({n})"
                )

        return cls(
            data=data_arr,
            statistic=statistic,
            n_resamples=n_resamples,
            method=method,
            statistic_type=statistic_type,
            strata=strata_arr,
            ran_gen=ran_gen,
            mle=mle,
            seed=seed,
            gpu_statistic=gpu_statistic,
        )


@dataclass(frozen=True)
class PermutationDesign:
    """
    Frozen design for permutation testing.

    Attributes:
        x: Group 1 data, shape (n1,) or (n1, p).
        y: Group 2 data, shape (n2,) or (n2, p).
        statistic: fn(x, y) -> float.
        n_resamples: Number of permutations.
        alternative: "two-sided", "less", or "greater".
        seed: Random seed for reproducibility.
        gpu_statistic: Explicit declaration that ``statistic`` is a GPU-supported
            closed form. Only ``"mean_diff"`` (mean(x) - mean(y)) is currently
            supported. ``None`` means an arbitrary CPU-only Python callable. The
            GPU backend never infers the statistic form — it must be declared.
    """
    x: NDArray[np.floating[Any]]
    y: NDArray[np.floating[Any]]
    statistic: Callable
    n_resamples: int
    alternative: str
    seed: int | None
    gpu_statistic: str | None = None

    @classmethod
    def for_permutation_test(
        cls,
        x,
        y,
        statistic: Callable,
        n_resamples: int = 9999,
        *,
        alternative: str = "two-sided",
        seed: int | None = None,
        gpu_statistic: str | None = None,
    ) -> PermutationDesign:
        """
        Create a permutation test design with validation.

        Args:
            x: Group 1 data.
            y: Group 2 data.
            statistic: fn(x, y) -> float. The test statistic.
            n_resamples: Number of permutations. Must be >= 1.
            alternative: "two-sided", "less", or "greater".
            seed: Random seed.

        Returns:
            Validated PermutationDesign.
        """
        x_arr = np.asarray(x, dtype=np.float64).copy()
        y_arr = np.asarray(y, dtype=np.float64).copy()

        if x_arr.ndim == 0 or y_arr.ndim == 0:
            raise ValidationError("x and y must be arrays, not scalars")

        if len(x_arr) < 1 or len(y_arr) < 1:
            raise ValidationError("x and y must each have at least 1 observation")

        if n_resamples < 1:
            raise ValidationError(f"n_resamples must be >= 1, got {n_resamples}")

        if alternative not in ("two-sided", "less", "greater"):
            raise ValidationError(
                f"alternative must be 'two-sided', 'less', or 'greater', "
                f"got {alternative!r}"
            )

        if gpu_statistic is not None and gpu_statistic != "mean_diff":
            raise ValidationError(
                f"gpu_statistic must be 'mean_diff' or None, got {gpu_statistic!r}"
            )

        return cls(
            x=x_arr,
            y=y_arr,
            statistic=statistic,
            n_resamples=n_resamples,
            alternative=alternative,
            seed=seed,
            gpu_statistic=gpu_statistic,
        )
