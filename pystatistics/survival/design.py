"""
SurvivalDesign: immutable container for time-to-event data.

Wraps time, event indicator, optional covariates, and optional strata.
Validates inputs at construction time — all downstream code trusts clean data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SurvivalDesign:
    """Immutable survival data container.

    Parameters
    ----------
    time : NDArray
        Time to event or censoring. Must be non-negative.
    event : NDArray
        Event indicator: 1 = event observed, 0 = censored.
    X : NDArray or None
        Covariate matrix (n, p). None for KM / log-rank.
    strata : NDArray or None
        Strata labels for stratified analyses.
    """

    time: NDArray
    event: NDArray
    X: NDArray | None
    strata: NDArray | None

    @classmethod
    def for_survival(
        cls,
        time,
        event,
        X=None,
        *,
        strata=None,
    ) -> SurvivalDesign:
        """Create and validate survival data.

        Parameters
        ----------
        time : array-like
            Time to event or censoring.
        event : array-like
            Event indicator (0/1).
        X : array-like or None
            Optional covariate matrix.
        strata : array-like or None
            Optional strata labels.

        Returns
        -------
        SurvivalDesign

        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        time = np.asarray(time, dtype=np.float64).ravel()
        event = np.asarray(event, dtype=np.float64).ravel()

        n = len(time)

        if n == 0:
            raise ValueError("time must have at least one observation")

        if len(event) != n:
            raise ValueError(
                f"time and event must have the same length: "
                f"got {n} and {len(event)}"
            )

        if np.any(time < 0):
            raise ValueError("time must be non-negative")

        # Allow event to be 0/1 or True/False
        unique_events = np.unique(event[~np.isnan(event)])
        if not np.all(np.isin(unique_events, [0.0, 1.0])):
            raise ValueError(
                f"event must contain only 0 and 1, "
                f"got unique values: {unique_events}"
            )

        event = event.astype(np.float64)

        X_arr = None
        if X is not None:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            if X_arr.ndim != 2:
                raise ValueError(
                    f"X must be 1D or 2D, got {X_arr.ndim}D"
                )
            if X_arr.shape[0] != n:
                raise ValueError(
                    f"X must have {n} rows to match time, "
                    f"got {X_arr.shape[0]}"
                )

        strata_arr = None
        if strata is not None:
            strata_arr = np.asarray(strata).ravel()
            if len(strata_arr) != n:
                raise ValueError(
                    f"strata must have {n} elements to match time, "
                    f"got {len(strata_arr)}"
                )

        return cls(
            time=time,
            event=event,
            X=X_arr,
            strata=strata_arr,
        )

    @property
    def n(self) -> int:
        """Number of observations."""
        return len(self.time)

    @property
    def p(self) -> int | None:
        """Number of covariates (None if no covariates)."""
        return self.X.shape[1] if self.X is not None else None

    @property
    def n_events(self) -> int:
        """Number of observed events."""
        return int(np.sum(self.event))
