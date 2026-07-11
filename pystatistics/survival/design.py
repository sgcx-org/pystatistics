"""
SurvivalDesign: immutable container for time-to-event data.

Wraps time, event indicator, optional covariates, and optional strata.
Validates inputs at construction time — all downstream code trusts clean data.
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SurvivalDesign:
    """Immutable survival data container.

    Parameters
    ----------
    time : NDArray
        Time to event or censoring (the risk-interval EXIT). Must be
        non-negative.
    event : NDArray
        Event indicator: 1 = event observed, 0 = censored.
    X : NDArray or None
        Covariate matrix (n, p). None for KM / log-rank.
    strata : NDArray or None
        Strata labels for stratified analyses.
    entry : NDArray or None
        Risk-interval ENTRY time per row; the row is at risk on
        ``(entry, time]``. None means at risk from time 0. Carries both the
        scalar delayed-entry (left-truncation) input of ``kaplan_meier`` and
        the counting-process ``start`` of ``coxph`` (CONVENTIONS A8: the two
        public names share this one validated field).
    """

    time: NDArray
    event: NDArray
    X: NDArray | None
    strata: NDArray | None
    entry: NDArray | None = None

    @classmethod
    def for_survival(
        cls,
        time,
        event,
        X=None,
        *,
        strata=None,
        entry=None,
    ) -> SurvivalDesign:
        """Create and validate survival data.

        Parameters
        ----------
        time : array-like
            Time to event or censoring (risk-interval exit).
        event : array-like
            Event indicator (0/1).
        X : array-like or None
            Optional covariate matrix.
        strata : array-like or None
            Optional strata labels.
        entry : array-like or None
            Optional risk-interval entry times. Each must be strictly less
            than the corresponding ``time`` (R NA-drops such rows with a
            warning; PyStatistics refuses loudly instead — A6).

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
            raise ValidationError("time must have at least one observation")

        if len(event) != n:
            raise ValidationError(
                f"time and event must have the same length: "
                f"got {n} and {len(event)}"
            )

        # Non-finite times must fail loud: NaN/inf slip past ``time < 0`` (both
        # comparisons are False) and would silently corrupt every risk set.
        if not np.all(np.isfinite(time)):
            raise ValidationError("time must be finite (no NaN or inf)")

        if np.any(time < 0):
            raise ValidationError("time must be non-negative")

        # event must be exactly 0/1 (or True/False) and finite — a NaN event
        # is neither a censoring nor an event, so it is rejected, not silently
        # kept (it would otherwise poison the death/at-risk counts).
        unique_events = np.unique(event)
        if not (np.all(np.isfinite(event))
                and np.all(np.isin(unique_events, [0.0, 1.0]))):
            raise ValidationError(
                f"event must contain only 0 and 1 (finite), "
                f"got unique values: {unique_events}"
            )

        event = event.astype(np.float64)

        X_arr = None
        if X is not None:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            if X_arr.ndim != 2:
                raise ValidationError(
                    f"X must be 1D or 2D, got {X_arr.ndim}D"
                )
            if X_arr.shape[0] != n:
                raise ValidationError(
                    f"X must have {n} rows to match time, "
                    f"got {X_arr.shape[0]}"
                )

        strata_arr = None
        if strata is not None:
            strata_arr = np.asarray(strata).ravel()
            if len(strata_arr) != n:
                raise ValidationError(
                    f"strata must have {n} elements to match time, "
                    f"got {len(strata_arr)}"
                )
            # A missing stratum label (NaN, or None in an object array) would be
            # grouped into its own phantom stratum or silently dropped — fail
            # loud instead of losing those rows from the fit.
            if strata_arr.dtype.kind == "f":
                if not np.all(np.isfinite(strata_arr)):
                    raise ValidationError(
                        "strata contains NaN/inf labels; every observation "
                        "must have a defined stratum"
                    )
            elif strata_arr.dtype.kind == "O":
                if any(s is None or (isinstance(s, float) and np.isnan(s))
                       for s in strata_arr):
                    raise ValidationError(
                        "strata contains missing (None/NaN) labels; every "
                        "observation must have a defined stratum"
                    )

        entry_arr = None
        if entry is not None:
            entry_arr = np.asarray(entry, dtype=np.float64).ravel()
            if len(entry_arr) != n:
                raise ValidationError(
                    f"entry must have {n} elements to match time, "
                    f"got {len(entry_arr)}"
                )
            if not np.all(np.isfinite(entry_arr)):
                raise ValidationError("entry times must be finite")
            bad = entry_arr >= time
            if np.any(bad):
                # R turns such rows into NA (with a warning) and silently
                # drops them from the fit; we refuse loudly instead (A6).
                idx = np.nonzero(bad)[0]
                raise ValidationError(
                    f"entry time must be strictly less than time (the risk "
                    f"interval is (entry, time]); violated at row(s) "
                    f"{idx[:5].tolist()}{'...' if len(idx) > 5 else ''}"
                )

        return cls(
            time=time,
            event=event,
            X=X_arr,
            strata=strata_arr,
            entry=entry_arr,
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
