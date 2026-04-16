"""
ETS state space model specifications and recursion.

Defines the ETSSpec frozen dataclass and the core forward recursion for
all supported ETS(error, trend, season) combinations. The naming convention
follows Hyndman et al. (2008):

    ETS(Error, Trend, Season)
    Error:  A (additive) or M (multiplicative)
    Trend:  N (none), A (additive), Ad (additive damped)
    Season: N (none), A (additive), M (multiplicative)

The recursion uses explicit if/elif branches per model type for clarity
and debuggability, matching the approach in R's forecast package.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import ValidationError


# ---------------------------------------------------------------------------
# ETS specification
# ---------------------------------------------------------------------------

_VALID_ERROR = frozenset({"A", "M"})
_VALID_TREND = frozenset({"N", "A", "Ad"})
_VALID_SEASON = frozenset({"N", "A", "M"})


@dataclass(frozen=True)
class ETSSpec:
    """
    Specification of an ETS model type.

    Attributes
    ----------
    error : str
        Error type: ``'A'`` (additive) or ``'M'`` (multiplicative).
    trend : str
        Trend type: ``'N'`` (none), ``'A'`` (additive), or ``'Ad'`` (damped).
    season : str
        Season type: ``'N'`` (none), ``'A'`` (additive), or ``'M'`` (multiplicative).
    period : int
        Seasonal period (1 when no seasonal component).
    damped : bool
        ``True`` when trend is ``'Ad'``.
    """

    error: str
    trend: str
    season: str
    period: int
    damped: bool

    @property
    def name(self) -> str:
        """Human-readable model name, e.g. ``'ETS(A,Ad,N)'``."""
        return f"ETS({self.error},{self.trend},{self.season})"

    @property
    def n_states(self) -> int:
        """Number of state variables: 1 (level) + trend? + season?."""
        n = 1  # level
        if self.trend in ("A", "Ad"):
            n += 1
        if self.season in ("A", "M"):
            n += self.period
        return n

    @property
    def n_params(self) -> int:
        """Number of smoothing parameters (alpha, beta, gamma, phi)."""
        n = 1  # alpha
        if self.trend in ("A", "Ad"):
            n += 1  # beta
        if self.season in ("A", "M"):
            n += 1  # gamma
        if self.damped:
            n += 1  # phi
        return n


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_FULL_RE = re.compile(
    r"^(?:ETS\()?"
    r"([AM])"          # error
    r",?\s*"
    r"(N|Ad?)"         # trend
    r",?\s*"
    r"(N|[AM])"        # season
    r"(?:\))?$",
    re.IGNORECASE,
)


def parse_ets_spec(model: str, period: int = 1) -> ETSSpec:
    """
    Parse an ETS model string into an :class:`ETSSpec`.

    Accepted formats (case-insensitive):

    * 3-char shorthand: ``'AAN'``
    * With damping: ``'AAdN'``
    * Comma-separated: ``'A,Ad,N'``
    * Full notation: ``'ETS(A,Ad,N)'``

    Parameters
    ----------
    model : str
        ETS model string.
    period : int
        Seasonal period (must be >= 2 when season is ``'A'`` or ``'M'``).

    Returns
    -------
    ETSSpec
        Parsed model specification.

    Raises
    ------
    ValidationError
        If the model string is invalid or period is inconsistent.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValidationError(f"model: expected non-empty string, got {model!r}")

    m = _FULL_RE.match(model.strip())
    if m is None:
        raise ValidationError(
            f"model: cannot parse {model!r}. "
            "Expected format like 'AAN', 'AAdN', 'A,A,M', or 'ETS(A,Ad,N)'"
        )

    error = m.group(1).upper()
    trend_raw = m.group(2)
    season = m.group(3).upper()

    # Normalise trend
    if trend_raw.upper() == "AD":
        trend = "Ad"
    else:
        trend = trend_raw.upper()

    if error not in _VALID_ERROR:
        raise ValidationError(f"model: invalid error type {error!r}, expected 'A' or 'M'")
    if trend not in _VALID_TREND:
        raise ValidationError(
            f"model: invalid trend type {trend!r}, expected 'N', 'A', or 'Ad'"
        )
    if season not in _VALID_SEASON:
        raise ValidationError(
            f"model: invalid season type {season!r}, expected 'N', 'A', or 'M'"
        )

    damped = trend == "Ad"

    if season in ("A", "M") and period < 2:
        raise ValidationError(
            f"period: seasonal model requires period >= 2, got {period}"
        )
    if season == "N":
        period = 1

    return ETSSpec(
        error=error,
        trend=trend,
        season=season,
        period=period,
        damped=damped,
    )


# ---------------------------------------------------------------------------
# Helpers to unpack parameter vector
# ---------------------------------------------------------------------------

def unpack_params(
    params: NDArray,
    spec: ETSSpec,
) -> tuple[float, float | None, float | None, float | None]:
    """
    Unpack a flat parameter vector into (alpha, beta, gamma, phi).

    Parameters
    ----------
    params : NDArray
        Smoothing parameters in canonical order.
    spec : ETSSpec
        Model specification.

    Returns
    -------
    tuple
        ``(alpha, beta_or_None, gamma_or_None, phi_or_None)``
    """
    idx = 0
    alpha = float(params[idx]); idx += 1

    beta = None
    if spec.trend in ("A", "Ad"):
        beta = float(params[idx]); idx += 1

    gamma = None
    if spec.season in ("A", "M"):
        gamma = float(params[idx]); idx += 1

    phi = None
    if spec.damped:
        phi = float(params[idx])

    return alpha, beta, gamma, phi


# ---------------------------------------------------------------------------
# Core recursion
# ---------------------------------------------------------------------------

def ets_recursion(
    y: NDArray,
    spec: ETSSpec,
    params: NDArray,
    init_states: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Run the ETS state space recursion forward through the data.

    Given smoothing parameters and initial states, produce fitted values,
    residuals, and the full state history.

    Parameters
    ----------
    y : NDArray
        Observed time series of length *n*.
    spec : ETSSpec
        Model specification.
    params : NDArray
        Smoothing parameters ``[alpha, beta?, gamma?, phi?]``.
    init_states : NDArray
        Initial state vector ``[l_0, b_0?, s_{1-m}, ..., s_0?]``.

    Returns
    -------
    fitted : NDArray
        One-step-ahead fitted values, shape ``(n,)``.
    residuals : NDArray
        Residuals (additive: ``y - fitted``; multiplicative: ``y/fitted - 1``),
        shape ``(n,)``.
    states : NDArray
        State history, shape ``(n + 1, n_states_core)`` where
        ``n_states_core`` is 1 (level) + 1 (trend, if any) + period (season,
        if any).  Row 0 holds the initial states.
    """
    n = len(y)
    alpha, beta, gamma, phi = unpack_params(params, spec)
    m = spec.period

    # Unpack initial states
    idx = 0
    l_prev = float(init_states[idx]); idx += 1

    b_prev = 0.0
    has_trend = spec.trend in ("A", "Ad")
    if has_trend:
        b_prev = float(init_states[idx]); idx += 1

    has_season = spec.season in ("A", "M")
    # Seasonal states: s_{1-m}, s_{2-m}, ..., s_0  (length m)
    s = np.zeros(m, dtype=np.float64)
    if has_season:
        s[:] = init_states[idx : idx + m]

    phi_val = phi if phi is not None else 1.0

    # Determine number of core state columns for storage
    n_cols = 1 + (1 if has_trend else 0) + (m if has_season else 0)
    states = np.empty((n + 1, n_cols), dtype=np.float64)

    # Store initial state row
    _store_state(states, 0, l_prev, b_prev, has_trend, s, has_season, m)

    fitted = np.empty(n, dtype=np.float64)
    residuals = np.empty(n, dtype=np.float64)

    mult_error = spec.error == "M"

    for t in range(n):
        s_idx = t % m  # index into seasonal ring buffer

        # --- one-step-ahead prediction (mu_t) ---
        if spec.season == "N":
            if spec.trend == "N":
                mu = l_prev
            else:
                mu = l_prev + phi_val * b_prev
        elif spec.season == "A":
            if spec.trend == "N":
                mu = l_prev + s[s_idx]
            else:
                mu = l_prev + phi_val * b_prev + s[s_idx]
        else:  # season == "M"
            if spec.trend == "N":
                mu = l_prev * s[s_idx]
            else:
                mu = (l_prev + phi_val * b_prev) * s[s_idx]

        fitted[t] = mu

        # --- error ---
        if mult_error:
            if abs(mu) < 1e-15:
                e = y[t] - mu
            else:
                e = (y[t] / mu) - 1.0
            residuals[t] = e
        else:
            e = y[t] - mu
            residuals[t] = e

        # --- state update ---
        s_old = s[s_idx]
        l_old = l_prev
        b_old = b_prev

        if spec.season == "N" and spec.trend == "N":
            # ETS(.,N,N)
            if mult_error:
                l_prev = l_old * (1.0 + alpha * e)
            else:
                l_prev = l_old + alpha * e

        elif spec.season == "N" and spec.trend in ("A", "Ad"):
            # ETS(.,A,N) or ETS(.,Ad,N)
            if mult_error:
                l_prev = (l_old + phi_val * b_old) * (1.0 + alpha * e)
                b_prev = phi_val * b_old + beta * (l_old + phi_val * b_old) * e
            else:
                l_prev = l_old + phi_val * b_old + alpha * e
                b_prev = phi_val * b_old + beta * e

        elif spec.season == "A" and spec.trend == "N":
            # ETS(.,N,A)
            if mult_error:
                l_prev = l_old + alpha * (l_old + s_old) * e
                s[s_idx] = s_old + gamma * (l_old + s_old) * e
            else:
                l_prev = l_old + alpha * e
                s[s_idx] = s_old + gamma * e

        elif spec.season == "A" and spec.trend in ("A", "Ad"):
            # ETS(.,A,A) or ETS(.,Ad,A)
            if mult_error:
                mu_val = l_old + phi_val * b_old + s_old
                l_prev = l_old + phi_val * b_old + alpha * mu_val * e
                b_prev = phi_val * b_old + beta * mu_val * e
                s[s_idx] = s_old + gamma * mu_val * e
            else:
                l_prev = l_old + phi_val * b_old + alpha * e
                b_prev = phi_val * b_old + beta * e
                s[s_idx] = s_old + gamma * e

        elif spec.season == "M" and spec.trend == "N":
            # ETS(.,N,M)
            if mult_error:
                l_prev = l_old * (1.0 + alpha * e)
                s[s_idx] = s_old * (1.0 + gamma * e)
            else:
                denom_s = s_old if abs(s_old) > 1e-15 else 1e-15
                denom_l = l_old if abs(l_old) > 1e-15 else 1e-15
                l_prev = l_old + alpha * e / denom_s
                s[s_idx] = s_old + gamma * e / denom_l

        elif spec.season == "M" and spec.trend in ("A", "Ad"):
            # ETS(.,A,M) or ETS(.,Ad,M)
            base = l_old + phi_val * b_old
            if mult_error:
                l_prev = base * (1.0 + alpha * e)
                b_prev = phi_val * b_old + beta * base * e
                s[s_idx] = s_old * (1.0 + gamma * e)
            else:
                denom_s = s_old if abs(s_old) > 1e-15 else 1e-15
                denom_l = base if abs(base) > 1e-15 else 1e-15
                l_prev = base + alpha * e / denom_s
                b_prev = phi_val * b_old + beta * e / denom_s
                s[s_idx] = s_old + gamma * e / denom_l

        _store_state(states, t + 1, l_prev, b_prev, has_trend, s, has_season, m)

    return fitted, residuals, states


def _store_state(
    states: NDArray,
    row: int,
    level: float,
    trend: float,
    has_trend: bool,
    season: NDArray,
    has_season: bool,
    m: int,
) -> None:
    """Write a single row of the state matrix."""
    idx = 0
    states[row, idx] = level; idx += 1
    if has_trend:
        states[row, idx] = trend; idx += 1
    if has_season:
        states[row, idx : idx + m] = season
