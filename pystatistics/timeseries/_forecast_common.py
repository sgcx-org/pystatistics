"""
Shared input validation for forecast prediction intervals.

Both :func:`forecast_ets` and :func:`forecast_arima` accept confidence
levels as fractions in ``(0, 1)`` (the library-wide ``conf_level``
convention). This module owns the single normalization/validation of
that argument so the two forecast entry points stay consistent.
"""

from __future__ import annotations

from collections.abc import Sequence

from pystatistics.core.exceptions import ValidationError


def _normalize_conf_levels(conf_level: float | Sequence[float]) -> list[float]:
    """Normalize a ``conf_level`` argument to a list of fractions.

    Accepts either a single float or a sequence of floats, each a
    confidence level expressed as a fraction in ``(0, 1)`` (e.g. ``0.95``).

    Parameters
    ----------
    conf_level : float or sequence of float
        One or more confidence levels as fractions in ``(0, 1)``.

    Returns
    -------
    list of float
        The validated confidence levels.

    Raises
    ------
    ValidationError
        If *conf_level* is empty, non-numeric, or any value is outside
        ``(0, 1)``. A value ``>= 1`` (e.g. the old whole-percent ``95``)
        is rejected with an explicit fraction-vs-percent message.
    """
    if isinstance(conf_level, bool):
        raise ValidationError(
            f"conf_level: must be a fraction in (0, 1) or a sequence "
            f"thereof, got {conf_level!r}"
        )
    if isinstance(conf_level, (int, float)):
        levels = [float(conf_level)]
    else:
        try:
            levels = [float(c) for c in conf_level]
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"conf_level: must be a float or a sequence of floats, "
                f"got {conf_level!r} ({exc})"
            ) from exc

    if not levels:
        raise ValidationError("conf_level: must supply at least one level")

    for c in levels:
        if c >= 1.0:
            raise ValidationError(
                f"conf_level: each must be a fraction in (0, 1), got {c}. "
                "Confidence levels are fractions (e.g. 0.95), not whole "
                "percents (e.g. 95)."
            )
        if c <= 0.0:
            raise ValidationError(
                f"conf_level: each must be a fraction in (0, 1), got {c}"
            )
    return levels
