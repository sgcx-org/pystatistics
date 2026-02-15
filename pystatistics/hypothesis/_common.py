"""
Common types for hypothesis testing.

Defines HTestParams (maps to R's htest class) and the Alternative enum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray


VALID_ALTERNATIVES = ("two.sided", "less", "greater")


@dataclass(frozen=True)
class HTestParams:
    """
    Parameter payload for hypothesis tests.

    Maps directly to R's htest structure. Every hypothesis test returns
    this same structure; test-specific extras go in the `extras` dict.

    Attributes
    ----------
    statistic : float or None
        Test statistic value (None for Fisher 2x2 exact test).
    statistic_name : str
        Name of the test statistic ("t", "X-squared", "W", "V", "D", "F").
    parameter : dict or None
        Distribution parameters, e.g. {"df": 9} or {"num df": 4, "denom df": 8}.
        None for exact tests with no degrees of freedom.
    p_value : float
        p-value of the test.
    conf_int : ndarray or None
        Confidence interval, shape (2,). None if not computed.
    conf_level : float
        Confidence level (e.g. 0.95).
    estimate : dict or None
        Point estimate(s), e.g. {"mean of x": 5.1, "mean of y": 3.2}.
    null_value : dict or None
        Hypothesized value under H0, e.g. {"difference in means": 0}.
    alternative : str
        "two.sided", "less", or "greater".
    method : str
        Human-readable method name, e.g. "Welch Two Sample t-test".
    data_name : str
        Description of the data, e.g. "x and y".
    extras : dict or None
        Test-specific additional outputs (e.g. observed/expected/residuals
        for chi-squared test).
    """
    statistic: float | None
    statistic_name: str
    parameter: dict[str, float] | None
    p_value: float
    conf_int: NDArray[np.floating[Any]] | None
    conf_level: float
    estimate: dict[str, float] | None
    null_value: dict[str, float] | None
    alternative: str
    method: str
    data_name: str
    extras: dict[str, Any] | None = None
