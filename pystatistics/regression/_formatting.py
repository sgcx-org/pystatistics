"""Shared formatting utilities for regression output."""

import numpy as np


def significance_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""
