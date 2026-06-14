"""
Registry of imputation methods.

A single source of truth mapping a method name (``'pmm'``, ``'norm'``, ...) to
its implementation. The chained-equations sweep resolves methods through here,
so adding a method (Stage 3) means registering a new entry — no edits to the
sweep, the solver, or the design.

State note (CLAUDE.md Rule 5): this module owns a module-level dict. That is a
deliberate, documented registry — the canonical exception the rule allows. It is
populated only at import time by method modules calling ``register`` on
themselves, never mutated at runtime by user code.
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError
from pystatistics.mice.methods.base import ImputationMethod

# name -> method instance. Populated at import by method modules (norm, pmm).
_REGISTRY: dict[str, ImputationMethod] = {}


def register(method: ImputationMethod) -> ImputationMethod:
    """Register a method instance under its ``name``. Returns it unchanged.

    Raises
    ------
    ValidationError
        If a different method is already registered under the same name.
    """
    name = method.name
    if name in _REGISTRY and _REGISTRY[name] is not method:
        raise ValidationError(
            f"Imputation method {name!r} is already registered. "
            f"Method names must be unique."
        )
    _REGISTRY[name] = method
    return method


def get_method(name: str) -> ImputationMethod:
    """Look up a registered method by name (fail loud on unknown — Rule 1)."""
    if name not in _REGISTRY:
        raise ValidationError(
            f"Unknown imputation method {name!r}. "
            f"Available methods: {available_methods()}."
        )
    return _REGISTRY[name]


def is_registered(name: str) -> bool:
    """Whether ``name`` is a known method."""
    return name in _REGISTRY


def available_methods() -> list[str]:
    """Sorted list of registered method names."""
    return sorted(_REGISTRY)
