"""
Imputation methods for MICE.

Importing this package registers every built-in method (via import side-effects
in ``norm`` and ``pmm``), so ``registry.get_method('pmm')`` works as soon as the
package is imported. Stage 3 adds categorical methods here the same way.
"""

from pystatistics.mice.methods.base import ImputationMethod
from pystatistics.mice.methods.registry import (
    available_methods,
    get_method,
    is_registered,
    register,
)

# Side-effect imports: each module registers its method on import.
from pystatistics.mice.methods import norm as _norm  # noqa: F401
from pystatistics.mice.methods import pmm as _pmm  # noqa: F401
from pystatistics.mice.methods import logreg as _logreg  # noqa: F401
from pystatistics.mice.methods import polyreg as _polyreg  # noqa: F401
from pystatistics.mice.methods import polr as _polr  # noqa: F401

__all__ = [
    "ImputationMethod",
    "available_methods",
    "get_method",
    "is_registered",
    "register",
]
