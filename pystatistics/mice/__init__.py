"""
MICE — Multiple Imputation by Chained Equations.

Iteratively impute each incomplete column from the others until the chain
stabilises, producing ``m`` completed datasets, then combine analyses across
them with Rubin's rules. R-pegged: numeric defaults follow R's ``mice`` package
(predictive mean matching for numeric columns).

Public API (Stage 1, CPU, numeric columns):

    >>> from pystatistics.mice import mice, pool
    >>> imp = mice(data, m=5, method='pmm', seed=0)
    >>> completed = imp.completed_datasets()      # list of m arrays

Stage 1 supports numeric columns only; categorical methods are planned.
"""

from pystatistics.mice.design import MICEDesign
from pystatistics.mice.pooling import PooledResult, pool
from pystatistics.mice.solution import MICESolution
from pystatistics.mice.solvers import mice

__all__ = [
    "MICEDesign",
    "MICESolution",
    "PooledResult",
    "mice",
    "pool",
]
