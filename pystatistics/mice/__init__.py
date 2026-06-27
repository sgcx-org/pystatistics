"""
MICE — Multiple Imputation by Chained Equations.

Iteratively impute each incomplete column from the others until the chain
stabilises, producing ``m`` completed datasets, then combine analyses across
them with Rubin's rules. R-pegged: numeric defaults follow R's ``mice`` package
(predictive mean matching for numeric columns).

Public API:

    >>> from pystatistics.mice import mice, pool
    >>> imp = mice(data, m=5, method='pmm', seed=0)
    >>> completed = imp.completed_datasets()      # list of m arrays

Supports numeric columns (``pmm``/``norm``) and categorical columns — binary
(``logreg``), unordered (``polyreg``), and ordered (``polr``) factors — on both
the CPU and GPU (CUDA / Apple Silicon MPS) backends.
"""

from pystatistics.mice.design import MICEDesign
from pystatistics.mice.pooling import PooledSolution, pool
from pystatistics.mice.solution import MICESolution
from pystatistics.mice.solvers import mice

__all__ = [
    "MICEDesign",
    "MICESolution",
    "PooledSolution",
    "mice",
    "pool",
]
