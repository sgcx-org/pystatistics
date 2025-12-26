"""
Linear and generalized linear models.

Usage:
    from pystatistics import DataSource
    from pystatistics.regression import Design, fit
    
    # From DataSource
    ds = DataSource.from_file("data.csv")
    design = Design.from_datasource(ds, y='target')
    result = fit(design)
    
    # From arrays (convenience)
    result = fit(X, y)
"""

from pystatistics.core.datasource import DataSource
from pystatistics.regression.design import Design
from pystatistics.regression.solution import LinearSolution, LinearParams
from pystatistics.regression.solvers import fit

__all__ = [
    "fit",
    "Design",
    "DataSource",
    "LinearSolution",
    "LinearParams",
]
