"""
PyStatistics: GPU-accelerated statistical computing for Python.

Usage:
    from pystatistics import DataSource
    from pystatistics.regression import Design, fit
    
    # DataSource: "I have data" (domain-agnostic)
    ds = DataSource.from_file("data.csv")
    ds = DataSource.build(X, y)
    
    # Design: "I'm building a regression" (wraps DataSource)
    design = Design.from_datasource(ds, y='target')
    
    # fit: "Here's your answer"
    result = fit(design)
"""

__version__ = "0.1.0"
__author__ = "Hai-Shuo"
__email__ = "contact@sgcx.org"

from pystatistics.core.datasource import DataSource
from pystatistics import regression
from pystatistics import mvnmle
from pystatistics import descriptive
from pystatistics import hypothesis
from pystatistics import montecarlo
from pystatistics import survival
from pystatistics import anova

__all__ = [
    "__version__",
    "DataSource",
    "regression",
    "mvnmle",
    "descriptive",
    "hypothesis",
    "montecarlo",
    "survival",
    "anova",
]
