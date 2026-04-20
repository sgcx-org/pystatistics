"""Nonparametric / distribution-free MCAR tests.

Alternatives to Little's (1988) MCAR test that do not assume multivariate
normality. Motivated by the 2026-04 Lacuna finding that cached Little's
features (under MLE or MoM plug-in) do not help mechanism classification
on heavy-tailed, skewed, categorical-laden tabular data where MVN is
badly violated.

Tests exposed:

- ``propensity_mcar_test``: supervised detection of non-MCAR via a
  classifier's ability to predict the missingness indicator column from
  the observed values — if a random forest / gradient boosting machine
  can separate "row j of column c is missing" from "row j of column c
  is observed" above chance, missingness depends on observed values and
  MCAR is rejected. Native handling of mixed types and nonlinearity;
  permutation-based calibration.
- ``hsic_mcar_test`` (planned): kernel independence test between the
  observed values and the missingness indicator matrix, with a Gaussian
  RBF kernel and median-heuristic bandwidth.
- ``missmech_mcar_test`` (planned): Jamshidian-Jalal (2011)
  Hawkins-test-on-covariance-homogeneity-across-missingness-patterns
  with k-NN imputation, recommended by the statistical community as the
  modern replacement for Little's when MVN fails.

Result type
-----------

All tests return a ``NonparametricMCARResult``. The field set overlaps
with ``pystatistics.mvnmle.MCARTestResult`` where meaningful (statistic,
p_value, rejected, alpha, method) but omits the MVN-specific fields
(``df``, ``ml_mean``, ``ml_cov``, ``patterns``) that do not apply.
"""

from pystatistics.nonparametric_mcar.result import NonparametricMCARResult
from pystatistics.nonparametric_mcar.propensity import propensity_mcar_test
from pystatistics.nonparametric_mcar.hsic import hsic_mcar_test
from pystatistics.nonparametric_mcar.missmech import missmech_mcar_test

__all__ = [
    "NonparametricMCARResult",
    "propensity_mcar_test",
    "hsic_mcar_test",
    "missmech_mcar_test",
]
