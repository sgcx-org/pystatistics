"""
Solver dispatch for descriptive statistics.

Provides describe() as the comprehensive entry point, plus individual
functions: cor(), cov(), var(), quantile(), summary().
"""

from __future__ import annotations

from typing import Literal
import numpy as np
from numpy.typing import ArrayLike

from pystatistics.core.compute.backend import resolve_backend
from pystatistics.core.exceptions import ValidationError
from pystatistics.descriptive.design import DescriptiveDesign
from pystatistics.descriptive.solution import DescriptiveSolution
from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend


NaAction = Literal['everything', 'complete', 'pairwise']
# Public na_action values map to the backends' internal R-style codes.
_NA_CODE = {'everything': 'everything', 'complete': 'complete.obs',
            'pairwise': 'pairwise.complete.obs'}
CorMethod = Literal['pearson', 'spearman', 'kendall']
# Descriptive stats have no GPU float64 path (the win is bandwidth-bound moment
# reductions, not precision); the honest subset omits 'gpu_fp64'.
BackendChoice = Literal['auto', 'cpu', 'gpu']


def _ensure_design(data: ArrayLike | DescriptiveDesign) -> DescriptiveDesign:
    """Convert raw array to DescriptiveDesign if needed."""
    if isinstance(data, DescriptiveDesign):
        return data
    return DescriptiveDesign.from_array(data)


def _get_backend(backend: BackendChoice | None):
    """Select backend from the resolved (device, precision) target."""
    target = resolve_backend(backend, supports_fp64=False)
    if target.device_type == 'cpu':
        return CPUDescriptiveBackend()

    from pystatistics.descriptive.backends.gpu import GPUDescriptiveBackend
    return GPUDescriptiveBackend(device=target.device)


def describe(
    data: ArrayLike | DescriptiveDesign,
    *,
    na_action: NaAction = 'everything',
    quantile_type: int = 7,
    backend: BackendChoice | None = None,
) -> DescriptiveSolution:
    """
    Compute comprehensive descriptive statistics.

    Computes: mean, variance, standard deviation, covariance matrix,
    Pearson correlation, quantiles (0, 0.25, 0.5, 0.75, 1), skewness,
    kurtosis, and six-number summary.

    Parameters
    ----------
    data : array-like or DescriptiveDesign
        1D or 2D data matrix.
    na_action : str
        Missing data handling. 'everything' (propagate NaN),
        'complete' (listwise deletion),
        'pairwise' (pairwise deletion for cor/cov).
    quantile_type : int
        R quantile type (1-9). Default 7 matches R default.
    backend : str or None
        Default None → 'cpu' (R-reference path). Explicit: 'cpu',
        'gpu', or 'auto' to prefer GPU when available.

    Returns
    -------
    DescriptiveSolution with all statistics populated.
    """
    design = _ensure_design(data)
    be = _get_backend(backend)

    compute = {
        'mean', 'var', 'sd', 'cov', 'cor_pearson',
        'quantiles', 'summary', 'skewness', 'kurtosis',
    }

    result = be.solve(
        design,
        compute=compute,
        use=_NA_CODE[na_action],
        quantile_probs=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        quantile_type=quantile_type,
    )

    return DescriptiveSolution(_result=result, _design=design)


def cor(
    x: ArrayLike | DescriptiveDesign,
    y: ArrayLike | None = None,
    *,
    method: CorMethod = 'pearson',
    na_action: NaAction = 'everything',
    backend: BackendChoice | None = None,
) -> DescriptiveSolution:
    """
    Compute correlation matrix. Matches R cor().

    Parameters
    ----------
    x : array-like or DescriptiveDesign
        2D data matrix (columns are variables), or DescriptiveDesign.
    y : array-like, optional
        Second variable (1D). If provided, computes cor(x, y) by
        stacking as a 2-column matrix.
    method : str
        'pearson', 'spearman', 'kendall'.
    na_action : str
        Missing data handling.
    backend : str or None
        Default None → 'cpu' (R-reference path). Explicit: 'cpu',
        'gpu', or 'auto' to prefer GPU when available.

    Returns
    -------
    DescriptiveSolution with correlation matrix populated.
    """
    if y is not None:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        design = DescriptiveDesign.from_array(np.column_stack([x_arr, y_arr]))
    else:
        design = _ensure_design(x)

    be = _get_backend(backend)

    compute_key = f'cor_{method}'
    if compute_key not in ('cor_pearson', 'cor_spearman', 'cor_kendall'):
        raise ValidationError(
            f"Unknown correlation method: {method!r}. "
            f"Must be 'pearson', 'spearman', or 'kendall'."
        )

    result = be.solve(
        design,
        compute={compute_key},
        use=_NA_CODE[na_action],
        cor_method=method,
    )

    return DescriptiveSolution(_result=result, _design=design)


def cov(
    x: ArrayLike | DescriptiveDesign,
    y: ArrayLike | None = None,
    *,
    na_action: NaAction = 'everything',
    backend: BackendChoice | None = None,
) -> DescriptiveSolution:
    """
    Compute covariance matrix (Bessel-corrected, n-1). Matches R cov().

    Parameters
    ----------
    x : array-like or DescriptiveDesign
        2D data matrix (columns are variables).
    y : array-like, optional
        Second variable (1D).
    na_action : str
        Missing data handling.
    backend : str or None
        Default None → 'cpu' (R-reference path). Explicit: 'cpu',
        'gpu', or 'auto' to prefer GPU when available.

    Returns
    -------
    DescriptiveSolution with covariance_matrix populated.
    """
    if y is not None:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        design = DescriptiveDesign.from_array(np.column_stack([x_arr, y_arr]))
    else:
        design = _ensure_design(x)

    be = _get_backend(backend)

    result = be.solve(design, compute={'cov'}, use=_NA_CODE[na_action])

    return DescriptiveSolution(_result=result, _design=design)


def var(
    x: ArrayLike | DescriptiveDesign,
    *,
    na_action: NaAction = 'everything',
    backend: BackendChoice | None = None,
) -> DescriptiveSolution:
    """
    Compute variance (Bessel-corrected, n-1). Matches R var().

    For 1D input: returns per-column variance.
    For 2D input with p > 1: returns covariance matrix (same as cov()).

    Parameters
    ----------
    x : array-like or DescriptiveDesign
        1D or 2D data.
    na_action : str
        Missing data handling.
    backend : str or None
        Default None → 'cpu' (R-reference path). Explicit: 'cpu',
        'gpu', or 'auto' to prefer GPU when available.

    Returns
    -------
    DescriptiveSolution with variance or covariance_matrix populated.
    """
    design = _ensure_design(x)
    be = _get_backend(backend)

    # R var() on a matrix returns cov(), but we always populate variance too
    if design.p > 1:
        result = be.solve(design, compute={'var', 'cov'}, use=_NA_CODE[na_action])
    else:
        result = be.solve(design, compute={'var'}, use=_NA_CODE[na_action])

    return DescriptiveSolution(_result=result, _design=design)


def quantile(
    x: ArrayLike | DescriptiveDesign,
    probs: ArrayLike | None = None,
    *,
    quantile_type: int = 7,
    na_action: NaAction = 'everything',
    backend: BackendChoice | None = None,
) -> DescriptiveSolution:
    """
    Compute quantiles. Matches R quantile() with all 9 types.

    Parameters
    ----------
    x : array-like or DescriptiveDesign
        1D or 2D data.
    probs : array-like, optional
        Probabilities in [0, 1]. Default (0, 0.25, 0.5, 0.75, 1.0).
    quantile_type : int
        R quantile type 1-9. Default 7 (R default).
    na_action : str
        Missing data handling.
    backend : str or None
        Default None → 'cpu' (R-reference path). Explicit: 'cpu',
        'gpu', or 'auto' to prefer GPU when available.

    Returns
    -------
    DescriptiveSolution with quantiles populated.
    """
    if quantile_type not in range(1, 10):
        raise ValidationError(f"Quantile type must be 1-9, got {quantile_type}")

    design = _ensure_design(x)
    be = _get_backend(backend)

    if probs is not None:
        q_probs = np.asarray(probs, dtype=np.float64)
    else:
        q_probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    # Fail loud on invalid probabilities, matching R's quantile()
    # ("'probs' outside [0,1]"). Silently clamping to the min/max would return a
    # plausible wrong number for an invalid request (Rule 1 / A6).
    finite = q_probs[~np.isnan(q_probs)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise ValidationError(
            f"'probs' outside [0,1]: got range "
            f"[{float(finite.min())}, {float(finite.max())}]"
        )

    result = be.solve(
        design,
        compute={'quantiles'},
        use=_NA_CODE[na_action],
        quantile_probs=q_probs,
        quantile_type=quantile_type,
    )

    return DescriptiveSolution(_result=result, _design=design)


def summary(
    x: ArrayLike | DescriptiveDesign,
    *,
    na_action: NaAction = 'everything',
    backend: BackendChoice | None = None,
) -> DescriptiveSolution:
    """
    Compute six-number summary. Matches R summary() for numeric vectors.

    Computes: Min, Q1, Median, Mean, Q3, Max (per column).

    Parameters
    ----------
    x : array-like or DescriptiveDesign
        1D or 2D data.
    na_action : str
        Missing data handling.
    backend : str or None
        Default None → 'cpu' (R-reference path). Explicit: 'cpu',
        'gpu', or 'auto' to prefer GPU when available.

    Returns
    -------
    DescriptiveSolution with summary_table populated.
    """
    design = _ensure_design(x)
    be = _get_backend(backend)

    result = be.solve(design, compute={'summary', 'mean'}, use=_NA_CODE[na_action])

    return DescriptiveSolution(_result=result, _design=design)
