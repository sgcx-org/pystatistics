"""
Solver dispatch for descriptive statistics.

Provides describe() as the comprehensive entry point, plus individual
functions: cor(), cov(), var(), quantile(), summary().
"""

from __future__ import annotations

from typing import Literal
import numpy as np
from numpy.typing import ArrayLike

from pystatistics.core.compute.device import select_device
from pystatistics.core.exceptions import ValidationError
from pystatistics.descriptive.design import DescriptiveDesign
from pystatistics.descriptive.solution import DescriptiveSolution
from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend


UseMethod = Literal['everything', 'complete.obs', 'pairwise.complete.obs']
CorMethod = Literal['pearson', 'spearman', 'kendall']
BackendChoice = Literal['auto', 'cpu', 'gpu']


def _ensure_design(data: ArrayLike | DescriptiveDesign) -> DescriptiveDesign:
    """Convert raw array to DescriptiveDesign if needed."""
    if isinstance(data, DescriptiveDesign):
        return data
    return DescriptiveDesign.from_array(data)


def _get_backend(backend: BackendChoice):
    """Select backend based on preference."""
    if backend == 'cpu':
        return CPUDescriptiveBackend()

    if backend == 'auto':
        device = select_device('auto')
        if device.device_type in ('cuda', 'mps'):
            try:
                from pystatistics.descriptive.backends.gpu import GPUDescriptiveBackend
                return GPUDescriptiveBackend(device=device)
            except ImportError:
                return CPUDescriptiveBackend()
        return CPUDescriptiveBackend()

    if backend == 'gpu':
        device = select_device('gpu')
        from pystatistics.descriptive.backends.gpu import GPUDescriptiveBackend
        return GPUDescriptiveBackend(device=device)

    raise ValidationError(f"Unknown backend: {backend!r}")


def describe(
    data: ArrayLike | DescriptiveDesign,
    *,
    use: UseMethod = 'everything',
    quantile_type: int = 7,
    backend: BackendChoice = 'auto',
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
    use : str
        Missing data handling. 'everything' (propagate NaN),
        'complete.obs' (listwise deletion),
        'pairwise.complete.obs' (pairwise deletion for cor/cov).
    quantile_type : int
        R quantile type (1-9). Default 7 matches R default.
    backend : str
        'auto', 'cpu', 'gpu'.

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
        use=use,
        quantile_probs=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        quantile_type=quantile_type,
    )

    return DescriptiveSolution(_result=result, _design=design)


def cor(
    x: ArrayLike | DescriptiveDesign,
    y: ArrayLike | None = None,
    *,
    method: CorMethod = 'pearson',
    use: UseMethod = 'everything',
    backend: BackendChoice = 'auto',
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
    use : str
        Missing data handling.
    backend : str
        'auto', 'cpu', 'gpu'.

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
        use=use,
        cor_method=method,
    )

    return DescriptiveSolution(_result=result, _design=design)


def cov(
    x: ArrayLike | DescriptiveDesign,
    y: ArrayLike | None = None,
    *,
    use: UseMethod = 'everything',
    backend: BackendChoice = 'auto',
) -> DescriptiveSolution:
    """
    Compute covariance matrix (Bessel-corrected, n-1). Matches R cov().

    Parameters
    ----------
    x : array-like or DescriptiveDesign
        2D data matrix (columns are variables).
    y : array-like, optional
        Second variable (1D).
    use : str
        Missing data handling.
    backend : str
        'auto', 'cpu', 'gpu'.

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

    result = be.solve(design, compute={'cov'}, use=use)

    return DescriptiveSolution(_result=result, _design=design)


def var(
    x: ArrayLike | DescriptiveDesign,
    *,
    use: UseMethod = 'everything',
    backend: BackendChoice = 'auto',
) -> DescriptiveSolution:
    """
    Compute variance (Bessel-corrected, n-1). Matches R var().

    For 1D input: returns per-column variance.
    For 2D input with p > 1: returns covariance matrix (same as cov()).

    Parameters
    ----------
    x : array-like or DescriptiveDesign
        1D or 2D data.
    use : str
        Missing data handling.
    backend : str
        'auto', 'cpu', 'gpu'.

    Returns
    -------
    DescriptiveSolution with variance or covariance_matrix populated.
    """
    design = _ensure_design(x)
    be = _get_backend(backend)

    # R var() on a matrix returns cov(), but we always populate variance too
    if design.p > 1:
        result = be.solve(design, compute={'var', 'cov'}, use=use)
    else:
        result = be.solve(design, compute={'var'}, use=use)

    return DescriptiveSolution(_result=result, _design=design)


def quantile(
    x: ArrayLike | DescriptiveDesign,
    probs: ArrayLike | None = None,
    *,
    type: int = 7,
    use: UseMethod = 'everything',
    backend: BackendChoice = 'auto',
) -> DescriptiveSolution:
    """
    Compute quantiles. Matches R quantile() with all 9 types.

    Parameters
    ----------
    x : array-like or DescriptiveDesign
        1D or 2D data.
    probs : array-like, optional
        Probabilities in [0, 1]. Default (0, 0.25, 0.5, 0.75, 1.0).
    type : int
        R quantile type 1-9. Default 7 (R default).
    use : str
        Missing data handling.
    backend : str
        'auto', 'cpu', 'gpu'.

    Returns
    -------
    DescriptiveSolution with quantiles populated.
    """
    if type not in range(1, 10):
        raise ValidationError(f"Quantile type must be 1-9, got {type}")

    design = _ensure_design(x)
    be = _get_backend(backend)

    if probs is not None:
        q_probs = np.asarray(probs, dtype=np.float64)
    else:
        q_probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    result = be.solve(
        design,
        compute={'quantiles'},
        use=use,
        quantile_probs=q_probs,
        quantile_type=type,
    )

    return DescriptiveSolution(_result=result, _design=design)


def summary(
    x: ArrayLike | DescriptiveDesign,
    *,
    use: UseMethod = 'everything',
    backend: BackendChoice = 'auto',
) -> DescriptiveSolution:
    """
    Compute six-number summary. Matches R summary() for numeric vectors.

    Computes: Min, Q1, Median, Mean, Q3, Max (per column).

    Parameters
    ----------
    x : array-like or DescriptiveDesign
        1D or 2D data.
    use : str
        Missing data handling.
    backend : str
        'auto', 'cpu', 'gpu'.

    Returns
    -------
    DescriptiveSolution with summary_table populated.
    """
    design = _ensure_design(x)
    be = _get_backend(backend)

    result = be.solve(design, compute={'summary', 'mean'}, use=use)

    return DescriptiveSolution(_result=result, _design=design)
