"""
QR decomposition with column pivoting.

CPU implementation uses SciPy's LAPACK-based QR to match R's behavior exactly.
"""

from dataclasses import dataclass
from typing import Literal, Any, Optional
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QRResult:
    """
    Result of QR decomposition with column pivoting.
    
    Attributes:
        Q: Orthogonal matrix (n x k where k = min(n, p) for reduced mode)
        R: Upper triangular matrix (k x p), columns are in PIVOTED order
        pivot: Permutation array (0-indexed). pivot[i] = original column index
               that was moved to position i during pivoting.
        rank: Numerical rank
    """
    Q: NDArray[np.floating[Any]]
    R: NDArray[np.floating[Any]]
    pivot: NDArray[np.intp]
    rank: int


def qr_decompose(
    X: NDArray[np.floating[Any]],
    mode: Literal['reduced', 'complete'] = 'reduced'
) -> QRResult:
    """
    QR decomposition with column pivoting using SciPy/LAPACK.
    
    Args:
        X: Matrix to decompose (n x p)
        mode: 'reduced' for economy QR, 'complete' for full QR
        
    Returns:
        QRResult with Q, R, pivot indices, and rank
    """
    from scipy.linalg import qr
    
    n, p = X.shape
    X = np.asarray(X, dtype=np.float64)
    
    scipy_mode = 'economic' if mode == 'reduced' else 'full'
    Q, R, pivot = qr(X, mode=scipy_mode, pivoting=True)
    
    # Rank = number of columns with non-negligible diagonal elements
    diag_R = np.abs(np.diag(R))
    if len(diag_R) > 0 and diag_R[0] > 0:
        threshold = max(n, p) * np.finfo(np.float64).eps * diag_R[0]
        rank = int(np.sum(diag_R > threshold))
    else:
        rank = 0
    
    return QRResult(Q=Q, R=R, pivot=pivot, rank=rank)


def qr_solve(
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], QRResult]:
    """
    Solve least squares via pivoted QR decomposition.
    
    Solves: min_β ||y - Xβ||²
    
    Args:
        X: Design matrix (n x p)
        y: Response vector (n,)
        
    Returns:
        Tuple of:
        - coefficients: Coefficient vector β (p,) in ORIGINAL column order
        - qr_result: The QRResult for downstream use (e.g., SE computation)
    """
    from scipy.linalg import solve_triangular
    
    p = X.shape[1]
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    qr_result = qr_decompose(X, mode='reduced')
    
    Qty = qr_result.Q.T @ y
    coef_pivoted = solve_triangular(
        qr_result.R[:p, :p],
        Qty[:p],
        lower=False
    )
    
    # Unpivot to original column order
    coef = np.empty(p, dtype=np.float64)
    coef[qr_result.pivot] = coef_pivoted
    
    return coef, qr_result