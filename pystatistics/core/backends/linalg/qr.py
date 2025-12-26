"""
QR decomposition implementations.

Provides consistent QR decomposition interface across CPU (LAPACK via NumPy)
and GPU (PyTorch). Used by regression, mixed models, and other domains
requiring least squares solutions.
"""

from dataclasses import dataclass
from typing import Literal, Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import SingularMatrixError

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class QRResult:
    """
    Result of QR decomposition.
    
    Attributes:
        Q: Orthogonal matrix (n x k where k = min(n, p) for reduced mode)
        R: Upper triangular matrix (k x p)
        rank: Numerical rank determined from R diagonal
    """
    Q: NDArray[np.floating[Any]]
    R: NDArray[np.floating[Any]]
    rank: int


def qr_cpu(
    X: NDArray[np.floating[Any]],
    mode: Literal['reduced', 'complete'] = 'reduced'
) -> QRResult:
    """
    QR decomposition using LAPACK (via NumPy).
    
    Computes X = QR where Q is orthogonal and R is upper triangular.
    
    Args:
        X: Matrix to decompose (n x p)
        mode: 'reduced' for economy QR (Q is n x k, R is k x p where k = min(n,p))
              'complete' for full QR (Q is n x n, R is n x p)
        
    Returns:
        QRResult with Q, R, and numerical rank
    """
    Q, R = np.linalg.qr(X, mode=mode)
    
    # Determine numerical rank from R diagonal
    diag_R = np.abs(np.diag(R))
    if len(diag_R) > 0 and diag_R[0] > 0:
        # Tolerance based on matrix size and machine epsilon
        tol = max(X.shape) * np.finfo(X.dtype).eps * diag_R[0]
        rank = int(np.sum(diag_R > tol))
    else:
        rank = 0
    
    return QRResult(Q=Q, R=R, rank=rank)


def qr_gpu(
    X: 'torch.Tensor',
    mode: Literal['reduced', 'complete'] = 'reduced'
) -> QRResult:
    """
    QR decomposition using PyTorch (GPU-accelerated).
    
    Args:
        X: Tensor to decompose (n x p), must already be on desired device
        mode: 'reduced' for economy QR, 'complete' for full QR
        
    Returns:
        QRResult with Q, R as NumPy arrays (moved to CPU), and numerical rank
    """
    import torch
    
    Q, R = torch.linalg.qr(X, mode=mode)
    
    # Determine numerical rank
    diag_R = torch.abs(torch.diag(R))
    if len(diag_R) > 0 and diag_R[0] > 0:
        tol = max(X.shape) * torch.finfo(X.dtype).eps * diag_R[0].item()
        rank = int(torch.sum(diag_R > tol).item())
    else:
        rank = 0
    
    return QRResult(
        Q=Q.cpu().numpy(),
        R=R.cpu().numpy(),
        rank=rank
    )


def qr_solve_cpu(
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    check_rank: bool
) -> NDArray[np.floating[Any]]:
    """
    Solve least squares via QR decomposition (CPU).
    
    Solves: min_β ||y - Xβ||² via QR decomposition of X.
    
    The solution is computed as:
        X = QR
        β = R⁻¹ Q'y
    
    Args:
        X: Design matrix (n x p), must have n >= p
        y: Response vector (n,)
        check_rank: If True, raise SingularMatrixError on rank-deficient X
        
    Returns:
        Coefficient vector β (p,)
        
    Raises:
        SingularMatrixError: If X is rank-deficient and check_rank=True
    """
    from scipy.linalg import solve_triangular
    
    n, p = X.shape
    qr_result = qr_cpu(X, mode='reduced')
    
    if check_rank and qr_result.rank < p:
        raise SingularMatrixError(
            f"Design matrix is rank-deficient: rank={qr_result.rank}, expected={p}. "
            f"This indicates perfect multicollinearity.",
            matrix_name='X',
            rank=qr_result.rank,
            expected_rank=p
        )
    
    # β = R⁻¹ Q'y
    # Compute Q'y first, then solve the triangular system
    Qty = qr_result.Q.T @ y
    
    # Solve R @ beta = Qty using back substitution
    # R is p x p upper triangular (for reduced QR with n >= p)
    beta = solve_triangular(qr_result.R[:p, :p], Qty[:p], lower=False)
    
    return beta


def qr_solve_gpu(
    X: 'torch.Tensor',
    y: 'torch.Tensor',
    check_rank: bool
) -> NDArray[np.floating[Any]]:
    """
    Solve least squares via QR decomposition (GPU).
    
    Args:
        X: Design matrix tensor (n x p), on GPU
        y: Response vector tensor (n,), on GPU
        check_rank: If True, raise SingularMatrixError on rank-deficient X
        
    Returns:
        Coefficient vector β as NumPy array (p,)
        
    Raises:
        SingularMatrixError: If X is rank-deficient and check_rank=True
    """
    import torch
    
    n, p = X.shape
    qr_result = qr_gpu(X, mode='reduced')
    
    if check_rank and qr_result.rank < p:
        raise SingularMatrixError(
            f"Design matrix is rank-deficient: rank={qr_result.rank}, expected={p}. "
            f"This indicates perfect multicollinearity.",
            matrix_name='X',
            rank=qr_result.rank,
            expected_rank=p
        )
    
    # Move QR results back to GPU for the solve
    Q_gpu = torch.from_numpy(qr_result.Q).to(X.device)
    R_gpu = torch.from_numpy(qr_result.R).to(X.device)
    
    # β = R⁻¹ Q'y
    Qty = Q_gpu.T @ y
    beta = torch.linalg.solve_triangular(R_gpu[:p, :p], Qty[:p].unsqueeze(1), upper=True)
    
    return beta.squeeze().cpu().numpy()
