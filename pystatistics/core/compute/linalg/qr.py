"""
Least-squares via single-pass, rank-revealing Householder QR.

This is the CPU least-squares core that backs OLS (`regression/backends/cpu.py`)
and GLM/IRLS (`regression/backends/cpu_glm.py`), and is available to any other
module that needs an R-equivalent linear solve.

Why a custom solver (CONVENTIONS.md — the Prime Directive). On the CPU path,
speed parity with R is a *requirement*. R's `lm.fit`/`glm.fit` use LINPACK
`dqrls`/`dqrdc2`: a Householder QR with *limited* pivoting — a column is deferred
only when its residual norm collapses (rank deficiency), so one light pass both
reveals rank AND solves. SciPy's column-pivoted QR (LAPACK `dgeqp3`) does full
Businger–Golub column-norm pivoting on every solve, which is strictly heavier and
made CPU OLS/GLM slower than R.

The method here matches R's behaviour with stock LAPACK in a single factorization:

  1. Non-pivoted QR of the augmented matrix ``[X | y]`` via ``numpy.linalg.qr``
     in ``mode='r'`` — Q is never formed. The leading ``p×p`` block of R is the
     factor of X; the last column's first ``p`` entries are ``Qᵀy``.
  2. Rank reveal using R's ``dqrdc2`` criterion, read off the computed factor:
     column ``k`` is dependent iff ``|R[k,k]| < tol · ‖X[:,k]‖`` with ``tol=1e-7``
     (R's `lm.fit` default). ``R[k,k]`` *is* column k's residual norm after the
     prior reflections, and ``‖X[:,k]‖`` is its original norm — so this is exactly
     dqrdc2's deferral test, just evaluated after a plain `dgeqrf`.
  3. Resolve from that single factor — no second factorization. Dropping matching
     row+column indices from an upper-triangular matrix preserves triangularity,
     so ``R[ix_(ind, ind)]`` is the factor of the independent columns and
     ``Qᵀy[ind]`` its right-hand side. Dependent columns are aliased to ``NaN``
     (R reports them as ``NA``).

Implementation note (do not regress): use ``numpy.linalg.qr``, not
``scipy.linalg.qr``. The SciPy wrapper makes internal copies that dominate the
runtime on tall designs (~311 ms vs ~200 ms for the same factorization); the
LAPACK kernel was never the bottleneck.
"""

from dataclasses import dataclass
from typing import Literal, Any, Optional
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.exceptions import DimensionError

# R's lm.fit / glm.fit default relative tolerance for rank detection (dqrdc2).
_RANK_TOL = 1e-7


@dataclass(frozen=True)
class QRResult:
    """
    Result of a rank-revealing QR least-squares factorization.

    Attributes:
        Q: Orthogonal matrix, or ``None``. The least-squares solver never forms
           Q (it applies Qᵀ implicitly via the augmented-R trick), so this is
           ``None`` on that path. ``qr_decompose`` still populates it.
        R: Upper-triangular factor (p x p) in PIVOTED order — the leading
           ``rank x rank`` block is the factor of the independent columns.
        pivot: Permutation array (0-indexed). ``pivot[i]`` is the original column
               index now at position ``i``; independent columns come first (in
               their original relative order), then any dependent columns.
        rank: Numerical rank.
    """
    Q: Optional[NDArray[np.floating[Any]]]
    R: NDArray[np.floating[Any]]
    pivot: NDArray[np.intp]
    rank: int


def qr_decompose(
    X: NDArray[np.floating[Any]],
    mode: Literal['reduced', 'complete'] = 'reduced'
) -> QRResult:
    """
    Column-pivoted QR decomposition using SciPy/LAPACK (``dgeqp3``).

    Retained for callers that need an explicit Q and full Businger–Golub
    pivoting. The least-squares hot path uses :func:`qr_solve`, which is faster
    and does not form Q.

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
    Solve ``min_β ||y - Xβ||²`` via a single-pass, rank-revealing QR.

    Matches R's `lm.fit`/`glm.fit` (LINPACK ``dqrls``) in one factorization:
    reveals rank, solves, and aliases rank-deficient columns to ``NaN`` exactly
    as R reports ``NA``. See the module docstring for the algorithm.

    Args:
        X: Design matrix (n x p). For weighted least squares (IRLS) pass the
           already-weighted design ``√W·X``; rank is then detected on the
           weighted columns, matching R's glm.fit.
        y: Response vector (n,). Pass ``√W·z`` for weighted least squares.

    Returns:
        Tuple of:
        - coefficients: β (p,) in ORIGINAL column order; dependent columns are
          ``NaN``.
        - qr_result: QRResult (Q is ``None``) carrying the pivoted R factor,
          pivot, and rank for downstream SE computation.

    Raises:
        ValueError: if X is not 2-D, y is not 1-D, or their lengths disagree.
    """
    from scipy.linalg import solve_triangular

    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if X.ndim != 2:
        raise DimensionError(f"X must be 2-D, got shape {X.shape}")
    if y.ndim != 1:
        raise DimensionError(f"y must be 1-D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise DimensionError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]} elements"
        )

    n, p = X.shape

    # Degenerate / underdetermined designs (n < p) are not the regression hot
    # path and break the augmented-R row assumptions below. Fall back to the
    # explicit pivoted decomposition, which handles any shape.
    if n < p:
        return _qr_solve_fallback(X, y)

    # Original column norms for R's relative rank-deficiency test (dqrdc2).
    col_norm = np.sqrt(np.einsum('ij,ij->j', X, X))

    # One non-pivoted QR of [X | y]; Q is never formed. The last column of R
    # holds Qᵀy, R[:p, :p] is the factor of X.
    aug = np.empty((n, p + 1), dtype=np.float64)
    aug[:, :p] = X
    aug[:, p] = y
    R_full = np.linalg.qr(aug, mode='r')

    Rxx = R_full[:p, :p]
    Qty = R_full[:p, p]

    diag_R = np.abs(np.diag(Rxx))
    with np.errstate(invalid='ignore'):
        keep = diag_R > (_RANK_TOL * col_norm)
    ind = np.nonzero(keep)[0]
    rank = int(ind.size)

    coef = np.full(p, np.nan, dtype=np.float64)

    if rank == p:
        coef[:] = solve_triangular(Rxx, Qty, lower=False, check_finite=False)
        pivot = np.arange(p, dtype=np.intp)
        R_out = Rxx
    else:
        dep = np.nonzero(~keep)[0]
        pivot = np.concatenate([ind, dep]).astype(np.intp)

        # A plain dgeqrf still forms a (garbage, near-zero) reflection at each
        # dependent column, which corrupts the R rows of any independent column
        # that follows it — so the naive submatrix Rxx[ix_(ind, ind)] is NOT the
        # factor of the independent columns. R's dqrdc2 instead defers dependent
        # columns and never forms those reflections.
        #
        # Recover the clean, R-equivalent pivoted factor by re-triangularizing
        # the reordered R together with Qᵀy. Since Rxx = QᵀX (Q orthogonal),
        # the QR of the reordered Rxx has the same R factor as the QR of the
        # reordered X — but this is a tiny p×p factorization, not n×p.
        aug2 = np.empty((p, p + 1), dtype=np.float64)
        aug2[:, :p] = Rxx[:, pivot]
        aug2[:, p] = Qty
        R2 = np.linalg.qr(aug2, mode='r')

        R_out = R2[:p, :p]
        Qty_piv = R2[:p, p]
        # Leading rank×rank block is the clean factor of the independent block;
        # solve only that (dependent columns stay NaN).
        coef[ind] = solve_triangular(
            R_out[:rank, :rank], Qty_piv[:rank], lower=False,
            check_finite=False,
        )

    return coef, QRResult(Q=None, R=R_out, pivot=pivot, rank=rank)


def _qr_solve_fallback(
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], QRResult]:
    """Pivoted-QR least squares for degenerate shapes (n < p).

    Not the hot path; correctness over speed. Uses the explicit column-pivoted
    decomposition so rank-deficient, underdetermined systems still solve.
    """
    from scipy.linalg import solve_triangular

    p = X.shape[1]
    qr_result = qr_decompose(X, mode='reduced')
    rank = qr_result.rank

    Qty = qr_result.Q.T @ y
    coef = np.full(p, np.nan, dtype=np.float64)
    if rank > 0:
        beta = solve_triangular(
            qr_result.R[:rank, :rank], Qty[:rank], lower=False,
            check_finite=False,
        )
        coef[qr_result.pivot[:rank]] = beta

    return coef, qr_result
