"""
NumPy forward-Cholesky backend for direct MVN MLE.

Runs the forward-Cholesky estimator (``NumpyObjectiveFP64``) through the shared
scipy optimiser path. It needs only NumPy and SciPy, and is the direct-method
CPU backend used when PyTorch is not installed. Defaults (BFGS, per-observation
gradient tolerance 1e-5, 100 iterations) and the result metadata match the
PyTorch FP64 CPU path.
"""

from pystatistics.core.result import Result
from pystatistics.mvnmle.design import MVNDesign
from pystatistics.mvnmle.solution import MVNParams
from pystatistics.mvnmle.backends._direct import run_direct_solve


class NumpyCholeskyMLEBackend:
    """Forward-Cholesky direct MVN MLE in float64 NumPy (CPU, no PyTorch)."""

    DEFAULT_METHOD = 'BFGS'
    DEFAULT_TOL = 1e-5

    @property
    def name(self) -> str:
        return 'cpu_cholesky_numpy_fp64'

    def solve(
        self,
        design: MVNDesign,
        *,
        method: str | None = None,
        tol: float | None = None,
        max_iter: int = 100,
    ) -> Result[MVNParams]:
        """
        Solve the MVN MLE.

        Parameters
        ----------
        design : MVNDesign
            Data design wrapper.
        method : str or None
            scipy optimiser method. Default 'BFGS'.
        tol : float or None
            Per-observation gradient tolerance. Default 1e-5.
        max_iter : int
            Maximum optimiser iterations.

        Returns
        -------
        Result[MVNParams]
        """
        from pystatistics.mvnmle._objectives.numpy_fp64 import NumpyObjectiveFP64

        return run_direct_solve(
            lambda: NumpyObjectiveFP64(design.data),
            method=method or self.DEFAULT_METHOD,
            tol=tol or self.DEFAULT_TOL,
            max_iter=max_iter,
            backend_name=self.name,
            parameterization='cholesky',
            device='cpu',
            precision='fp64',
        )
