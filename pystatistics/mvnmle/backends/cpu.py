"""
CPU backend for MVN MLE using BFGS optimization.

Uses the R-exact CPUObjectiveFP64 with scipy.optimize.minimize.
"""

from pystatistics.core.result import Result
from pystatistics.mvnmle.design import MVNDesign
from pystatistics.mvnmle.solution import MVNParams
from pystatistics.mvnmle._objectives.cpu import CPUObjectiveFP64
from pystatistics.mvnmle.backends._direct import run_direct_solve


class CPUMLEBackend:
    """
    CPU backend for MVN MLE.

    Uses R-exact inverse Cholesky parameterization with BFGS optimization.
    This is the reference implementation matching R's mvnmle package.
    """

    @property
    def name(self) -> str:
        return 'cpu_bfgs_fp64'

    def solve(
        self,
        design: MVNDesign,
        *,
        method: str = 'BFGS',
        tol: float = 1e-5,
        max_iter: int = 100,
    ) -> Result[MVNParams]:
        """
        Solve MVN MLE using CPU.

        Parameters
        ----------
        design : MVNDesign
            Data design wrapper
        method : str
            Optimization method ('BFGS', 'L-BFGS-B', 'Nelder-Mead', 'Powell')
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations

        Returns
        -------
        Result[MVNParams]
        """
        return run_direct_solve(
            lambda: CPUObjectiveFP64(design.data, validate=False),
            method=method,
            tol=tol,
            max_iter=max_iter,
            backend_name=self.name,
            parameterization='inverse_cholesky',
        )
