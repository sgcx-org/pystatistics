"""
CPU reference backend for linear regression.

Uses QR decomposition via LAPACK (through NumPy/SciPy) to solve
the normal equations. This is the reference implementation that
replicates R's lm() behavior.
"""

from typing import Any
import numpy as np

from pystatistics.core.result import Result
from pystatistics.core.backends.timing import Timer
from pystatistics.core.backends.linalg.qr import qr_solve_cpu, qr_cpu
from pystatistics.regression.design import RegressionDesign
from pystatistics.regression.solution import LinearParams


class CPUQRBackend:
    """
    CPU backend using QR decomposition.
    
    Implements the Backend protocol for RegressionDesign -> LinearParams.
    
    This is the reference implementation that should match R's lm() to
    machine precision for regulatory compliance.
    """
    
    @property
    def name(self) -> str:
        return 'cpu_qr'
    
    def solve(self, design: RegressionDesign) -> Result[LinearParams]:
        """
        Solve OLS via QR decomposition.
        
        Algorithm:
            1. Compute QR decomposition: X = QR
            2. Solve: β = R⁻¹ Q'y
            3. Compute residuals, fitted values, and diagnostics
            
        Args:
            design: Validated regression design
            
        Returns:
            Result containing LinearParams
            
        Raises:
            SingularMatrixError: If X is rank-deficient
        """
        timer = Timer()
        timer.start()
        
        X = design.X
        y = design.y
        n, p = design.n, design.p
        
        # === QR Decomposition and Solve ===
        with timer.section('qr_decomposition'):
            qr_result = qr_cpu(X, mode='reduced')
            
        with timer.section('solve'):
            coefficients = qr_solve_cpu(X, y, check_rank=True)
        
        # === Compute Residuals and Fitted Values ===
        with timer.section('residuals'):
            fitted_values = X @ coefficients
            residuals = y - fitted_values
        
        # === Compute Summary Statistics ===
        with timer.section('statistics'):
            rss = float(residuals @ residuals)
            y_mean = np.mean(y)
            tss = float(np.sum((y - y_mean) ** 2))
        
        timer.stop()
        
        # === Construct Result ===
        params = LinearParams(
            coefficients=coefficients,
            residuals=residuals,
            fitted_values=fitted_values,
            rss=rss,
            tss=tss,
            rank=qr_result.rank,
            df_residual=n - qr_result.rank,
        )
        
        info: dict[str, Any] = {
            'method': 'qr',
            'rank': qr_result.rank,
        }
        
        return Result(
            params=params,
            info=info,
            timing=timer.result(),
            backend_name=self.name,
            warnings=(),
        )
