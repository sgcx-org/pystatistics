"""
CPU reference backend for linear regression.
"""

from typing import Any
import numpy as np

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.core.compute.linalg.qr import qr_solve_cpu, qr_cpu
from pystatistics.regression.design import Design
from pystatistics.regression.solution import LinearParams


class CPUQRBackend:
    """CPU backend using QR decomposition."""
    
    @property
    def name(self) -> str:
        return 'cpu_qr'
    
    def solve(self, design: Design) -> Result[LinearParams]:
        timer = Timer()
        timer.start()
        
        X, y = design.X, design.y
        n, p = design.n, design.p
        
        with timer.section('qr_decomposition'):
            qr_result = qr_cpu(X, mode='reduced')
            
        with timer.section('solve'):
            coefficients = qr_solve_cpu(X, y, check_rank=True)
        
        with timer.section('residuals'):
            fitted_values = X @ coefficients
            residuals = y - fitted_values
        
        with timer.section('statistics'):
            rss = float(residuals @ residuals)
            tss = float(np.sum((y - np.mean(y)) ** 2))
        
        timer.stop()
        
        params = LinearParams(
            coefficients=coefficients,
            residuals=residuals,
            fitted_values=fitted_values,
            rss=rss,
            tss=tss,
            rank=qr_result.rank,
            df_residual=n - qr_result.rank,
        )
        
        return Result(
            params=params,
            info={'method': 'qr', 'rank': qr_result.rank},
            timing=timer.result(),
            backend_name=self.name,
            warnings=(),
        )
