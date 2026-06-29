"""
CPU reference backend for linear regression.

Uses the single-pass, rank-revealing QR least-squares solver (`qr_solve`) to
match R's lm() exactly — see core/compute/linalg/qr.py.
"""

from typing import Any
import numpy as np

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.core.compute.linalg.qr import qr_solve
from pystatistics.regression.design import Design
from pystatistics.regression.solution import LinearParams


class CPUQRBackend:
    """CPU backend using QR decomposition with column pivoting."""
    
    @property
    def name(self) -> str:
        return 'cpu_qr'
    
    def solve(
        self,
        design: Design,
        weights: 'np.ndarray | None' = None,
        offset: 'np.ndarray | None' = None,
    ) -> Result[LinearParams]:
        """Fit OLS, or weighted least squares when ``weights`` is given.

        ``weights`` are per-observation prior weights (WLS): the fit minimizes
        Σ wᵢ(yᵢ − xᵢ·β − offsetᵢ)². ``offset`` is a fixed additive term in the
        linear predictor (η = Xβ + offset), not estimated. Both ``None`` is the
        plain-OLS fast path. The stored QR factor is of the weighted design, so
        the downstream (XᵀWX)⁻¹ standard errors are correct for free.
        """
        timer = Timer()
        timer.start()

        X, y = design.X, design.y
        n, p = design.n, design.p

        off = offset
        y_fit = y if off is None else y - off

        with timer.section('qr_solve'):
            if weights is None:
                coefficients, qr_result = qr_solve(X, y_fit)
            else:
                sqrt_w = np.sqrt(weights)
                coefficients, qr_result = qr_solve(
                    X * sqrt_w[:, np.newaxis], y_fit * sqrt_w
                )

        with timer.section('residuals'):
            fitted_values = X @ coefficients
            if off is not None:
                fitted_values = fitted_values + off
            residuals = y - fitted_values

        with timer.section('statistics'):
            # R² as R's summary.lm() defines it: mss / (mss + rss), with the
            # model SS taken about the (weighted) mean of the FITTED values.
            # tss = mss + rss, so r_squared = 1 - rss/tss reproduces it. With no
            # offset this equals the usual Σw(y-ȳ)²; with an offset the residual
            # is not weighted-orthogonal to the offset, so the two differ and
            # this (R's) definition is the correct one.
            if weights is None:
                rss = float(residuals @ residuals)
                f_mean = float(np.mean(fitted_values))
                mss = float(np.sum((fitted_values - f_mean) ** 2))
            else:
                rss = float(np.sum(weights * residuals ** 2))
                sw = float(np.sum(weights))
                f_mean = float(np.sum(weights * fitted_values) / sw)
                mss = float(np.sum(weights * (fitted_values - f_mean) ** 2))
            tss = mss + rss

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
            info={
                'method': 'qr_rank_revealing',
                'rank': qr_result.rank,
                'pivot': qr_result.pivot.tolist(),
                'R': qr_result.R,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=(),
        )