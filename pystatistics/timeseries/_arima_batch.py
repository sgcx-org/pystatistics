"""Batched ARIMA fitting over K independent time series.

``arima_batch(Y, order=(p, d, q))`` fits K ARMA(p, q) models to the
rows of a (K, n) matrix simultaneously. Non-seasonal, Whittle-only
for 1.9.0 — the Kalman / CSS paths are not yet batched.

On GPU, one batched rFFT + one batched Adam loop fits all K series
together (see :mod:`backends/whittle_batch_gpu`). On CPU the
default backend falls back to a Python loop over the existing
single-series :func:`arima` with ``method='Whittle'``. The GPU path
is the one that makes this worth shipping — expected 10-100×
speedups for K ≳ 50.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import Result, SolutionReprMixin
from pystatistics.core.compute.backend import (
    resolve_backend, valid_backends, unknown_backend_message,
    FP64_REQUIRES_CUDA_MSG,
)


@dataclass(frozen=True)
class ARMABatchParams:
    """Immutable parameter payload for a batched ARMA fit.

    Attributes
    ----------
    order : tuple[int, int, int]
        ``(p, d, q)``. The ``d`` that was applied before the ARMA fit.
    ar : NDArray
        AR coefficients, shape ``(K, p)``.
    ma : NDArray
        MA coefficients, shape ``(K, q)``.
    sigma2 : NDArray
        Innovation variance per series, shape ``(K,)``.
    mean : NDArray | None
        Per-series sample mean of the differenced series (None if
        ``include_mean=False``).
    n_iter : int
        Maximum iteration count across all series.
    converged : NDArray
        Per-series boolean convergence flag, shape ``(K,)``.
    n_series : int
        Number of series K.
    n_used : int
        Length of each (post-differencing) series.
    method : str
        Which fitter ran: ``'Whittle-batch-GPU'``, ``'Whittle-loop-CPU'``.
    """

    order: tuple[int, int, int]
    ar: NDArray
    ma: NDArray
    sigma2: NDArray
    mean: NDArray | None
    n_iter: int
    converged: NDArray
    n_series: int
    n_used: int
    method: str


@dataclass
class ARMABatchSolution(SolutionReprMixin):
    """Result from a batched ARMA fit.

    Wraps a :class:`Result` ``[ARMABatchParams]`` envelope; every datum is
    exposed via a read-only ``@property`` so the public attribute surface is
    unchanged from the previous flat dataclass.

    Attributes
    ----------
    order : tuple[int, int, int]
        ``(p, d, q)``. The ``d`` that was applied before the ARMA fit.
    ar : NDArray
        AR coefficients, shape ``(K, p)``.
    ma : NDArray
        MA coefficients, shape ``(K, q)``.
    sigma2 : NDArray
        Innovation variance per series, shape ``(K,)``.
    mean : NDArray | None
        Per-series sample mean of the differenced series (None if
        ``include_mean=False``).
    n_iter : int
        Maximum iteration count across all series.
    converged : NDArray
        Per-series boolean convergence flag, shape ``(K,)``.
    n_series : int
        Number of series K.
    n_used : int
        Length of each (post-differencing) series.
    method : str
        Which fitter ran: ``'Whittle-batch-GPU'``, ``'Whittle-loop-CPU'``.
    """

    _result: Result[ARMABatchParams]

    @property
    def order(self) -> tuple[int, int, int]:
        return self._result.params.order

    @property
    def ar(self) -> NDArray:
        return self._result.params.ar

    @property
    def ma(self) -> NDArray:
        return self._result.params.ma

    @property
    def sigma2(self) -> NDArray:
        return self._result.params.sigma2

    @property
    def mean(self) -> NDArray | None:
        return self._result.params.mean

    @property
    def n_iter(self) -> int:
        return self._result.params.n_iter

    @property
    def converged(self) -> NDArray:
        return self._result.params.converged

    @property
    def n_series(self) -> int:
        return self._result.params.n_series

    @property
    def n_used(self) -> int:
        return self._result.params.n_used

    @property
    def method(self) -> str:
        return self._result.params.method

    @property
    def info(self) -> dict:
        return self._result.info

    @property
    def timing(self) -> dict[str, float] | None:
        return self._result.timing

    @property
    def backend_name(self) -> str:
        return self._result.backend_name

    @property
    def warnings(self) -> tuple[str, ...]:
        return self._result.warnings

    def summary(self) -> str:
        """Compact summary of a batched ARMA fit across the K series."""
        p, d, q = self.order
        n_conv = int(np.sum(self.converged))
        lines = [
            f"Batched ARIMA({p},{d},{q}) — {self.n_series} series",
            "=" * 56,
            f"Fitter: {self.method}",
            f"Observations per series (after differencing): {self.n_used}",
            f"Converged: {n_conv}/{self.n_series}",
            "",
            f"sigma^2  mean={float(np.mean(self.sigma2)):.4g}  "
            f"min={float(np.min(self.sigma2)):.4g}  "
            f"max={float(np.max(self.sigma2)):.4g}",
        ]
        if p > 0:
            lines.append(
                f"AR[1]    mean={float(np.mean(self.ar[:, 0])):.4f}  "
                f"(across series)"
            )
        if q > 0:
            lines.append(
                f"MA[1]    mean={float(np.mean(self.ma[:, 0])):.4f}  "
                f"(across series)"
            )
        return "\n".join(lines)


def _yule_walker_batch(Y_centered: NDArray, p: int) -> NDArray:
    """Yule-Walker AR(p) starts per series — shape ``(K, p)``.

    Mirrors the single-series ``_yule_walker_start`` used in
    ``_arima_fit.py`` but vectorised across the batch dimension. The
    lag-0 autocovariance drops out of the Toeplitz system so we solve
    a per-series p × p linear system; for typical p ≤ 5 this is
    negligible next to the downstream Adam loop.
    """
    K, n = Y_centered.shape
    if p == 0:
        return np.zeros((K, 0), dtype=np.float64)
    # Per-series autocovariances at lags 0..p.
    y = Y_centered
    gammas = np.empty((K, p + 1), dtype=np.float64)
    for lag in range(p + 1):
        if lag == 0:
            gammas[:, 0] = np.mean(y * y, axis=1)
        else:
            gammas[:, lag] = np.mean(y[:, lag:] * y[:, :-lag], axis=1)
    # Solve Toeplitz system per series. For small p this batched
    # loop is faster than building a (K, p, p) Toeplitz tensor.
    phi = np.zeros((K, p), dtype=np.float64)
    for k in range(K):
        g = gammas[k]
        if g[0] <= 0:
            continue
        R = np.empty((p, p))
        for i in range(p):
            for j in range(p):
                R[i, j] = g[abs(i - j)]
        try:
            phi[k] = np.linalg.solve(R, g[1 : p + 1])
        except np.linalg.LinAlgError:
            phi[k] = 0.0
    return np.clip(phi, -0.99, 0.99)


def _difference_batch(Y: NDArray, d: int) -> NDArray:
    """Apply ``d`` successive lag-1 differences per row."""
    for _ in range(d):
        Y = Y[:, 1:] - Y[:, :-1]
    return Y


def arima_batch(
    Y: ArrayLike,
    *,
    order: tuple[int, int, int] = (0, 0, 0),
    include_mean: bool = True,
    method: str = "Whittle",
    tol: float = 1e-5,
    max_iter: int = 300,
    lr: float = 0.05,
    backend: str | None = None,
) -> ARMABatchSolution:
    """Fit K independent ARMA(p, d, q) models on the rows of ``Y``.

    Parameters
    ----------
    Y : ArrayLike or torch.Tensor
        Shape ``(K, n)``. Each row is an independent time series.
    order : tuple[int, int, int]
        ``(p, d, q)``. Differencing ``d`` is applied per-series
        before the ARMA fit.
    include_mean : bool
        Whether to report the per-series sample mean of the
        differenced series. Default ``True``. Whittle is centred
        internally regardless.
    method : str
        Only ``'Whittle'`` is supported for batch fitting in 1.9.0.
    tol : float
        Per-series gradient-norm (L∞) convergence tolerance.
    max_iter : int
        Maximum Adam iterations (batched) / maximum per-series
        L-BFGS-B iterations (CPU loop).
    lr : float
        Adam learning rate (GPU path only). 0.05 is the default
        tuned for typical Whittle NLL curvature — smaller if you
        see per-series non-convergence, larger if you want faster
        wall time on easy problems.
    backend : str or None
        Compute backend = (device, precision). ``'cpu'`` (default — loop over
        :func:`arima` with ``method='Whittle'``, float64), ``'gpu'`` (float32,
        require GPU), ``'gpu_fp64'`` (float64, CUDA only — raises on MPS),
        ``'auto'`` (GPU-float32 if CUDA present, else CPU loop). A torch.Tensor
        input routes to its own device automatically (see CONVENTIONS.md).

    Returns
    -------
    ARMABatchSolution
    """
    if method != "Whittle":
        raise ValidationError(
            f"arima_batch: only method='Whittle' is supported in 1.9.0, "
            f"got {method!r}. Use arima() in a Python loop for other methods."
        )
    p, d, q = order
    if p < 0 or d < 0 or q < 0:
        raise ValidationError(f"order: non-negative ints, got {order}")

    # Tensor-input → GPU default (shared convention).
    import sys as _sys
    is_tensor = (
        "torch" in _sys.modules
        and isinstance(Y, _sys.modules["torch"].Tensor)
    )
    if is_tensor:
        import torch
        if Y.ndim != 2:
            raise ValidationError(f"Y: expected 2-D, got {Y.ndim}-D")
        if backend in (None, "auto"):
            backend = "gpu" if Y.device.type != "cpu" else "cpu"
        if backend == "cpu":
            raise ValidationError(
                "backend='cpu' was specified but Y is a torch.Tensor on "
                f"device {Y.device}. Move the tensor with .to('cpu') "
                "explicitly, or drop backend= to use the device Y is on."
            )
        if backend not in ("gpu", "gpu_fp64"):
            raise ValidationError(
                unknown_backend_message(backend, valid_backends(True))
            )
        # Precision lives in the backend string; guard fp64-needs-CUDA against
        # the tensor's own device.
        use_fp64 = backend == "gpu_fp64"
        if use_fp64 and Y.device.type != "cuda":
            raise RuntimeError(FP64_REQUIRES_CUDA_MSG)
        gpu_device_type = Y.device.type
        Y_host = Y
    else:
        _target = resolve_backend(backend, supports_fp64=True)
        backend = _target.backend
        use_fp64 = _target.use_fp64
        gpu_device_type = _target.device_type
        Y_host = np.asarray(Y, dtype=np.float64)
        if Y_host.ndim != 2:
            raise ValidationError(f"Y: expected 2-D, got {Y_host.ndim}-D")

    # Per-series differencing. Done on CPU numpy for clarity — it's
    # one vectorised np.diff-equivalent per d-step, negligible cost.
    if is_tensor:
        Y_np = Y_host.detach().cpu().numpy().astype(np.float64)
    else:
        Y_np = Y_host
    Y_diff = _difference_batch(Y_np, d)
    K, n_used = Y_diff.shape
    if K < 1 or n_used < 2 * (p + q + 1) + 4:
        raise ValidationError(
            f"arima_batch: each series needs n >= "
            f"{2 * (p + q + 1) + 4} observations after differencing; "
            f"got n={n_used}."
        )

    # Route to GPU when possible / requested. Device + precision were resolved
    # above (resolve_backend already raised for an explicit GPU request with no
    # GPU and downgraded 'auto' to CPU when no CUDA is present).
    run_gpu = backend != "cpu"
    device_type = gpu_device_type

    if include_mean:
        mu_batch = Y_diff.mean(axis=1)
    else:
        mu_batch = None

    if run_gpu:
        from pystatistics.timeseries.backends.whittle_batch_gpu import (
            BatchedWhittleGPU,
        )
        Y_centered = Y_diff - Y_diff.mean(axis=1, keepdims=True)
        fitter = BatchedWhittleGPU(
            Y_centered if not is_tensor else Y_host,
            p, q, device=device_type, use_fp64=use_fp64,
        )
        # Yule-Walker AR starts; zero-start on AR is unsafe for
        # Whittle (mirror-basin at the reciprocal AR roots).
        ar_start = _yule_walker_batch(Y_centered, p) if p > 0 else np.zeros((K, 0))
        ma_start = np.full((K, q), -0.1) if q > 0 else np.zeros((K, 0))
        start_batch = np.concatenate([ar_start, ma_start], axis=1)
        ar, ma, sigma2, n_iter, converged = fitter.fit(
            start_batch, lr=lr, max_iter=max_iter, tol=tol,
        )
        method_str = f"Whittle-batch-GPU ({device_type}, " \
                     f"{'fp64' if use_fp64 else 'fp32'})"
    else:
        # CPU path: loop over arima() with method='Whittle'. Exactly
        # the same per-series result as the standalone call, just
        # packaged for the batch API. For K ≫ 1 users on CPU this
        # doesn't speed anything up — they should use this API on GPU.
        from pystatistics.timeseries._arima_fit import arima as _arima
        ar_rows: list[NDArray] = []
        ma_rows: list[NDArray] = []
        sigma2_vals: list[float] = []
        conv_vals: list[bool] = []
        n_iter = 0
        for k in range(K):
            res = _arima(
                Y_np[k], order=order, include_mean=include_mean,
                method="Whittle", tol=tol, max_iter=max_iter,
            )
            ar_rows.append(res.ar if p > 0 else np.zeros(0))
            ma_rows.append(res.ma if q > 0 else np.zeros(0))
            sigma2_vals.append(res.sigma2)
            conv_vals.append(res.converged)
            n_iter = max(n_iter, res.n_iter)
        ar = np.array(ar_rows) if p > 0 else np.zeros((K, 0))
        ma = np.array(ma_rows) if q > 0 else np.zeros((K, 0))
        sigma2 = np.array(sigma2_vals)
        converged = np.array(conv_vals, dtype=bool)
        method_str = "Whittle-loop-CPU"

    if run_gpu:
        backend_name = f"whittle_batch_gpu ({device_type}, " \
                       f"{'fp64' if use_fp64 else 'fp32'})"
    else:
        backend_name = "cpu"

    return ARMABatchSolution(
        _result=Result(
            params=ARMABatchParams(
                order=order,
                ar=ar,
                ma=ma,
                sigma2=sigma2,
                mean=mu_batch,
                n_iter=int(n_iter),
                converged=converged,
                n_series=K,
                n_used=n_used,
                method=method_str,
            ),
            info={"method": method_str},
            timing=None,
            backend_name=backend_name,
            warnings=(),
        )
    )
