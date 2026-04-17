"""GPU backend for Whittle approximate MLE of ARMA models.

The CPU path in ``_whittle.py`` does one real FFT, one elementwise
``|MA|²/|AR|²`` per frequency, and a small reduction — zero
per-iteration state. Every one of those operations is a GPU sweet
spot. The win is particularly large for long series (n ≫ 10⁴) where
``torch.fft.rfft`` is much faster than numpy's pocketfft and the
subsequent reductions run in effectively one kernel each.

This class mirrors ``MultinomialGPULikelihood`` / ``PolrGPULikelihood``:
scipy L-BFGS-B wants ``fun`` and ``jac`` as two separate callables, so
we cache the forward+backward pass on each unique ``params_flat`` and
serve both from the cache.

Two-tier validation:
    - The CPU Whittle path is validated against its own closed-form
      (FFT-based) CPU reference and — on long stationary series —
      against the exact-ML arima path to within Whittle's O(1/n)
      approximation floor.
    - GPU Whittle is validated against the CPU Whittle path at the
      GPU_FP32 tier on log-likelihood and sigma² (rtol=1e-4,
      atol=1e-5); FP64 on CUDA matches to machine precision.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


class WhittleGPULikelihood:
    """Stateful holder of GPU tensors for a Whittle ARMA fit.

    The periodogram ``I(ω_k)`` and the Fourier frequencies ``ω_k`` are
    computed once on device at construction; every subsequent L-BFGS-B
    evaluation is just one (p + q)-coefficient elementwise build of
    ``log|AR|²`` and ``log|MA|²`` plus two reductions.
    """

    def __init__(
        self,
        y: NDArray[np.floating[Any]] | Any,   # numpy or torch.Tensor
        p: int,
        q: int,
        device: str = "cuda",
        use_fp64: bool = False,
    ) -> None:
        import torch

        if device == "mps" and use_fp64:
            raise RuntimeError(
                "GPU Whittle: MPS does not support FP64. "
                "Use use_fp64=False or backend='cpu'."
            )

        self._torch = torch
        self._device = torch.device(device)
        self._dtype = torch.float64 if use_fp64 else torch.float32
        self._p = p
        self._q = q

        # Accept numpy or device-resident tensor for y. Centre on device
        # so the DC bin is numerically the mean component — skipping it
        # in the periodogram then really drops the mean contribution.
        if isinstance(y, torch.Tensor):
            y_gpu = y.to(device=self._device, dtype=self._dtype)
        else:
            y_gpu = torch.as_tensor(
                y, device=self._device, dtype=self._dtype,
            )
        y_gpu = y_gpu - y_gpu.mean()
        n = y_gpu.shape[0]
        self._n = n

        # Periodogram at non-zero, non-Nyquist frequencies.
        fft_y = torch.fft.rfft(y_gpu)
        spec = (fft_y.real * fft_y.real + fft_y.imag * fft_y.imag) / n
        m = (n - 1) // 2
        self._periodogram = spec[1 : 1 + m]   # (m,)
        self._m = m

        freqs = 2.0 * math.pi * torch.arange(
            1, m + 1, device=self._device, dtype=self._dtype,
        ) / n
        self._freqs = freqs  # (m,)

        # Precompute the j = 1..max(p, q) lag tensor once — used to
        # build the e^{-ijω} matrix shared by both AR and MA evals.
        k = max(p, q, 1)
        j = torch.arange(
            1, k + 1, device=self._device, dtype=self._dtype,
        )
        # (m, k) complex exponentials. Store real and imag parts
        # separately to avoid complex-tensor autograd quirks on some
        # torch builds; we only use the `dtype`-real path.
        angle = -freqs.unsqueeze(1) * j.unsqueeze(0)       # (m, k)
        self._cos = torch.cos(angle)
        self._sin = torch.sin(angle)

        self._cache_params: NDArray | None = None
        self._cache_nll: float | None = None
        self._cache_grad: NDArray | None = None

    # ---- core likelihood ------------------------------------------------

    def _nll_from_params(self, params_gpu):
        """Concentrated Whittle NLL as a scalar torch tensor."""
        torch = self._torch
        p = self._p
        q = self._q

        # Split params into φ and θ.
        phi = params_gpu[:p]
        theta = params_gpu[p : p + q]

        # AR(e^{iω}) = 1 - Σ φ_j e^{-ijω}.  Use the precomputed (m, k)
        # cos/sin tables. Real part: 1 - Σ φ_j cos(jω); imag: -Σ φ_j sin(jω).
        if p > 0:
            cos_p = self._cos[:, :p]
            sin_p = self._sin[:, :p]
            ar_re = 1.0 - (cos_p * phi.unsqueeze(0)).sum(dim=1)
            ar_im = -(sin_p * phi.unsqueeze(0)).sum(dim=1)
            log_ar_mag2 = torch.log(ar_re * ar_re + ar_im * ar_im)
        else:
            log_ar_mag2 = torch.zeros_like(self._freqs)

        # MA(e^{iω}) = 1 + Σ θ_j e^{-ijω}.
        if q > 0:
            cos_q = self._cos[:, :q]
            sin_q = self._sin[:, :q]
            ma_re = 1.0 + (cos_q * theta.unsqueeze(0)).sum(dim=1)
            ma_im = (sin_q * theta.unsqueeze(0)).sum(dim=1)
            log_ma_mag2 = torch.log(ma_re * ma_re + ma_im * ma_im)
        else:
            log_ma_mag2 = torch.zeros_like(self._freqs)

        log_g = log_ma_mag2 - log_ar_mag2
        log_g = torch.clamp(log_g, min=-50.0, max=50.0)
        g = torch.exp(log_g)

        mean_I_over_g = (self._periodogram / g).mean()
        # Concentrated NLL: m · log(σ²_hat / (2π)) + Σ log g, dropping
        # additive constants that don't affect the minimizer.
        return self._m * torch.log(
            torch.clamp(mean_I_over_g, min=1e-20)
        ) + log_g.sum()

    def _compute(
        self, params_flat: NDArray[np.floating[Any]],
    ) -> tuple[float, NDArray[np.floating[Any]]]:
        torch = self._torch
        params_gpu = torch.as_tensor(
            params_flat, device=self._device, dtype=self._dtype,
        ).detach().clone().requires_grad_(True)
        nll_t = self._nll_from_params(params_gpu)
        nll_t.backward()
        return (
            float(nll_t.detach().cpu().item()),
            params_gpu.grad.detach().to(torch.float64).cpu().numpy(),
        )

    def _ensure_cached(self, params_flat: NDArray[np.floating[Any]]) -> None:
        if (
            self._cache_params is not None
            and np.array_equal(self._cache_params, params_flat)
        ):
            return
        nll, grad = self._compute(params_flat)
        self._cache_params = params_flat.copy()
        self._cache_nll = nll
        self._cache_grad = grad

    def fun(self, params_flat: NDArray[np.floating[Any]]) -> float:
        self._ensure_cached(params_flat)
        return float(self._cache_nll)   # type: ignore[arg-type]

    def jac(
        self, params_flat: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        self._ensure_cached(params_flat)
        return self._cache_grad         # type: ignore[return-value]

    # ---- post-fit ------------------------------------------------------

    def sigma2(self, params_flat: NDArray[np.floating[Any]]) -> float:
        """Back out σ² from the converged parameters.

        σ² = 2π · mean(I(ω) / g(ω))  where g = |MA|²/|AR|².
        """
        torch = self._torch
        p = self._p
        q = self._q
        with torch.no_grad():
            params_gpu = torch.as_tensor(
                params_flat, device=self._device, dtype=self._dtype,
            )
            phi = params_gpu[:p]
            theta = params_gpu[p : p + q]
            if p > 0:
                cos_p = self._cos[:, :p]
                sin_p = self._sin[:, :p]
                ar_re = 1.0 - (cos_p * phi.unsqueeze(0)).sum(dim=1)
                ar_im = -(sin_p * phi.unsqueeze(0)).sum(dim=1)
                log_ar_mag2 = torch.log(ar_re * ar_re + ar_im * ar_im)
            else:
                log_ar_mag2 = torch.zeros_like(self._freqs)
            if q > 0:
                cos_q = self._cos[:, :q]
                sin_q = self._sin[:, :q]
                ma_re = 1.0 + (cos_q * theta.unsqueeze(0)).sum(dim=1)
                ma_im = (sin_q * theta.unsqueeze(0)).sum(dim=1)
                log_ma_mag2 = torch.log(ma_re * ma_re + ma_im * ma_im)
            else:
                log_ma_mag2 = torch.zeros_like(self._freqs)
            log_g = torch.clamp(
                log_ma_mag2 - log_ar_mag2, min=-50.0, max=50.0,
            )
            g = torch.exp(log_g)
            # Convention matched with the CPU ``_whittle.py`` path:
            # f(ω) = σ² · g(ω) on the discrete Fourier grid, so
            # σ² = mean(I / g).  See CPU module for derivation.
            s2 = (self._periodogram / g).mean()
        return float(s2.detach().cpu().item())
