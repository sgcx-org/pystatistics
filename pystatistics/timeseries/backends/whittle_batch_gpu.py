"""Batched GPU Whittle ARMA fit.

Users doing per-series ARMA fits (panel data, financial asset
universes, sensor fleets) pay one outer Python call per series with
the non-batched ``arima()``. For K = 100 series at n = 1000 that's
100 × (rFFT + 40 L-BFGS-B evaluations) = a lot of Python-level
overhead plus K independent CUDA launches per evaluation.

This module fits all K series simultaneously on GPU. One batched
``torch.fft.rfft`` computes the (K, m) periodogram in a single call;
per-iteration the batched NLL evaluates the (K, m) log-spectrum in
one elementwise kernel, and the sum-of-K-NLLs scalar's backward
pass produces the full (K, p+q) gradient at once (the independence
of per-series losses means the off-diagonal Hessian blocks are
exactly zero — no gradient bleed between series).

scipy L-BFGS-B can't wrap a batched objective cleanly (each series
would need its own line search), so we use a batched Adam optimizer
instead. Adam converges reliably on the smooth concentrated Whittle
NLL with Yule-Walker starts; a per-row gradient-norm convergence
check freezes series that have converged so Adam doesn't keep
wiggling them.

Two-tier validation: against the single-series Whittle path (the
``method='Whittle'`` in ``arima()``), which itself is validated
against exact time-domain ML. GPU FP32 matches CPU Whittle at the
``GPU_FP32`` tier on σ² and log-likelihood.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


class BatchedWhittleGPU:
    """Fit K independent ARMA(p, q) models in parallel on GPU.

    Public API:
        ``fit()`` returns ``(phi, theta, sigma2, n_iter, converged)``
        arrays / tensors of batch-leading shape ``(K, …)``.

    State stored on device:
        - periodogram ``(K, m)`` from one batched rFFT at construction
        - Fourier-frequency cos/sin tables ``(m, k)`` shared across
          the batch
        - mean-subtraction vector ``(K,)`` so ``sigma2`` recovery uses
          the centred series
    """

    def __init__(
        self,
        Y: NDArray[np.floating[Any]] | Any,     # (K, n) numpy or torch.Tensor
        p: int,
        q: int,
        device: str = "cuda",
        use_fp64: bool = False,
    ) -> None:
        import torch

        if device == "mps" and use_fp64:
            raise RuntimeError(
                "Batched GPU Whittle: MPS does not support FP64. "
                "Use use_fp64=False or backend='cpu'."
            )

        self._torch = torch
        self._device = torch.device(device)
        self._dtype = torch.float64 if use_fp64 else torch.float32
        self._p = p
        self._q = q

        if isinstance(Y, torch.Tensor):
            Y_gpu = Y.to(device=self._device, dtype=self._dtype)
        else:
            Y_gpu = torch.as_tensor(
                Y, device=self._device, dtype=self._dtype,
            )
        if Y_gpu.ndim != 2:
            raise ValueError(
                f"Batched Whittle: Y must be 2-D (K, n), got {Y_gpu.ndim}-D"
            )

        K, n = Y_gpu.shape
        self._K = K
        self._n = n

        # Centre per-series so the DC bin (which we drop) is exactly
        # the per-series mean. One reduction over dim=1.
        self._mu_batch = Y_gpu.mean(dim=1)        # (K,)
        Y_centered = Y_gpu - self._mu_batch.unsqueeze(1)

        # One batched rFFT → (K, n//2+1) complex. Periodogram magnitude,
        # drop DC and Nyquist bins.
        fft_y = torch.fft.rfft(Y_centered, dim=1)
        spec = (fft_y.real * fft_y.real + fft_y.imag * fft_y.imag) / n
        m = (n - 1) // 2
        self._periodogram = spec[:, 1 : 1 + m]     # (K, m)
        self._m = m

        # Shared Fourier-frequency cos/sin tables (no K dim — identical
        # across the batch).
        freqs = 2.0 * math.pi * torch.arange(
            1, m + 1, device=self._device, dtype=self._dtype,
        ) / n
        self._freqs = freqs
        k_max = max(p, q, 1)
        j = torch.arange(
            1, k_max + 1, device=self._device, dtype=self._dtype,
        )
        angle = -freqs.unsqueeze(1) * j.unsqueeze(0)
        self._cos = torch.cos(angle)               # (m, k_max)
        self._sin = torch.sin(angle)

    # -------- batched NLL --------------------------------------------

    def _log_g_batched(self, phi_batch, theta_batch):
        """Return ``log g(ω)`` batched over K: shape ``(K, m)``.

        ``phi_batch`` is ``(K, p)``, ``theta_batch`` is ``(K, q)``.
        The einsum builds ``Σ_j φ_{k,j} cos(jω)`` in one pass per
        coefficient type — fully GPU-parallel over K, m, j.
        """
        torch = self._torch
        p = self._p
        q = self._q

        if p > 0:
            ar_re = 1.0 - torch.einsum(
                "kp,mp->km", phi_batch, self._cos[:, :p],
            )
            ar_im = -torch.einsum(
                "kp,mp->km", phi_batch, self._sin[:, :p],
            )
            log_ar_mag2 = torch.log(ar_re * ar_re + ar_im * ar_im)
        else:
            log_ar_mag2 = torch.zeros(
                (self._K, self._m),
                device=self._device, dtype=self._dtype,
            )
        if q > 0:
            ma_re = 1.0 + torch.einsum(
                "kq,mq->km", theta_batch, self._cos[:, :q],
            )
            ma_im = torch.einsum(
                "kq,mq->km", theta_batch, self._sin[:, :q],
            )
            log_ma_mag2 = torch.log(ma_re * ma_re + ma_im * ma_im)
        else:
            log_ma_mag2 = torch.zeros(
                (self._K, self._m),
                device=self._device, dtype=self._dtype,
            )
        return log_ma_mag2 - log_ar_mag2

    def _nll_per_series(self, params_batch):
        """Concentrated Whittle NLL per series: shape ``(K,)``."""
        torch = self._torch
        p = self._p
        q = self._q
        phi = params_batch[:, :p]
        theta = params_batch[:, p : p + q]
        log_g = self._log_g_batched(phi, theta)
        log_g = torch.clamp(log_g, min=-50.0, max=50.0)
        g = torch.exp(log_g)
        mean_I_over_g = (self._periodogram / g).mean(dim=1)        # (K,)
        return (
            self._m * torch.log(torch.clamp(mean_I_over_g, min=1e-20))
            + log_g.sum(dim=1)
        )

    # -------- batched optimizer --------------------------------------

    def fit(
        self,
        start_params_batch: NDArray[np.floating[Any]],
        *,
        lr: float = 0.05,
        max_iter: int = 300,
        tol: float = 1e-5,
    ) -> tuple[NDArray, NDArray, NDArray, int, NDArray]:
        """Run batched Adam on ``(K, p+q)`` params.

        A per-series convergence flag freezes rows once their gradient
        L∞-norm falls below ``tol``; "freezing" means the Adam state
        for that row stops updating so we don't keep wiggling around
        its optimum while the harder series are still improving.
        Returns ``(phi, theta, sigma2, n_iter, converged)`` with
        ``n_iter`` = the maximum iteration count across series.
        """
        torch = self._torch
        p = self._p
        q = self._q
        n_params = p + q

        if n_params == 0:
            # ARMA(0, 0) batched — σ² per series = var(y_k). No Adam.
            sigma2 = self._periodogram.mean(dim=1).detach()
            zeros_K0 = np.zeros((self._K, 0), dtype=np.float64)
            return (
                zeros_K0, zeros_K0,
                sigma2.to(torch.float64).cpu().numpy(),
                0,
                np.ones(self._K, dtype=bool),
            )

        params = torch.as_tensor(
            start_params_batch, device=self._device, dtype=self._dtype,
        ).detach().clone().requires_grad_(True)

        # Adam state — same broadcast rules as the usual Adam update.
        m_state = torch.zeros_like(params)
        v_state = torch.zeros_like(params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        converged = torch.zeros(
            (self._K,), device=self._device, dtype=torch.bool,
        )
        n_iter = 0

        for it in range(1, max_iter + 1):
            nll_per_k = self._nll_per_series(params)
            # Sum of independent per-series losses → backward gives
            # per-row gradients (off-diagonal Hessian between series
            # is exactly zero for this objective).
            total = nll_per_k.sum()
            if params.grad is not None:
                params.grad = None
            total.backward()
            grad = params.grad.detach()

            # Per-row gradient-norm convergence. We use L∞ because it's
            # scale-invariant across different series' parameter magnitudes.
            grad_norm = grad.abs().amax(dim=1)                     # (K,)
            just_converged = (grad_norm < tol) & ~converged
            converged = converged | just_converged

            # Mask frozen rows so their Adam state stops advancing.
            active = (~converged).to(self._dtype).unsqueeze(1)    # (K, 1)
            m_state = beta1 * m_state + (1 - beta1) * grad * active
            v_state = beta2 * v_state + (1 - beta2) * grad * grad * active
            m_hat = m_state / (1 - beta1 ** it)
            v_hat = v_state / (1 - beta2 ** it)
            with torch.no_grad():
                params -= lr * active * m_hat / (torch.sqrt(v_hat) + eps)

            n_iter = it
            if bool(converged.all().item()):
                break

        # Recover per-series σ² from the concentrated form.
        with torch.no_grad():
            phi = params[:, :p].detach()
            theta = params[:, p : p + q].detach()
            log_g = self._log_g_batched(phi, theta)
            log_g = torch.clamp(log_g, min=-50.0, max=50.0)
            g = torch.exp(log_g)
            sigma2 = (self._periodogram / g).mean(dim=1)

        return (
            phi.to(torch.float64).cpu().numpy(),
            theta.to(torch.float64).cpu().numpy(),
            sigma2.to(torch.float64).cpu().numpy(),
            n_iter,
            converged.cpu().numpy(),
        )
