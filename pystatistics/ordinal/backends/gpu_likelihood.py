"""GPU likelihood + gradient + Hessian for cumulative link models (polr).

The CPU path in ``_likelihood.py`` computes the cumulative-link NLL and
its analytical gradient with per-level numpy masks, then builds the
Hessian for vcov via ``scipy.approx_fprime`` on each row of the gradient
— an ``O(n_params²)`` sweep of gradient evaluations that dominates fit
time once the model gets past toy sizes.

On GPU we build the computation graph once for the NLL, use autograd to
obtain the gradient in a single backward pass per L-BFGS-B evaluation,
and use ``torch.autograd.functional.hessian`` for vcov — ``O(n_params)``
backward passes total, all batched over the full sample.

Two-tier validation (see README "Design Philosophy"):
    - CPU is validated against R ``MASS::polr``.
    - GPU is validated against CPU at the ``GPU_FP32`` tier (rtol=1e-4,
      atol=1e-5). FP64 on CUDA matches CPU to machine precision.

The ``PolrGPULikelihood`` class holds the (n, p) design matrix and the
integer ``y_codes`` vector on device between optimizer iterations, so
the bulk data crosses the PCIe bus once per fit (or zero times on the
amortized DataSource path), not once per scipy evaluation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


class PolrGPULikelihood:
    """Stateful holder of GPU tensors for a cumulative link fit.

    Mirrors ``MultinomialGPULikelihood``: scipy's L-BFGS-B wants ``fun``
    and ``jac`` as two separate callables, so back-to-back calls with
    the same ``params`` share one forward+backward pass via an internal
    cache.
    """

    def __init__(
        self,
        X: NDArray[np.floating[Any]] | Any,   # numpy OR torch.Tensor
        y_codes: NDArray[np.integer[Any]],
        n_levels: int,
        link_name: str,
        device: str = "cuda",
        use_fp64: bool = False,
    ) -> None:
        import torch

        if device == "mps" and use_fp64:
            raise RuntimeError(
                "GPU polr: MPS does not support FP64. "
                "Use use_fp64=False or backend='cpu'."
            )

        self._torch = torch
        self._device = torch.device(device)
        self._dtype = torch.float64 if use_fp64 else torch.float32

        n, p = X.shape
        self._n = n
        self._p = p
        self._n_levels = n_levels
        self._n_thresh = n_levels - 1
        # Accept either the polr method name ('logistic') or the Link
        # class's own ``.name`` attribute ('logit') — they mean the same
        # distribution in polr.
        link_lc = link_name.lower()
        if link_lc in ("logistic", "logit"):
            self._link_name = "logistic"
        elif link_lc == "probit":
            self._link_name = "probit"
        elif link_lc == "cloglog":
            self._link_name = "cloglog"
        else:
            raise RuntimeError(
                f"GPU polr: unsupported link {link_name!r}. "
                "Supported: logistic, probit, cloglog."
            )

        # Accept numpy or device-resident torch.Tensor for X. In the
        # tensor case we only cast / move if dtype or device differs,
        # to avoid an extra GPU copy on the amortized DataSource path.
        if isinstance(X, torch.Tensor):
            if X.device != self._device or X.dtype != self._dtype:
                self._X = X.to(device=self._device, dtype=self._dtype)
            else:
                self._X = X
        else:
            self._X = torch.as_tensor(
                X, device=self._device, dtype=self._dtype,
            )
        self._y_codes = torch.as_tensor(
            y_codes, device=self._device, dtype=torch.long,
        )

        # Per-call cache: last (params, nll, grad) tuple, reused by
        # back-to-back fun/jac calls with the same params.
        self._cache_params: NDArray | None = None
        self._cache_nll: float | None = None
        self._cache_grad: NDArray | None = None

    # ---- link inverse (torch-native) ------------------------------------

    def _linkinv(self, x):
        """Inverse link. ``x`` is any broadcast-compatible torch tensor."""
        torch = self._torch
        if self._link_name == "logistic":
            return torch.sigmoid(x)
        if self._link_name == "probit":
            # Normal CDF = 0.5 * (1 + erf(x / sqrt(2))).
            return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        # cloglog: 1 - exp(-exp(eta)). Clamp to mirror the CPU path's
        # overflow guard on ``eta``.
        x_clamped = torch.clamp(x, min=-500.0, max=500.0)
        return 1.0 - torch.exp(-torch.exp(x_clamped))

    # ---- core NLL computation -------------------------------------------

    def _nll_from_params(self, params_gpu):
        """NLL as a scalar torch tensor, autograd-friendly."""
        torch = self._torch
        n_thresh = self._n_thresh

        raw = params_gpu[:n_thresh]
        beta = params_gpu[n_thresh:]

        # Unconstrained threshold parameterization (see
        # ``raw_to_thresholds`` in _likelihood.py):
        #   alpha_0 = raw_0
        #   alpha_j = alpha_{j-1} + exp(raw_j)  for j >= 1
        # Written as cumsum of (raw_0, exp(raw_1), ..., exp(raw_{K-2})).
        increments = torch.cat([raw[:1], torch.exp(raw[1:])], dim=0)
        alpha = torch.cumsum(increments, dim=0)                  # (K-1,)

        eta = self._X @ beta                                     # (n,)
        # cum_args[i, j] = alpha_j - eta_i
        cum_args = alpha.unsqueeze(0) - eta.unsqueeze(1)         # (n, K-1)
        cum_probs = self._linkinv(cum_args)                      # (n, K-1)

        # Pad to (n, K+1) with 0 on the left, 1 on the right so that
        # P(Y=j|x) = cum_ext[:, j+1] - cum_ext[:, j] for all j. This
        # avoids the CPU path's "first / middle / last" case split and
        # plays nicely with autograd.
        zeros_col = torch.zeros_like(cum_probs[:, :1])
        ones_col = torch.ones_like(cum_probs[:, :1])
        cum_ext = torch.cat([zeros_col, cum_probs, ones_col], dim=1)
        cat_probs = cum_ext[:, 1:] - cum_ext[:, :-1]             # (n, K)

        prob = cat_probs.gather(
            1, self._y_codes.unsqueeze(1),
        ).squeeze(1)
        prob = torch.clamp(prob, min=1e-15)
        return -torch.sum(torch.log(prob))

    # ---- fun / jac cache driver -----------------------------------------

    def _compute(
        self, params_flat: NDArray[np.floating[Any]],
    ) -> tuple[float, NDArray[np.floating[Any]]]:
        """Forward + backward on GPU. Returns (nll, grad as flat float64)."""
        torch = self._torch
        params_gpu = torch.as_tensor(
            params_flat, device=self._device, dtype=self._dtype,
        ).detach().clone().requires_grad_(True)

        nll_t = self._nll_from_params(params_gpu)
        nll_t.backward()
        nll_val = float(nll_t.detach().cpu().item())
        grad_val = params_gpu.grad.detach().to(torch.float64).cpu().numpy()
        return nll_val, grad_val

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
        """Negative log-likelihood (scipy-compatible callable)."""
        self._ensure_cached(params_flat)
        return float(self._cache_nll)   # type: ignore[arg-type]

    def jac(
        self, params_flat: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Gradient (scipy-compatible callable)."""
        self._ensure_cached(params_flat)
        return self._cache_grad          # type: ignore[return-value]

    # ---- post-fit quantities --------------------------------------------

    def compute_vcov(
        self, params_flat: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Hessian → inverse → vcov, fully on GPU.

        The CPU path in ``_solver._compute_vcov`` runs
        ``scipy.approx_fprime`` on each component of the gradient — an
        O(n_params²) sweep of finite-differenced gradient evaluations.
        On GPU we use ``torch.autograd.functional.hessian`` which does
        one backward pass per parameter (O(n_params)), all batched over
        the n-row data tensor. For MASS::housing-scale data this cuts
        a ~300 ms vcov step down to milliseconds.
        """
        torch = self._torch
        from torch.autograd.functional import hessian as torch_hessian

        params_gpu = torch.as_tensor(
            params_flat, device=self._device, dtype=self._dtype,
        ).detach().clone()

        H = torch_hessian(
            self._nll_from_params, params_gpu, create_graph=False,
        )
        # Symmetrize defensively (autograd output can be off by 1 ulp).
        H = 0.5 * (H + H.T)

        try:
            vcov = torch.linalg.inv(H)
        except RuntimeError:
            vcov = torch.linalg.pinv(H)

        return vcov.to(torch.float64).cpu().numpy()

    def compute_log_lik(
        self, params_flat: NDArray[np.floating[Any]],
    ) -> float:
        """Final log-likelihood using the device-resident state."""
        torch = self._torch
        with torch.no_grad():
            params_gpu = torch.as_tensor(
                params_flat, device=self._device, dtype=self._dtype,
            )
            nll_t = self._nll_from_params(params_gpu)
        return -float(nll_t.detach().cpu().item())
