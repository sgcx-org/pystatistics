"""GPU likelihood + gradient for multinomial logistic regression.

The CPU path in ``_likelihood.py`` computes the softmax cross-entropy
NLL and its analytical gradient with two separate numpy passes. On
GPU we instead build the full computation graph once, use PyTorch's
``log_softmax`` + autograd to get NLL and gradient in a single
forward+backward, and only pay host↔device transfer cost for the
parameter vector going in and (scalar NLL, gradient vector) coming
out per scipy L-BFGS-B evaluation.

Two-tier validation contract (README "Design Philosophy"):
    - CPU is validated against R ``nnet::multinom``.
    - GPU is validated against CPU at the ``GPU_FP32`` tier (rtol=1e-4,
      atol=1e-5). FP64 on CUDA matches CPU to machine precision.

The ``MultinomialGPULikelihood`` class holds the GPU tensors for X and
the one-hot y between optimizer iterations, so the (n, p)-sized data
crosses the PCIe bus once, not once per scipy evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class MultinomialGPULikelihood:
    """Stateful holder of GPU tensors for a multinomial fit.

    ``fun_and_grad(params_flat)`` returns ``(float, NDArray)``. Scipy
    wants ``fun`` and ``jac`` as two separate callables, so we also
    expose ``.fun`` and ``.jac`` methods that each cache a forward pass
    and serve both from the cache — the autograd forward+backward
    happens at most once per unique ``params_flat``.
    """

    def __init__(
        self,
        X: NDArray[np.floating[Any]],
        y_codes: NDArray[np.int64],
        n_classes: int,
        device: str = "cuda",
        use_fp64: bool = False,
    ) -> None:
        import torch

        if device == "mps" and use_fp64:
            raise RuntimeError(
                "GPU multinomial: MPS does not support FP64. "
                "Use use_fp64=False or backend='cpu'."
            )

        self._torch = torch
        self._device = torch.device(device)
        self._dtype = torch.float64 if use_fp64 else torch.float32

        n, p = X.shape
        self._n = n
        self._p = p
        self._n_classes = n_classes
        self._n_nonref = n_classes - 1

        # Accept either a numpy array (pay per-call H2D transfer now)
        # or a torch.Tensor already on the target device (amortized
        # transfer via DataSource.to('cuda') — the common pattern for
        # multi-fit workflows). In the tensor case we only cast dtype
        # if it doesn't already match, to avoid an extra GPU copy.
        if isinstance(X, torch.Tensor):
            if X.device != self._device or X.dtype != self._dtype:
                self._X = X.to(device=self._device, dtype=self._dtype)
            else:
                self._X = X
        else:
            self._X = torch.as_tensor(
                X, device=self._device, dtype=self._dtype,
            )
        y_onehot = np.zeros((n, n_classes), dtype=np.float64)
        y_onehot[np.arange(n), y_codes] = 1.0
        self._y_onehot = torch.as_tensor(
            y_onehot, device=self._device, dtype=self._dtype,
        )
        # Zero column that augments eta for the reference class.
        self._zero_ref = torch.zeros(n, 1, device=self._device, dtype=self._dtype)

        # Cache the last (params, nll, grad) so fun/jac back-to-back
        # calls with the same params share one forward+backward pass.
        self._cache_params: NDArray | None = None
        self._cache_nll: float | None = None
        self._cache_grad: NDArray | None = None

    def _compute(
        self, params_flat: NDArray[np.floating[Any]],
    ) -> tuple[float, NDArray[np.floating[Any]]]:
        """Forward + backward on GPU. Returns (nll, grad as flat float64)."""
        torch = self._torch

        # Copy to GPU as a leaf tensor requiring grad.
        params_gpu = torch.as_tensor(
            params_flat, device=self._device, dtype=self._dtype,
        ).detach().clone().requires_grad_(True)

        beta = params_gpu.reshape(self._n_nonref, self._p)
        # Linear predictors for non-reference classes: (n, J-1).
        eta_nonref = self._X @ beta.T
        # Full eta with zeros appended for reference class: (n, J).
        eta = torch.cat([eta_nonref, self._zero_ref], dim=1)
        # Log-softmax handles the numerical stability internally
        # (equivalent to the CPU path's log-sum-exp trick).
        log_probs = torch.log_softmax(eta, dim=1)
        nll_t = -(self._y_onehot * log_probs).sum()

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

    def compute_probs(
        self, params_flat: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Predicted probabilities (n, J) at converged params."""
        torch = self._torch
        with torch.no_grad():
            params_gpu = torch.as_tensor(
                params_flat, device=self._device, dtype=self._dtype,
            )
            beta = params_gpu.reshape(self._n_nonref, self._p)
            eta_nonref = self._X @ beta.T
            eta = torch.cat([eta_nonref, self._zero_ref], dim=1)
            probs = torch.softmax(eta, dim=1)
        return probs.to(torch.float64).cpu().numpy()

    def compute_vcov(
        self, params_flat: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Numerical Hessian → inverse → vcov, fully on GPU.

        The CPU path runs ``multinomial_gradient`` in a Python loop
        over every parameter — for J=10, p=30 that is 300 evaluations
        of an O(n · J · p) matmul (~3 s on 50k observations). On GPU
        we can compute the Hessian analytically in closed form:

            H = X' · (diag(p) − p p') · X  for each class pair (j, k).

        More precisely, the softmax Hessian with respect to the flat
        parameter vector for classes 0..J-2 is the block matrix

            H[j, k] = X' · W_{jk} · X

        where W_{jk} is an (n, n) diagonal: W_{jj} = diag(p_j · (1 − p_j))
        and W_{jk} = diag(−p_j · p_k) for j ≠ k. We never materialize
        the n × n diagonals — the weighted X'WX is an elementwise
        weight + matmul that fits in GPU memory even for large n.
        """
        torch = self._torch
        n_nonref = self._n_nonref
        p = self._p
        n = self._n
        n_params = n_nonref * p

        with torch.no_grad():
            params_gpu = torch.as_tensor(
                params_flat, device=self._device, dtype=self._dtype,
            )
            beta = params_gpu.reshape(n_nonref, p)
            eta_nonref = self._X @ beta.T
            eta = torch.cat([eta_nonref, self._zero_ref], dim=1)
            probs = torch.softmax(eta, dim=1)[:, :n_nonref]   # (n, J-1)

            # Build the Hessian block-by-block.
            hessian = torch.empty(
                (n_params, n_params),
                device=self._device, dtype=self._dtype,
            )
            for j in range(n_nonref):
                p_j = probs[:, j]
                for k in range(j, n_nonref):
                    if j == k:
                        w = p_j * (1.0 - p_j)
                    else:
                        w = -p_j * probs[:, k]
                    # (p, p) block: X' · diag(w) · X  =  (X * w[:,None])' · X
                    block = (self._X * w.unsqueeze(1)).T @ self._X
                    row_slice = slice(j * p, (j + 1) * p)
                    col_slice = slice(k * p, (k + 1) * p)
                    hessian[row_slice, col_slice] = block
                    if k != j:
                        hessian[col_slice, row_slice] = block.T

            # Invert on GPU. Fall back to pseudo-inverse if singular.
            try:
                vcov = torch.linalg.inv(hessian)
            except RuntimeError:
                vcov = torch.linalg.pinv(hessian)

        return vcov.to(torch.float64).cpu().numpy()
