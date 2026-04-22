"""GPU backend for Generalized Additive Model P-IRLS fitting.

The CPU GAM fit in ``_fit.py`` / ``_gcv.py`` follows a two-level
structure:

1. An outer L-BFGS-B over ``log(lambda)`` (the smoothing parameters)
   chooses the penalty strength via GCV or REML — ~50 evaluations
   per fit.
2. Each evaluation runs a full P-IRLS loop to convergence at the
   candidate ``lambda`` vector, then assembles the scoring criterion
   from the converged ``(beta, W, deviance)``.

Every outer step rebuilds ``X'WX``, ``sum lambda_j S_j`` and the
Cholesky solve on the CPU. On GPU we keep ``X_aug``, ``y`` and the
stacked penalty tensor ``S_stack`` device-resident for the whole
fit, then run both P-IRLS and the hat-trace / log-det diagnostics in
batched tensor ops. The outer optimizer only pays transfer for the
``log_lambdas`` vector going in and the scalar criterion coming out
per evaluation.

Two-tier validation (see README "Design Philosophy"):
    - CPU is validated against R ``mgcv::gam``.
    - GPU is validated against CPU at the ``GPU_FP32`` tier
      (rtol = 1e-4, atol = 1e-5). FP64 on CUDA matches CPU to
      machine precision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.gam.backends._gpu_family import (
    GPUFamilyOps,
    resolve_gpu_family,
)


class GAMGPUFitter:
    """Device-resident GAM P-IRLS fitter.

    Holds ``X_aug`` (n, p), ``y`` (n,), and the stacked penalty tensor
    (n_smooths, p, p) on device for the entire fit. Exposes:

    - :meth:`fit_fixed` — full P-IRLS at a given lambda vector,
      returning numpy arrays for result assembly.
    - :meth:`gcv_score` / :meth:`reml_score` — scipy-compatible
      scalar scorers for the outer L-BFGS-B over ``log_lambdas``.
    """

    def __init__(
        self,
        y: NDArray,
        X_aug: NDArray | Any,              # numpy or torch.Tensor
        S_penalties: list[NDArray],
        family_name: str,
        parametric_cols: int,
        device: str,
        use_fp64: bool,
    ) -> None:
        import torch

        if device == "mps" and use_fp64:
            raise RuntimeError(
                "GPU GAM: MPS does not support FP64. "
                "Use use_fp64=False or backend='cpu'."
            )

        self._torch = torch
        self._device = torch.device(device)
        self._dtype = torch.float64 if use_fp64 else torch.float32
        self._parametric_cols = parametric_cols
        self._fam: GPUFamilyOps = resolve_gpu_family(family_name)

        # X may arrive as a torch.Tensor (amortized DataSource path) or
        # a numpy array (per-call H2D). Avoid an extra GPU copy if the
        # tensor is already on the right device with the right dtype.
        if isinstance(X_aug, torch.Tensor):
            if (
                X_aug.device != self._device
                or X_aug.dtype != self._dtype
            ):
                self._X = X_aug.to(device=self._device, dtype=self._dtype)
            else:
                self._X = X_aug
        else:
            self._X = torch.as_tensor(
                X_aug, device=self._device, dtype=self._dtype,
            )
        self._y = torch.as_tensor(
            y, device=self._device, dtype=self._dtype,
        )

        n, p = self._X.shape
        self._n = n
        self._p = p
        self._n_smooths = len(S_penalties)

        # Stack padded penalty matrices into a (n_smooths, p, p) tensor
        # so the weighted sum sum_j lambda_j S_j becomes a single
        # broadcasted multiply + reduction. The penalties only change
        # when lambda changes; the matrices themselves are constant.
        if self._n_smooths > 0:
            S_np = np.stack(
                [np.asarray(S, dtype=np.float64) for S in S_penalties],
                axis=0,
            )
            self._S_stack = torch.as_tensor(
                S_np, device=self._device, dtype=self._dtype,
            )
        else:
            self._S_stack = None

    # ---- core P-IRLS primitives -----------------------------------------

    def _penalty_sum(self, lambdas_gpu: Any) -> Any:
        """Compute sum_j lambda_j * S_j as a (p, p) tensor."""
        torch = self._torch
        if self._S_stack is None:
            return torch.zeros(
                (self._p, self._p),
                device=self._device, dtype=self._dtype,
            )
        return (lambdas_gpu.view(-1, 1, 1) * self._S_stack).sum(dim=0)

    def _solve_posdef(self, A: Any, b: Any) -> Any:
        """Solve ``A x = b`` for the P-IRLS inner step.

        Fast path (LU) with a small-ridge fallback for numerically
        singular A. The inner P-IRLS step only needs a stable beta
        update — it doesn't care about null-space directions — so
        LU is fine here and 2-3× faster than the SVD-based alternative
        we use for hat-trace.
        """
        torch = self._torch
        try:
            return torch.linalg.solve(A, b)
        except RuntimeError:
            A = A + torch.eye(
                A.shape[0], device=self._device, dtype=self._dtype,
            ) * 1e-8
            return torch.linalg.solve(A, b)

    def _canonicalise_beta(self, A: Any, b: Any) -> Any:
        """Recompute beta via the CPU's Cholesky-with-LU-fallback path.

        The in-loop P-IRLS solve uses ``torch.linalg.solve`` (LU on
        device) for speed, which is fine while P-IRLS is still iterating
        — it only needs a stable Newton direction. But the *final*
        converged beta needs to be the same null-space representative
        that the CPU path produces, because downstream code (smooth-term
        chi-squared, coefficient reporting, R-cross-validation tests)
        compares coefficients directly.

        The penalised normal matrix ``A = X'WX + sum lam_j S_j`` has
        condition number up to ~1e17 when λ is small — the penalty
        does not fully eliminate the design matrix's null space. On
        such A, ``torch.linalg.solve`` and ``np.linalg.cholesky``
        converge to the same fitted values (the null-space components
        don't affect ``X · beta``) but pick DIFFERENT
        null-space-representative coefficients. Example on a simple
        sine GAM: CPU vs GPU fitted values agree to 7e-8 but the
        intercept and smooth coefficients differ by a constant ±1.73
        shift. That invalidates every downstream comparison of
        individual coefficients.

        Fix: route the final beta solve through numpy LAPACK using the
        exact same Cholesky-first / LU-fallback logic the CPU ``_pirls_step``
        uses. The A and b tensors are small (p×p / p×1), so the D2H
        round-trip is cheap. Output is guaranteed to match CPU to
        FP64 precision.
        """
        torch = self._torch
        A_np = A.detach().to(torch.float64).cpu().numpy()
        b_np = b.detach().to(torch.float64).cpu().numpy()
        try:
            L = np.linalg.cholesky(A_np)
            beta_np = np.linalg.solve(L.T, np.linalg.solve(L, b_np))
        except np.linalg.LinAlgError:
            A_np = A_np + 1e-8 * np.eye(A_np.shape[0])
            beta_np = np.linalg.solve(A_np, b_np)
        return torch.as_tensor(
            beta_np, device=self._device, dtype=self._dtype,
        )

    def _solve_for_edf(self, A: Any, XtWX: Any) -> Any:
        """Compute ``F = A^{-1} XtWX`` for EDF / hat-trace, using numpy
        LAPACK to match the CPU reference bit-for-bit.

        The penalised normal-equation matrix has condition number up to
        ~1e17 (in FP64) when the optimizer probes small lambda — the
        penalty does not fully eliminate the design matrix's null
        space, and numerical noise gives the smallest eigenvalue an
        arbitrary sign. LU on such a matrix produces implementation-
        dependent answers: LAPACK getrf and cuSOLVER getrf pivot
        differently on near-singular matrices and diverge by factors
        of two (including sign flips) for lambdas differing by 1e-6.

        The hat-trace is a p×p solve with p typically 30-80 — tiny.
        Routing it through numpy LAPACK costs ~100 µs per call but
        guarantees CPU/GPU agreement on this notoriously-unstable
        quantity. The dominant GPU work (the n×p GEMM forming
        ``X'WX``) stays on device; only the p×p solve goes via host.
        """
        torch = self._torch
        A_np = A.detach().to(torch.float64).cpu().numpy()
        XtWX_np = XtWX.detach().to(torch.float64).cpu().numpy()
        try:
            F_np = np.linalg.solve(A_np, XtWX_np)
        except np.linalg.LinAlgError:
            A_np = A_np + 1e-8 * np.eye(A_np.shape[0])
            F_np = np.linalg.solve(A_np, XtWX_np)
        return torch.as_tensor(
            F_np, device=self._device, dtype=self._dtype,
        )

    def _pirls_step(
        self, mu: Any, lambdas_gpu: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """One penalised IRLS step, returning ``(beta, mu, eta, w, XtWX, A)``.

        ``XtWX`` and ``A = XtWX + penalty`` are returned so the outer
        scorers can reuse them for hat-trace / log-determinant without
        a second formation pass.
        """
        torch = self._torch
        eta = self._fam.link_fn(mu)
        dmu_deta = self._fam.mu_eta(eta)
        var_mu = self._fam.variance(mu)
        w = torch.clamp(dmu_deta * dmu_deta / torch.clamp(var_mu, min=1e-20),
                        min=1e-20)
        z = eta + (self._y - mu) / torch.clamp(dmu_deta, min=1e-20)

        XtW = self._X.T * w.unsqueeze(0)              # (p, n)
        XtWX = XtW @ self._X                          # (p, p)
        penalty = self._penalty_sum(lambdas_gpu)
        A = XtWX + penalty
        b = XtW @ z

        beta = self._solve_posdef(A, b)
        new_eta = self._X @ beta
        new_mu = self._fam.linkinv(new_eta)
        return beta, new_mu, new_eta, w, XtWX, A

    def _fit_loop(
        self, lambdas_gpu: Any, tol: float, max_iter: int,
    ) -> tuple[Any, Any, Any, Any, Any, int, bool, Any, Any]:
        """Run P-IRLS to convergence at fixed ``lambdas_gpu``.

        Returns ``(beta, mu, eta, w, deviance, n_iter, converged,
        XtWX, A)`` — all tensors, for reuse by the scoring helpers.
        """
        torch = self._torch
        mu = self._fam.mu_from_y(self._y)
        dev_old_t = self._fam.deviance(self._y, mu)
        beta = torch.zeros(self._p, device=self._device, dtype=self._dtype)
        w = torch.ones(self._n, device=self._device, dtype=self._dtype)
        XtWX = None
        A = None
        eta = None
        n_iter = 0
        converged = False
        for it in range(1, max_iter + 1):
            beta, mu, eta, w, XtWX, A = self._pirls_step(mu, lambdas_gpu)
            dev_new_t = self._fam.deviance(self._y, mu)
            # Convergence on relative deviance change. One scalar D2H
            # per iteration, same as the CPU loop's scalar compare.
            dev_old = float(dev_old_t.detach().cpu().item())
            dev_new = float(dev_new_t.detach().cpu().item())
            if abs(dev_old) > 0:
                rel_change = abs(dev_new - dev_old) / (abs(dev_old) + 1e-20)
            else:
                rel_change = abs(dev_new - dev_old)
            dev_old_t = dev_new_t
            n_iter = it
            if rel_change < tol and it > 1:
                converged = True
                break
        # Final W = diag weights (after last step's update we need
        # weights consistent with the returned mu).
        eta_f = self._fam.link_fn(mu)
        dmu_deta = self._fam.mu_eta(eta_f)
        var_mu = self._fam.variance(mu)
        w = torch.clamp(dmu_deta * dmu_deta / torch.clamp(var_mu, min=1e-20),
                        min=1e-20)
        return beta, mu, eta, w, dev_old_t, n_iter, converged, XtWX, A

    # ---- public API -----------------------------------------------------

    def fit_fixed(
        self, lambdas: NDArray, tol: float, max_iter: int,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, float, int, bool]:
        """Full P-IRLS at a fixed ``lambdas`` vector.

        Returns numpy arrays / scalars matching the CPU
        ``_fit_gam_fixed_lambda`` contract — including coefficient-
        level equivalence, not just fitted-value equivalence. See
        ``_canonicalise_beta``'s docstring for why the final solve is
        re-routed through numpy LAPACK.
        """
        torch = self._torch
        lambdas_gpu = torch.as_tensor(
            np.asarray(lambdas, dtype=np.float64),
            device=self._device, dtype=self._dtype,
        )
        (_beta_loop, mu, eta, w, dev_t, n_iter, converged,
         _XtWX, _A) = self._fit_loop(lambdas_gpu, tol, max_iter)

        # Canonicalise final beta through CPU's Cholesky path so
        # coefficient-level comparisons (smooth-term chi_sq, R cross-
        # validation) match the CPU backend bit-for-bit. The in-loop
        # beta is good enough for mu / eta / deviance but can land on
        # a shifted null-space representative when the penalised
        # normal matrix is near-singular (small λ regime).
        XtWX_final, A_final = self._final_XtWX_A(w, lambdas_gpu)
        dmu_deta = self._fam.mu_eta(eta)
        z_final = eta + (self._y - mu) / torch.clamp(dmu_deta, min=1e-20)
        XtW_final = self._X.T * w.unsqueeze(0)
        b_final = XtW_final @ z_final
        beta = self._canonicalise_beta(A_final, b_final)

        # Refresh mu / eta from the canonicalised beta. Fitted values
        # should match the loop's output to FP64 precision by
        # construction (A beta = b is the same equation P-IRLS was
        # converging to); recomputing keeps the tuple internally
        # consistent.
        eta = self._X @ beta
        mu = self._fam.linkinv(eta)

        return (
            beta.detach().to(torch.float64).cpu().numpy(),
            mu.detach().to(torch.float64).cpu().numpy(),
            eta.detach().to(torch.float64).cpu().numpy(),
            w.detach().to(torch.float64).cpu().numpy(),
            float(dev_t.detach().cpu().item()),
            n_iter,
            converged,
        )

    def _final_XtWX_A(self, w: Any, lambdas_gpu: Any) -> tuple[Any, Any]:
        """Rebuild ``XtWX`` and ``A = XtWX + penalty`` from the final
        weights ``w``. The ``_fit_loop`` returns the XtWX computed
        *before* the last beta update, but GCV / REML / EDF all want
        the quantities consistent with the final ``mu`` — so we
        recompute once, matching the CPU ``_compute_hat_matrix_trace``
        which also rebuilds from the post-convergence weights.
        """
        XtW = self._X.T * w.unsqueeze(0)
        XtWX = XtW @ self._X
        A = XtWX + self._penalty_sum(lambdas_gpu)
        return XtWX, A

    def hat_trace(
        self, lambdas: NDArray, tol: float, max_iter: int,
    ) -> tuple[float, float]:
        """Fit P-IRLS then return ``(total_edf, deviance)`` as floats.

        Used by the GCV/REML scorers below; a convenience that avoids
        a second P-IRLS when the scorer wants both pieces.
        """
        torch = self._torch
        lambdas_gpu = torch.as_tensor(
            np.asarray(lambdas, dtype=np.float64),
            device=self._device, dtype=self._dtype,
        )
        (_beta, _mu, _eta, w, dev_t, _n_iter, _conv,
         _XtWX_stale, _A_stale) = self._fit_loop(lambdas_gpu, tol, max_iter)
        XtWX, A = self._final_XtWX_A(w, lambdas_gpu)
        F = self._solve_for_edf(A, XtWX)
        total_edf = float(torch.diagonal(F).sum().detach().cpu().item())
        return total_edf, float(dev_t.detach().cpu().item())

    def edf_per_term(
        self,
        lambdas: NDArray,
        term_indices: list[tuple[int, int]],
        tol: float,
        max_iter: int,
    ) -> tuple[NDArray, float, NDArray, NDArray, NDArray, NDArray, float, int, bool]:
        """Fit + compute per-term EDF, returning everything for result
        assembly at end of fit. Avoids a redundant P-IRLS pass vs.
        calling ``fit_fixed`` then a separate EDF call.

        Returns
        -------
        (edf_per, total_edf, beta, mu, eta, w, deviance, n_iter, converged)
        """
        torch = self._torch
        lambdas_gpu = torch.as_tensor(
            np.asarray(lambdas, dtype=np.float64),
            device=self._device, dtype=self._dtype,
        )
        (_beta_loop, mu, eta, w, dev_t, n_iter, converged,
         XtWX, A) = self._fit_loop(lambdas_gpu, tol, max_iter)

        # Recompute XtWX / A with the final W (W can drift in the last
        # iteration because P-IRLS's stored XtWX is pre-update).
        XtW = self._X.T * w.unsqueeze(0)
        XtWX_final = XtW @ self._X
        A_final = XtWX_final + self._penalty_sum(lambdas_gpu)

        # Canonicalise the final beta through CPU's Cholesky path so
        # coefficient-level comparisons match bit-for-bit even when the
        # penalised normal matrix is near-singular (small-λ regime) and
        # torch vs numpy would land on different null-space
        # representatives. See ``_canonicalise_beta`` docstring.
        dmu_deta = self._fam.mu_eta(eta)
        z_final = eta + (self._y - mu) / torch.clamp(dmu_deta, min=1e-20)
        b_final = XtW @ z_final
        beta = self._canonicalise_beta(A_final, b_final)
        # Refresh mu / eta from canonicalised beta for internal
        # consistency (identical to the loop's output modulo FP64 noise
        # by construction of the canonicalisation).
        eta = self._X @ beta
        mu = self._fam.linkinv(eta)

        F = self._solve_for_edf(A_final, XtWX_final)
        F_diag = torch.diagonal(F)
        total_edf = float(F_diag.sum().detach().cpu().item())
        edf_list = [
            float(F_diag[s:e].sum().detach().cpu().item())
            for (s, e) in term_indices
        ]
        return (
            np.asarray(edf_list, dtype=np.float64),
            total_edf,
            beta.detach().to(torch.float64).cpu().numpy(),
            mu.detach().to(torch.float64).cpu().numpy(),
            eta.detach().to(torch.float64).cpu().numpy(),
            w.detach().to(torch.float64).cpu().numpy(),
            float(dev_t.detach().cpu().item()),
            n_iter,
            converged,
        )

    # ---- scoring criteria for outer L-BFGS-B ----------------------------

    def gcv_score(
        self, log_lambdas: NDArray, tol: float, max_iter: int,
    ) -> float:
        """GCV = n * deviance / (n - edf)^2.

        Callable signature matches scipy L-BFGS-B: one numpy input, one
        float output. All tensor work is device-resident between calls.
        """
        torch = self._torch
        lambdas = np.exp(np.asarray(log_lambdas, dtype=np.float64))
        lambdas_gpu = torch.as_tensor(
            lambdas, device=self._device, dtype=self._dtype,
        )
        (_beta, _mu, _eta, w, dev_t, _n_iter, _conv,
         _XtWX_stale, _A_stale) = self._fit_loop(lambdas_gpu, tol, max_iter)
        XtWX, A = self._final_XtWX_A(w, lambdas_gpu)
        F = self._solve_for_edf(A, XtWX)
        edf_total = float(torch.diagonal(F).sum().detach().cpu().item())
        denom = self._n - edf_total
        if denom <= 0.0:
            return 1e20
        deviance = float(dev_t.detach().cpu().item())
        return float(self._n) * deviance / (denom * denom)

    def reml_score(
        self, log_lambdas: NDArray, tol: float, max_iter: int,
    ) -> float:
        """Approximate REML criterion (matches CPU ``_reml_score``)."""
        torch = self._torch
        lambdas = np.exp(np.asarray(log_lambdas, dtype=np.float64))
        lambdas_gpu = torch.as_tensor(
            lambdas, device=self._device, dtype=self._dtype,
        )
        (_beta, _mu, _eta, w, dev_t, _n_iter, _conv,
         _XtWX_stale, _A_stale) = self._fit_loop(lambdas_gpu, tol, max_iter)
        XtWX, A = self._final_XtWX_A(w, lambdas_gpu)
        F = self._solve_for_edf(A, XtWX)
        edf_total = float(torch.diagonal(F).sum().detach().cpu().item())
        deviance = float(dev_t.detach().cpu().item())
        scale = max(deviance / max(self._n - edf_total, 1.0), 1e-20)

        # log|A| and log|XtWX| via Cholesky. log|M| = 2 * sum log diag(L).
        eye_p = torch.eye(
            self._p, device=self._device, dtype=self._dtype,
        ) * 1e-10
        try:
            L_a = torch.linalg.cholesky(A)
            L_b = torch.linalg.cholesky(XtWX + eye_p)
        except RuntimeError:
            return 1e20
        logdet_a = 2.0 * float(torch.log(
            torch.diagonal(L_a)
        ).sum().detach().cpu().item())
        logdet_b = 2.0 * float(torch.log(
            torch.diagonal(L_b)
        ).sum().detach().cpu().item())
        return deviance / scale + logdet_a - logdet_b
