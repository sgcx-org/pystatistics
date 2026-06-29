"""Levenberg-Marquardt-damped Cholesky IRLS on MPS float32.

One job: solve the binomial-logit MLE on the device using a damped Newton step
(XtWX + λ·diag(XtWX)) δ = score, with λ adapted by trust-region accept/reject and
driven toward 0 as the iteration converges. The damping GUARANTEES the Cholesky
factor exists (the damped matrix is PD for λ>0), which kills the not-PD crash —
but because λ→0 at the optimum, the fixed point is the *unmodified* MLE. This is
SOLVER DAMPING, not ridge regression: nothing is added to the estimator at
convergence, so the coefficients must match the CPU fp64 fit (verified by the
study, not assumed). Contrast ridge (l2>0), which changes the estimate.
"""

from __future__ import annotations

import time as _time
import numpy as np
import torch

from .irls_common import to_dev, working_wz, binom_deviance_dev, DEVICE, sync


def run_lm(X_np, y_np, *, init="zero", max_outer=200, lam0=1e-3,
           lam_down=0.3, lam_up=3.0, lam_min=1e-12, lam_max=1e8,
           max_inner=40, rel_dev_tol=1e-7, lam_converged=1e-6):
    """Run LM-damped float32 IRLS. Returns (coef_np_f64, info dict).

    λ starts at lam0·mean(diag) scale, shrinks on accepted steps (toward the
    undamped Newton step / true MLE) and grows on rejected ones. Stops when the
    relative deviance change of an accepted step falls below rel_dev_tol or λ is
    driven to lam_min with a converged step.
    """
    sync()
    t0 = _time.perf_counter()
    X = to_dev(X_np)
    y = to_dev(y_np)
    p = X.shape[1]
    if init == "zero":
        b = torch.zeros(p, device=DEVICE, dtype=torch.float32)
    elif init == "mustart":
        mu0 = (y + 0.5) / 2.0
        b = torch.zeros(p, device=DEVICE, dtype=torch.float32)
        eta0 = torch.log(mu0 / (1 - mu0))
        # one undamped-ish step from the mustart eta to seed b
        w, z = working_wz(y, eta0)
        wX = w.unsqueeze(1) * X
        XtWX = X.T @ wX
        d = torch.diag(XtWX)
        damp = torch.diag(d * 1e-3 + 1e-8)
        try:
            L = torch.linalg.cholesky(XtWX + damp)
            rhs = (X.T @ (w * z)).unsqueeze(1)
            sol = torch.linalg.solve_triangular(L, rhs, upper=False)
            b = torch.linalg.solve_triangular(L.T, sol, upper=True).squeeze(1)
        except Exception:
            pass
    else:
        raise ValueError(f"unknown init {init!r}")

    eta = X @ b
    dev = binom_deviance_dev(y, eta)
    lam = lam0
    outer = 0
    inner_total = 0
    chol_fail = 0
    for outer in range(1, max_outer + 1):
        w, z = working_wz(y, eta)
        wX = w.unsqueeze(1) * X
        XtWX = X.T @ wX
        XtWz = X.T @ (w * z)
        score = XtWz - XtWX @ b          # gradient of the IRLS quadratic at b
        diag = torch.diag(XtWX).clamp(min=1e-12)

        accepted = False
        for _ in range(max_inner):
            inner_total += 1
            M = XtWX + lam * torch.diag(diag)
            try:
                L = torch.linalg.cholesky(M)
            except Exception:
                chol_fail += 1
                lam = min(lam * lam_up, lam_max)
                continue
            delta = torch.linalg.solve_triangular(L, score.unsqueeze(1), upper=False)
            delta = torch.linalg.solve_triangular(L.T, delta, upper=True).squeeze(1)
            b_try = b + delta
            eta_try = X @ b_try
            dev_try = binom_deviance_dev(y, eta_try)
            # fp32-aware accept: near the optimum the deviance sits on a √n·eps
            # round-off floor where a strictly-monotone test rejects good steps as
            # fp32 noise (the same trap 4.2.3 fixed on the main path). Accept any
            # step that does not increase the deviance beyond that floor.
            dev_floor = abs(dev) * 3e-6 + 1e-6
            if np.isfinite(dev_try) and dev_try <= dev + dev_floor:
                accepted = True
                break
            lam = min(lam * lam_up, lam_max)   # reject: damp harder, retry

        if not accepted:
            break                              # cannot make progress -> stop, gate decides

        rel_dev = abs(dev - dev_try) / (abs(dev) + 0.1)
        b, eta, dev = b_try, eta_try, dev_try
        lam = max(lam * lam_down, lam_min)     # success: relax damping toward 0
        # Converged only when the deviance has stopped moving AND we are taking
        # near-undamped (Newton) steps -- otherwise a tiny over-damped step can
        # masquerade as convergence far from the optimum.
        if rel_dev < rel_dev_tol and lam <= lam_converged:
            break

    sync()
    wall = _time.perf_counter() - t0
    coef = b.detach().to("cpu").numpy().astype(np.float64)
    info = {"outer_iters": outer, "wall_s": wall, "raised": False, "reason": "",
            "final_dev": dev, "lm_inner_total": inner_total,
            "lm_chol_fail": chol_fail, "lm_final_lambda": lam}
    return coef, info
