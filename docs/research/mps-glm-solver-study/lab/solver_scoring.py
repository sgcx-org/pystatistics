"""Fisher-scoring IRLS on MPS float32 with a pluggable inner solve.

One job: drive IRLS (eta -> w,z -> solve (XtWX)b = XtWz) on the MPS device in
float32, with monotone-descent step-halving on the deviance, and let the caller
pick the inner solve: Cholesky, matrix-free CG, CPU-QR, or a Cholesky->CG hybrid.
Each inner solve returns the FULL updated coefficient vector; the driver wraps it
in step-halving and a relative-step early stop. Acceptance is decided afterward
by the host fp64 gate in irls_common (not here) — this driver only iterates.
"""

from __future__ import annotations

import time as _time
import numpy as np
import torch

from .irls_common import (
    to_dev, sigmoid, working_wz, binom_deviance_dev, rel_step, DEVICE, sync,
)


# ---- inner solves: (X, w, z, b_prev) -> b_new (all device float32) -----------

def inner_cholesky(X, w, z, b_prev, state):
    """Form XtWX (squares condition number) and Cholesky-solve. Raises if not PD."""
    wX = w.unsqueeze(1) * X
    XtWX = X.T @ wX
    XtWz = X.T @ (w * z)
    L = torch.linalg.cholesky(XtWX)          # raises on not-PD
    sol = torch.linalg.solve_triangular(L, XtWz.unsqueeze(1), upper=False)
    b = torch.linalg.solve_triangular(L.T, sol, upper=True).squeeze(1)
    return b


def _cg(matvec, rhs, x0, tol, max_iter, state, tag):
    """Plain CG for SPD operator. Records iters + final relative residual."""
    x = x0.clone()
    r = rhs - matvec(x)
    p = r.clone()
    rs = torch.dot(r, r)
    rhs_norm = torch.linalg.vector_norm(rhs).clamp(min=1e-30)
    it = 0
    for it in range(1, max_iter + 1):
        Ap = matvec(p)
        denom = torch.dot(p, Ap)
        if float(denom.item()) <= 0.0:          # lost SPD-ness in fp32
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        relres = float((torch.sqrt(rs_new) / rhs_norm).item())
        if relres < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    state.setdefault("cg_iters", []).append(it)
    state.setdefault("cg_relres", []).append(
        float((torch.linalg.vector_norm(rhs - matvec(x)) / rhs_norm).item()))
    return x


def inner_cg(X, w, z, b_prev, state):
    """Matrix-free CG on H v = Xt(W(Xv)); rhs = Xt(W z). Warm-started from b_prev.

    NOTE: this is the *ungated* inner solve. The silent-wrong hazard (small CG
    residual, large solution error under fp32 ill-conditioning) is caught only by
    the host fp64 Newton-decrement gate AFTER the IRLS loop — that is what makes
    the overall path 'gated CG'. CG here never decides acceptance.
    """
    rhs = X.T @ (w * z)
    matvec = lambda v: X.T @ (w * (X @ v))
    return _cg(matvec, rhs, b_prev, tol=1e-7, max_iter=300, state=state, tag="cg")


def make_inner_cpu_qr(X_host_f64):
    """CPU float64 QR inner solve. Keeps logits/w/z on MPS; ships only the two
    length-n vectors (w, z) to the host, then solves the n×p WLS with the
    host-resident X via QR (np.linalg.lstsq). X never leaves the host."""
    def inner_cpu_qr(X, w, z, b_prev, state):
        t0 = _time.perf_counter()
        w_h = w.detach().to("cpu").numpy().astype(np.float64)
        z_h = z.detach().to("cpu").numpy().astype(np.float64)
        state["xfer_s"] = state.get("xfer_s", 0.0) + (_time.perf_counter() - t0)
        sw = np.sqrt(w_h)
        A = sw[:, None] * X_host_f64
        b = sw * z_h
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)   # QR, no normal equations
        state["cpu_solve_s"] = state.get("cpu_solve_s", 0.0) + (
            _time.perf_counter() - t0)
        return torch.from_numpy(sol.astype(np.float32)).to(X.device)
    return inner_cpu_qr


def make_inner_hybrid():
    """Try Cholesky; on not-PD fall back to CG for that step. Records fallbacks."""
    def inner_hybrid(X, w, z, b_prev, state):
        try:
            b = inner_cholesky(X, w, z, b_prev, state)
            state["chol_ok"] = state.get("chol_ok", 0) + 1
            return b
        except Exception:
            state["cg_fallbacks"] = state.get("cg_fallbacks", 0) + 1
            return inner_cg(X, w, z, b_prev, state)
    return inner_hybrid


# ---- the scoring IRLS driver -------------------------------------------------

def run_scoring(X_np, y_np, inner, *, init="zero", max_iter=60,
                rel_step_tol=1e-7, max_halve=20):
    """Run float32 scoring IRLS. Returns (coef_np_f64, info dict).

    info: outer_iters, wall_s, inner state (cg_iters, fallbacks, transfer time),
    raised (bool) + reason if the inner solve threw (e.g. Cholesky not-PD).
    """
    sync()
    t0 = _time.perf_counter()
    X = to_dev(X_np)
    y = to_dev(y_np)
    p = X.shape[1]
    if init == "zero":
        b = torch.zeros(p, device=DEVICE, dtype=torch.float32)
    elif init == "mustart":
        # R's binomial mustart: mu=(y+0.5)/2 -> eta=logit(mu); seed via one WLS.
        mu0 = (y + 0.5) / 2.0
        eta0 = torch.log(mu0 / (1 - mu0))
        w, z = working_wz(y, eta0)
        try:
            b = inner(X, w, z, torch.zeros(p, device=DEVICE), {})
        except Exception:
            b = torch.zeros(p, device=DEVICE, dtype=torch.float32)
    else:
        raise ValueError(f"unknown init {init!r}")

    state: dict = {}
    eta = X @ b
    dev = binom_deviance_dev(y, eta)
    outer = 0
    raised = False
    reason = ""
    for outer in range(1, max_iter + 1):
        w, z = working_wz(y, eta)
        try:
            b_full = inner(X, w, z, b, state)
        except Exception as e:               # e.g. Cholesky not-PD
            raised = True
            reason = f"{type(e).__name__}: {e}"
            break
        # monotone-descent step-halving on the deviance
        step = b_full - b
        accepted = False
        for h in range(max_halve + 1):
            b_try = b + step / (2 ** h)
            eta_try = X @ b_try
            dev_try = binom_deviance_dev(y, eta_try)
            if np.isfinite(dev_try) and dev_try <= dev + 1e-8:
                accepted = True
                break
        if not accepted:                      # could not decrease at all
            b_try, eta_try, dev_try = b_full, X @ b_full, binom_deviance_dev(y, X @ b_full)
        rstep = rel_step(b_try, b)
        b, eta, dev = b_try, eta_try, dev_try
        if rstep < rel_step_tol:
            break
    sync()
    wall = _time.perf_counter() - t0
    coef = b.detach().to("cpu").numpy().astype(np.float64)
    info = {"outer_iters": outer, "wall_s": wall, "raised": raised,
            "reason": reason, "final_dev": dev, **state}
    return coef, info
