"""
GPU backend for Generalized Linear Models via IRLS.

Same IRLS algorithm as cpu_glm.py, but the weighted least squares (WLS)
inner step is performed on GPU using PyTorch. Each iteration:
    1. Compute working response z and weights w (GPU)
    2. Form √w·X and √w·z (GPU)
    3. Solve WLS via torch.linalg.lstsq (GPU)
    4. Update η = X @ β, μ = linkinv(η) (GPU)
    5. Compute deviance on GPU, check convergence on CPU (scalar)

Final coefficients, fitted values, and residuals are transferred back
to CPU as float64 numpy arrays. All intermediate IRLS computations
use float32 for performance on consumer GPUs.

Convergence: the float64 (CUDA) and ridge-float32 paths stop on R's relative
deviance-change criterion. The plain float32 path instead stops on the relative
Newton decrement (stationarity), because the float32 deviance cannot fall to the
strict tolerance — only to a √n·eps round-off floor that would stop the iteration
early, leaving slowly-converging directions non-stationary. See ``_irls_step.py``.
Whether a stopped float32 fit is ACCEPTED is decided by a strict float64 Newton-
decrement gate after the loop (A6: accept the fits float32 can produce, refuse the
rest loudly).

Inner solve: the WLS step (XᵀWX)β = Xᵀ(Wz) is solved by Cholesky of the
on-device weighted normal-equations matrix — EXCEPT on the MPS plain-float32 path,
which solves it matrix-free by conjugate gradient on the operator H v = Xᵀ(W(X v))
so XᵀWX is never formed. Forming XᵀWX squares the condition number, and on a cold
MPS context the float32 Gram matrix at person-period quarterly scale can come out
not-positive-definite and crash Cholesky; the squaring-free CG step removes that
failure at its root. CUDA Cholesky is robust and is left unchanged. See
``_mps_cg_solve`` and ``docs`` for the design study. The MPS solver path is
validated per shipped torch version (kernel behaviour is version-sensitive); the
host float64 Newton-decrement acceptance gate is the version-independent guarantee
that an unreliable fit fails loud rather than returning silently wrong.

Supports CUDA and MPS (Apple Silicon).
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.core.exceptions import NumericalError, ValidationError
from pystatistics.core.compute.timing import Timer
from pystatistics.regression.backends._irls_step import (
    relative_newton_decrement, step_halve,
)
from pystatistics.regression.design import Design
from pystatistics.regression.families import Family
from pystatistics.regression.solution import GLMParams


# Acceptance threshold for a float32 GPU GLM fit, expressed as a RELATIVE Newton
# decrement (λ² / (|deviance| + 0.1)). The Newton decrement λ² = Uᵀ(XᵀWX)⁻¹U is
# ≈ twice the remaining deviance gap to the optimum, so this is the fraction of
# the deviance still "on the table". A correct float32 fit that has reached the
# float32 floor sits orders of magnitude below this; an ill-conditioned fit whose
# float32 solve landed off the optimum sits far above it. Calibrated empirically
# (well-conditioned binomial fits land ~1e-12; ill-conditioned log-link fits at
# scale land ~1e-1 or diverge) — see tests/regression/test_gpu_fp32_acceptance.py.
_FP32_REL_DECREMENT_TOL = 1e-6


def _newton_decrement(X_np, y_np, wt_np, coef, link, family, offset=None):
    """Newton decrement λ² = Uᵀ(XᵀWX)⁻¹U at ``coef`` (float64, on the host).

    The standard affine-invariant measure of distance to the optimum in
    objective (deviance/2) units: ≈ 0 at a true optimum, large when the
    coefficients sit off it. Used to decide whether a float32 GPU fit that
    stopped at the float32 deviance floor actually reached the optimum (accept)
    or merely stalled off it (fail loud). One O(n·p²) pass, evaluated once.

    Returns ``inf`` if the float64 information matrix is itself singular.
    """
    eta = X_np @ coef + (0.0 if offset is None else offset)
    mu = link.linkinv(eta)
    mu_eta = link.mu_eta(eta)            # dμ/dη
    var = family.variance(mu)            # V(μ)
    score = X_np.T @ (wt_np * mu_eta / var * (y_np - mu))          # U = ∂ℓ/∂β
    w = np.maximum(wt_np * (mu_eta ** 2) / var, 1e-30)             # IRLS weights
    XtWX = X_np.T @ (w[:, None] * X_np)                            # observed info
    try:
        step = np.linalg.solve(XtWX, score)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(score @ step)


def _mps_cg_solve(X_gpu, w_gpu, rhs_gpu, b_prev_gpu, tol=1e-7, max_iter=300):
    """Matrix-free conjugate-gradient solve of (XᵀWX) b = Xᵀ(W z) on-device.

    Solves the IRLS weighted normal-equations step WITHOUT ever forming XᵀWX,
    using only the operator ``H v = Xᵀ(W(X v))``. Not squaring the design's
    condition number is what keeps the float32 solve off the not-positive-definite
    knife edge that crashes a cold MPS Cholesky context at person-period quarterly
    scale (study §4 — a *reliability* improvement, not a correctness fix: the old
    Cholesky path always failed loud, never silently wrong). Warm-started from
    ``b_prev_gpu`` (the current IRLS iterate) so successive IRLS steps need few CG
    iterations.

    A non-converged or SPD-lost iterate is deliberately NOT special-cased here: it
    is returned as-is and judged by the host float64 Newton-decrement acceptance
    gate after the IRLS loop, which refuses it loudly (A6). CG never decides
    acceptance; it only proposes a step.

    Args:
        X_gpu: Design matrix (n×p), device float32.
        w_gpu: IRLS weights (n,), device float32.
        rhs_gpu: Right-hand side Xᵀ(W z) (p,), device float32.
        b_prev_gpu: Warm-start coefficients (p,), device float32.
        tol: Relative-residual stop ‖rhs − H x‖ / ‖rhs‖.
        max_iter: Iteration cap (≥ p guarantees convergence in exact arithmetic).

    Returns:
        Device float32 tensor (p,) — the warm-started CG iterate.
    """
    import torch

    def matvec(v):
        return X_gpu.T @ (w_gpu * (X_gpu @ v))   # H v = Xᵀ(W(X v)); XᵀWX never formed

    x = b_prev_gpu.clone()
    r = rhs_gpu - matvec(x)
    pvec = r.clone()
    rs = torch.dot(r, r)
    rhs_norm = torch.linalg.vector_norm(rhs_gpu).clamp(min=1e-30)
    for _ in range(max_iter):
        Ap = matvec(pvec)
        denom = torch.dot(pvec, Ap)
        if float(denom.item()) <= 0.0:        # SPD lost in float32 — stop and let
            break                              # the host gate refuse if it matters
        alpha = rs / denom
        x = x + alpha * pvec
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if float((torch.sqrt(rs_new) / rhs_norm).item()) < tol:
            break
        pvec = r + (rs_new / rs) * pvec
        rs = rs_new
    return x


class GPUIRLSBackend:
    """GPU backend using IRLS with torch.linalg.lstsq inner solve.

    Matches the CPU IRLS algorithm but leverages GPU parallelism for
    the matrix operations. Uses FP32 by default (FP64 not supported on MPS).

    For the WLS step, we solve min_β ||√w·z - √w·X·β||² via lstsq.
    Standard errors are computed on CPU from X'WX after IRLS converges.
    """

    def __init__(self, device: str = 'cuda', use_fp64: bool = False):
        """Initialize GPU IRLS backend.

        Args:
            device: GPU device type ('cuda', 'cuda:0', 'mps')
            use_fp64: Run the IRLS in float64. Valid only on CUDA (Metal/MPS has
                no float64). This is a *correctness* path — numerically equivalent
                to the CPU reference, but slow on consumer GPUs (fast double
                precision needs a data-center card). The default float32 path is
                the speed path. Selected via backend='gpu_fp64'.
        """
        import torch

        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA not available. Install PyTorch with CUDA support, "
                    "or use backend='cpu'."
                )
            self.device = torch.device(device)
            self.dtype = torch.float64 if use_fp64 else torch.float32
            props = torch.cuda.get_device_properties(self.device)
            self.device_name = props.name

        elif device == 'mps':
            if use_fp64:
                from pystatistics.core.compute.backend import FP64_REQUIRES_CUDA_MSG
                raise RuntimeError(FP64_REQUIRES_CUDA_MSG)
            self.device = torch.device('mps')
            self.dtype = torch.float32
            self.device_name = 'Apple Silicon GPU (MPS)'

        else:
            raise ValidationError(
                f"Unknown GPU device: {device!r}. Use 'cuda' or 'mps'."
            )
        self.use_fp64 = use_fp64

    @property
    def name(self) -> str:
        return 'gpu_irls_fp64' if self.use_fp64 else 'gpu_irls_fp32'

    def solve(
        self,
        design: Design,
        family: Family,
        tol: float = 1e-8,
        max_iter: int = 25,
        force: bool = False,
        penalty: 'NDArray[np.float64] | None' = None,
        weights: 'NDArray[np.float64] | None' = None,
        offset: 'NDArray[np.float64] | None' = None,
    ) -> Result[GLMParams]:
        """Run IRLS on GPU to fit the GLM.

        ``penalty`` (optional, length p): an L2 ridge penalty added to the diagonal
        of XᵀWX each iteration (``diag(penalty)``). It both regularizes the fit and
        makes the float32 Cholesky well-conditioned — the intended way to run a
        stable GLM on the GPU at very large scale. The caller is responsible for
        standardizing the design and back-transforming coefficients (see
        ``solvers._fit_glm_ridge``); the design passed here is the standardized one.

        Args:
            design: Design object with X and y
            family: GLM family specification
            tol: Convergence tolerance (relative deviance change)
            max_iter: Maximum IRLS iterations
            force: If True, return the GPU result even when IRLS does not
                converge in float32 (the caller accepts a possibly-inaccurate
                fit). If False (default), a non-converged fit raises
                NumericalError instead of returning unreliable coefficients.

        Returns:
            Result[GLMParams] with coefficients, deviance, residuals, etc.
        """
        import torch

        timer = Timer()
        timer.start()

        X_np, y_np = design.X.astype(np.float64), design.y.astype(np.float64)
        n, p = design.n, design.p
        link = family.link

        # Prior weights and offset (fast default: unit weights / no offset).
        wt_np = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        off_np = None if offset is None else np.asarray(offset, dtype=np.float64)

        warnings_list: list[str] = []

        # ------------------------------------------------------------------
        # Transfer to GPU
        # ------------------------------------------------------------------
        with timer.section('data_transfer_to_gpu'):
            X_gpu = torch.from_numpy(X_np).to(device=self.device, dtype=self.dtype)
            y_gpu = torch.from_numpy(y_np).to(device=self.device, dtype=self.dtype)
            # Optional ridge penalty added to the XᵀWX diagonal (stabilizes fp32).
            pen_gpu = None
            if penalty is not None:
                pen_gpu = torch.from_numpy(
                    np.asarray(penalty, dtype=np.float64)
                ).to(device=self.device, dtype=self.dtype)
            wt_gpu = torch.from_numpy(wt_np).to(device=self.device, dtype=self.dtype)
            off_gpu = (None if off_np is None else
                       torch.from_numpy(off_np).to(device=self.device, dtype=self.dtype))

        # ------------------------------------------------------------------
        # Initialize μ and η (on CPU using family/link, then transfer)
        # ------------------------------------------------------------------
        with timer.section('initialize'):
            mu_np = family.initialize(y_np, wt_np)
            eta_np = link.link(mu_np)
            mu_gpu = torch.from_numpy(mu_np).to(device=self.device, dtype=self.dtype)
            eta_gpu = torch.from_numpy(eta_np).to(device=self.device, dtype=self.dtype)

        # ------------------------------------------------------------------
        # IRLS loop
        # ------------------------------------------------------------------
        converged = False
        solve_failed = False
        # Initial deviance on CPU (family methods use numpy)
        dev_old = family.deviance(y_np, mu_np, wt_np)
        dev_new = dev_old

        # float64 host-side state for μ/η. The mixed-precision paths (ridge and
        # plain float32) maintain these across iterations (float32 solve, float64
        # working quantities) so the convergence signal isn't degraded by a float32
        # GPU round-trip. Only the pure float64 (CUDA) path reads μ/η back from the
        # device each iteration.
        mu_cpu = mu_np.astype(np.float64)
        eta_cpu = eta_np.astype(np.float64)

        # The plain (unpenalized) float32 path stops on a Newton-decrement
        # stationarity test rather than a deviance-change tolerance — see the loop.
        # ``coef_cur_np`` is the iterate that defines the current η (the point the
        # decrement is measured at, and the step-halving target). None until the
        # first step is taken; the stationarity test therefore runs from iteration 2.
        plain_fp32 = self.dtype == torch.float32 and pen_gpu is None
        coef_cur_np: 'NDArray[np.float64] | None' = None

        # MPS plain-float32 inner solve: matrix-free conjugate gradient on the
        # operator H v = Xᵀ(W(X v)) instead of Cholesky-of-XᵀWX. Squaring-free, so
        # it removes the not-PD cold-context Cholesky crash at the quarterly knife
        # edge at its root (study §4/§8). CUDA keeps Cholesky (robust there, 24/24);
        # ridge and fp64 keep their existing paths. NOTE: MPS kernel behaviour is
        # torch-version-sensitive, so this solver path is validated per shipped
        # torch version; the host float64 acceptance gate after the loop is the
        # version-INDEPENDENT guarantee that a wrong fit fails loud rather than
        # returning silently wrong.
        mps_cg = plain_fp32 and self.device.type == 'mps'

        # Effective convergence tolerance for the deviance-change paths (pure
        # float64 → the strict tol it can reach; ridge float32 → the √n·eps
        # round-off floor, where the penalty guarantees the stopped fit is
        # accurate). The plain float32 path does NOT use eff_tol: the same √n·eps
        # floor would stop it early, while slowly-converging low-hazard directions
        # are still moving, leaving a non-stationary iterate that the strict gate
        # then (correctly) refuses — a false negative on designs float32 can fit.
        eff_tol = tol
        if self.dtype == torch.float32 and pen_gpu is not None:
            fp32_floor = float(np.sqrt(n)) * float(torch.finfo(torch.float32).eps)
            eff_tol = max(tol, 2.0 * fp32_floor)

        with timer.section('irls'):
            for iteration in range(1, max_iter + 1):
                # Host float64 working state. Only the pure float64 (CUDA) path
                # reads μ/η back from the device; the mixed-precision paths (ridge,
                # plain float32) already hold them in float64 on the host (updated
                # below), so they skip this round-trip that would truncate them.
                if self.dtype == torch.float64:
                    mu_cpu = mu_gpu.cpu().numpy().astype(np.float64)
                    eta_cpu = eta_gpu.cpu().numpy().astype(np.float64)

                # Working quantities (CPU, using family/link)
                mu_eta_val = link.mu_eta(eta_cpu)    # dμ/dη
                var_mu = family.variance(mu_cpu)

                # Working response with the offset removed before the solve:
                # z = (η - offset) + (y - μ)/(dμ/dη). The offset is added back
                # into η when reconstructing it from the coefficients below.
                z_np = eta_cpu + (y_np - mu_cpu) / mu_eta_val
                if off_np is not None:
                    z_np = z_np - off_np

                # Working weights: w = wt * (dμ/dη)² / V(μ)
                w_np = wt_np * (mu_eta_val ** 2) / var_mu
                # NUMERICAL GUARD: prevents division by zero in weighted regression
                w_np = np.maximum(w_np, 1e-30)

                # Transfer to GPU
                z_gpu = torch.from_numpy(z_np).to(device=self.device, dtype=self.dtype)
                w_gpu = torch.from_numpy(w_np).to(device=self.device, dtype=self.dtype)

                # The plain float32 path's stationarity stop is the relative Newton
                # decrement at the CURRENT iterate (η = X·coef_cur). On MPS it is
                # recovered matrix-free from the CG result (mps_rel_decr below); on
                # the other paths it uses the host-copied XᵀWX / XᵀWz formed here.
                # Evaluated AFTER this iteration's step so the returned coefficients
                # are the freshest (most converged) ones.
                mps_rel_decr = None
                if mps_cg:
                    # Matrix-free CG solve of (XᵀWX) b = Xᵀ(W z) via H v = Xᵀ(W(X v)).
                    # XᵀWX is NEVER formed → the float32 condition number is not
                    # squared → no cold-context not-PD Cholesky crash. Warm-started
                    # from the current iterate.
                    b_prev_gpu = (
                        torch.from_numpy(coef_cur_np).to(
                            device=self.device, dtype=self.dtype)
                        if coef_cur_np is not None
                        else torch.zeros(p, device=self.device, dtype=self.dtype)
                    )
                    rhs_gpu = X_gpu.T @ (w_gpu * z_gpu)
                    coef_gpu = _mps_cg_solve(X_gpu, w_gpu, rhs_gpu, b_prev_gpu)

                    # Relative Newton decrement of the pre-step iterate (coef_cur =
                    # b_prev_gpu), recovered matrix-free from the CG result: the
                    # score is U = rhs − H·coef_cur and the Newton step
                    # Δ = coef_gpu − coef_cur = H⁻¹U, so λ² = UᵀΔ — algebraically the
                    # SAME quantity relative_newton_decrement computes, but on-device
                    # without forming XᵀWX. This is only the STOP signal; the
                    # load-bearing acceptance decision is the host float64 gate after
                    # the loop (unchanged).
                    if iteration > 1:
                        H_bprev = X_gpu.T @ (w_gpu * (X_gpu @ b_prev_gpu))
                        score_gpu = rhs_gpu - H_bprev
                        delta_gpu = coef_gpu - b_prev_gpu
                        lam2 = max(float(torch.dot(score_gpu, delta_gpu).item()), 0.0)
                        mps_rel_decr = lam2 / (abs(dev_old) + 0.1)
                else:
                    # WLS via the weighted normal equations (Xᵀ W X) β = Xᵀ W z,
                    # solved by Cholesky + two triangular solves. This is
                    # mathematically the same weighted least-squares step as the
                    # √w·X / √w·z lstsq formulation, but uses only Cholesky and
                    # triangular solves — which ARE supported on MPS (lstsq is NOT,
                    # so the old path failed outright on Apple Silicon) and are far
                    # cheaper than a full lstsq on the n×p system on CUDA. The p×p
                    # Gram solve also stays on-device, removing the lstsq's larger
                    # work and (on MPS) its unsupported-op fallback.
                    wX_gpu = w_gpu.unsqueeze(1) * X_gpu          # W X  (n×p)
                    XtWX = X_gpu.T @ wX_gpu                       # p×p
                    XtWz = X_gpu.T @ (w_gpu * z_gpu)             # p
                    if pen_gpu is not None:
                        # Ridge: add λ to the diagonal. Raises the smallest
                        # eigenvalue, so the float32 Cholesky is well-conditioned.
                        XtWX = XtWX + torch.diag(pen_gpu)

                    if plain_fp32 and iteration > 1:
                        XtWX_h = XtWX.detach().cpu().numpy().astype(np.float64)
                        XtWz_h = XtWz.detach().cpu().numpy().astype(np.float64)

                    try:
                        L = torch.linalg.cholesky(XtWX)
                        sol = torch.linalg.solve_triangular(
                            L, XtWz.unsqueeze(1), upper=False)
                        coef_gpu = torch.linalg.solve_triangular(
                            L.T, sol, upper=True).squeeze(1)
                    except torch._C._LinAlgError:
                        # The float32 weighted normal equations are not
                        # positive-definite — typically an ill-conditioned design at
                        # large n in float32. Stop and fall back to the exact CPU
                        # path below rather than raising; the caller still gets a
                        # result. (CUDA/ridge only — the MPS plain path uses CG.)
                        solve_failed = True
                        break

                # Update η = X @ β, μ = linkinv(η), and the deviance.
                if plain_fp32:
                    # Stationarity of the PRE-step iterate (coef_cur), from the
                    # XᵀWX / XᵀWz formed at it above. Computed here (after the solve)
                    # so that if coef_cur is already stationary we still return the
                    # fresher post-step coefficients rather than stopping one step
                    # short — which previously landed slightly off the CPU fit.
                    rel_decr = None
                    if iteration > 1:
                        rel_decr = (
                            mps_rel_decr if mps_cg
                            else relative_newton_decrement(
                                XtWX_h, XtWz_h, coef_cur_np, dev_old)
                        )
                    # R-style step-halving on the host (float64): damp a Newton step
                    # that overshoots (float32 mode on extreme early iterates: μ→0/1,
                    # tiny weights, large working responses) back toward the iterate
                    # we stepped from. require_decrease enforces monotone descent from
                    # iteration 2 (a valid predecessor); iteration 1 halves only on a
                    # non-finite deviance. Keeping η/μ float64 gives a clean signal.
                    coef_np = coef_gpu.detach().cpu().numpy().astype(np.float64)
                    halve_target = (
                        coef_cur_np if coef_cur_np is not None
                        else np.zeros(p, dtype=np.float64)
                    )
                    coef_np, eta_cpu, mu_cpu, dev_new = step_halve(
                        coef_np, halve_target, X_np, link, family, y_np, wt_np,
                        dev_old, require_decrease=iteration > 1, offset=off_np,
                    )
                    coef_gpu = torch.from_numpy(coef_np).to(
                        device=self.device, dtype=self.dtype)
                    eta_gpu = torch.from_numpy(eta_cpu).to(
                        device=self.device, dtype=self.dtype)
                    mu_gpu = torch.from_numpy(mu_cpu).to(
                        device=self.device, dtype=self.dtype)
                    coef_cur_np = coef_np   # defines next iteration's η
                elif pen_gpu is not None:
                    # Ridge path: the penalty makes the float32 solve accurate, so
                    # the coefficients are genuinely fp32-precision. Evaluate η in
                    # float64 on the host (X is well-conditioned here) so the
                    # convergence signal — the deviance — is not polluted by float32
                    # matmul noise. This is what lets a ridge GLM converge cleanly
                    # on the GPU at very large n.
                    coef_cpu = coef_gpu.detach().cpu().numpy().astype(np.float64)
                    eta_cpu = X_np @ coef_cpu
                    if off_np is not None:
                        eta_cpu = eta_cpu + off_np
                    eta_gpu = torch.from_numpy(eta_cpu).to(
                        device=self.device, dtype=self.dtype)
                    mu_cpu = link.linkinv(eta_cpu)
                    mu_gpu = torch.from_numpy(mu_cpu).to(
                        device=self.device, dtype=self.dtype)
                    dev_new = family.deviance(y_np, mu_cpu, wt_np)
                else:
                    # Pure float64 (CUDA) path: η on-device, read back at float64.
                    eta_gpu = X_gpu @ coef_gpu
                    if off_gpu is not None:
                        eta_gpu = eta_gpu + off_gpu
                    eta_cpu = eta_gpu.cpu().numpy().astype(np.float64)
                    mu_cpu = link.linkinv(eta_cpu)
                    mu_gpu = torch.from_numpy(mu_cpu).to(
                        device=self.device, dtype=self.dtype)
                    dev_new = family.deviance(y_np, mu_cpu, wt_np)

                # Deviance-change convergence for the eff_tol paths. The plain
                # float32 path stops once the pre-step iterate is stationary (its
                # post-step refinement, coef_gpu, is then returned).
                if plain_fp32:
                    if rel_decr is not None and rel_decr < _FP32_REL_DECREMENT_TOL:
                        converged = True
                        break
                elif abs(dev_new - dev_old) / (abs(dev_old) + 0.1) < eff_tol:
                    converged = True
                    break

                dev_old = dev_new

        # --- float32 acceptance gate: stationarity, not an fp64 tolerance -----
        # A6: a float32 fit that reached a stationary point at the float32 floor
        # IS converged and must not be refused for failing to meet an unreachable
        # float64 tolerance. We decide via the relative Newton decrement at the
        # final coefficients (float64, host) rather than the deviance-change flag:
        # a correct fit sits far below the threshold; an ill-conditioned fit whose
        # float32 solve stalled off the optimum sits far above it (and still fails
        # loud — A6's other half: no silent wrong answers). float64 paths already
        # reach the strict tol and keep their deviance-based `converged` flag.
        rel_decrement = None
        if self.dtype == torch.float32 and not solve_failed:
            coef_np = coef_gpu.detach().cpu().numpy().astype(np.float64)
            decrement = _newton_decrement(X_np, y_np, wt_np, coef_np, link, family, off_np)
            rel_decrement = decrement / (abs(dev_new) + 0.1)
            converged = bool(rel_decrement < _FP32_REL_DECREMENT_TOL)

        # A broken inner solve leaves no usable coefficients, so it always raises
        # (force cannot salvage it). A non-stationary float32 fit (or an
        # unconverged float64 fit) raises unless force=True. PyStatistics does NOT
        # silently fall back to CPU here (A6) — it names the explicit options.
        if solve_failed or (not converged and not force):
            if solve_failed:
                reason = ("the float32 inner solve broke down (XᵀWX not "
                          "positive-definite in float32)")
            elif self.dtype == torch.float32:
                reason = ("the float32 fit did not reach a stationary point "
                          f"(relative Newton decrement {rel_decrement:.2e} exceeds "
                          f"the float32 acceptance threshold "
                          f"{_FP32_REL_DECREMENT_TOL:.0e})")
            else:
                reason = f"IRLS did not converge within {max_iter} iterations"
            # Explicit remedies. gpu_fp64 is CUDA-only (Metal/MPS has no float64),
            # so on MPS it is omitted — suggesting it to an Apple-Silicon user would
            # just produce a guaranteed "CUDA required" error. The only same-machine
            # higher-precision fallback on MPS is backend='cpu'. We never substitute
            # the device automatically (A6) — the user chooses.
            options = [
                "  - backend='cpu' for a correct double-precision fit (slower)"
            ]
            if self.device.type != 'mps':
                options.append(
                    "  - backend='gpu_fp64' on CUDA for an exact GPU fit"
                )
            options.append(
                "  - a ridge-penalized fit (a different, better-conditioned "
                "estimator)"
            )
            options.append(
                "  - force=True to return the float32 fit anyway (fast, but may "
                "be inaccurate / unstable)"
            )
            raise NumericalError(
                f"GPU GLM did not produce a reliable fit: {reason}. This happens "
                f"for ill-conditioned designs at large n in float32 — typically a "
                f"log-link family (Poisson/Gamma) — where the float32 solve cannot "
                f"locate the optimum (most often on Apple Silicon / MPS, which has "
                f"no float64). PyStatistics does not silently substitute a CPU or "
                f"lower-precision result; choose explicitly:\n"
                + "\n".join(options)
            )

        if not converged:
            # force=True: the caller explicitly accepts a possibly-inaccurate
            # float32 fit. Return it, but flag it loudly on the result.
            warnings_list.append(
                f"GPU GLM IRLS did not converge in {max_iter} float32 iterations; "
                f"force=True so the (possibly inaccurate) GPU fit is returned. "
                f"Deviance={dev_new:.6f}."
            )

        n_iter = iteration if converged else max_iter
        dev = dev_new

        # ------------------------------------------------------------------
        # Transfer final results to CPU (float64)
        # ------------------------------------------------------------------
        with timer.section('data_transfer_to_cpu'):
            coefficients = coef_gpu.cpu().numpy().astype(np.float64)
            eta_final = (X_gpu @ coef_gpu).cpu().numpy().astype(np.float64)
            if off_np is not None:
                eta_final = eta_final + off_np

        # Recompute μ on CPU at float64 for maximum accuracy
        mu_final = link.linkinv(eta_final)

        # ------------------------------------------------------------------
        # Null deviance (intercept-only, matching R's glm.fit)
        # ------------------------------------------------------------------
        with timer.section('null_deviance'):
            null_deviance = self._null_deviance(y_np, wt_np, family, off_np)

        # ------------------------------------------------------------------
        # Dispersion
        # ------------------------------------------------------------------
        final_rank = p  # GPU lstsq doesn't give rank easily
        df_residual = n - final_rank
        if family.dispersion_is_fixed:
            dispersion = 1.0
        else:
            dispersion = dev / df_residual if df_residual > 0 else float('nan')

        # ------------------------------------------------------------------
        # AIC
        # ------------------------------------------------------------------
        with timer.section('aic'):
            aic = family.aic(y_np, mu_final, wt_np, final_rank, dispersion)

        # ------------------------------------------------------------------
        # Residuals (CPU, float64)
        # ------------------------------------------------------------------
        with timer.section('residuals'):
            resid_response = y_np - mu_final

            var_mu = family.variance(mu_final)
            resid_pearson = resid_response / np.sqrt(var_mu)

            resid_deviance = self._deviance_residuals(
                y_np, mu_final, wt_np, family
            )

            mu_eta_final = link.mu_eta(eta_final)
            resid_working = (y_np - mu_final) / mu_eta_final

        # ------------------------------------------------------------------
        # X'WX for SE computation (compute on CPU at float64 for accuracy)
        # ------------------------------------------------------------------
        with timer.section('XtWX'):
            w_final = wt_np * (mu_eta_final ** 2) / family.variance(mu_final)
            # NUMERICAL GUARD: prevents division by zero in weighted regression
            w_final = np.maximum(w_final, 1e-30)
            XtWX = (X_np * w_final[:, np.newaxis]).T @ X_np

        timer.stop()

        # ------------------------------------------------------------------
        # Build result
        # ------------------------------------------------------------------
        params = GLMParams(
            coefficients=coefficients,
            fitted_values=mu_final,
            linear_predictor=eta_final,
            residuals_working=resid_working,
            residuals_deviance=resid_deviance,
            residuals_pearson=resid_pearson,
            residuals_response=resid_response,
            deviance=dev,
            null_deviance=null_deviance,
            aic=aic,
            ic_param_count=final_rank + family.n_ic_dispersion_params,
            dispersion=dispersion,
            rank=final_rank,
            df_residual=df_residual,
            df_null=n - 1,
            n_iter=n_iter,
            converged=converged,
            family_name=family.name,
            link_name=link.name,
        )

        return Result(
            params=params,
            info={
                'method': (
                    'ridge_irls_gpu' if penalty is not None
                    else 'irls_cg_gpu' if mps_cg
                    else 'irls_cholesky_gpu'
                ),
                'penalized': penalty is not None,
                'device': str(self.device),
                'dtype': str(self.dtype),
                'device_name': self.device_name,
                'XtWX': XtWX,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=tuple(warnings_list),
        )

    # ------------------------------------------------------------------
    # Helpers (same logic as CPU backend)
    # ------------------------------------------------------------------

    @staticmethod
    def _null_deviance(
        y: NDArray, wt: NDArray, family: Family,
        offset: NDArray | None = None,
    ) -> float:
        """Compute null deviance matching R's glm.fit().

        R's glm.fit() with intercept=FALSE uses mu_null = linkinv(offset)
        (linkinv(0) when there is no offset).
        """
        n = len(y)
        link = family.link
        eta_null = np.zeros(n, dtype=np.float64) if offset is None else offset
        mu_null = link.linkinv(eta_null)
        return family.deviance(y, mu_null, wt)

    @staticmethod
    def _deviance_residuals(
        y: NDArray, mu: NDArray, wt: NDArray, family: Family
    ) -> NDArray:
        """Compute signed deviance residuals."""
        sign = np.sign(y - mu)

        if family.name == 'gaussian':
            d = (y - mu) ** 2
        elif family.name == 'binomial':
            mu_c = np.clip(mu, 1e-10, 1 - 1e-10)
            with np.errstate(divide='ignore', invalid='ignore'):
                term1 = np.where(y > 0, y * np.log(y / mu_c), 0.0)
                term2 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu_c)), 0.0)
            d = 2.0 * (term1 + term2)
        elif family.name == 'poisson':
            mu_c = np.maximum(mu, 1e-10)
            with np.errstate(divide='ignore', invalid='ignore'):
                term = np.where(y > 0, y * np.log(y / mu_c), 0.0)
            d = 2.0 * (term - (y - mu_c))
        elif family.name == 'Gamma':
            mu_c = np.maximum(mu, 1e-10)
            y_safe = np.maximum(y, 1e-10)
            d = 2.0 * ((y - mu_c) / mu_c - np.log(y_safe / mu_c))
        elif family.name == 'negative.binomial':
            theta = family.theta
            mu_c = np.maximum(mu, 1e-10)
            with np.errstate(divide='ignore', invalid='ignore'):
                term1 = np.where(y > 0, y * np.log(y / mu_c), 0.0)
                term2 = (y + theta) * np.log((y + theta) / (mu_c + theta))
            d = 2.0 * (term1 - term2)
        else:
            n = len(y)
            d = np.zeros(n, dtype=np.float64)
            ones = np.ones(1, dtype=np.float64)
            for i in range(n):
                yi = y[i:i+1]
                mui = mu[i:i+1]
                d[i] = family.deviance(yi, mui, ones)

        return sign * np.sqrt(np.maximum(wt * d, 0.0))
