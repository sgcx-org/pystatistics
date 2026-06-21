"""
Batched proportional-odds (ordinal logistic) imputation on the GPU (MICE ``polr``).

GPU counterpart of ``methods/polr.py``, vectorized over the ``m`` imputation
chains. R mice's default for ordered factors: fit a cumulative-logit model
``P(Y <= j | x) = sigmoid(alpha_j - x'beta)`` of the K-level ordered target on the
predictors, draw the threshold+slope vector once from its posterior normal
approximation, compute category probabilities for the missing rows, and sample.

Faithful to the CPU path in two load-bearing details:

  * **Raw (unconstrained) threshold parameterization for the fit.** The fit
    optimizes ``raw = [alpha_0, log(alpha_1 - alpha_0), ...]`` (the same transform
    as ``_likelihood.raw_to_thresholds``), so the thresholds stay strictly ordered
    by construction — no clipping, no FP32 ordering hazard.
  * **Natural-coordinate posterior draw.** Exactly as ``methods/polr.py`` +
    ``OrdinalSolution.vcov``, which (since the issue #5 fix) is the
    natural-coordinate covariance ``MASS::polr`` reports — NOT the raw-coordinate
    one. The Hessian is observed in raw coordinates, so the raw-coordinate draw
    deviation ``L^{-T} z ~ N(0, H_raw^{-1})`` is mapped to natural coordinates by
    the raw->natural Jacobian ``J`` (delta method): ``J L^{-T} z ~ N(0,
    J H_raw^{-1} J^T) = vcov_natural``. Drawing in raw coordinates (the pre-fix
    behaviour) gives the wrong between-imputation threshold variance.

The CPU path fits with L-BFGS-B; here we use a **batched damped Newton in raw
coords**. The gradient and the (observed) Hessian come from autograd on the batched
NLL: because the chains are independent, differentiating ``grad[:, d].sum()`` once
gives Hessian row ``d`` for *every* chain, so ``P+1`` backward passes yield all ``m``
Hessians — no per-chain loop and no ``torch.func`` (unavailable on this MPS build).
The Newton solve goes through ``cholesky_ex`` + the shared device-split SPD apply
(matmul-series inverse on MPS). Each step is globalised by a per-chain backtracking
line search (``_backtracking_step``): the full Newton step is halved until the
penalised NLL decreases for that chain. This is load-bearing — without it the
*unpenalised thresholds* overshoot into the saturated tail of the cumulative logits
on imbalanced ordinals (a sparse extreme category under (quasi-)separation), the
gradient vanishes, and the iterate sticks at a degenerate ``|alpha| ~ 1e6`` fit that
assigns every missing row one category (issue #8). Per-chain convergence freezing;
bounded iterations.

This is the heaviest GPU method (autograd double-backward per Newton step); an
analytical Hessian is a possible future optimization. Like the CPU method there is
no per-chain marginal fallback here — the slope objective ridge plus the line-search
globalisation keep the fit finite and on the CPU penalised MLE, and a genuinely
degenerate fit would still surface as a non-finite imputation caught by the
backend's end-of-sweep guard. Matches the CPU reference distributionally at the
GPU/FP32 tolerance tier. All randomness flows through the passed generator.
"""

from __future__ import annotations

from pystatistics.mice.backends._gpu_linreg import _cholesky_ridged
from pystatistics.mice.backends._gpu_spd import apply_inv_factor_T, solve_spd

# Relative ridge on the observed Hessian (matches the other GPU GLM methods).
_RIDGE = 1e-5
_MAX_NEWTON_ITER = 100
_NEWTON_TOL = 1e-8
_PROB_FLOOR = 1e-12
# Backtracking line-search budget for the damped Newton step. The full Newton
# step is halved until the penalised NLL decreases per chain; 50 halvings reach
# a step of ~1e-15, well past any useful progress, so a chain still not
# decreasing there is at a degenerate stall and is frozen with finite params.
_MAX_BACKTRACK = 50


def _raw_to_alpha(raw):
    """raw (m, K-1) -> ordered thresholds (m, K-1), autograd-friendly.
    ``increments = [raw_0, exp(raw_1), ...]``; ``alpha = cumsum(increments)``."""
    import torch

    inc = torch.cat([raw[:, :1], raw[:, 1:].exp()], dim=1)
    return torch.cumsum(inc, dim=1)


def _raw_to_natural_jacobian(raw, P):
    """Batched raw->natural threshold Jacobian ``d[alpha, beta] / d[raw, beta]``,
    shape (m, P, P). Mirrors ``ordinal._information.raw_to_natural_jacobian`` (the
    CPU delta-method transform) batched over the m chains, so the GPU draw uses the
    same natural-coordinate covariance as the CPU ``polr`` and ``MASS::polr``.

    With ``alpha_0 = raw_0`` and ``alpha_j = alpha_{j-1} + exp(raw_j)``:
    ``d(alpha_j)/d(raw_0) = 1`` for all thresholds j (column 0), and
    ``d(alpha_j)/d(raw_k) = exp(raw_k)`` for ``j >= k`` (k >= 1), else 0. The slope
    block is the identity and the threshold/slope cross-blocks are zero."""
    import torch

    m, knr = raw.shape
    jac = torch.eye(P, dtype=raw.dtype, device=raw.device).expand(m, P, P).clone()
    idx = torch.arange(knr, device=raw.device)
    lower = (idx[:, None] >= idx[None, :]).to(raw.dtype)            # (knr, knr) j>=k
    block = lower[None] * raw.exp()[:, None, :]                    # col k = exp(raw_k), j>=k
    block[:, :, 0] = 1.0                                            # col 0: all-ones (alpha_0=raw_0)
    jac[:, :knr, :knr] = block
    return jac


def _cat_logprobs(alpha, eta, K):
    """Cumulative-logit category log-probs. ``alpha`` (m, K-1), ``eta`` (m, n) ->
    (m, n, K). ``P(Y=j) = sigmoid(alpha_j - eta) - sigmoid(alpha_{j-1} - eta)`` with
    the j=0 / j=K-1 boundaries differenced against 0 / 1."""
    import torch

    cum = torch.sigmoid(alpha[:, None, :] - eta[:, :, None])        # (m, n, K-1)
    ones = torch.ones_like(cum[:, :, :1])
    zeros = torch.zeros_like(cum[:, :, :1])
    cum_hi = torch.cat([cum, ones], dim=2)                         # P(Y<=j), j=0..K-1
    cum_lo = torch.cat([zeros, cum], dim=2)                        # P(Y<=j-1)
    return (cum_hi - cum_lo).clamp_min(_PROB_FLOOR).log()


def _nll_per_chain(params, y_obs, X_obs, K, slope_ridge):
    """Per-chain penalised NLL, shape (m,). ``params`` (m, (K-1)+q).

    ``slope_ridge`` (m,) is the per-chain objective ridge coefficient. The
    penalty ``0.5 * lambda_m * ||beta_m||^2`` is added on the SLOPES only
    (thresholds unpenalised), mirroring the CPU ``polr`` objective ridge in
    ``methods/polr.py`` (``cumulative_negloglik`` adds ``0.5*ridge*beta@beta``).

    This is the single source of truth for the optimisation objective: the
    Newton gradient/Hessian differentiate its sum (``_batched_nll``) and the
    line search compares it per chain, so the damped step is accepted against
    exactly the objective the step minimises."""
    import torch

    knr = K - 1
    raw, beta = params[:, :knr], params[:, knr:]
    eta = (X_obs @ beta.unsqueeze(-1)).squeeze(-1)                  # (m, n)
    logp = _cat_logprobs(_raw_to_alpha(raw), eta, K)               # (m, n, K)
    logp_y = torch.gather(logp, 2, y_obs.to(torch.int64).unsqueeze(-1)).squeeze(-1)
    penalty = 0.5 * slope_ridge * (beta * beta).sum(dim=1)         # (m,) slopes only
    return -logp_y.sum(dim=1) + penalty


def _batched_nll(params, y_obs, X_obs, K, slope_ridge):
    """Scalar total NLL over all chains (the sum of ``_nll_per_chain``).

    Because the chains are independent and the total NLL sums over them,
    differentiating this contributes ``lambda_m * beta_m`` to chain ``m``'s
    gradient and ``lambda_m`` to its beta-block Hessian diagonal — so the slope
    penalty curvature flows into the observed Hessian (and hence ``L`` / the
    posterior covariance) via autograd, exactly as the penalised CPU fit. The
    slope penalty keeps the proportional-odds slopes finite under separation; it
    is complementary to (a) the Hessian-solve ridge in ``batched_polr_newton``
    (linear-solve stability only) and (b) the line-search globalisation there
    (which is what bounds the *unpenalised thresholds*, the dominant runaway on
    imbalanced real-survey ordinals)."""
    return _nll_per_chain(params, y_obs, X_obs, K, slope_ridge).sum()


def _grad_and_hessian(params, y_obs, X_obs, K, slope_ridge):
    """Exact batched gradient (m, P) and observed Hessian (m, P, P) via P+1
    backward passes (chains independent -> one backward per Hessian row covers
    all chains). ``slope_ridge`` (m,) is threaded into the penalised NLL so the
    returned gradient and Hessian both include the objective-ridge curvature."""
    import torch

    P = params.shape[1]
    params = params.detach().requires_grad_(True)
    nll = _batched_nll(params, y_obs, X_obs, K, slope_ridge)
    g = torch.autograd.grad(nll, params, create_graph=True)[0]      # (m, P)
    rows = [
        torch.autograd.grad(g[:, d].sum(), params, retain_graph=(d < P - 1))[0]
        for d in range(P)
    ]
    return g.detach(), torch.stack(rows, dim=1).detach()


def _starting_raw(y_obs, K):
    """Empirical-proportion raw thresholds, shared across chains (the observed
    target is identical per chain). Reuses the CPU starting-value helper so the
    GPU fit starts exactly where the CPU fit does."""
    import numpy as np
    from pystatistics.ordinal._solver import _compute_starting_values
    from pystatistics.regression.families import LogitLink

    y_codes = y_obs[0].detach().to("cpu").numpy().astype(np.intp)
    raw = _compute_starting_values(y_codes, int(K), LogitLink(), 0)  # p=0 -> raw only
    return raw[: K - 1]


def _backtracking_step(params, delta, y_obs, X_obs, K, slope_ridge, frozen):
    """Per-chain backtracking line-search scale for the Newton step ``delta``.

    Returns ``step`` (m,): the largest of ``1, 1/2, 1/4, ...`` for which
    ``NLL(params - step*delta) <= NLL(params)`` per chain (a small relative
    slack absorbs floating-point noise at the full step near convergence).
    ``frozen`` chains (already converged) are not searched — their step is held
    at 1 and ignored by the caller.

    This globalises the Newton iteration. A full undamped step overshoots the
    *unpenalised thresholds* into the saturated tail of the cumulative logits on
    imbalanced ordinals (a sparse extreme category), where the gradient vanishes
    and the iterate is stuck at a degenerate |alpha| ~ 1e6 point — the issue #8
    collapse. Requiring a per-chain decrease keeps every step in the basin of
    the finite penalised MLE, matching the CPU L-BFGS-B globalisation. Evaluated
    without autograd (objective values only)."""
    import torch

    with torch.no_grad():
        f0 = _nll_per_chain(params, y_obs, X_obs, K, slope_ridge)   # (m,)
        slack = 1e-8 * (1.0 + f0.abs())
        step = torch.ones_like(f0)
        for _ in range(_MAX_BACKTRACK):
            trial = params - step.unsqueeze(-1) * delta
            f1 = _nll_per_chain(trial, y_obs, X_obs, K, slope_ridge)
            ok = frozen | (f1 <= f0 + slack)
            if bool(ok.all()):
                break
            step = torch.where(ok, step, step * 0.5)
    return step


def batched_polr_newton(y_obs, X_obs, n_classes):
    """Batched damped Newton in raw coords. ``y_obs`` (m, n_obs) of 0..K-1 ordered
    indices, ``X_obs`` (m, n_obs, q) WITHOUT intercept (thresholds are the
    intercepts). Returns ``(alpha_hat (m, K-1), beta_hat (m, q), raw_hat (m, K-1),
    L)`` — natural thresholds, slopes, the raw thresholds at the optimum (for the
    raw->natural Jacobian), and the Cholesky of the ridged observed Hessian in
    *raw* coords (whose inverse is the raw-coordinate covariance).

    Each Newton step is globalised by a per-chain backtracking line search
    (``_backtracking_step``) so the unpenalised thresholds cannot overshoot into
    saturation under (quasi-)separation — the issue #8 failure mode. Combined
    with the slope objective ridge, the fit reproduces the CPU penalised MLE."""
    import torch

    K = int(n_classes)
    m, n, q = X_obs.shape
    knr = K - 1
    P = knr + q
    dtype, device = X_obs.dtype, X_obs.device

    params = torch.zeros((m, P), dtype=dtype, device=device)
    raw0 = _starting_raw(y_obs, K)
    params[:, :knr] = torch.as_tensor(raw0, dtype=dtype, device=device)

    # mean over predictors of the per-column second moment, sum_i X_ij^2.
    second_moment = (X_obs * X_obs).sum(dim=1).mean(dim=1)          # (m,)
    ridge_diag = _RIDGE * second_moment.clamp_min(1e-12) / max(n, 1)  # Hessian-solve ridge (m,)
    # Scale-aware objective ridge on the SLOPES, mirroring the CPU
    # ``methods/polr._slope_ridge`` exactly: lambda = _RIDGE * max(diag_scale,
    # 1e-12) * n_obs with diag_scale = second_moment / n_obs. The n_obs factor
    # cancels algebraically but is kept explicit so the formula reads as the CPU
    # one and stays correct if either side changes. This penalises the
    # proportional-odds slopes so the MLE stays finite under (quasi-)complete
    # separation; complementary to ``ridge_diag`` above (solve stability only).
    slope_ridge = _RIDGE * (second_moment / max(n, 1)).clamp_min(1e-12) * max(n, 1)  # (m,)
    eye = torch.eye(P, dtype=dtype, device=device)

    converged = torch.zeros(m, dtype=torch.bool, device=device)
    for _ in range(_MAX_NEWTON_ITER):
        g, H = _grad_and_hessian(params, y_obs, X_obs, K, slope_ridge)
        L = _cholesky_ridged(H + ridge_diag[:, None, None] * eye)
        delta = solve_spd(L, g.unsqueeze(-1)).squeeze(-1)          # (m, P) Newton step
        step = _backtracking_step(params, delta, y_obs, X_obs, K, slope_ridge, converged)
        apply = (~converged).to(dtype) * step                     # frozen -> 0 step
        update = apply.unsqueeze(-1) * delta
        small = update.abs().amax(dim=1) < _NEWTON_TOL
        params = params - update                                  # damped, minimise NLL
        converged = converged | (~converged & small)
        if bool(converged.all()):
            break

    g, H = _grad_and_hessian(params, y_obs, X_obs, K, slope_ridge)
    L = _cholesky_ridged(H + ridge_diag[:, None, None] * eye)
    raw_hat = params[:, :knr]
    return _raw_to_alpha(raw_hat), params[:, knr:], raw_hat, L


def draw_natural_theta(alpha_hat, beta_hat, raw_hat, L, gen):
    """One natural-coordinate posterior draw of ``[alpha, beta]`` per chain,
    ``theta* ~ N([alpha_hat, beta_hat], vcov_natural)`` — matching CPU ``polr`` /
    ``MASS::polr``. Returns (m, P).

    The Hessian is observed in raw coordinates, so the deviation is drawn there
    (``L^{-T} z ~ N(0, H_raw^{-1})``) and mapped to natural coordinates by the
    raw->natural Jacobian (delta method): ``J L^{-T} z ~ N(0, J H_raw^{-1} J^T) =
    vcov_natural``. Drawing in raw coordinates (the pre-fix behaviour) gives the
    wrong between-imputation threshold variance."""
    import torch

    knr = alpha_hat.shape[1]
    m, q = beta_hat.shape
    P = knr + q
    mean = torch.cat([alpha_hat, beta_hat], dim=1)                 # (m, P)
    z = torch.randn((m, P), generator=gen, dtype=L.dtype, device=L.device)
    delta_raw = apply_inv_factor_T(L, z.unsqueeze(-1))             # (m, P, 1)
    jac = _raw_to_natural_jacobian(raw_hat, P)                     # (m, P, P)
    return mean + (jac @ delta_raw).squeeze(-1)


def _sample_categories(probs, gen):
    """Inverse-CDF sample one class per row. probs (m, n, K) -> (m, n) indices.
    Clips tiny negatives from a draw whose thresholds nudged out of order."""
    import torch

    probs = probs.clamp_min(0.0)
    probs = probs / probs.sum(dim=2, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)
    cdf = probs.cumsum(dim=2)
    u = torch.rand(
        probs.shape[:2] + (1,), generator=gen, dtype=probs.dtype, device=probs.device
    )
    return (cdf >= u).to(torch.int64).argmax(dim=2)


def gpu_polr_impute(y_obs, X_obs, X_mis, gen, *, donors=None, n_classes=None):
    """Batched proportional-odds imputation (R ``polr``).

    ``y_obs`` (m, n_obs) of 0..K-1 ordered class indices, ``X_obs`` (m, n_obs, q),
    ``X_mis`` (m, n_mis, q); ``n_classes`` = K required (passed by the sweep).
    Returns (m, n_mis) of 0..K-1 indices. ``donors`` is accepted and ignored.
    """
    if n_classes is None:
        raise ValueError("gpu_polr_impute requires n_classes (number of levels)")
    K = int(n_classes)
    knr = K - 1

    alpha_hat, beta_hat, raw_hat, L = batched_polr_newton(y_obs, X_obs, K)

    theta = draw_natural_theta(alpha_hat, beta_hat, raw_hat, L, gen)
    alpha_s, beta_s = theta[:, :knr], theta[:, knr:]

    eta = (X_mis @ beta_s.unsqueeze(-1)).squeeze(-1)
    probs = _cat_logprobs(alpha_s, eta, K).exp()
    return _sample_categories(probs, gen).to(X_obs.dtype)
