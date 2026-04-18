"""SQUAREM acceleration for EM (Varadhan & Roland 2008).

Algorithm SqS3 from the paper, specialized for MVN MLE. Takes three
EM steps, extrapolates the sequence via a Steffensen-like update,
then guards monotonicity by back-halving the step until the
observed-data log-likelihood is non-decreasing.

Typical effect on well-behaved EM problems: 2–4× reduction in total
EM-step equivalents. Preserves the MLE — convergence point is
unchanged; only the path through parameter space is accelerated.

References
----------
Varadhan, R. & Roland, C. (2008). Simple and globally convergent
methods for accelerating the convergence of any EM algorithm.
Scandinavian Journal of Statistics, 35(2), 335-353.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def _pack(mu: np.ndarray, sigma: np.ndarray, p: int) -> np.ndarray:
    """Flatten (mu, sigma_lower_triangle) into a single vector."""
    return np.concatenate([mu, sigma[np.tril_indices(p)]])


def _unpack(theta: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of _pack."""
    mu = theta[:p]
    sigma = np.zeros((p, p))
    sigma[np.tril_indices(p)] = theta[p:]
    sigma = sigma + sigma.T - np.diag(np.diag(sigma))
    return mu, sigma


def squarem_step(
    mu: np.ndarray,
    sigma: np.ndarray,
    p: int,
    em_step: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    loglik_fn: Callable[[np.ndarray, np.ndarray], float],
    ensure_pd: Callable[[np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """One SQUAREM cycle.

    Runs three underlying EM steps plus (usually) one more monotonicity
    check, so each call costs roughly 3–4 EM-step-equivalents — but
    advances the iterate by substantially more than that in terms of
    gradient progress.

    Parameters
    ----------
    mu, sigma : current iterate
    p : int
        Dimension of mu (redundant but avoids recomputing).
    em_step : callable
        Function that performs one EM step: (mu, sigma) -> (mu', sigma').
    loglik_fn : callable
        Returns the observed-data log-likelihood at (mu, sigma).
    ensure_pd : callable
        Validates / projects sigma to the PD cone; raises NumericalError
        on irrecoverable non-PD.

    Returns
    -------
    (mu_new, sigma_new, em_steps_used)
        The accelerated iterate, plus the number of underlying EM-step
        calls this cycle consumed (for bookkeeping against max_iter).
    """
    # Three base EM steps (θ0 → θ1 → θ2 → θ3).
    theta0 = _pack(mu, sigma, p)

    mu1, sigma1 = em_step(mu, sigma)
    theta1 = _pack(mu1, sigma1, p)

    mu2, sigma2 = em_step(mu1, sigma1)
    theta2 = _pack(mu2, sigma2, p)

    r = theta1 - theta0
    v = theta2 - theta1 - r
    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))

    em_steps_used = 2

    # If the EM iteration has stalled (r or v near zero), fall back to
    # the plain EM iterate — no point extrapolating zero.
    if r_norm < 1e-14 or v_norm < 1e-14:
        return mu2, sigma2, em_steps_used

    # Steffensen-like step length. α = -||r|| / ||v||.
    alpha = -r_norm / v_norm

    # Safeguard loop: halve alpha toward -1 (plain EM) until the
    # extrapolated iterate is PD and the log-likelihood is non-
    # decreasing relative to theta2 (the plain-EM result).
    theta2_mu, theta2_sigma = mu2, sigma2
    ll_base = loglik_fn(theta2_mu, theta2_sigma)

    # Start with the full Steffensen step and back off if it hurts.
    for _ in range(20):
        theta_new = theta0 - 2.0 * alpha * r + (alpha ** 2) * v
        mu_new, sigma_new = _unpack(theta_new, p)

        try:
            sigma_new = ensure_pd(sigma_new)
        except Exception:
            alpha = (alpha - 1.0) / 2.0
            if alpha >= -1.0:
                alpha = -1.0
                # α = -1 is equivalent to the plain EM step from theta0,
                # which we already know is valid (= theta1). No further
                # back-off helps; accept theta2 as the safe iterate.
                return theta2_mu, theta2_sigma, em_steps_used
            continue

        # One more EM step to stabilise and evaluate likelihood.
        mu_refined, sigma_refined = em_step(mu_new, sigma_new)
        em_steps_used += 1

        try:
            sigma_refined = ensure_pd(sigma_refined)
        except Exception:
            alpha = (alpha - 1.0) / 2.0
            continue

        ll_new = loglik_fn(mu_refined, sigma_refined)
        if np.isfinite(ll_new) and ll_new >= ll_base - 1e-10:
            # Accept the accelerated iterate.
            return mu_refined, sigma_refined, em_steps_used

        # Monotonicity violated — shrink alpha toward -1.
        alpha = (alpha - 1.0) / 2.0
        if alpha >= -1.0:
            alpha = -1.0
            return theta2_mu, theta2_sigma, em_steps_used

    # Safeguard loop exhausted (shouldn't happen in practice). Return
    # the plain-EM iterate so we at least guarantee monotonicity.
    return theta2_mu, theta2_sigma, em_steps_used


def squarem_step_torch(
    mu, sigma, p, em_step, loglik_fn, torch_mod, device, dtype,
):
    """Torch-tensor SQUAREM step — same algorithm as ``squarem_step``
    but all intermediate state stays on the GPU.

    ``em_step`` is a closure that takes (mu, sigma) torch tensors and
    returns updated (mu, sigma) torch tensors; ``loglik_fn`` likewise.
    No numpy fallbacks in the iteration — SQUAREM's bookkeeping
    consists entirely of vector arithmetic, pack/unpack, and norm
    computations, all of which run natively on torch.

    Returns ``(mu_new, sigma_new, em_steps_used)``; ``em_steps_used``
    is a Python int for host-side bookkeeping against ``max_iter``.
    """
    torch = torch_mod
    tril_i, tril_j = torch.tril_indices(p, p, device=device).unbind(0)

    def pack(mu_, sigma_):
        return torch.cat([mu_, sigma_[tril_i, tril_j]])

    def unpack(theta_):
        mu_ = theta_[:p]
        tril_values = theta_[p:]
        sigma_ = torch.zeros((p, p), device=device, dtype=dtype)
        sigma_[tril_i, tril_j] = tril_values
        sigma_ = sigma_ + sigma_.T - torch.diag(torch.diag(sigma_))
        return mu_, sigma_

    theta0 = pack(mu, sigma)

    mu1, sigma1 = em_step(mu, sigma)
    theta1 = pack(mu1, sigma1)

    mu2, sigma2 = em_step(mu1, sigma1)
    theta2 = pack(mu2, sigma2)

    r = theta1 - theta0
    v = theta2 - theta1 - r
    r_norm = float(torch.linalg.norm(r).item())
    v_norm = float(torch.linalg.norm(v).item())

    em_steps_used = 2

    # Stall guard.
    if r_norm < 1e-14 or v_norm < 1e-14:
        return mu2, sigma2, em_steps_used

    alpha = -r_norm / v_norm

    ll_base_t = loglik_fn(mu2, sigma2)
    ll_base = float(ll_base_t.item())

    for _ in range(20):
        theta_new = theta0 - 2.0 * alpha * r + (alpha ** 2) * v
        mu_new, sigma_new = unpack(theta_new)

        # Safeguard against non-PD extrapolated iterates by attempting
        # Cholesky; back off alpha toward -1 on failure. We use try /
        # except because torch's cholesky raises a RuntimeError rather
        # than returning an error status on the GPU.
        try:
            _ = torch.linalg.cholesky(sigma_new)
        except RuntimeError:
            alpha = (alpha - 1.0) / 2.0
            if alpha >= -1.0:
                alpha = -1.0
                return mu2, sigma2, em_steps_used
            continue

        mu_refined, sigma_refined = em_step(mu_new, sigma_new)
        em_steps_used += 1

        try:
            _ = torch.linalg.cholesky(sigma_refined)
        except RuntimeError:
            alpha = (alpha - 1.0) / 2.0
            continue

        ll_new = float(loglik_fn(mu_refined, sigma_refined).item())
        if not (ll_new != ll_new) and ll_new >= ll_base - 1e-10:
            # Not NaN and non-decreasing.
            return mu_refined, sigma_refined, em_steps_used

        alpha = (alpha - 1.0) / 2.0
        if alpha >= -1.0:
            alpha = -1.0
            return mu2, sigma2, em_steps_used

    return mu2, sigma2, em_steps_used
