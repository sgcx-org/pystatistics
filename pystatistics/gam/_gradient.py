"""Analytic gradient of the GAM smoothing-parameter criterion (Gaussian).

For the Gaussian-identity model the outer smoothing-parameter search can be
driven by the EXACT gradient of the GCV / Laplace-REML score with respect to
``rho = log(lambda)`` instead of finite differences, cutting the per-outer-step
cost from ``2m+1`` inner fits to a single fit for ``m`` smooths (mgcv's
``gam.fit3``/``newton`` does the same; this is the A.3 pattern of the mixed
module — Wood 2011 — specialised to constant IRLS weights).

Because the weights are constant (``W = I``), the criterion depends on ``rho``
only through the closed-form penalized solve, with no implicit
``d beta / d rho`` weight term. Writing ``A = X'X + S_lambda`` (``A = R'R`` for
the augmented triangle ``R``), ``H = A^{-1} X'X`` the influence matrix and
``beta`` the penalized coefficients:

    d edf / d rho_j       = -lambda_j * tr(A^{-1} S_j H)
    d D   / d rho_j       =  2 * lambda_j * (A^{-1} S_lambda beta)' (S_j beta)
    d(D + penalty)/d rho_j =  lambda_j * beta' S_j beta        (envelope theorem)

from which GCV = n D / (n - edf)^2 and the profiled Laplace-REML score follow
by the chain rule (see each function). Every term is an O(p^3) operation on the
p-by-p factors the fit already produced — ``A^{-1}`` is the posterior
covariance evaluated at unit scale. Pure numpy.

Scope: Gaussian-identity ONLY. For GLM families the IRLS weights depend on
``beta``, so the score gains an implicit ``d beta / d rho`` term (the full
Wood 2011 form) — that lives in ``_gradient_glm``, which shares this
module's ``_penalty_terms``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatistics.gam._edf import influence_matrix, posterior_covariance
from pystatistics.gam._pirls import PenaltyRoot, PirlsFit


def _penalty_terms(
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
    p: int,
) -> tuple[list[tuple[float, NDArray[np.floating[Any]], int]],
           NDArray[np.floating[Any]]]:
    """Per-smooth ``(lambda_j, S_j, rank_j)`` and the assembled ``S_lambda``.

    ``S_j`` is the full ``(p, p)`` penalty (zero outside the smooth's block),
    reconstructed from the cached eigen square root (``rows' rows = S_j``);
    ``S_lambda = sum_j lambda_j S_j`` matches the fit's ``B_lam' B_lam``.
    """
    terms: list[tuple[float, NDArray[np.floating[Any]], int]] = []
    s_lam = np.zeros((p, p), dtype=np.float64)
    for root, lam in zip(roots, lambdas):
        s, e = root.block
        sj = np.zeros((p, p), dtype=np.float64)
        sj[s:e, s:e] = root.rows.T @ root.rows
        s_lam += float(lam) * sj
        terms.append((float(lam), sj, root.rank))
    return terms, s_lam


def gcv_gradient(
    fit: PirlsFit,
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
    n: int,
    edf: float,
) -> NDArray[np.floating[Any]]:
    """``d GCV / d rho`` for the Gaussian-identity model, ``rho = log lambda``.

    ``GCV = n D / (n - edf)^2`` so, with ``tau = n - edf``,

        d GCV / d rho_j = n * D'_j / tau^2 + 2 n D * edf'_j / tau^3

    using ``D'_j = 2 lambda_j (A^{-1} S_lambda beta)'(S_j beta)`` and
    ``edf'_j = -lambda_j tr(A^{-1} S_j H)``.
    """
    p = fit.R.shape[0]
    a_inv = posterior_covariance(fit.R, fit.piv, fit.rank, 1.0)  # A^{-1}
    h = influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank)
    terms, s_lam = _penalty_terms(roots, lambdas, p)
    beta = fit.beta
    d = fit.deviance
    tau = n - edf
    # p_vec = A^{-1} S_lambda beta  (so D'_j = 2 lambda_j p_vec' S_j beta)
    p_vec = a_inv @ (s_lam @ beta)

    grad = np.empty(len(terms), dtype=np.float64)
    for j, (lam, sj, _rank_j) in enumerate(terms):
        sj_beta = sj @ beta
        d_dev = 2.0 * lam * float(p_vec @ sj_beta)
        # edf'_j = -lambda_j tr(A^{-1} S_j H)
        d_edf = -lam * float(np.einsum("ab,ba->", a_inv @ sj, h))
        grad[j] = n * d_dev / tau**2 + 2.0 * n * d * d_edf / tau**3
    return grad


def reml_gradient_gauss(
    fit: PirlsFit,
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
    n: int,
) -> NDArray[np.floating[Any]]:
    """``d V / d rho`` for the Gaussian-identity Laplace-REML score.

    The score reduces (constants dropped) to

        V = (n - M_p)/2 * log(D + penalty) + (log|A| - log|S_lambda|_+)/2

    so, with ``phi = (D + penalty)/(n - M_p)``, ``d(D+penalty)/d rho_j``
    collapsing to ``lambda_j beta' S_j beta`` (envelope),
    ``d log|A|/d rho_j = lambda_j tr(A^{-1} S_j)`` and
    ``d log|S_lambda|_+/d rho_j = rank_j``:

        d V / d rho_j = lambda_j beta'S_j beta / (2 phi)
                        + lambda_j tr(A^{-1} S_j) / 2 - rank_j / 2
    """
    p = fit.R.shape[0]
    a_inv = posterior_covariance(fit.R, fit.piv, fit.rank, 1.0)  # A^{-1}
    terms, _s_lam = _penalty_terms(roots, lambdas, p)
    beta = fit.beta
    rank_s = sum(r.rank for r in roots)
    m_p = max(fit.rank - rank_s, 0)
    d_p = fit.deviance + fit.penalty
    phi = d_p / (n - m_p)

    grad = np.empty(len(terms), dtype=np.float64)
    for j, (lam, sj, rank_j) in enumerate(terms):
        b_sj_b = float(beta @ (sj @ beta))
        tr_ainv_sj = float(np.einsum("ab,ba->", a_inv, sj))
        grad[j] = (
            lam * b_sj_b / (2.0 * phi)
            + 0.5 * lam * tr_ainv_sj
            - 0.5 * rank_j
        )
    return grad
