"""Analytic gradient of the GAM smoothing-parameter criterion (GLM families).

Extends the Gaussian-identity analytic gradient (``_gradient.py``) to the
GLM families, so the outer smoothing-parameter search for Poisson/binomial
(UBRE), free-dispersion GCV (e.g. Gamma) and fixed-dispersion Laplace REML
is driven by the exact criterion gradient with ONE inner P-IRLS fit per
outer step instead of the ``2m+1`` finite-difference fits (mgcv's
``gam.fit3``/``newton`` pattern — Wood 2011, the full implicit-derivative
form the Gaussian module deliberately left out).

For a GLM the IRLS weights depend on ``beta``, so ``rho = log(lambda)``
reaches the criterion along two routes: the direct ``S_lambda`` dependence
(as in the Gaussian case) and the implicit ``d beta / d rho`` route through
the P-IRLS fixed point. Writing the penalized score equation
``X'u(beta) = S_lambda beta`` (``u_i = (y_i - mu_i) mu'(eta_i) / V(mu_i)``
is the working score; the fixed point of Fisher-scoring P-IRLS satisfies it
exactly) and differentiating it implicitly:

    d beta / d rho_j = -lambda_j (X' Wn X + S_lambda)^{-1} S_j beta

where ``Wn = diag(-du_i/deta_i)`` are the FULL NEWTON weights — using the
fit's Fisher weights here is exact only for canonical links and biases the
gradient by up to several percent for non-canonical ones (probit,
Gamma-log; panel-verified), which would silently shift the selected
smoothness. With ``deta_j = X dbeta_j`` and ``omega_i = dw_i/deta_i`` the
derivative of the FISHER weight (the criterion's ``A = X'WX + S_lambda``
is built from Fisher weights, matching the P-IRLS factorization):

    dD/drho_j        = -2 u' deta_j
    d edf/drho_j     = -lambda_j tr(A^{-1} S_j H)
                       + sum_i omega_i deta_{j,i} (X A^{-1} S_lambda A^{-1} X')_ii
    d log|A|/drho_j  =  lambda_j tr(A^{-1} S_j)
                       + sum_i omega_i deta_{j,i} (X A^{-1} X')_ii
    d(D + pen)/drho_j = lambda_j beta' S_j beta          (envelope theorem)

from which the UBRE / GCV / fixed-dispersion REML gradients follow by the
chain rule (see each function). Every term is O(n p^2) or O(p^3) on the
factors the fit already produced — never an extra inner fit.

``u``, ``omega`` and ``Wn`` are obtained by DETERMINISTIC central
differences of the family's own scalar functions in ``eta`` (step
``1e-5 * max(|eta|, 1)``; the weight functions are smooth, so the FD error
is ~1e-10 relative — far below the inner P-IRLS convergence noise). This
keeps the module correct for EVERY family/link pair the fitter accepts
without growing the ``Family``/``Link`` contract; the whole gradient is
verified against finite differences of the actual criterion in the tests.

The Newton system is formed as normal equations on the pivoted rank block
(dropped columns stay dropped, their derivative is zero). That squares the
design's condition number, which is acceptable HERE because the result only
steers the optimizer — the criterion values it line-searches on still come
from the QR-stable path. A NUMERICALLY SINGULAR Newton system (reachable:
non-canonical links at the near-zero-penalty optima mgcv itself selects —
panel-verified on binomial-probit, n=60, where mgcv's optimum sits exactly
where the Newton eigenvalues hit ~4e-8) falls back, WITH a warning, to the
Fisher-weight implicit solve for that iterate: always defined (the fit's
stable pivoted factor), exact for canonical links, and only steering the
optimizer — never aborting a fit mgcv completes.

CONTRACT: the gradient is exact only at a CONVERGED P-IRLS fixed point (the
derivation differentiates the score equation), and only for the P-IRLS
branch the fit actually sits on — at near-zero penalty the inner problem
can be multimodal, which is why ``select_lambdas`` resolves the branch at
the accepted optimum (warm-continued vs fresh, better criterion wins) and
hands the winning branch's mu back for the caller's final fit. The
reported fit therefore always carries the criterion the search accepted,
never a silently different branch.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve

from pystatistics.gam._edf import influence_matrix, posterior_covariance
from pystatistics.gam._gradient import _penalty_terms
from pystatistics.gam._penalty_group import penalty_logdet_grad
from pystatistics.gam._pirls import PenaltyRoot, PirlsFit

if TYPE_CHECKING:
    from pystatistics.regression.families import Family

_FD_ETA_STEP = 1e-5   # relative central-difference step in eta (deterministic)
_FD_ETA_STEP2 = 1e-4  # step for the SECOND difference (optimal ~eps^(1/4))

# Canonical (family, link) pairs: there the Newton weights equal the Fisher
# weights exactly, so the criterion/gradient can stay on the fit's exact
# QR-stable factors. negative.binomial's canonical link (log(mu/(mu+theta)))
# is not offered, so nb is always non-canonical.
_CANONICAL_LINKS = {
    "gaussian": "identity",
    "poisson": "log",
    "binomial": "logit",
    "Gamma": "inverse",
}


def is_canonical(family: Family) -> bool:
    """True when the family uses its canonical link (Newton == Fisher)."""
    return _CANONICAL_LINKS.get(family.name) == family.link.name


def _eta_derivatives(
    family: Family,
    y: NDArray[np.floating[Any]],
    eta: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]],
           NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Working score ``u``, Fisher-weight derivative ``omega = dw/deta``,
    full Newton weight ``Wn = -du/deta`` and its eta-derivative
    ``omega_n = dWn/deta = -d2u/deta2``, elementwise in ``eta``.

    ``w(eta) = mu'(eta)^2 / V(mu(eta))`` and
    ``u(eta) = (y - mu(eta)) mu'(eta) / V(mu(eta))`` are evaluated through
    the family's own ``linkinv`` / ``mu_eta`` / ``variance``; derivatives
    are deterministic central differences (see module docstring). The
    second difference uses its own larger step (``_FD_ETA_STEP2``, the
    eps^(1/4) optimum for second derivatives — the first-difference step
    would put its roundoff at ~1e-6 relative).
    """
    h = _FD_ETA_STEP * np.maximum(np.abs(eta), 1.0)
    h2 = _FD_ETA_STEP2 * np.maximum(np.abs(eta), 1.0)

    def w_of(e: NDArray) -> NDArray:
        mu = family.link.linkinv(e)
        me = family.link.mu_eta(e)
        return me * me / np.maximum(family.variance(mu), 1e-300)

    def u_of(e: NDArray) -> NDArray:
        mu = family.link.linkinv(e)
        me = family.link.mu_eta(e)
        return (y - mu) * me / np.maximum(family.variance(mu), 1e-300)

    u = u_of(eta)
    omega = (w_of(eta + h) - w_of(eta - h)) / (2.0 * h)
    w_newton = -(u_of(eta + h) - u_of(eta - h)) / (2.0 * h)
    omega_n = -(u_of(eta + h2) - 2.0 * u + u_of(eta - h2)) / (h2 * h2)
    return u, omega, w_newton, omega_n


def _implicit_derivatives(
    fit: PirlsFit,
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    family: Family,
) -> tuple[list[tuple[float, NDArray[np.floating[Any]], int]],
           NDArray[np.floating[Any]], NDArray[np.floating[Any]],
           NDArray[np.floating[Any]], NDArray[np.floating[Any]],
           NDArray[np.floating[Any]]]:
    """Shared per-smooth implicit-derivative pieces.

    Returns ``(terms, s_lam, a_inv, u, omega, deta)`` where ``terms`` is
    the ``(lambda_j, S_j, rank_j)`` list, ``a_inv`` the unit-scale Fisher
    posterior covariance and ``deta`` the ``(n, m)`` matrix whose column
    ``j`` is ``X dbeta/drho_j`` from the Newton-weight implicit solve.

    A numerically singular Newton system falls back to the Fisher-weight
    implicit solve for the affected evaluation, with a warning (see module
    docstring) — never inf/nan into the optimizer, never an aborted fit.
    """
    p = fit.R.shape[0]
    terms, s_lam = _penalty_terms(roots, lambdas, p)
    a_inv = posterior_covariance(fit.R, fit.piv, fit.rank, 1.0)
    u, omega, w_newton, _omega_n = _eta_derivatives(family, y, fit.eta)

    # Newton system on the kept (pivoted rank-block) coordinates; dropped
    # columns have beta pinned at zero, so their derivative is zero too.
    kept = np.asarray(fit.piv[: fit.rank])
    Xk = X[:, kept]
    A_kk = (Xk * w_newton[:, None]).T @ Xk + s_lam[np.ix_(kept, kept)]
    # NOTE: LAPACK getrf only WARNS on an exactly-zero pivot (lu_solve then
    # yields inf/nan), so singularity is gated on finiteness below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # getrf zero-pivot RuntimeWarning
        lu = lu_factor(A_kk)

        n = y.shape[0]
        deta = np.empty((n, len(terms)), dtype=np.float64)
        for j, (lam, sj, _rank_j) in enumerate(terms):
            rhs = (sj @ fit.beta)[kept]
            dbeta = np.zeros(p, dtype=np.float64)
            dbeta[kept] = -lam * lu_solve(lu, rhs)
            deta[:, j] = X @ dbeta
    if not np.all(np.isfinite(deta)):
        # Singular Newton system: reachable for non-canonical links at the
        # near-zero-penalty optima mgcv itself selects (module docstring).
        # Use the always-defined Fisher implicit solve for this evaluation
        # (exact for canonical links, approximate otherwise) rather than
        # aborting a fit mgcv completes; the warning keeps it non-silent
        # and select_lambdas' fresh-refit check guards the accepted result.
        warnings.warn(
            "analytic smoothing-parameter gradient: the Newton system "
            f"X'WnX + S_lambda is numerically singular at lambda={lambdas!r}"
            "; using the Fisher-weight implicit solve for this evaluation",
            UserWarning, stacklevel=3,
        )
        for j, (lam, sj, _rank_j) in enumerate(terms):
            deta[:, j] = X @ (-lam * (a_inv @ (sj @ fit.beta)))
    return terms, s_lam, a_inv, u, omega, deta


def _deviance_edf_derivatives(
    fit: PirlsFit,
    terms: list[tuple[float, NDArray[np.floating[Any]], int]],
    s_lam: NDArray[np.floating[Any]],
    a_inv: NDArray[np.floating[Any]],
    u: NDArray[np.floating[Any]],
    omega: NDArray[np.floating[Any]],
    deta: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Per-smooth ``dD/drho`` and ``d edf/drho`` (see module docstring)."""
    h_mat = influence_matrix(fit.R, fit.R_x, fit.piv, fit.rank)
    m_mat = X @ a_inv                                       # X A^{-1}
    s_diag = np.einsum("ij,jk,ik->i", m_mat, s_lam, m_mat)  # (XA'SlamA'X')_ii
    d_dev = np.empty(len(terms), dtype=np.float64)
    d_edf = np.empty(len(terms), dtype=np.float64)
    for j, (lam, sj, _rank_j) in enumerate(terms):
        d_dev[j] = -2.0 * float(u @ deta[:, j])
        tr_term = -lam * float(np.einsum("ab,ba->", a_inv @ sj, h_mat))
        d_edf[j] = tr_term + float(np.sum(omega * deta[:, j] * s_diag))
    return d_dev, d_edf


def ubre_gradient_glm(
    fit: PirlsFit,
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    family: Family,
) -> NDArray[np.floating[Any]]:
    """``d UBRE / d rho`` for fixed-dispersion families (``phi = 1``).

    ``UBRE = D/n - 1 + 2 edf / n`` so
    ``d UBRE/d rho_j = D'_j / n + 2 edf'_j / n``.
    """
    n = y.shape[0]
    terms, s_lam, a_inv, u, omega, deta = _implicit_derivatives(
        fit, roots, lambdas, y, X, family)
    d_dev, d_edf = _deviance_edf_derivatives(
        fit, terms, s_lam, a_inv, u, omega, deta, X)
    return d_dev / n + 2.0 * d_edf / n


def gcv_gradient_glm(
    fit: PirlsFit,
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    family: Family,
    edf: float,
) -> NDArray[np.floating[Any]]:
    """``d GCV / d rho`` for free-dispersion GLM families.

    ``GCV = n D / (n - edf)^2`` so, with ``tau = n - edf``,
    ``d GCV/d rho_j = n D'_j / tau^2 + 2 n D edf'_j / tau^3``.
    """
    n = y.shape[0]
    terms, s_lam, a_inv, u, omega, deta = _implicit_derivatives(
        fit, roots, lambdas, y, X, family)
    d_dev, d_edf = _deviance_edf_derivatives(
        fit, terms, s_lam, a_inv, u, omega, deta, X)
    tau = n - edf
    return n * d_dev / tau**2 + 2.0 * n * fit.deviance * d_edf / tau**3


def _newton_hessian_kept(
    fit: PirlsFit,
    s_lam: NDArray[np.floating[Any]],
    w_newton: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]],
           tuple[NDArray[np.floating[Any]], bool] | None]:
    """``A_n = X'WnX + S_lambda`` on the kept (pivoted rank) block.

    Returns ``(A_n, Xk, chol)`` where ``chol`` is the ``cho_factor`` result
    when ``A_n`` is positive definite and None otherwise. The PD test is a
    Cholesky attempt — NOT a slogdet sign (an even-dimensional negative-
    definite matrix has POSITIVE determinant). The score and the gradient
    both go through here so they make the SAME deterministic PD decision.
    """
    kept = np.asarray(fit.piv[: fit.rank])
    Xk = X[:, kept]
    A_n = (Xk * w_newton[:, None]).T @ Xk + s_lam[np.ix_(kept, kept)]
    try:
        chol = cho_factor(A_n)
        if not np.all(np.isfinite(chol[0])):
            chol = None
    except np.linalg.LinAlgError:
        chol = None
    return A_n, Xk, chol


def reml_logdet_glm(
    fit: PirlsFit,
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    family: Family,
) -> float:
    """``log|X'WnX + S_lambda|`` for the non-canonical fixed-dispersion
    Laplace-REML score — mgcv's NEWTON-weight Hessian convention (verified:
    matches mgcv's reported REML to ~1e-8 on probit and nb where the Fisher
    determinant was 0.03–0.04 off). Falls back, WITH a warning, to the
    Fisher determinant from the fit's stable factor when the Newton Hessian
    is not positive definite (the Laplace approximation is then dubious for
    either convention; mgcv completes such fits — never a crash, never
    silent).
    """
    from pystatistics.gam._edf import logdet_penalized

    terms, s_lam = _penalty_terms(roots, lambdas, fit.R.shape[0])
    _u, _omega, w_newton, _omega_n = _eta_derivatives(family, y, fit.eta)
    _A_n, _Xk, chol = _newton_hessian_kept(fit, s_lam, w_newton, X)
    if chol is None:
        warnings.warn(
            "REML: the Newton Hessian X'WnX + S_lambda is not positive "
            f"definite at lambda={lambdas!r}; using the Fisher determinant "
            "for this evaluation",
            UserWarning, stacklevel=3,
        )
        return logdet_penalized(fit.R, fit.rank)
    return float(2.0 * np.sum(np.log(np.diag(chol[0]))))


def reml_gradient_glm(
    fit: PirlsFit,
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    family: Family,
) -> NDArray[np.floating[Any]]:
    """``d V / d rho`` for the fixed-dispersion Laplace-REML score.

    With ``phi = 1`` the score is
    ``V = -l(beta) + pen/2 + (log|A| - log|S_lambda|_+)/2 - (M_p/2) log 2pi``
    and ``d(-l + pen/2)/d rho_j = lambda_j beta' S_j beta / 2`` by the
    envelope theorem (the fixed point zeroes the beta-derivative), so

        d V/d rho_j = lambda_j beta' S_j beta / 2
                      + d log|A| / d rho_j / 2
                      - rank_j / 2

    ``A`` follows the criterion's convention: the fit's Fisher factor at
    canonical links (Newton == Fisher there), the NEWTON Hessian otherwise
    (``d log|A_n|/d rho_j = lambda_j tr(A_n^{-1} S_j) + sum_i omega_n_i
    deta_{j,i} (X A_n^{-1} X')_ii`` with ``omega_n = dWn/deta`` and
    ``deta_j = -lambda_j X A_n^{-1} S_j beta``). A non-PD Newton Hessian
    falls back to the Fisher terms, mirroring ``reml_logdet_glm`` exactly
    so value and gradient always describe the same criterion.
    """
    if is_canonical(family):
        return _reml_gradient_fisher(fit, roots, lambdas, y, X, family)

    terms, s_lam = _penalty_terms(roots, lambdas, fit.R.shape[0])
    _u, _omega, w_newton, omega_n = _eta_derivatives(family, y, fit.eta)
    _A_n, Xk, chol = _newton_hessian_kept(fit, s_lam, w_newton, X)
    if chol is None:
        # reml_logdet_glm warned and used the Fisher determinant for this
        # evaluation; differentiate the SAME (Fisher) criterion.
        return _reml_gradient_fisher(fit, roots, lambdas, y, X, family)

    kept = np.asarray(fit.piv[: fit.rank])
    m_k = cho_solve(chol, Xk.T).T                       # X A_n^{-1} (kept)
    a_diag = np.einsum("ij,ij->i", m_k, Xk)             # (X A_n^{-1} X')_ii
    beta = fit.beta
    d_logdet_s = penalty_logdet_grad(roots, lambdas)    # joint per-group
    grad = np.empty(len(terms), dtype=np.float64)
    for j, (lam, sj, _rank_j) in enumerate(terms):
        sj_beta = sj @ beta
        b_sj_b = float(beta @ sj_beta)
        sj_kk = sj[np.ix_(kept, kept)]
        tr_term = float(np.trace(cho_solve(chol, sj_kk)))
        deta = Xk @ (-lam * cho_solve(chol, sj_beta[kept]))
        d_logdet_a = (
            lam * tr_term + float(np.sum(omega_n * deta * a_diag))
        )
        grad[j] = 0.5 * lam * b_sj_b + 0.5 * d_logdet_a - 0.5 * d_logdet_s[j]
    return grad


def _reml_gradient_fisher(
    fit: PirlsFit,
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    family: Family,
) -> NDArray[np.floating[Any]]:
    """The Fisher-determinant REML gradient (exact at canonical links; the
    warned fallback otherwise — see ``reml_gradient_glm``)."""
    terms, _s_lam, a_inv, _u, omega, deta = _implicit_derivatives(
        fit, roots, lambdas, y, X, family)
    m_mat = X @ a_inv
    a_diag = np.einsum("ij,ij->i", m_mat, X)                # (XA^{-1}X')_ii
    beta = fit.beta
    d_logdet_s = penalty_logdet_grad(roots, lambdas)        # joint per-group
    grad = np.empty(len(terms), dtype=np.float64)
    for j, (lam, sj, _rank_j) in enumerate(terms):
        b_sj_b = float(beta @ (sj @ beta))
        tr_ainv_sj = float(np.einsum("ab,ba->", a_inv, sj))
        d_logdet_a = (
            lam * tr_ainv_sj + float(np.sum(omega * deta[:, j] * a_diag))
        )
        grad[j] = 0.5 * lam * b_sj_b + 0.5 * d_logdet_a - 0.5 * d_logdet_s[j]
    return grad
