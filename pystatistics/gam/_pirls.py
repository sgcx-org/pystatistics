"""Stable penalized IRLS for GAMs via augmented / reduced QR.

The 4.5.x fitter solved the penalized normal equations
``(X'WX + sum lambda_j S_j) beta = X'Wz`` — squaring the design's condition
number and, because 4.5.x smooths were unconstrained (each span contained the
constant), inverting an *exactly singular* matrix. This module replaces that
with the orthogonal approach mgcv itself uses (Wood 2011):

    min_beta || sqrt(w) (z - X beta) ||^2  +  sum_j lambda_j beta' S_j beta
    ==  min_beta || [ sqrt(w) X ]        [ sqrt(w) z ]      ||^2
                  || [ B_lam    ] beta - [     0     ]      ||

where ``B_lam' B_lam = sum_j lambda_j S_j`` (per-smooth eigen square roots,
cached once per fit). The QR of the augmented matrix has condition number
~sqrt of the normal equations', and every downstream quantity (beta, EDF,
log|A|, posterior covariance) comes from triangular solves against ``R``.

Reduced form: with ``M = sqrt(w) X = Q_x R_x`` and ``c1 = Q_x' sqrt(w) z``,
the problem is equivalent to ``min || [R_x; B_lam] beta - [c1; 0] ||^2`` —
a (p + r) x p problem independent of n. For Gaussian-identity fits the
weights never change, so the outer smoothing-parameter search reuses one
cached ``(R_x, c1)`` and each lambda evaluation costs O(p^3), not O(n p^2).

Robustness (both panel-verified failure modes):

- The penalized solve always uses COLUMN-PIVOTED QR (LAPACK ``dgeqp3``);
  a non-pivoted R-diagonal test does not reliably reveal rank (Kahan).
  All downstream consumers receive the pivoted factor plus the permutation.
- mu is clamped inside the family's valid domain and the linear predictor is
  bounded before the weight/working-response computation (mgcv ``gam.fit3``
  does the same); complete separation therefore surfaces as an R-style
  "fitted probabilities numerically 0 or 1" warning and a finite fit, never
  as inf/NaN inside the QR. Any residual non-finite value fails loud.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr_multiply, qr as _sqr, solve_triangular

from pystatistics.core.exceptions import ConvergenceError

if TYPE_CHECKING:
    from pystatistics.regression.families import Family

_RANK_TOL = 1e-11   # |R_ii| / max|R_jj| below this => dependent column
_MAX_HALVINGS = 30
_MU_EPS = 1e-10     # domain clamp distance (binomial: [eps, 1-eps])
_ETA_MAX = 700.0    # exp overflow guard on the linear predictor


# ---------------------------------------------------------------------------
# Penalty square roots (cached once per fit)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PenaltyRoot:
    """Eigen square root of one penalty block.

    ``rows`` is ``(r_j, k_j)`` with ``rows' rows = S_j`` (block coordinates);
    ``block = (start, end)`` locates the smooth's columns in the full design.
    ``group`` ties penalties that share a coefficient block: an ordinary
    smooth owns ONE root in its own group, a tensor-product smooth owns one
    root PER MARGIN — all in a single group over the same ``block``, so the
    REML penalty determinant is taken JOINTLY over the group (their null
    spaces overlap and do not decompose per root; see ``_penalty_group``).
    """

    rows: NDArray[np.floating[Any]]
    rank: int
    block: tuple[int, int]
    logdet_pos: float  # log pseudo-determinant of the unit-lambda block
    group: int         # penalties sharing a block (tensor margins) share this


def make_penalty_roots(
    S_blocks: list[NDArray[np.floating[Any]]],
    blocks: list[tuple[int, int]],
    groups: list[int] | None = None,
) -> list[PenaltyRoot]:
    """Eigen-decompose each (block-coordinate) penalty once.

    ``groups`` assigns each penalty to a determinant group (tensor margins
    share one); when ``None`` every penalty is its own singleton group —
    the block-orthogonal case, reproducing the pre-tensor numbers exactly.
    """
    if groups is None:
        groups = list(range(len(S_blocks)))
    roots: list[PenaltyRoot] = []
    for S, blk, grp in zip(S_blocks, blocks, groups):
        ev, U = np.linalg.eigh(S)
        tol = ev.max() * 1e-12 if ev.size else 0.0
        pos = ev > tol
        rows = np.sqrt(ev[pos])[:, None] * U[:, pos].T  # (r_j, k_j)
        roots.append(PenaltyRoot(
            rows=rows, rank=int(pos.sum()), block=blk,
            logdet_pos=float(np.sum(np.log(ev[pos]))),
            group=int(grp),
        ))
    return roots


def stack_penalty(
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
    p: int,
) -> NDArray[np.floating[Any]]:
    """Assemble ``B_lam`` with ``B_lam' B_lam = sum_j lambda_j S_j``."""
    if not roots:
        return np.zeros((0, p), dtype=np.float64)
    parts = []
    for root, lam in zip(roots, lambdas):
        s, e = root.block
        row = np.zeros((root.rows.shape[0], p), dtype=np.float64)
        row[:, s:e] = np.sqrt(lam) * root.rows
        parts.append(row)
    return np.vstack(parts)


# ---------------------------------------------------------------------------
# Reduced penalized weighted least squares
# ---------------------------------------------------------------------------

def reduce_wls(
    X: NDArray[np.floating[Any]],
    w: NDArray[np.floating[Any]],
    z: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], float]:
    """QR-reduce the weighted problem to p-space without materializing Q.

    Returns ``(R_x, c1, rss_perp)`` where ``M = sqrt(w) X = Q_x R_x``,
    ``c1 = Q_x' sqrt(w) z`` (via ``dormqr``, no explicit Q) and
    ``rss_perp = ||sqrt(w) z||^2 - ||c1||^2`` >= 0 (clamped against
    round-off; it is a squared norm by construction).

    Raises:
        ConvergenceError: if the weighted working response contains
            non-finite values (a diverged IRLS must fail loud, not feed
            NaN into LAPACK).
    """
    sw = np.sqrt(w)
    M = sw[:, None] * X
    zt = sw * z
    if not np.all(np.isfinite(zt)):
        raise ConvergenceError(
            "P-IRLS produced a non-finite weighted working response "
            "(diverging linear predictor); the fit cannot continue"
        )
    p = X.shape[1]
    # c1 = Q_x' zt computed as (zt' Q_x)' via dormqr — no explicit Q.
    ztq, R_x = qr_multiply(M, zt[None, :], mode="right")
    c1 = ztq[0, :p]
    rss_perp = float(max(zt @ zt - c1 @ c1, 0.0))
    return R_x[:p, :], c1, rss_perp


@dataclass(frozen=True)
class PenalizedSolve:
    """Pivoted triangular factorization of one penalized WLS solve.

    ``R`` is the (p, p) upper triangle of the PIVOTED augmented system;
    column i of ``R`` corresponds to original design column ``piv[i]``.
    ``rank < p`` means dependent columns were detected: their coefficients
    are zero and downstream H/covariance rows/cols are zero.
    """

    beta: NDArray[np.floating[Any]]   # original column order
    R: NDArray[np.floating[Any]]
    piv: NDArray[np.integer[Any]]
    rank: int


def solve_penalized(
    R_x: NDArray[np.floating[Any]],
    c1: NDArray[np.floating[Any]],
    B_lam: NDArray[np.floating[Any]],
    smooth_names: list[str] | None = None,
) -> PenalizedSolve:
    """Solve the reduced penalized LS problem via column-pivoted QR.

    Rank deficiency (concurvity, over-specified k surviving the penalty)
    is detected on the pivoted diagonal, reported with a warning naming
    the smooths, and handled by dropping the dependent columns — mirroring
    mgcv's proceed-with-warning. Never a silent ridge.
    """
    aug = np.vstack([R_x, B_lam])
    rhs = np.concatenate([c1, np.zeros(B_lam.shape[0])])
    p = R_x.shape[1]

    Q, R, piv = _sqr(aug, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    max_d = diag.max() if diag.size else 0.0
    rank = int(np.sum(diag > _RANK_TOL * max_d)) if max_d > 0.0 else 0

    if rank < p:
        names = ", ".join(smooth_names) if smooth_names else "the model"
        warnings.warn(
            f"GAM design is rank deficient (rank {rank} < {p} columns) — "
            f"coefficients of dependent columns in {names} set to zero "
            f"(concurvity or over-specified k)",
            UserWarning, stacklevel=3,
        )

    qtb = Q.T @ rhs
    beta_piv = np.zeros(p, dtype=np.float64)
    beta_piv[:rank] = solve_triangular(R[:rank, :rank], qtb[:rank])
    beta = np.zeros(p, dtype=np.float64)
    beta[piv] = beta_piv
    return PenalizedSolve(beta=beta, R=R, piv=piv, rank=rank)


# ---------------------------------------------------------------------------
# Family-domain guards
# ---------------------------------------------------------------------------

def _clamp_mu(
    family: Family, mu: NDArray[np.floating[Any]],
) -> tuple[NDArray[np.floating[Any]], bool]:
    """Clamp mu inside the family's valid open domain (mgcv gam.fit3 style).

    Returns the clamped mu and whether any clamping occurred (used for the
    R-style separation warning).
    """
    name = family.name
    if name == "binomial":
        clipped = np.clip(mu, _MU_EPS, 1.0 - _MU_EPS)
    elif name in ("poisson", "Gamma", "negative.binomial", "inverse.gaussian"):
        clipped = np.maximum(mu, _MU_EPS)
    else:
        return mu, False
    return clipped, bool(np.any(clipped != mu))


# ---------------------------------------------------------------------------
# P-IRLS loop
# ---------------------------------------------------------------------------

@dataclass
class PirlsFit:
    """Everything the criteria/EDF layers need from one fixed-lambda fit."""

    beta: NDArray[np.floating[Any]]
    mu: NDArray[np.floating[Any]]
    eta: NDArray[np.floating[Any]]
    w: NDArray[np.floating[Any]]
    deviance: float
    penalty: float                       # beta' S_lambda beta
    R: NDArray[np.floating[Any]]         # pivoted augmented triangle (p, p)
    R_x: NDArray[np.floating[Any]]       # weighted-design triangle (p, p)
    piv: NDArray[np.integer[Any]]        # column permutation for R
    rank: int
    n_iter: int
    converged: bool


def fit_fixed_lambda(
    y: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    roots: list[PenaltyRoot],
    lambdas: NDArray[np.floating[Any]],
    family: Family,
    tol: float,
    max_iter: int,
    smooth_names: list[str] | None = None,
    gaussian_cache: tuple[NDArray, NDArray, float] | None = None,
    mu_start: NDArray[np.floating[Any]] | None = None,
) -> PirlsFit:
    """Fit the GAM at fixed smoothing parameters via stable P-IRLS.

    Convergence is on the relative change of the PENALIZED deviance with
    step halving when it increases (mgcv ``gam.fit3`` behaviour). For the
    Gaussian-identity case the loop is one exact solve; pass
    ``gaussian_cache=(R_x, c1, rss_perp)`` to skip the O(n p^2) reduction.
    ``mu_start`` warm-starts the loop (the outer lambda search passes the
    previous evaluation's mu; nearby lambdas need ~2 instead of ~6 steps).

    Raises:
        ConvergenceError: if step halving cannot find an acceptable step or
            the working response diverges (loud, never a silent answer).
    """
    n = y.shape[0]
    p = X.shape[1]
    B_lam = stack_penalty(roots, lambdas, p)

    def pen_of(beta: NDArray) -> float:
        v = B_lam @ beta
        return float(v @ v)

    is_gauss_identity = (
        family.name == "gaussian" and family.link.name == "identity"
    )

    if is_gauss_identity:
        if gaussian_cache is not None:
            R_x, c1, rss_perp = gaussian_cache
        else:
            R_x, c1, rss_perp = reduce_wls(X, np.ones(n), y)
        sol = solve_penalized(R_x, c1, B_lam, smooth_names)
        eta = X @ sol.beta
        resid = c1 - R_x @ sol.beta
        deviance = rss_perp + float(resid @ resid)
        return PirlsFit(
            beta=sol.beta, mu=eta, eta=eta, w=np.ones(n), deviance=deviance,
            penalty=pen_of(sol.beta), R=sol.R, R_x=R_x, piv=sol.piv,
            rank=sol.rank, n_iter=1, converged=True,
        )

    wt = np.ones(n, dtype=np.float64)
    if mu_start is not None:
        mu = np.asarray(mu_start, dtype=np.float64).copy()
    else:
        mu = np.asarray(family.initialize(y), dtype=np.float64)
    mu, _ = _clamp_mu(family, mu)
    eta = np.clip(family.link.link(mu), -_ETA_MAX, _ETA_MAX)
    beta_old: NDArray | None = None
    pendev_old = family.deviance(y, mu, wt)  # penalty is 0 at beta=0
    converged = False
    sol: PenalizedSolve | None = None
    R_x = np.eye(p)
    mu_hit_boundary = False
    it = 0

    for it in range(1, max_iter + 1):
        dmu = family.link.mu_eta(eta)
        var = family.variance(mu)
        w = dmu * dmu / np.maximum(var, 1e-300)
        z = eta + (y - mu) / np.where(np.abs(dmu) > 1e-300, dmu, 1e-300)

        R_x, c1, _rss = reduce_wls(X, w, z)
        sol = solve_penalized(R_x, c1, B_lam, smooth_names)
        beta_new = sol.beta

        # Step halving on penalized-deviance increase or invalid state
        # (mgcv gam.fit3).
        step = 1.0
        for _ in range(_MAX_HALVINGS + 1):
            beta_try = (
                beta_new if beta_old is None
                else beta_old + step * (beta_new - beta_old)
            )
            eta_try = np.clip(X @ beta_try, -_ETA_MAX, _ETA_MAX)
            mu_try = family.link.linkinv(eta_try)
            mu_try, hit = _clamp_mu(family, mu_try)
            dev_try = family.deviance(y, mu_try, wt)
            pendev_try = dev_try + pen_of(beta_try)
            if np.isfinite(pendev_try) and (
                beta_old is None or pendev_try <= pendev_old * (1.0 + 1e-7)
            ):
                break
            step *= 0.5
        else:
            raise ConvergenceError(
                f"P-IRLS diverged at iteration {it}: penalized deviance "
                f"would not decrease after {_MAX_HALVINGS} step halvings"
            )

        mu_hit_boundary = mu_hit_boundary or hit
        beta, eta, mu = beta_try, eta_try, mu_try
        pendev_new = pendev_try
        rel = abs(pendev_new - pendev_old) / (abs(pendev_new) + 0.1)
        pendev_old = pendev_new
        beta_old = beta
        if rel < tol and it > 1:
            converged = True
            break

    if mu_hit_boundary and family.name == "binomial":
        warnings.warn(
            "fitted probabilities numerically 0 or 1 occurred "
            "(possible complete separation)",
            UserWarning, stacklevel=3,
        )

    # Final weights consistent with the returned mu; refresh BOTH triangular
    # factors so EDF / covariance / log|A| are computed at these weights
    # (the loop's factors belong to the previous iteration's weights).
    dmu = family.link.mu_eta(eta)
    var = family.variance(mu)
    w = dmu * dmu / np.maximum(var, 1e-300)
    R_x, c1, _ = reduce_wls(X, w, eta)
    sol_final = solve_penalized(R_x, c1, B_lam, smooth_names)
    deviance = family.deviance(y, mu, wt)

    return PirlsFit(
        beta=beta, mu=mu, eta=eta, w=w, deviance=deviance,
        penalty=pen_of(beta), R=sol_final.R, R_x=R_x, piv=sol_final.piv,
        rank=sol_final.rank, n_iter=it, converged=converged,
    )
