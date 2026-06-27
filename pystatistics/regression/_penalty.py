"""L2 (ridge) penalty support for linear and generalized linear models.

One job: the math of an L2-penalized fit — standardize the design, build the
penalized (weighted) least-squares system, and back-transform coefficients to the
original scale. The solver dispatch (which backend, CPU/GPU) stays in ``solvers``;
this module only knows penalties and scaling.

Convention (matches ``glmnet(alpha=0)`` / ``MASS::lm.ridge``):

- The intercept column (a leading all-ones column) is NEVER penalized.
- Non-intercept predictors are standardized (centered, scaled to unit standard
  deviation) before the penalty is applied, so the penalty strength ``l2`` is
  scale-invariant; coefficients are back-transformed to the original units.
- The penalized objective on the standardized predictors ``Z`` is
  ``||y - Zβ||² + l2·||β||²`` (weighted analogue for GLM), solved by augmenting the
  system with ``√l2`` rows rather than forming ``ZᵀZ + l2 I`` directly — the
  augmented QR is backward-stable and does not square the condition number.

A penalized fit does NOT carry the usual frequentist standard errors / t / p
values: they are not valid for a biased (penalized) estimator. Callers mark the
solution accordingly (see ``solvers``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Standardization:
    """How the design was standardized, for back-transforming coefficients.

    ``intercept_col`` is the index of the all-ones intercept column (or None if
    the design has no intercept). ``center``/``scale`` are per-column; the
    intercept column has center 0 and scale 1 (left untouched).
    """

    center: NDArray[np.float64]
    scale: NDArray[np.float64]
    intercept_col: int | None
    y_center: float


def _find_intercept(X: NDArray[np.float64]) -> int | None:
    """Index of an all-ones column (the intercept), or None."""
    for j in range(X.shape[1]):
        if np.allclose(X[:, j], 1.0):
            return j
    return None


def standardize(X: NDArray[np.float64], y: NDArray[np.float64],
                *, standardize_x: bool = True) -> tuple[NDArray[np.float64],
                                                        NDArray[np.float64],
                                                        Standardization]:
    """Center y and (optionally) standardize the non-intercept predictors.

    Returns ``(Z, y_centered, info)`` where ``Z`` is the standardized
    *predictor* matrix WITHOUT the intercept column (the intercept is handled
    analytically via the centering). ``info`` back-transforms coefficients.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    icol = _find_intercept(X)

    pred_cols = [j for j in range(p) if j != icol]
    center = np.zeros(p)
    scale = np.ones(p)
    y_center = float(y.mean()) if icol is not None else 0.0

    Xp = X[:, pred_cols]
    center_p = Xp.mean(axis=0)
    if standardize_x:
        sd = Xp.std(axis=0, ddof=0)
        sd = np.where(sd > 0, sd, 1.0)
    else:
        sd = np.ones(Xp.shape[1])
    Z = (Xp - center_p) / sd

    for k, j in enumerate(pred_cols):
        center[j] = center_p[k]
        scale[j] = sd[k]

    y_c = y - y_center
    return Z, y_c, Standardization(center=center, scale=scale,
                                   intercept_col=icol, y_center=y_center)


def back_transform(beta_z: NDArray[np.float64], info: Standardization,
                   p: int) -> NDArray[np.float64]:
    """Map standardized-predictor coefficients back to original-scale, full-length.

    ``beta_z`` are the coefficients on the standardized predictor columns (in
    predictor order, excluding the intercept). Returns a length-``p`` coefficient
    vector aligned to the original design columns, with the intercept recovered.
    """
    beta = np.zeros(p)
    pred_cols = [j for j in range(p) if j != info.intercept_col]
    beta_raw = beta_z / info.scale[pred_cols]
    for k, j in enumerate(pred_cols):
        beta[j] = beta_raw[k]
    if info.intercept_col is not None:
        beta[info.intercept_col] = info.y_center - float(
            np.dot(beta_raw, info.center[pred_cols]))
    return beta


def augmented_ridge_solve(Z: NDArray[np.float64], y_c: NDArray[np.float64],
                          l2: float, *, weights: NDArray[np.float64] | None = None
                          ) -> NDArray[np.float64]:
    """Solve ``min ||√w(y_c - Zβ)||² + l2·||β||²`` via an augmented QR.

    Stacks ``√l2·I`` under ``√w·Z`` (and zeros under ``√w·y_c``) so the solve is a
    plain least-squares problem — backward-stable, no normal-equations squaring.
    ``weights`` (the IRLS working weights) default to 1 (ordinary ridge).
    """
    if l2 < 0:
        raise ValueError(f"l2 penalty must be non-negative, got {l2}")
    n, k = Z.shape
    if weights is not None:
        sw = np.sqrt(np.maximum(weights, 0.0))
        Zw = Z * sw[:, None]
        yw = y_c * sw
    else:
        Zw, yw = Z, y_c
    Z_aug = np.vstack([Zw, np.sqrt(l2) * np.eye(k)])
    y_aug = np.concatenate([yw, np.zeros(k)])
    beta_z, *_ = np.linalg.lstsq(Z_aug, y_aug, rcond=None)
    return beta_z


def standardized_design(X: NDArray[np.float64]) -> tuple[NDArray[np.float64],
                                                         NDArray[np.float64],
                                                         NDArray[np.float64],
                                                         int | None]:
    """Standardize predictor columns in place of the design, keeping the intercept.

    Returns ``(A, center, scale, intercept_col)`` where ``A`` is ``X`` with each
    non-intercept column centered and scaled to unit population sd, and the
    intercept (all-ones) column left untouched. Used by penalized IRLS, which
    keeps the intercept in the design (unpenalized) rather than centering it out.
    """
    X = np.asarray(X, dtype=np.float64)
    p = X.shape[1]
    icol = _find_intercept(X)
    A = X.copy()
    center = np.zeros(p)
    scale = np.ones(p)
    for j in range(p):
        if j == icol:
            continue
        c = float(X[:, j].mean())
        s = float(X[:, j].std(ddof=0))
        s = s if s > 0 else 1.0
        A[:, j] = (X[:, j] - c) / s
        center[j] = c
        scale[j] = s
    return A, center, scale, icol


def weighted_augmented_solve(A: NDArray[np.float64], target: NDArray[np.float64],
                             penalty_diag: NDArray[np.float64],
                             weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve ``min ||√w(target - Aβ)||² + Σ penalty_diag_j β_j²`` (augmented LS).

    ``A`` includes the intercept column; ``penalty_diag`` is 0 on the intercept
    and ``l2`` elsewhere. The augmented form is backward-stable.
    """
    sw = np.sqrt(np.maximum(weights, 0.0))
    Aw = A * sw[:, None]
    tw = target * sw
    p = A.shape[1]
    A_aug = np.vstack([Aw, np.diag(np.sqrt(penalty_diag))])
    t_aug = np.concatenate([tw, np.zeros(p)])
    beta, *_ = np.linalg.lstsq(A_aug, t_aug, rcond=None)
    return beta


def back_transform_in_design(beta_A: NDArray[np.float64],
                             center: NDArray[np.float64],
                             scale: NDArray[np.float64],
                             intercept_col: int | None) -> NDArray[np.float64]:
    """Map coefficients on the standardized in-design parameterization to raw scale."""
    raw = beta_A / scale
    if intercept_col is not None:
        adj = sum(raw[j] * center[j] for j in range(len(raw)) if j != intercept_col)
        raw[intercept_col] = beta_A[intercept_col] - adj
    return raw
