"""IRLS step-control helpers for the float32 GPU GLM backend.

The float32 IRLS path needs more care than the float64 (CPU/CUDA-fp64) path: its
early iterates have the worst dynamic range (μ→0/1, tiny weights, large working
responses), and its deviance can only fall to a √n·eps round-off floor, not to the
strict float64 tolerance. These two helpers make the float32 iteration land at a
genuinely stationary point so the strict acceptance gate accepts the fits float32
can actually produce — instead of quitting early and (correctly, but uselessly)
refusing them. The CPU/float64 path needs neither and uses neither.

  - ``step_halve`` — damped Newton step for the float32 path. With
    ``require_decrease=True`` it enforces monotone deviance descent: a float32 step
    that overshoots (raising the deviance) is halved back toward the iterate it
    stepped from, which keeps the iteration inside the stationary basin so the
    Newton decrement can settle below the acceptance threshold. With
    ``require_decrease=False`` (R's actual rule) it only intervenes on a *non-finite*
    deviance — a defensive backstop that, in practice, the families' μ/η clipping
    (η∈[-500,500]) already pre-empts, so it is effectively a no-op there. A finite
    increasing step is accepted, exactly as R's ``glm.fit`` does (Fisher scoring is
    not a descent method); this is why the CPU path, which must match R, does not
    use the damped mode.

  - ``relative_newton_decrement`` — the affine-invariant stationarity measure
    λ²/(|dev|+0.1) computed from the IRLS normal-equation pieces XᵀWX and XᵀWz that
    the backend already forms each iteration. Used by the float32 GPU path as its
    *stopping* signal so the iteration runs until it is genuinely stationary, rather
    than quitting at the noisy √n·eps float32 deviance-change floor (which stops
    early, while slowly-converging low-hazard directions are still moving). The
    final float32 *acceptance* gate is computed independently in float64 — this
    helper only decides when to stop iterating.

Both functions are deterministic and operate on host float64 arrays.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray

from pystatistics.regression.families import Family, Link


def step_halve(
    coef_new: NDArray[np.float64],
    coef_prev: NDArray[np.float64],
    X: NDArray[np.float64],
    link: Link,
    family: Family,
    y: NDArray[np.float64],
    wt: NDArray[np.float64],
    dev_old: float,
    max_halvings: int = 12,
    require_decrease: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Halve the IRLS step toward ``coef_prev`` until the deviance is acceptable.

    Two trigger modes:

    - ``require_decrease=False`` (default, **R-exact**): halve only when the full
      Newton step yields a *non-finite* deviance (η overflow / boundary μ). A
      finite but larger deviance is accepted as-is — Fisher scoring is not a
      descent method and R's ``glm.fit`` deliberately accepts a deviance-increasing
      step. Required for CPU coefficient parity with R. On a well-behaved fit this
      never triggers and returns the full step unchanged (bit-identical).

    - ``require_decrease=True`` (**damped**): additionally halve while the (finite)
      deviance exceeds the previous iterate's, i.e. enforce monotone descent. Used
      by the float32 GPU path, where float32 noise makes some Newton steps overshoot
      and damping them keeps the iteration inside the stationary basin so the Newton
      decrement can settle below the acceptance threshold. Only meaningful with a
      genuine previous iterate — call with the default (False) on the first
      iteration, where ``coef_prev`` is the zero placeholder.

    Args:
        coef_new: Full Newton step coefficients β (p,), host float64.
        coef_prev: Previous iteration's coefficients (p,); zeros on the first
            iteration (a finite in-subspace fallback for the non-finite case — R
            instead errors when the first step is non-finite, having no predecessor).
        X: Design matrix (n×p), host float64.
        link, family: GLM link / family (their methods compute on host float64).
        y, wt: Response and prior weights (n,).
        dev_old: Deviance at ``coef_prev`` — the descent reference for
            ``require_decrease`` (ignored otherwise).
        max_halvings: Maximum number of halvings before giving up and returning
            the last iterate — the caller's convergence / acceptance logic then
            decides what to do with it.
        require_decrease: Enforce monotone deviance descent (see above).

    Returns:
        ``(coef, eta, mu, dev)`` for the accepted iterate: the coefficients, their
        linear predictor ``X@coef``, fitted mean ``linkinv(eta)``, and deviance.
    """
    coef = np.asarray(coef_new, dtype=np.float64)
    eta = X @ coef
    mu = link.linkinv(eta)
    dev = family.deviance(y, mu, wt)

    k = 0
    # Non-finite is always a halving trigger (R's condition). The descent guard is
    # optional (float32 GPU path); the 1e-8 slack absorbs benign round-off at the
    # optimum so a converged fit is never needlessly halved.
    while (
        not np.isfinite(dev)
        or (require_decrease and dev > dev_old + 1e-8)
    ) and k < max_halvings:
        coef = 0.5 * (coef + coef_prev)
        eta = X @ coef
        mu = link.linkinv(eta)
        dev = family.deviance(y, mu, wt)
        k += 1

    return coef, eta, mu, dev


def relative_newton_decrement(
    XtWX: NDArray[np.float64],
    XtWz: NDArray[np.float64],
    coef: NDArray[np.float64],
    deviance: float,
) -> float:
    """Relative Newton decrement λ²/(|dev|+0.1) at ``coef`` from IRLS normal eqns.

    Given the weighted normal-equation pieces formed at ``coef`` (so that the IRLS
    weights/working response correspond to η = X@coef), the Fisher score is
    ``U = XᵀWz − (XᵀWX)·coef`` and the Newton decrement is
    ``λ² = Uᵀ (XᵀWX)⁻¹ U`` — the standard affine-invariant distance-to-optimum in
    deviance/2 units (≈0 at a stationary point). Dividing by ``|dev|+0.1`` makes it
    the fraction of the deviance still "on the table", directly comparable to the
    acceptance threshold.

    This reuses matrices the backend already formed for the current iteration, so it
    adds only a p×p solve. It is intended as a *stopping* signal; trustworthy
    acceptance is decided separately in float64 by the backend's gate.

    Returns ``inf`` if XᵀWX is singular (no usable step → not stationary).
    """
    A = np.asarray(XtWX, dtype=np.float64)
    b = np.asarray(XtWz, dtype=np.float64)
    score = b - A @ np.asarray(coef, dtype=np.float64)
    try:
        step = np.linalg.solve(A, score)
    except np.linalg.LinAlgError:
        return float("inf")
    lam2 = float(score @ step)
    # Numerical noise in the float32-formed normal equations can yield a tiny
    # negative quadratic form; clamp at 0 (a stationary point, not a spurious pass).
    lam2 = max(lam2, 0.0)
    return lam2 / (abs(deviance) + 0.1)
