"""
Factored (ma1, sma1, ...) parameterization for SARIMA optimization.

For seasonal ARIMA the "effective" MA polynomial has length q + sq * m
with most coefficients zero or deterministic products of the non-
seasonal and seasonal factors. Optimizing over the full effective
vector wastes scipy's finite-difference budget on a 13-dimensional
surface when only 2 parameters are truly free (for the Box-Jenkins
airline model). R's ``stats::arima`` optimizes in the factored space;
this module brings the same pattern to pystatistics.

Key functions:
    ``_factored_to_effective(...)``  — expand (ar, ma, sar, sma, mean)
        to the effective (ar_eff, ma_eff, mean) vector the likelihood
        operates on.
    ``arima_negloglik_factored(...)`` — NLL in factored space.
    ``optimize_arima_factored(...)``  — L-BFGS-B optimize in factored
        space, delegating the actual CSS/ML computation to the effective
        likelihood.

Sign conventions (critical — see the ``_multiply_ma_polynomials``
docstring below):

    AR:  e_t = y_t - Σ ar_i y_{t-i}       (AR polynomial 1 − Σ ar_i B^i)
    MA:  e_t = y_t - Σ ma_j e_{t-j}       (MA polynomial 1 + Σ ma_j B^j)

so the seasonal-expansion helpers for AR and MA differ by a sign on the
cross term.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from pystatistics.core.exceptions import ConvergenceError
from pystatistics.timeseries._arima_likelihood import arima_negloglik


# ---------------------------------------------------------------------------
# Seasonal polynomial multiplication
# ---------------------------------------------------------------------------

def _multiply_polynomials(
    nonseasonal: NDArray,
    seasonal: NDArray,
    period: int,
) -> NDArray:
    """
    Multiply non-seasonal and seasonal polynomials.

    For AR polynomials, computes the product:
        phi(B) * Phi(B^m) = (1 - phi_1*B - ... - phi_p*B^p)
                          * (1 - Phi_1*B^m - ... - Phi_P*B^{Pm})

    and returns the coefficients of the result (excluding the leading 1).

    Parameters
    ----------
    nonseasonal : NDArray
        Non-seasonal coefficients [c_1, ..., c_p] where the polynomial
        is ``1 - c_1*B - ... - c_p*B^p``.
    seasonal : NDArray
        Seasonal coefficients [C_1, ..., C_P] where the polynomial
        is ``1 - C_1*B^m - ... - C_P*B^{Pm}``.
    period : int
        Seasonal period *m*.

    Returns
    -------
    NDArray
        Combined coefficients of the product polynomial (excluding
        the leading 1 term), negated so they follow the sign convention
        ``1 - result_1*B - result_2*B^2 - ...``.
    """
    # Build polynomial representations with leading 1
    # For poly1 = 1 - c_1*B - c_2*B^2 ..., we store [1, -c_1, -c_2, ...]
    poly1 = np.zeros(1 + len(nonseasonal))
    poly1[0] = 1.0
    for i, c in enumerate(nonseasonal):
        poly1[i + 1] = -c

    max_seasonal_lag = len(seasonal) * period
    poly2 = np.zeros(1 + max_seasonal_lag)
    poly2[0] = 1.0
    for i, c in enumerate(seasonal):
        poly2[(i + 1) * period] = -c

    # Convolve the two polynomials
    product = np.convolve(poly1, poly2)

    # Return coefficients after leading 1, negated back to positive convention
    return -product[1:]


def _multiply_ma_polynomials(
    nonseasonal: NDArray,
    seasonal: NDArray,
    period: int,
) -> NDArray:
    """Multiply two MA polynomials under pystatistics' MA sign convention.

    pystatistics stores AR and MA with OPPOSITE sign conventions:
        AR:  e_t = y_t - Σ ar_i y_{t-i}       (AR polynomial 1 − Σ ar_i B^i)
        MA:  e_t = y_t - Σ ma_j e_{t-j}       (MA polynomial 1 + Σ ma_j B^j)

    ``_multiply_polynomials`` above handles the AR case (all signs
    negated). For MA we need the product of
        (1 + ma_1 B + … + ma_q B^q)(1 + sma_1 B^m + … + sma_Q B^{Qm})
    returned as ``[ma_eff_1, ma_eff_2, …]``. A straight convolution of
    ``[1, ma_1, …, ma_q]`` with ``[1, sma_1 at position m, …]`` yields
    this directly — no double negation.
    """
    poly1 = np.zeros(1 + len(nonseasonal))
    poly1[0] = 1.0
    for i, c in enumerate(nonseasonal):
        poly1[i + 1] = c

    max_seasonal_lag = len(seasonal) * period
    poly2 = np.zeros(1 + max_seasonal_lag)
    poly2[0] = 1.0
    for i, c in enumerate(seasonal):
        poly2[(i + 1) * period] = c

    product = np.convolve(poly1, poly2)
    return product[1:]



def _factored_to_effective(
    factored: NDArray,
    p: int, q: int, sp: int, sq: int, period: int,
    include_mean: bool,
    multiply_ar_polynomials,
    multiply_ma_polynomials,
) -> NDArray:
    """Expand factored SARIMA parameters to the effective ARMA vector.

    Factored layout: ``[ar_1..ar_p, ma_1..ma_q, sar_1..sar_P,
    sma_1..sma_Q, mean?]``. Effective layout:
    ``[ar_eff_1..ar_eff_{p_eff}, ma_eff_1..ma_eff_{q_eff}, mean?]``
    with p_eff = p + sp*period, q_eff = q + sq*period.

    AR and MA expansion helpers are passed in from ``_arima_fit`` so
    this module doesn't import polynomial math from its parent.
    """
    ar = factored[:p]
    ma = factored[p:p + q]
    sar = factored[p + q:p + q + sp]
    sma = factored[p + q + sp:p + q + sp + sq]

    ar_eff = multiply_ar_polynomials(ar, sar, period) if sp > 0 else ar
    ma_eff = multiply_ma_polynomials(ma, sma, period) if sq > 0 else ma

    parts = [ar_eff, ma_eff]
    if include_mean:
        parts.append(factored[p + q + sp + sq:p + q + sp + sq + 1])
    return np.concatenate(parts) if parts else np.array([])


def arima_negloglik_factored(
    factored: NDArray,
    y: NDArray,
    shape: tuple[int, int, int, int],  # (p, q, sp, sq)
    period: int,
    include_mean: bool,
    method: str,
    multiply_ar_polynomials,
    multiply_ma_polynomials,
) -> float:
    """Negative log-likelihood at factored SARIMA parameters.

    Expands to effective ARMA form (via the supplied polynomial-product
    helpers) and delegates to ``arima_negloglik``. Used by
    ``optimize_arima_factored`` so scipy's L-BFGS-B optimizes on the
    small factored surface.
    """
    p, q, sp, sq = shape
    eff_params = _factored_to_effective(
        factored, p, q, sp, sq, period, include_mean,
        multiply_ar_polynomials, multiply_ma_polynomials,
    )
    p_eff = p + sp * period
    q_eff = q + sq * period
    return arima_negloglik(
        eff_params, y, (p_eff, q_eff), include_mean, method,
    )


def optimize_arima_factored(
    y_diff: NDArray,
    p: int, q: int, sp: int, sq: int, period: int,
    include_mean: bool,
    method: str,
    tol: float,
    max_iter: int,
    start_factored: NDArray,
    multiply_ar_polynomials,
    multiply_ma_polynomials,
) -> tuple[NDArray, float, bool, int, str]:
    """L-BFGS-B on the factored SARIMA parameter vector.

    Same semantics as ``_optimize_arima`` (CSS, ML, CSS-ML pipelines),
    but the optimizer sees (p + q + sp + sq + [1]) parameters instead
    of (p_eff + q_eff + [1]). On the Box-Jenkins airline model that is
    2 dims vs 14; the likelihood is the same and the per-evaluation
    cost is the same, so the savings come from ~10× fewer scipy
    evaluations to converge.
    """
    order = (p, q, sp, sq)
    ll_args = (
        y_diff, order, period, include_mean,
    )
    opts = {"maxiter": max_iter, "ftol": tol}

    if method in ("CSS", "CSS-ML"):
        result_css = minimize(
            arima_negloglik_factored,
            start_factored,
            args=ll_args + ("CSS", multiply_ar_polynomials, multiply_ma_polynomials),
            method="L-BFGS-B",
            options=opts,
        )
        opt_factored = result_css.x
        nll = result_css.fun
        converged = result_css.success
        n_iter = result_css.nit
        method_used = "CSS"

        if method == "CSS-ML":
            css_factored = opt_factored.copy()
            try:
                result_ml = minimize(
                    arima_negloglik_factored,
                    css_factored,
                    args=ll_args + ("ML", multiply_ar_polynomials, multiply_ma_polynomials),
                    method="L-BFGS-B",
                    options=opts,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise ConvergenceError(
                    "ARIMA CSS-ML: ML refinement failed numerically "
                    f"({type(exc).__name__}: {exc}).",
                    iterations=n_iter,
                    reason="ml_refinement_failed",
                )
            if not np.isfinite(result_ml.fun):
                raise ConvergenceError(
                    "ARIMA CSS-ML: ML refinement produced a non-finite "
                    "log-likelihood.",
                    iterations=result_ml.nit,
                    reason="ml_refinement_nonfinite",
                )

            best_x = result_ml.x
            best_nll = result_ml.fun
            best_success = result_ml.success
            total_nit = result_ml.nit

            # Second ML start from the original starting values, kept
            # only if strictly better — the (ar, mean) surface has a
            # flat canyon toward the AR unit root and the CSS stage can
            # hand ML a drifted-mean basin (see _optimize_arima). Only
            # mean-carrying fits (d = D = 0) are exposed.
            if include_mean:
                try:
                    result_ml2 = minimize(
                        arima_negloglik_factored,
                        start_factored,
                        args=ll_args + ("ML", multiply_ar_polynomials,
                                        multiply_ma_polynomials),
                        method="L-BFGS-B",
                        options=opts,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    result_ml2 = None
                if result_ml2 is not None:
                    total_nit += result_ml2.nit
                    # Take the second optimum when it is strictly
                    # better, or when it converged and the first start
                    # did not (a converged optimum in hand must win
                    # over a failed one regardless of its nll).
                    if (result_ml2.success and np.isfinite(result_ml2.fun)
                            and (result_ml2.fun < best_nll
                                 or not best_success)):
                        best_x = result_ml2.x
                        best_nll = result_ml2.fun
                        best_success = True

            return best_x, best_nll, best_success, total_nit, "CSS-ML"
        return opt_factored, nll, converged, n_iter, method_used

    # method == "ML"
    result_ml = minimize(
        arima_negloglik_factored,
        start_factored,
        args=ll_args + ("ML", multiply_ar_polynomials, multiply_ma_polynomials),
        method="L-BFGS-B",
        options=opts,
    )
    return (
        result_ml.x, result_ml.fun, result_ml.success, result_ml.nit, "ML",
    )
