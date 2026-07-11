"""
cox.zph — test of the proportional-hazards assumption.

Implements the survival 3.x formulation (Grambsch-Therneau, as rewritten in
survival >= 3.0): a SCORE TEST for adding the time-interaction covariates
``x * g(t)`` to the fitted model, evaluated at the fitted ``beta`` and
interaction coefficient 0 — not the pre-3.0 correlation test.

For each stratum and distinct event time, the risk-set moments S0/S1/S2 give
the per-event-time covariate mean and covariance V(t) (Efron or Breslow tie
handling, matching the fit). With g(t) the centered transformed time:

    u2     = sum over deaths of  g(t) * (x_i - xbar(t))      (augmented score)
    I11    = sum V(t);  I12 = sum g V(t);  I22 = sum g^2 V(t)

Per-covariate test j:  chisq = u' A^{-1} u  on the submatrix A of the augmented
information over columns [all original, j-th interaction], u = (0,...,0, u2_j);
GLOBAL uses the full augmented system. Scaled Schoenfeld residuals for
inspection/plotting are ``V_bar^{-1} s_k + beta`` with ``V_bar = I11 / d`` the
average per-event covariance (R's ``zph$y``).

Matches ``survival::cox.zph(fit, transform=)`` for transforms
km / rank / identity / log.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from pystatistics.core.exceptions import ValidationError
from pystatistics.core.result import SolutionReprMixin
from pystatistics.survival._cox import (
    _death_group_sums, _entry_adjustment, _reverse_cumsum,
)
from pystatistics.survival._cox_strata import _build_strata


def _km_left_cdf(
    all_times: NDArray, all_events: NDArray, all_entry: NDArray | None,
) -> tuple[NDArray, NDArray]:
    """Pooled (stratum-ignoring) left-continuous KM CDF, per row.

    Returns ``(unique_times, cdf_left)`` where ``cdf_left[k] = 1 - S(t_k-)``:
    the KM estimate of P(T < t_k), i.e. survival just BEFORE each unique
    observed time. This is R's ``transform="km"``: the survival curve is fit to
    the whole sample (strata pooled, as ``cox.zph`` does). With
    counting-process rows the risk sets account for delayed entry.
    """
    order = np.argsort(all_times, kind="mergesort")
    t = all_times[order]
    e = all_events[order]
    ut, first = np.unique(t, return_index=True)
    m = len(ut)
    # deaths and at-risk count per unique time
    d = np.zeros(m)
    np.add.at(d, np.searchsorted(ut, t), e)
    n_risk = (len(t) - first).astype(np.float64)  # rows with time >= ut[k]
    if all_entry is not None:
        entry_sorted = np.sort(all_entry)
        n_risk -= len(t) - np.searchsorted(entry_sorted, ut, side="left")
    factor = 1.0 - d / n_risk
    surv = np.cumprod(factor)               # S(t_k), right-continuous
    surv_left = np.concatenate([[1.0], surv[:-1]])   # S(t_k-)
    return ut, 1.0 - surv_left


def _transform_times(
    all_times: NDArray, all_events: NDArray, all_entry: NDArray | None,
    transform: str,
) -> NDArray:
    """Per-row transformed times g0(t) (before centering)."""
    if transform == "identity":
        return all_times.astype(np.float64)
    if transform == "log":
        return np.log(all_times)
    if transform == "rank":
        return stats.rankdata(all_times, method="average")
    if transform == "km":
        ut, cdf_left = _km_left_cdf(all_times, all_events, all_entry)
        return cdf_left[np.searchsorted(ut, all_times)]
    raise ValidationError(
        f"transform must be 'km', 'rank', 'identity', or 'log', "
        f"got '{transform}'"
    )


def cox_zph_compute(
    time: NDArray,
    event: NDArray,
    X: NDArray,
    beta: NDArray,
    strata: NDArray | None,
    ties: str,
    transform: str,
    entry: NDArray | None = None,
) -> dict:
    """Score-test statistics + scaled Schoenfeld residuals. See module doc."""
    n, p = X.shape
    # Mean-center the covariates (invariant to the score test; stabilizes the
    # risk-set moments, matching the fit — see _cox.cox_fit).
    X = X - X.mean(axis=0)
    ttimes = _transform_times(time, event, entry, transform)
    g_row = ttimes - np.mean(ttimes[event == 1])   # centered over ALL events

    if strata is None:
        strata = np.zeros(n, dtype=np.intp)
    strata_list = _build_strata(time, event, X, strata, entry=entry)

    # Row-level transformed value shared by every row tied at a given time:
    # look up per unique time (any representative row works for all transforms).
    ut_all, first_all = np.unique(time, return_index=True)
    g_of_time = g_row[first_all]          # centered (used in the score test)
    t_of_time = ttimes[first_all]         # uncentered (reported as .x)

    I11 = np.zeros((p, p)); I12 = np.zeros((p, p)); I22 = np.zeros((p, p))
    u2 = np.zeros(p)
    d_total = 0.0
    schoen_rows: list[NDArray] = []
    x_out: list[float] = []
    time_out: list[float] = []

    for s in strata_list:
        m = len(s.unique_event_times)
        if m == 0:
            continue
        eta = s.X @ beta
        eta_c = eta - np.max(eta)
        w = np.exp(eta_c)
        wx = w[:, None] * s.X
        cs0 = _reverse_cumsum(w)
        cs1 = _reverse_cumsum(wx)
        cs2 = _reverse_cumsum(wx[:, :, None] * s.X[:, None, :])
        first = np.searchsorted(s.time, s.unique_event_times, side="left")
        S0 = cs0[first]; S1 = cs1[first]; S2 = cs2[first]
        if s.entry is not None:
            uet = s.unique_event_times
            S0 = S0 - _entry_adjustment(w, s.entry, uet)
            S1 = S1 - _entry_adjustment(wx, s.entry, uet)
            S2 = S2 - _entry_adjustment(wx[:, :, None] * s.X[:, None, :],
                                        s.entry, uet)
        dg = _death_group_sums(s.time, s.event, s.X, eta_c, w,
                               s.unique_event_times)
        d, dS0, dS1, dS2 = dg["d"], dg["dS0"], dg["dS1"], dg["dS2"]
        eX_sum = dg["eX_sum"]

        death_mask = s.event == 1
        death_times = s.time[death_mask]
        death_X = s.X[death_mask]
        grp = np.searchsorted(s.unique_event_times, death_times)

        g_j = g_of_time[np.searchsorted(ut_all, s.unique_event_times)]

        mean_bar = np.empty((m, p))
        for k in range(m):
            dk = int(d[k])
            if ties == "efron" and dk > 1:
                sum_mean = np.zeros(p)
                sum_V = np.zeros((p, p))
                for a in range(dk):
                    frac = a / dk
                    denom = S0[k] - frac * dS0[k]
                    mean_a = (S1[k] - frac * dS1[k]) / denom
                    V_a = (S2[k] - frac * dS2[k]) / denom - np.outer(mean_a, mean_a)
                    sum_mean += mean_a
                    sum_V += V_a
            else:
                mean_0 = S1[k] / S0[k]
                V_0 = S2[k] / S0[k] - np.outer(mean_0, mean_0)
                sum_mean = dk * mean_0
                sum_V = dk * V_0
            u2 += g_j[k] * (eX_sum[k] - sum_mean)
            I11 += sum_V
            I12 += g_j[k] * sum_V
            I22 += g_j[k] * g_j[k] * sum_V
            mean_bar[k] = sum_mean / dk
            d_total += dk

        # Schoenfeld residual per death: x_i - (average per-death risk mean).
        # Deaths are already time-sorted within the stratum.
        schoen_rows.append(death_X - mean_bar[grp])
        time_out.extend(death_times.tolist())
        x_out.extend(t_of_time[np.searchsorted(ut_all, death_times)].tolist())

    schoen = np.vstack(schoen_rows) if schoen_rows else np.zeros((0, p))

    # Augmented information and tests.
    imat_aug = np.block([[I11, I12], [I12, I22]])
    chisq = np.empty(p + 1)
    for j in range(p):
        kk = list(range(p)) + [p + j]
        A = imat_aug[np.ix_(kk, kk)]
        u_vec = np.zeros(p + 1)
        u_vec[-1] = u2[j]
        chisq[j] = u_vec @ np.linalg.solve(A, u_vec)
    u_full = np.concatenate([np.zeros(p), u2])
    chisq[p] = u_full @ np.linalg.solve(imat_aug, u_full)
    df = np.concatenate([np.ones(p), [p]])
    p_values = stats.chi2.sf(chisq, df)

    # Scaled Schoenfeld residuals: V_bar^{-1} s_k + beta.
    v_bar = I11 / d_total
    y = np.linalg.solve(v_bar, schoen.T).T + beta

    return {
        "chisq": chisq,
        "df": df,
        "p_values": p_values,
        "y": y,
        "x": np.asarray(x_out),
        "time": np.asarray(time_out),
        "var": np.linalg.inv(v_bar),
        "transform": transform,
    }


class CoxZphSolution(SolutionReprMixin):
    """Proportional-hazards test result (one row per covariate + GLOBAL).

    Mirrors R's ``cox.zph`` object: ``.chisq`` / ``.df`` / ``.p_values`` are
    aligned with ``.row_names`` (covariates first, ``"GLOBAL"`` last);
    ``.residuals`` holds the scaled Schoenfeld residuals (one row per event,
    time-ordered within stratum), ``.x`` the transformed event times, ``.time``
    the raw event times.
    """

    __slots__ = ("_data", "_names")

    def __init__(self, _data: dict, _names: tuple[str, ...]) -> None:
        self._data = _data
        self._names = _names

    @property
    def row_names(self) -> tuple[str, ...]:
        return self._names + ("GLOBAL",)

    @property
    def chisq(self) -> NDArray:
        return self._data["chisq"]

    @property
    def df(self) -> NDArray:
        return self._data["df"]

    @property
    def p_values(self) -> NDArray:
        return self._data["p_values"]

    @property
    def residuals(self) -> NDArray:
        """Scaled Schoenfeld residuals, shape (n_events, p)."""
        return self._data["y"]

    @property
    def x(self) -> NDArray:
        """Transformed event times aligned with ``residuals`` rows."""
        return self._data["x"]

    @property
    def time(self) -> NDArray:
        """Raw event times aligned with ``residuals`` rows."""
        return self._data["time"]

    @property
    def var(self) -> NDArray:
        """Inverse average per-event covariance (R's ``zph$var``)."""
        return self._data["var"]

    @property
    def transform(self) -> str:
        return self._data["transform"]

    def summary(self) -> str:
        lines = [f"Proportional-hazards test (transform: {self.transform})", ""]
        w = max(10, max(len(nm) for nm in self.row_names))
        lines.append(f"  {'':{w}s}  {'chisq':>10s}  {'df':>4s}  {'p':>10s}")
        for i, nm in enumerate(self.row_names):
            lines.append(
                f"  {nm:>{w}s}  {self.chisq[i]:10.4f}  {int(self.df[i]):4d}  "
                f"{self.p_values[i]:10.4g}"
            )
        return "\n".join(lines)
