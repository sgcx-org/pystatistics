"""
Stratified Kaplan-Meier.

``kaplan_meier(strata=g)`` estimates a separate product-limit survival curve for
each stratum — the survival analog of ``survfit(Surv(t, e) ~ g)``. Each curve is
computed by the existing single-curve estimator (``_km.kaplan_meier_fit``) on the
stratum's observations; this module only partitions the data and packages the
per-stratum curves into one ``StratifiedKMSolution``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import SolutionReprMixin
from pystatistics.survival._common import KMParams
from pystatistics.survival._km import kaplan_meier_fit


def stratified_km_curves(
    time: NDArray,
    event: NDArray,
    strata: NDArray,
    conf_level: float,
    conf_type: str,
    entry: NDArray | None = None,
) -> list[tuple[object, KMParams]]:
    """Per-stratum Kaplan-Meier curves.

    Returns a list of ``(label, KMParams)`` in ascending label order (matching
    R's ``survfit`` stratum ordering). Each stratum's curve is the single-curve
    product-limit estimate over that stratum's observations (with per-stratum
    delayed entry when ``entry`` is given).
    """
    strata_flat = np.asarray(strata).ravel()
    labels = np.unique(strata_flat)
    curves: list[tuple[object, KMParams]] = []
    for lab in labels:
        idx = strata_flat == lab
        params = kaplan_meier_fit(
            time[idx], event[idx], conf_level=conf_level, conf_type=conf_type,
            entry=entry[idx] if entry is not None else None,
        )
        # Render numpy scalar labels (np.int64(1)) as plain Python values.
        key = lab.item() if hasattr(lab, "item") else lab
        curves.append((key, params))
    return curves


class StratifiedKMSolution(SolutionReprMixin):
    """Stratified Kaplan-Meier result: one survival curve per stratum.

    Wraps an ordered mapping ``{stratum_label: KMSolution}``. Index by label
    (``sol["A"]``) or iterate ``sol.curves.items()`` to reach each stratum's
    ordinary :class:`KMSolution`.
    """

    __slots__ = ("_curves", "_timing")

    def __init__(self, _curves: dict, _timing=None) -> None:
        # _curves: ordered {label: KMSolution}
        self._curves = _curves
        self._timing = _timing

    @property
    def strata(self) -> tuple:
        """Stratum labels, in curve order."""
        return tuple(self._curves.keys())

    @property
    def n_strata(self) -> int:
        return len(self._curves)

    @property
    def curves(self) -> dict:
        """Ordered ``{label: KMSolution}`` mapping."""
        return dict(self._curves)

    def __getitem__(self, label):
        return self._curves[label]

    def stratum(self, label):
        """The :class:`KMSolution` for one stratum."""
        return self._curves[label]

    @property
    def timing(self):
        return self._timing

    @property
    def warnings(self) -> tuple[str, ...]:
        """Union of the per-stratum warnings, prefixed by stratum label."""
        out: list[str] = []
        for label, km in self._curves.items():
            out.extend(f"[stratum {label}] {w}" for w in km.warnings)
        return tuple(out)

    def summary(self) -> str:
        lines = [f"Call: kaplan_meier(strata=...)  ({self.n_strata} strata)", ""]
        for label, km in self._curves.items():
            lines.append(f"── stratum {label} "
                         f"(n={km.n_observations}, events={km.n_events_total}) ──")
            lines.append(km.summary())
            lines.append("")
        return "\n".join(lines).rstrip()
