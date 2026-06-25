"""Assemble one comparable benchmark record.

One job: take an engine's outcome + timing and produce a single flat dict with a
uniform schema, so different engines (pystatistics CPU/GPU, R, other tools) can be
compared field-for-field. This is the GENERIC envelope shared by every subsystem;
subsystem-specific numeric summaries (e.g. a covariance's Frobenius norm) are passed
in via ``summary`` (see :mod:`pystatsval.estimates` for reusable summarizers) and
free-form fields via ``extra``.

Matches the ``record`` shape in the validation artifact schema
(``validation-run/v1``).
"""

from __future__ import annotations

from typing import Any


def make_record(*,
                engine: str,
                dataset: str,
                n: int | None = None,
                p: int | None = None,
                loglik: float | None = None,
                n_iter: int | None = None,
                converged: bool | None = None,
                wall: dict[str, Any] | None = None,
                backend_name: str | None = None,
                precision: str | None = None,
                parameterization: str | None = None,
                summary: dict[str, Any] | None = None,
                extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build one uniform benchmark record.

    Parameters
    ----------
    engine
        Engine identifier, e.g. ``"pystatistics:cpu"``, ``"pystatistics:gpu"``,
        ``"r:mvnmle"``. The convention ``"<tool>:<backend>"`` lets the renderer
        pair CPU/accelerator and SUT/reference rows.
    dataset
        Problem/dataset identifier (the renderer pairs rows by this + ``p``).
    wall
        A timing summary from :func:`pystatsval.timing.time_call` (or ``None`` on
        failure). Its ``median_s``/``min_s``/``max_s``/``times_s`` are flattened
        into the record.
    summary
        Subsystem-specific numeric summaries (already JSON-friendly), merged in.
    extra
        Free-form fields (e.g. ``{"error": "..."}`` on a recorded failure).

    Notes
    -----
    This function never raises on a failed fit — callers pass ``wall=None`` and an
    ``extra={"error": ...}`` so a sweep records the failure and continues. It does
    validate its own output contract (types of the core fields).
    """
    if not engine or not isinstance(engine, str):
        raise ValueError(f"engine must be a non-empty str, got {engine!r}")
    if not dataset or not isinstance(dataset, str):
        raise ValueError(f"dataset must be a non-empty str, got {dataset!r}")

    w = wall or {}
    rec: dict[str, Any] = {
        "engine": engine,
        "dataset": dataset,
        "n": (None if n is None else int(n)),
        "p": (None if p is None else int(p)),
        "backend_name": backend_name,
        "precision": precision,
        "parameterization": parameterization,
        "loglik": (None if loglik is None else float(loglik)),
        "n_iter": (None if n_iter is None else int(n_iter)),
        "converged": (None if converged is None else bool(converged)),
        "wall_median_s": w.get("median_s"),
        "wall_min_s": w.get("min_s"),
        "wall_max_s": w.get("max_s"),
        "wall_times_s": w.get("times_s"),
    }
    if summary:
        rec.update(summary)
    if extra:
        rec.update(extra)
    return rec
