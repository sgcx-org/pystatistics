"""
The chained-equations sweep — one imputation chain.

A single chain produces one completed dataset:

  1. *Initialise* every missing cell by a random draw from the observed values of
     its own column (R mice's default start), so all predictor columns are fully
     populated from iteration 1.
  2. *Sweep* ``maxit`` times. In each iteration, visit the incomplete columns in
     the visit sequence; for each, fit the column's assigned method on the rows
     where it is observed (using all other columns as predictors) and overwrite
     its missing cells with fresh draws.
  3. Record, per iteration and per incomplete column, the mean and variance of
     the imputed cells — the trace MICE convergence diagnostics are read from.

This module owns orchestration only. The per-variable statistical work lives
behind the :class:`ImputationMethod` protocol, so the sweep never hard-codes a
method and a GPU backend can swap in batched method implementations without
touching this control flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pystatistics.mice.design import MICEDesign
from pystatistics.mice.methods import get_method


@dataclass(frozen=True)
class ChainResult:
    """Output of one chain: completed data plus convergence traces."""

    completed: np.ndarray            # (n, p) — no NaN
    chain_mean: np.ndarray           # (maxit, n_incomplete)
    chain_var: np.ndarray            # (maxit, n_incomplete)
    incomplete_columns: tuple[int, ...]


def run_chain(
    design: MICEDesign,
    rng: np.random.Generator,
    maxit: int,
    visit_sequence: tuple[int, ...],
) -> ChainResult:
    """Run one chained-equations chain. See module docstring."""
    data = design.data.copy()
    mask = design.missing_mask
    p = design.p
    incomplete = design.incomplete_columns

    _initialise(data, mask, incomplete, rng)

    chain_mean = np.empty((maxit, len(incomplete)), dtype=np.float64)
    chain_var = np.empty((maxit, len(incomplete)), dtype=np.float64)
    incomplete_pos = {j: k for k, j in enumerate(incomplete)}

    for it in range(maxit):
        for j in visit_sequence:
            mis_rows = mask[:, j]
            obs_rows = ~mis_rows

            predictors = _predictor_columns(data, j, p)
            X_obs = predictors[obs_rows]
            X_mis = predictors[mis_rows]
            y_obs = data[obs_rows, j]

            method = get_method(design.method_for(j))
            imputed = method.impute(y_obs, X_obs, X_mis, rng)
            data[mis_rows, j] = imputed

        # Trace: summarise this iteration's imputed cells per incomplete column.
        for j in incomplete:
            cells = data[mask[:, j], j]
            k = incomplete_pos[j]
            chain_mean[it, k] = float(np.mean(cells))
            chain_var[it, k] = float(np.var(cells))

    return ChainResult(
        completed=data,
        chain_mean=chain_mean,
        chain_var=chain_var,
        incomplete_columns=incomplete,
    )


def _initialise(
    data: np.ndarray,
    mask: np.ndarray,
    incomplete: tuple[int, ...],
    rng: np.random.Generator,
) -> None:
    """Fill missing cells by sampling (with replacement) from observed values of
    the same column. Mutates ``data`` in place."""
    for j in incomplete:
        mis_rows = mask[:, j]
        observed = data[~mis_rows, j]
        n_missing = int(np.count_nonzero(mis_rows))
        data[mis_rows, j] = rng.choice(observed, size=n_missing, replace=True)


def _predictor_columns(data: np.ndarray, j: int, p: int) -> np.ndarray:
    """All columns except ``j`` (the current target), as a 2D predictor matrix.

    Columns are fully populated at this point (initialised on entry, kept
    complete by every visit), so the returned matrix is finite. Fancy-indexing
    with a list always yields a 2D matrix, including the single-predictor
    (p == 2) case.
    """
    cols = [c for c in range(p) if c != j]
    return data[:, cols]
