"""
CPU backend for MICE.

Owns the single backend entrypoint ``run`` that the solver calls. It runs the
``m`` imputation chains — each with its own pre-spawned, independent RNG stream
— and assembles the completed datasets and convergence traces into a Result.

This is the seam Stage 2 mirrors: a GPU backend implements the same ``run``
signature and batches the chain loop, while the solver, design, methods, and
pooling stay untouched.
"""

from __future__ import annotations

import numpy as np

from pystatistics.core.compute.timing import Timer
from pystatistics.core.result import Result
from pystatistics.mice._chain import run_chain
from pystatistics.mice._rng import spawn_streams
from pystatistics.mice.design import MICEDesign
from pystatistics.mice.solution import MICEParams


class CPUMiceBackend:
    """Runs MICE chains on the CPU (reference path)."""

    @property
    def name(self) -> str:
        return "cpu"

    def run(
        self,
        design: MICEDesign,
        *,
        m: int,
        maxit: int,
        visit_sequence: tuple[int, ...],
        seed: int,
    ) -> Result[MICEParams]:
        """Run ``m`` chains and package the imputations.

        Parameters
        ----------
        design : MICEDesign
        m : int
            Number of imputations (chains).
        maxit : int
            Iterations per chain.
        visit_sequence : tuple of int
            Column visit order within each iteration.
        seed : int
            Master RNG seed. The backend spawns one independent, reproducible
            stream per chain from it.
        """
        timer = Timer()
        timer.start()

        streams = spawn_streams(seed, m)

        completed = np.empty((m, design.n, design.p), dtype=np.float64)
        chain_means = np.empty((m, maxit, len(design.incomplete_columns)))
        chain_vars = np.empty((m, maxit, len(design.incomplete_columns)))

        with timer.section("chains"):
            for i in range(m):
                result = run_chain(design, streams[i], maxit, visit_sequence)
                completed[i] = result.completed
                chain_means[i] = result.chain_mean
                chain_vars[i] = result.chain_var

        timer.stop()

        params = MICEParams(
            completed=completed,
            chain_mean=chain_means,
            chain_var=chain_vars,
            incomplete_columns=design.incomplete_columns,
            m=m,
            maxit=maxit,
            visit_sequence=visit_sequence,
        )
        return Result(
            params=params,
            info={
                "method": "chained_equations",
                "m": m,
                "maxit": maxit,
                "device": "cpu",
                "visit_sequence": list(visit_sequence),
            },
            timing=timer.result(),
            backend_name="cpu",
            warnings=(),
        )
