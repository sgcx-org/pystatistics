"""
CPU backends for bootstrap and permutation test.

CPUBootstrapBackend: Ordinary, balanced, and parametric bootstrap.
CPUPermutationBackend: Permutation test with Phipson-Smyth correction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.montecarlo._common import (
    BootParams, PermutationParams, perm_pvalue,
)
from pystatistics.montecarlo.design import BootstrapDesign, PermutationDesign


class CPUBootstrapBackend:
    """
    CPU backend for bootstrap resampling.

    Supports ordinary, balanced, and parametric simulation types.
    """

    @property
    def name(self) -> str:
        return 'cpu_bootstrap'

    def solve(self, design: BootstrapDesign) -> Result[BootParams]:
        """Run bootstrap and return Result[BootParams]."""
        timer = Timer()
        timer.start()

        data = design.data
        statistic = design.statistic
        n_resamples = design.n_resamples
        method = design.method
        statistic_type = design.statistic_type
        seed = design.seed

        n = data.shape[0]
        rng = np.random.default_rng(seed)

        # Compute t0: statistic on original data
        with timer.section('t0_computation'):
            if method == "parametric":
                t0 = np.atleast_1d(np.asarray(
                    statistic(data), dtype=np.float64
                ))
            else:
                # For nonparametric: pass original indices
                original_indices = np.arange(n)
                if statistic_type == "i":
                    t0 = np.atleast_1d(np.asarray(
                        statistic(data, original_indices), dtype=np.float64
                    ))
                elif statistic_type == "f":
                    freqs = np.ones(n, dtype=np.float64)
                    t0 = np.atleast_1d(np.asarray(
                        statistic(data, freqs), dtype=np.float64
                    ))
                elif statistic_type == "w":
                    weights = np.ones(n, dtype=np.float64) / n
                    t0 = np.atleast_1d(np.asarray(
                        statistic(data, weights), dtype=np.float64
                    ))

        k = len(t0)
        t = np.empty((n_resamples, k), dtype=np.float64)
        warnings_list: list[str] = []

        with timer.section('bootstrap_replicates'):
            if method == "ordinary":
                self._ordinary_bootstrap(
                    data, statistic, n_resamples, n, statistic_type, rng,
                    design.strata, t,
                )
            elif method == "balanced":
                self._balanced_bootstrap(
                    data, statistic, n_resamples, n, statistic_type, rng,
                    design.strata, t,
                )
            elif method == "parametric":
                self._parametric_bootstrap(
                    data, statistic, n_resamples, design.ran_gen,
                    design.mle, rng, t,
                )

        # Compute bias and SE
        with timer.section('summary_statistics'):
            bias = np.mean(t, axis=0) - t0
            standard_errors = np.std(t, axis=0, ddof=1)

        timer.stop()

        params = BootParams(
            t0=t0,
            t=t,
            n_resamples=n_resamples,
            bias=bias,
            standard_errors=standard_errors,
            conf_int=None,
            conf_level=None,
        )

        return Result(
            params=params,
            info={
                'method': method,
                'statistic_type': statistic_type,
                'n': n,
                'k': k,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=tuple(warnings_list),
        )

    def _ordinary_bootstrap(
        self,
        data: NDArray,
        statistic,
        n_resamples: int,
        n: int,
        statistic_type: str,
        rng: np.random.Generator,
        strata: NDArray | None,
        t: NDArray,
    ) -> None:
        """Ordinary nonparametric bootstrap (sampling with replacement)."""
        for b in range(n_resamples):
            if strata is not None:
                indices = self._stratified_sample(n, strata, rng)
            else:
                indices = rng.choice(n, size=n, replace=True)

            if statistic_type == "i":
                t[b] = statistic(data, indices)
            elif statistic_type == "f":
                freqs = np.bincount(indices, minlength=n).astype(np.float64)
                t[b] = statistic(data, freqs)
            elif statistic_type == "w":
                freqs = np.bincount(indices, minlength=n).astype(np.float64)
                weights = freqs / n
                t[b] = statistic(data, weights)

    def _balanced_bootstrap(
        self,
        data: NDArray,
        statistic,
        n_resamples: int,
        n: int,
        statistic_type: str,
        rng: np.random.Generator,
        strata: NDArray | None,
        t: NDArray,
    ) -> None:
        """
        Balanced bootstrap: each observation appears exactly n_resamples times.

        Pre-generates a pool of n*n_resamples indices where each of {0,...,n-1}
        appears exactly n_resamples times, then shuffles and splits into
        n_resamples samples of size n.
        """
        if strata is not None:
            # Stratified balanced: balance within each stratum
            unique_strata = np.unique(strata)
            all_indices = np.empty((n_resamples, n), dtype=int)
            for s in unique_strata:
                mask = strata == s
                ns = int(mask.sum())
                s_indices = np.where(mask)[0]
                pool = np.tile(s_indices, n_resamples)
                rng.shuffle(pool)
                for b in range(n_resamples):
                    all_indices[b, mask] = pool[b * ns:(b + 1) * ns]
        else:
            pool = np.tile(np.arange(n), n_resamples)
            rng.shuffle(pool)
            all_indices = pool.reshape(n_resamples, n)

        for b in range(n_resamples):
            indices = all_indices[b]
            if statistic_type == "i":
                t[b] = statistic(data, indices)
            elif statistic_type == "f":
                freqs = np.bincount(indices, minlength=n).astype(np.float64)
                t[b] = statistic(data, freqs)
            elif statistic_type == "w":
                freqs = np.bincount(indices, minlength=n).astype(np.float64)
                weights = freqs / n
                t[b] = statistic(data, weights)

    def _parametric_bootstrap(
        self,
        data: NDArray,
        statistic,
        n_resamples: int,
        ran_gen,
        mle,
        rng: np.random.Generator,
        t: NDArray,
    ) -> None:
        """Parametric bootstrap: generate data from ran_gen, compute statistic."""
        for b in range(n_resamples):
            sim_data = ran_gen(data, mle, rng)
            t[b] = statistic(sim_data)

    def _stratified_sample(
        self,
        n: int,
        strata: NDArray,
        rng: np.random.Generator,
    ) -> NDArray:
        """Sample with replacement within each stratum."""
        indices = np.empty(n, dtype=int)
        unique_strata = np.unique(strata)
        for s in unique_strata:
            mask = strata == s
            s_indices = np.where(mask)[0]
            ns = len(s_indices)
            sampled = rng.choice(s_indices, size=ns, replace=True)
            indices[mask] = sampled
        return indices


class CPUPermutationBackend:
    """
    CPU backend for permutation testing.

    Shuffles combined data and computes test statistic R times.
    P-value uses Phipson-Smyth correction: (count + 1) / (n_resamples + 1).
    """

    @property
    def name(self) -> str:
        return 'cpu_permutation'

    def solve(self, design: PermutationDesign) -> Result[PermutationParams]:
        """Run permutation test and return Result[PermutationParams]."""
        timer = Timer()
        timer.start()

        x = design.x
        y = design.y
        statistic = design.statistic
        n_resamples = design.n_resamples
        alternative = design.alternative
        seed = design.seed

        rng = np.random.default_rng(seed)

        # Compute observed statistic
        with timer.section('observed_stat'):
            observed = float(statistic(x, y))

        # Generate permutation distribution
        with timer.section('permutation_replicates'):
            combined = np.concatenate([x, y])
            n1 = len(x)
            perm_stats = np.empty(n_resamples, dtype=np.float64)

            for b in range(n_resamples):
                shuffled = rng.permutation(combined)
                perm_stats[b] = statistic(
                    shuffled[:n1], shuffled[n1:]
                )

        # Compute p-value with Phipson-Smyth correction
        with timer.section('p_value'):
            p_value = perm_pvalue(perm_stats, observed, alternative, n_resamples)

        timer.stop()

        params = PermutationParams(
            observed_stat=observed,
            perm_stats=perm_stats,
            p_value=p_value,
            n_resamples=n_resamples,
            alternative=alternative,
        )

        return Result(
            params=params,
            info={
                'n1': len(x),
                'n2': len(y),
                'alternative': alternative,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=(),
        )
