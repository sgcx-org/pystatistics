"""
CPU reference backend for descriptive statistics.

Validated against R to rtol=1e-10.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.descriptive.design import DescriptiveDesign
from pystatistics.descriptive.solution import DescriptiveParams
from pystatistics.descriptive._missing import (
    apply_use_policy, pairwise_mask, columnwise_clean,
)
from pystatistics.descriptive._quantile_types import r_quantile


class CPUDescriptiveBackend:
    """CPU reference backend for descriptive statistics."""

    @property
    def name(self) -> str:
        return 'cpu_descriptive'

    def solve(
        self,
        design: DescriptiveDesign,
        *,
        compute: set[str],
        use: str = 'everything',
        cor_method: str = 'pearson',
        quantile_probs: NDArray | None = None,
        quantile_type: int = 7,
    ) -> Result[DescriptiveParams]:
        """
        Compute requested descriptive statistics.

        Parameters
        ----------
        design : DescriptiveDesign
        compute : set of str
            Which statistics to compute. Valid entries:
            'mean', 'var', 'sd', 'cov', 'cor_pearson', 'cor_spearman',
            'cor_kendall', 'quantiles', 'summary', 'skewness', 'kurtosis'
        use : str
            Missing data policy.
        cor_method : str
            Correlation method (for 'cor_*' entries).
        quantile_probs : array-like or None
            Quantile probabilities. Default (0, 0.25, 0.5, 0.75, 1.0).
        quantile_type : int
            R quantile type 1-9.
        """
        timer = Timer()
        timer.start()

        data = design.data
        warnings_list: list[str] = []

        # Apply missing data policy for column-wise statistics
        with timer.section('missing_data'):
            clean_data, n_complete = apply_use_policy(data, use)

        # Compute requested statistics
        mean = None
        variance = None
        sd = None
        skewness = None
        kurtosis = None
        covariance_matrix = None
        cor_pearson = None
        cor_spearman = None
        cor_kendall = None
        quantiles = None
        q_probs = None
        q_type = None
        summary_table = None
        pairwise_n = None

        if 'mean' in compute:
            with timer.section('mean'):
                mean = self._compute_mean(clean_data, use)

        if 'var' in compute:
            with timer.section('variance'):
                variance = self._compute_variance(clean_data, use)

        if 'sd' in compute:
            with timer.section('sd'):
                if variance is not None:
                    sd = np.sqrt(variance)
                else:
                    var_tmp = self._compute_variance(clean_data, use)
                    sd = np.sqrt(var_tmp)

        if 'cov' in compute:
            with timer.section('covariance'):
                cov_result = self._compute_covariance(clean_data, use, data)
                if isinstance(cov_result, tuple):
                    covariance_matrix, pairwise_n = cov_result
                else:
                    covariance_matrix = cov_result

        if 'cor_pearson' in compute:
            with timer.section('cor_pearson'):
                cor_result = self._compute_cor_pearson(clean_data, use, data)
                if isinstance(cor_result, tuple):
                    cor_pearson, pairwise_n = cor_result
                else:
                    cor_pearson = cor_result

        # Placeholders for phases 3-4
        if 'quantiles' in compute:
            with timer.section('quantiles'):
                q_probs = quantile_probs
                if q_probs is None:
                    q_probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                q_type = quantile_type
                quantiles = self._compute_quantiles(clean_data, use, q_probs, q_type)

        if 'summary' in compute:
            with timer.section('summary'):
                summary_table = self._compute_summary(clean_data, use)

        if 'cor_spearman' in compute:
            with timer.section('cor_spearman'):
                cor_result = self._compute_cor_spearman(clean_data, use, data)
                if isinstance(cor_result, tuple):
                    cor_spearman, pw_n = cor_result
                    if pairwise_n is None:
                        pairwise_n = pw_n
                else:
                    cor_spearman = cor_result

        if 'cor_kendall' in compute:
            with timer.section('cor_kendall'):
                cor_result = self._compute_cor_kendall(clean_data, use, data)
                if isinstance(cor_result, tuple):
                    cor_kendall, pw_n = cor_result
                    if pairwise_n is None:
                        pairwise_n = pw_n
                else:
                    cor_kendall = cor_result

        if 'skewness' in compute:
            with timer.section('skewness'):
                skewness = self._compute_skewness(clean_data, use)

        if 'kurtosis' in compute:
            with timer.section('kurtosis'):
                kurtosis = self._compute_kurtosis(clean_data, use)

        timer.stop()

        params = DescriptiveParams(
            mean=mean,
            variance=variance,
            sd=sd,
            skewness=skewness,
            kurtosis=kurtosis,
            covariance_matrix=covariance_matrix,
            correlation_pearson=cor_pearson,
            correlation_spearman=cor_spearman,
            correlation_kendall=cor_kendall,
            quantiles=quantiles,
            quantile_probs=q_probs,
            quantile_type=q_type,
            summary_table=summary_table,
            n_complete=n_complete,
            pairwise_n=pairwise_n,
        )

        return Result(
            params=params,
            info={'use': use, 'computed': sorted(compute)},
            timing=timer.result(),
            backend_name=self.name,
            warnings=tuple(warnings_list),
            provenance={'algorithm': 'direct'},
        )

    # --- Column-wise statistics ---

    def _compute_mean(self, data: NDArray, use: str) -> NDArray:
        """Compute column means. Matches R colMeans()."""
        if use == 'everything':
            # NaN propagates (matches R colMeans with na.rm=FALSE)
            return np.mean(data, axis=0)
        else:
            # For complete.obs, data is already cleaned (no NaN).
            # For pairwise, compute per-column ignoring NaN.
            return np.nanmean(data, axis=0)

    def _compute_variance(self, data: NDArray, use: str) -> NDArray:
        """Compute column variance with Bessel correction (n-1). Matches R var()."""
        if use == 'everything':
            return np.var(data, axis=0, ddof=1)
        else:
            return np.nanvar(data, axis=0, ddof=1)

    # --- Bivariate statistics ---

    def _compute_covariance(
        self, data: NDArray, use: str, original_data: NDArray
    ) -> NDArray | tuple[NDArray, NDArray]:
        """
        Compute covariance matrix. Matches R cov().

        For use='pairwise.complete.obs', returns (cov_matrix, pairwise_n).
        Otherwise returns cov_matrix.
        """
        if use == 'pairwise.complete.obs':
            return self._cov_pairwise(original_data)
        else:
            # data is already cleaned for complete.obs, or raw for everything
            p = data.shape[1]
            if p == 1:
                var_val = np.var(data[:, 0], ddof=1) if use != 'everything' else np.var(data[:, 0], ddof=1)
                return np.array([[var_val]])
            return np.cov(data, rowvar=False, ddof=1)

    def _cov_pairwise(self, data: NDArray) -> tuple[NDArray, NDArray]:
        """Covariance with use='pairwise.complete.obs'. Matches R exactly."""
        n, p = data.shape
        cov_mat = np.empty((p, p), dtype=np.float64)
        n_pairs = np.empty((p, p), dtype=np.int64)

        for i in range(p):
            for j in range(i, p):
                mask = pairwise_mask(data[:, i], data[:, j])
                ni = int(mask.sum())
                n_pairs[i, j] = n_pairs[j, i] = ni

                if ni < 2:
                    cov_mat[i, j] = cov_mat[j, i] = np.nan
                else:
                    xi = data[mask, i]
                    xj = data[mask, j]
                    cov_mat[i, j] = cov_mat[j, i] = np.sum(
                        (xi - xi.mean()) * (xj - xj.mean())
                    ) / (ni - 1)

        return cov_mat, n_pairs

    def _compute_cor_pearson(
        self, data: NDArray, use: str, original_data: NDArray
    ) -> NDArray | tuple[NDArray, NDArray]:
        """Pearson correlation. Matches R cor(method='pearson')."""
        if use == 'pairwise.complete.obs':
            return self._cor_pearson_pairwise(original_data)
        else:
            p = data.shape[1]
            if p == 1:
                return np.array([[1.0]])
            cov_mat = np.cov(data, rowvar=False, ddof=1)
            sd = np.sqrt(np.diag(cov_mat))
            # R returns NaN when a variable has zero variance
            with np.errstate(divide='ignore', invalid='ignore'):
                cor_mat = cov_mat / np.outer(sd, sd)
            np.fill_diagonal(cor_mat, 1.0)
            return cor_mat

    def _cor_pearson_pairwise(self, data: NDArray) -> tuple[NDArray, NDArray]:
        """
        Pairwise Pearson correlation. Matches R cor(use='pairwise.complete.obs').

        For each (i,j) pair, uses only the rows where BOTH variables are non-NaN,
        and computes the correlation using the SD from those shared rows only.
        """
        n, p = data.shape
        cor_mat = np.eye(p, dtype=np.float64)
        n_pairs = np.empty((p, p), dtype=np.int64)

        for i in range(p):
            n_pairs[i, i] = int((~np.isnan(data[:, i])).sum())
            for j in range(i + 1, p):
                mask = pairwise_mask(data[:, i], data[:, j])
                ni = int(mask.sum())
                n_pairs[i, j] = n_pairs[j, i] = ni

                if ni < 2:
                    cor_mat[i, j] = cor_mat[j, i] = np.nan
                else:
                    xi = data[mask, i]
                    xj = data[mask, j]
                    # Compute correlation on shared observations
                    xi_c = xi - xi.mean()
                    xj_c = xj - xj.mean()
                    num = np.sum(xi_c * xj_c)
                    denom = np.sqrt(np.sum(xi_c ** 2) * np.sum(xj_c ** 2))
                    if denom == 0:
                        cor_mat[i, j] = cor_mat[j, i] = np.nan
                    else:
                        cor_mat[i, j] = cor_mat[j, i] = num / denom

        return cor_mat, n_pairs

    # --- Placeholder methods for phases 3-4 ---

    def _compute_quantiles(
        self, data: NDArray, use: str,
        probs: NDArray, qtype: int,
    ) -> NDArray:
        """
        Compute quantiles using R's algorithm. Matches R quantile() exactly.

        Returns shape (n_probs, p).
        """
        n, p = data.shape
        result = np.empty((len(probs), p), dtype=np.float64)

        for j in range(p):
            if use == 'everything':
                col = data[:, j]
                if np.any(np.isnan(col)):
                    result[:, j] = np.nan
                    continue
                col_sorted = np.sort(col)
            else:
                col_sorted = np.sort(columnwise_clean(data[:, j]))

            if len(col_sorted) == 0:
                result[:, j] = np.nan
            else:
                result[:, j] = r_quantile(col_sorted, probs, qtype)

        return result

    def _compute_summary(self, data: NDArray, use: str) -> NDArray:
        """
        Compute six-number summary: Min, Q1, Median, Mean, Q3, Max.

        Returns shape (6, p). Uses R quantile type 7 for Q1/Median/Q3.
        """
        n, p = data.shape
        summary = np.empty((6, p), dtype=np.float64)

        for j in range(p):
            if use == 'everything':
                col = data[:, j]
                if np.any(np.isnan(col)):
                    summary[:, j] = np.nan
                    continue
                col_sorted = np.sort(col)
            else:
                col_sorted = np.sort(columnwise_clean(data[:, j]))

            if len(col_sorted) == 0:
                summary[:, j] = np.nan
                continue

            # Quantiles for Q1, Median, Q3 using R type 7 (R default)
            q_probs = np.array([0.25, 0.5, 0.75])
            q_vals = r_quantile(col_sorted, q_probs, qtype=7)

            summary[0, j] = col_sorted[0]           # Min
            summary[1, j] = q_vals[0]                # Q1 (1st Qu.)
            summary[2, j] = q_vals[1]                # Median
            summary[3, j] = np.mean(col_sorted)      # Mean
            summary[4, j] = q_vals[2]                # Q3 (3rd Qu.)
            summary[5, j] = col_sorted[-1]           # Max

        return summary

    def _compute_cor_spearman(
        self, data: NDArray, use: str, original_data: NDArray,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """
        Spearman rank correlation. Matches R cor(method='spearman').

        Approach: rank each column (average ties), then Pearson on ranks.
        """
        from scipy.stats import rankdata

        if use == 'pairwise.complete.obs':
            return self._cor_spearman_pairwise(original_data)

        p = data.shape[1]
        if p == 1:
            return np.array([[1.0]])

        # Rank each column with average tie-breaking (matches R)
        ranked = np.empty_like(data)
        for j in range(p):
            col = data[:, j]
            if use == 'everything' and np.any(np.isnan(col)):
                ranked[:, j] = np.nan
            else:
                clean = col if use == 'everything' else col[~np.isnan(col)]
                # For complete.obs, data is already cleaned, so this is fine
                ranked[:, j] = rankdata(col, method='average')

        # Pearson on ranks
        with np.errstate(divide='ignore', invalid='ignore'):
            cov_mat = np.cov(ranked, rowvar=False, ddof=1)
            sd = np.sqrt(np.diag(cov_mat))
            cor_mat = cov_mat / np.outer(sd, sd)
        np.fill_diagonal(cor_mat, 1.0)
        return cor_mat

    def _cor_spearman_pairwise(self, data: NDArray) -> tuple[NDArray, NDArray]:
        """Pairwise Spearman correlation. Matches R cor(method='spearman', use='pairwise.complete.obs')."""
        from scipy.stats import rankdata

        n, p = data.shape
        cor_mat = np.eye(p, dtype=np.float64)
        n_pairs = np.empty((p, p), dtype=np.int64)

        for i in range(p):
            n_pairs[i, i] = int((~np.isnan(data[:, i])).sum())
            for j in range(i + 1, p):
                mask = pairwise_mask(data[:, i], data[:, j])
                ni = int(mask.sum())
                n_pairs[i, j] = n_pairs[j, i] = ni

                if ni < 2:
                    cor_mat[i, j] = cor_mat[j, i] = np.nan
                else:
                    xi = data[mask, i]
                    xj = data[mask, j]
                    ri = rankdata(xi, method='average')
                    rj = rankdata(xj, method='average')
                    ri_c = ri - ri.mean()
                    rj_c = rj - rj.mean()
                    num = np.sum(ri_c * rj_c)
                    denom = np.sqrt(np.sum(ri_c ** 2) * np.sum(rj_c ** 2))
                    if denom == 0:
                        cor_mat[i, j] = cor_mat[j, i] = np.nan
                    else:
                        cor_mat[i, j] = cor_mat[j, i] = num / denom

        return cor_mat, n_pairs

    def _compute_cor_kendall(
        self, data: NDArray, use: str, original_data: NDArray,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """
        Kendall tau-b rank correlation. Matches R cor(method='kendall').

        Uses scipy.stats.kendalltau which computes tau-b (handles ties).
        """
        from scipy.stats import kendalltau

        if use == 'pairwise.complete.obs':
            return self._cor_kendall_pairwise(original_data)

        p = data.shape[1]
        cor_mat = np.eye(p, dtype=np.float64)

        for i in range(p):
            for j in range(i + 1, p):
                xi = data[:, i]
                xj = data[:, j]
                if use == 'everything' and (np.any(np.isnan(xi)) or np.any(np.isnan(xj))):
                    cor_mat[i, j] = cor_mat[j, i] = np.nan
                else:
                    tau, _ = kendalltau(xi, xj)
                    cor_mat[i, j] = cor_mat[j, i] = tau

        return cor_mat

    def _cor_kendall_pairwise(self, data: NDArray) -> tuple[NDArray, NDArray]:
        """Pairwise Kendall correlation. Matches R cor(method='kendall', use='pairwise.complete.obs')."""
        from scipy.stats import kendalltau

        n, p = data.shape
        cor_mat = np.eye(p, dtype=np.float64)
        n_pairs = np.empty((p, p), dtype=np.int64)

        for i in range(p):
            n_pairs[i, i] = int((~np.isnan(data[:, i])).sum())
            for j in range(i + 1, p):
                mask = pairwise_mask(data[:, i], data[:, j])
                ni = int(mask.sum())
                n_pairs[i, j] = n_pairs[j, i] = ni

                if ni < 2:
                    cor_mat[i, j] = cor_mat[j, i] = np.nan
                else:
                    xi = data[mask, i]
                    xj = data[mask, j]
                    tau, _ = kendalltau(xi, xj)
                    cor_mat[i, j] = cor_mat[j, i] = tau

        return cor_mat, n_pairs

    def _compute_skewness(self, data: NDArray, use: str) -> NDArray:
        """
        Compute bias-adjusted skewness. Matches R e1071::skewness(type=2).

        Formula (type 2, bias-adjusted):
            G1 = m3 / m2^1.5  (sample skewness)
            skewness = G1 * sqrt(n*(n-1)) / (n-2)

        where m2 = sum((x-mean)^2)/n, m3 = sum((x-mean)^3)/n.
        Requires n >= 3.
        """
        p = data.shape[1]
        result = np.empty(p, dtype=np.float64)

        for j in range(p):
            if use == 'everything':
                col = data[:, j]
                if np.any(np.isnan(col)):
                    result[j] = np.nan
                    continue
            else:
                col = columnwise_clean(data[:, j])

            n = len(col)
            if n < 3:
                result[j] = np.nan
                continue

            mean = np.mean(col)
            diffs = col - mean
            m2 = np.sum(diffs ** 2) / n
            m3 = np.sum(diffs ** 3) / n

            if m2 == 0:
                result[j] = np.nan
                continue

            g1 = m3 / (m2 ** 1.5)  # sample skewness
            # Bias-adjusted (e1071 type 2)
            result[j] = g1 * np.sqrt(n * (n - 1)) / (n - 2)

        return result

    def _compute_kurtosis(self, data: NDArray, use: str) -> NDArray:
        """
        Compute bias-adjusted excess kurtosis. Matches R e1071::kurtosis(type=2).

        Formula (type 2, bias-adjusted):
            G2 = m4/m2^2 - 3  (excess kurtosis, sample)
            kurtosis = ((n-1)/((n-2)*(n-3))) * ((n+1)*G2 + 6)

        where m2 = sum((x-mean)^2)/n, m4 = sum((x-mean)^4)/n.
        Requires n >= 4.
        """
        p = data.shape[1]
        result = np.empty(p, dtype=np.float64)

        for j in range(p):
            if use == 'everything':
                col = data[:, j]
                if np.any(np.isnan(col)):
                    result[j] = np.nan
                    continue
            else:
                col = columnwise_clean(data[:, j])

            n = len(col)
            if n < 4:
                result[j] = np.nan
                continue

            mean = np.mean(col)
            diffs = col - mean
            m2 = np.sum(diffs ** 2) / n
            m4 = np.sum(diffs ** 4) / n

            if m2 == 0:
                result[j] = np.nan
                continue

            g2 = m4 / (m2 ** 2) - 3.0  # excess kurtosis (sample)
            # Bias-adjusted (e1071 type 2)
            result[j] = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0)

        return result
