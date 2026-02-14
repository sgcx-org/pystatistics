"""
GPU backend for descriptive statistics using PyTorch.

Performance path for large datasets — validated against CPU reference.
Supports CUDA (Linux/Windows) and MPS (macOS Apple Silicon).

FP32 by default. Spearman/Kendall fall back to CPU (scipy-dependent).
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.core.compute.device import DeviceInfo
from pystatistics.descriptive.design import DescriptiveDesign
from pystatistics.descriptive.solution import DescriptiveParams
from pystatistics.descriptive._missing import apply_use_policy, columnwise_clean
from pystatistics.descriptive._quantile_types import r_quantile


class GPUDescriptiveBackend:
    """
    GPU backend for descriptive statistics using PyTorch.

    Accelerates mean, variance, covariance, Pearson correlation, quantiles,
    skewness, kurtosis, and summary. Spearman and Kendall fall back to
    the CPU backend (scipy-dependent).

    FP32 by default for performance. Returns FP64 numpy arrays for
    consistency with the CPU reference backend.
    """

    def __init__(self, device: DeviceInfo | None = None):
        """
        Initialize GPU backend.

        Parameters
        ----------
        device : DeviceInfo, optional
            Device info from select_device(). If None, auto-selects.
        """
        import torch

        if device is not None:
            if device.device_type == 'cuda':
                self.device = torch.device(f'cuda:{device.device_index or 0}')
                self.dtype = torch.float32
                self.device_name = device.name
            elif device.device_type == 'mps':
                self.device = torch.device('mps')
                self.dtype = torch.float32
                self.device_name = 'Apple Silicon GPU (MPS)'
            else:
                raise ValueError(f"GPUDescriptiveBackend requires GPU device, got {device.device_type}")
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.dtype = torch.float32
                self.device_name = torch.cuda.get_device_properties(0).name
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                self.dtype = torch.float32
                self.device_name = 'Apple Silicon GPU (MPS)'
            else:
                raise RuntimeError(
                    "No GPU available. Use backend='cpu' instead."
                )

    @property
    def name(self) -> str:
        return 'gpu_descriptive_fp32'

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
        """Compute requested descriptive statistics on GPU."""
        import torch

        timer = Timer()
        timer.start()

        data_np = design.data
        warnings_list: list[str] = []

        # Apply missing data policy (CPU — lightweight)
        with timer.section('missing_data'):
            clean_np, n_complete = apply_use_policy(data_np, use)

        # Transfer to GPU
        with timer.section('data_transfer_to_gpu'):
            data_gpu = torch.from_numpy(clean_np).to(
                device=self.device, dtype=self.dtype
            )

        n, p = data_gpu.shape

        # Results (all start as None)
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

        # --- GPU-accelerated computations ---

        if 'mean' in compute:
            with timer.section('mean'):
                if use == 'everything':
                    m = torch.mean(data_gpu, dim=0)
                else:
                    m = torch.nanmean(data_gpu, dim=0)
                mean = m.cpu().numpy().astype(np.float64)

        if 'var' in compute:
            with timer.section('variance'):
                if use == 'everything':
                    # ddof=1 (Bessel correction)
                    v = torch.var(data_gpu, dim=0, unbiased=True)
                else:
                    # For complete.obs, no NaN in clean data
                    v = torch.var(data_gpu, dim=0, unbiased=True)
                variance = v.cpu().numpy().astype(np.float64)

        if 'sd' in compute:
            with timer.section('sd'):
                if variance is not None:
                    sd = np.sqrt(variance)
                else:
                    v = torch.var(data_gpu, dim=0, unbiased=True)
                    sd = np.sqrt(v.cpu().numpy().astype(np.float64))

        if 'cov' in compute:
            with timer.section('covariance'):
                if use == 'pairwise.complete.obs':
                    # Fall back to CPU for pairwise
                    from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend
                    cpu = CPUDescriptiveBackend()
                    cov_result = cpu._compute_covariance(clean_np, use, data_np)
                    if isinstance(cov_result, tuple):
                        covariance_matrix, pairwise_n = cov_result
                    else:
                        covariance_matrix = cov_result
                else:
                    # Center data
                    mu = torch.mean(data_gpu, dim=0, keepdim=True)
                    X_c = data_gpu - mu
                    cov_gpu = (X_c.T @ X_c) / (n - 1)
                    covariance_matrix = cov_gpu.cpu().numpy().astype(np.float64)

        if 'cor_pearson' in compute:
            with timer.section('cor_pearson'):
                if use == 'pairwise.complete.obs':
                    from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend
                    cpu = CPUDescriptiveBackend()
                    cor_result = cpu._cor_pearson_pairwise(data_np)
                    if isinstance(cor_result, tuple):
                        cor_pearson, pairwise_n = cor_result
                    else:
                        cor_pearson = cor_result
                else:
                    mu = torch.mean(data_gpu, dim=0, keepdim=True)
                    X_c = data_gpu - mu
                    cov_gpu = (X_c.T @ X_c) / (n - 1)
                    sd_gpu = torch.sqrt(torch.diag(cov_gpu))
                    outer_sd = sd_gpu.unsqueeze(1) * sd_gpu.unsqueeze(0)
                    cor_gpu = cov_gpu / outer_sd
                    # Force diagonal to exactly 1.0
                    cor_gpu.fill_diagonal_(1.0)
                    cor_pearson = cor_gpu.cpu().numpy().astype(np.float64)

        # Spearman and Kendall fall back to CPU (scipy-dependent)
        if 'cor_spearman' in compute:
            with timer.section('cor_spearman'):
                from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend
                cpu = CPUDescriptiveBackend()
                cor_result = cpu._compute_cor_spearman(clean_np, use, data_np)
                if isinstance(cor_result, tuple):
                    cor_spearman, pw_n = cor_result
                    if pairwise_n is None:
                        pairwise_n = pw_n
                else:
                    cor_spearman = cor_result

        if 'cor_kendall' in compute:
            with timer.section('cor_kendall'):
                from pystatistics.descriptive.backends.cpu import CPUDescriptiveBackend
                cpu = CPUDescriptiveBackend()
                cor_result = cpu._compute_cor_kendall(clean_np, use, data_np)
                if isinstance(cor_result, tuple):
                    cor_kendall, pw_n = cor_result
                    if pairwise_n is None:
                        pairwise_n = pw_n
                else:
                    cor_kendall = cor_result

        # Quantiles — CPU for exact R matching (sort-based, per-column)
        if 'quantiles' in compute:
            with timer.section('quantiles'):
                q_probs = quantile_probs
                if q_probs is None:
                    q_probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                q_type = quantile_type
                # Use CPU r_quantile for exact R matching
                quantiles = np.empty((len(q_probs), p), dtype=np.float64)
                for j in range(p):
                    if use == 'everything':
                        col = clean_np[:, j]
                        if np.any(np.isnan(col)):
                            quantiles[:, j] = np.nan
                            continue
                        col_sorted = np.sort(col)
                    else:
                        col_sorted = np.sort(columnwise_clean(clean_np[:, j]))
                    if len(col_sorted) == 0:
                        quantiles[:, j] = np.nan
                    else:
                        quantiles[:, j] = r_quantile(col_sorted, q_probs, q_type)

        if 'summary' in compute:
            with timer.section('summary'):
                summary_table = np.empty((6, p), dtype=np.float64)
                for j in range(p):
                    if use == 'everything':
                        col = clean_np[:, j]
                        if np.any(np.isnan(col)):
                            summary_table[:, j] = np.nan
                            continue
                        col_sorted = np.sort(col)
                    else:
                        col_sorted = np.sort(columnwise_clean(clean_np[:, j]))
                    if len(col_sorted) == 0:
                        summary_table[:, j] = np.nan
                        continue
                    q_vals = r_quantile(col_sorted, np.array([0.25, 0.5, 0.75]), 7)
                    summary_table[0, j] = col_sorted[0]
                    summary_table[1, j] = q_vals[0]
                    summary_table[2, j] = q_vals[1]
                    summary_table[3, j] = np.mean(col_sorted)
                    summary_table[4, j] = q_vals[2]
                    summary_table[5, j] = col_sorted[-1]

        if 'skewness' in compute:
            with timer.section('skewness'):
                skewness = self._compute_skewness_gpu(data_gpu, clean_np, use, n, p, torch)

        if 'kurtosis' in compute:
            with timer.section('kurtosis'):
                kurtosis = self._compute_kurtosis_gpu(data_gpu, clean_np, use, n, p, torch)

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
            info={
                'use': use,
                'computed': sorted(compute),
                'device': str(self.device),
                'dtype': str(self.dtype),
                'device_name': self.device_name,
            },
            timing=timer.result(),
            backend_name=self.name,
            warnings=tuple(warnings_list),
            provenance={'algorithm': 'gpu'},
        )

    def _compute_skewness_gpu(self, data_gpu, data_np, use, n, p, torch):
        """GPU-accelerated skewness computation."""
        result = np.empty(p, dtype=np.float64)
        for j in range(p):
            if use == 'everything':
                col = data_np[:, j]
                if np.any(np.isnan(col)):
                    result[j] = np.nan
                    continue
            else:
                col = columnwise_clean(data_np[:, j])
            nj = len(col)
            if nj < 3:
                result[j] = np.nan
                continue
            col_gpu = data_gpu[:nj, j] if use != 'everything' else data_gpu[:, j]
            mu = torch.mean(col_gpu)
            diffs = col_gpu - mu
            m2 = torch.mean(diffs ** 2)
            m3 = torch.mean(diffs ** 3)
            if m2.item() == 0:
                result[j] = np.nan
                continue
            g1 = (m3 / (m2 ** 1.5)).item()
            result[j] = g1 * np.sqrt(nj * (nj - 1)) / (nj - 2)
        return result

    def _compute_kurtosis_gpu(self, data_gpu, data_np, use, n, p, torch):
        """GPU-accelerated kurtosis computation."""
        result = np.empty(p, dtype=np.float64)
        for j in range(p):
            if use == 'everything':
                col = data_np[:, j]
                if np.any(np.isnan(col)):
                    result[j] = np.nan
                    continue
            else:
                col = columnwise_clean(data_np[:, j])
            nj = len(col)
            if nj < 4:
                result[j] = np.nan
                continue
            col_gpu = data_gpu[:nj, j] if use != 'everything' else data_gpu[:, j]
            mu = torch.mean(col_gpu)
            diffs = col_gpu - mu
            m2 = torch.mean(diffs ** 2)
            m4 = torch.mean(diffs ** 4)
            if m2.item() == 0:
                result[j] = np.nan
                continue
            g2 = (m4 / (m2 ** 2)).item() - 3.0
            result[j] = ((nj - 1.0) / ((nj - 2.0) * (nj - 3.0))) * ((nj + 1.0) * g2 + 6.0)
        return result
