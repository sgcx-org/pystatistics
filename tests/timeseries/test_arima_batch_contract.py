"""Tests for the arima_batch per-series failure contract.

The contract (``_arima_batch_contract.py``) must be identical on every
backend: all-failed raises ``ConvergenceError``; partially-failed rows
come back NaN with ``converged=False`` and a loud ``UserWarning``; a
clean batch passes through untouched with no warning.

The DGP below is the validation suite's near-unit-root ARMA(1,1)
generator. Per-series pass/fail maps for seeds 5000+i were measured
empirically: rows i in {0, 1, 5} (ar=0.97, ma=0.3, n=1500) fail on
BOTH the CPU L-BFGS-B path and the GPU Adam path with a wide margin
(GPU AR estimates 1.029-1.040, far past the stationarity boundary),
while the ar=0.6/ma=0.4 seeds fit cleanly everywhere. The two
backends' fail sets are NOT identical in general (different
optimizers fail different borderline series) — the tests assert
semantic parity, never row-set equality.
"""

import warnings

import numpy as np
import pytest

from pystatistics.core.exceptions import ConvergenceError

CONTRACT_MSG = "failed to produce a valid stationary"


def _arma_dgp(n, seed, ar, ma):
    """The validation suite's ARMA(1,1) generator (50-obs burn-in)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 50)
    y = np.zeros(n + 50)
    for t in range(1, n + 50):
        y[t] = ar * y[t - 1] + e[t] + ma * e[t - 1]
    return y[50:]


def _failing_rows(k_list):
    """Series that converge to a non-stationary Whittle optimum on
    both backends (measured, see module docstring)."""
    return np.stack([_arma_dgp(1500, 5000 + i, 0.97, 0.3) for i in k_list])


def _healthy_rows(K):
    """Series that fit cleanly on both backends."""
    return np.stack([_arma_dgp(1500, 9000 + i, 0.6, 0.4) for i in range(K)])


def _gpu_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


def _no_contract_warning(record):
    return not any(CONTRACT_MSG in str(w.message) for w in record)


# =====================================================================
# Unit tests — the contract layer itself (no optimizer involved)
# =====================================================================


class TestNonstationaryRows:

    def test_flags_explosive_ar(self):
        from pystatistics.timeseries._arima_batch_contract import (
            nonstationary_rows,
        )
        ar = np.array([[0.5], [1.05], [-0.3], [0.999], [1.0]])
        np.testing.assert_array_equal(
            nonstationary_rows(ar),
            [False, True, False, False, True],
        )

    def test_ar2_case(self):
        from pystatistics.timeseries._arima_batch_contract import (
            nonstationary_rows,
        )
        # phi=(0.5, 0.3) stationary; phi=(0.9, 0.2) has phi1+phi2 > 1.
        ar = np.array([[0.5, 0.3], [0.9, 0.2]])
        np.testing.assert_array_equal(nonstationary_rows(ar), [False, True])

    def test_empty_ar_never_fails(self):
        from pystatistics.timeseries._arima_batch_contract import (
            nonstationary_rows,
        )
        assert not nonstationary_rows(np.zeros((4, 0))).any()

    def test_fp32_input_checked_in_fp64(self):
        from pystatistics.timeseries._arima_batch_contract import (
            nonstationary_rows,
        )
        ar = np.array([[0.4], [1.03]], dtype=np.float32)
        np.testing.assert_array_equal(nonstationary_rows(ar), [False, True])


class TestEnforceContract:

    def _raw(self, K=4, p=1, q=1):
        return dict(
            ar=np.full((K, p), 0.5),
            ma=np.full((K, q), -0.2),
            sigma2=np.ones(K),
            mean=np.zeros(K),
            converged=np.ones(K, dtype=bool),
            n_iter=10,
            backend_label="test-backend",
        )

    def test_no_failures_is_identity_no_warning(self):
        from pystatistics.timeseries._arima_batch_contract import (
            enforce_batch_failure_contract,
        )
        raw = self._raw()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = enforce_batch_failure_contract(
                failed=np.zeros(4, dtype=bool), **raw,
            )
        assert _no_contract_warning(rec)
        ar, ma, sigma2, mean, converged, wtuple = out
        # Clean batches pass through as the SAME objects — zero
        # behavior change on healthy data by construction.
        assert ar is raw["ar"] and ma is raw["ma"]
        assert sigma2 is raw["sigma2"] and mean is raw["mean"]
        assert converged is raw["converged"]
        assert wtuple == ()

    def test_partial_failure_nans_flags_and_warns(self):
        from pystatistics.timeseries._arima_batch_contract import (
            enforce_batch_failure_contract,
        )
        raw = self._raw()
        failed = np.array([True, False, True, False])
        with pytest.warns(UserWarning, match="2 of 4 series"):
            ar, ma, sigma2, mean, converged, wtuple = (
                enforce_batch_failure_contract(failed=failed, **raw)
            )
        for arr in (ar[:, 0], ma[:, 0], sigma2, mean):
            np.testing.assert_array_equal(np.isnan(arr), failed)
        np.testing.assert_array_equal(converged, ~failed)
        assert len(wtuple) == 1 and CONTRACT_MSG in wtuple[0]

    def test_partial_failure_does_not_mutate_inputs(self):
        from pystatistics.timeseries._arima_batch_contract import (
            enforce_batch_failure_contract,
        )
        raw = self._raw()
        with pytest.warns(UserWarning):
            enforce_batch_failure_contract(
                failed=np.array([True, False, False, False]), **raw,
            )
        assert np.isfinite(raw["ar"]).all()
        assert np.isfinite(raw["sigma2"]).all()
        assert raw["converged"].all()

    def test_partial_failure_mean_none(self):
        from pystatistics.timeseries._arima_batch_contract import (
            enforce_batch_failure_contract,
        )
        raw = self._raw()
        raw["mean"] = None
        with pytest.warns(UserWarning):
            *_, mean, _, _ = enforce_batch_failure_contract(
                failed=np.array([True, False, False, False]), **raw,
            )
        assert mean is None

    def test_all_failed_raises(self):
        from pystatistics.timeseries._arima_batch_contract import (
            enforce_batch_failure_contract,
        )
        with pytest.raises(ConvergenceError, match="all 4 series failed"):
            enforce_batch_failure_contract(
                failed=np.ones(4, dtype=bool), **self._raw(),
            )


# =====================================================================
# Integration — CPU backend
# =====================================================================


class TestArimaBatchContractCPU:

    def test_partial_failure_nan_warning_good_rows_exact(self):
        """Failed rows NaN + flagged; surviving rows identical to the
        single-series fit (the CPU loop is the same code path)."""
        from pystatistics.timeseries import arima, arima_batch
        Y = np.vstack([_failing_rows([0, 1]), _healthy_rows(2)])
        # tol pinned on both calls: arima_batch defaults to 1e-5,
        # single-series arima to 1e-8 — exact equality needs the same
        # stopping rule (same convention as the existing batch tests).
        with pytest.warns(UserWarning, match="2 of 4 series"):
            r = arima_batch(Y, order=(1, 0, 1), method="whittle",
                            backend="cpu", tol=1e-8)
        np.testing.assert_array_equal(
            np.isnan(r.sigma2), [True, True, False, False],
        )
        assert np.isnan(r.ar[:2]).all() and np.isnan(r.ma[:2]).all()
        assert np.isnan(r.mean[:2]).all()
        np.testing.assert_array_equal(
            r.converged, [False, False, True, True],
        )
        assert len(r.warnings) == 1 and CONTRACT_MSG in r.warnings[0]
        for k in (2, 3):
            rs = arima(Y[k], order=(1, 0, 1), method="whittle", tol=1e-8)
            np.testing.assert_array_equal(r.ar[k], rs.ar)
            np.testing.assert_array_equal(r.ma[k], rs.ma)
            assert r.sigma2[k] == rs.sigma2

    def test_all_fail_raises(self):
        from pystatistics.timeseries import arima_batch
        Y = _failing_rows([0, 1, 5])
        with pytest.raises(ConvergenceError, match="all 3 series failed"):
            arima_batch(Y, order=(1, 0, 1), method="whittle", backend="cpu")

    def test_all_good_unchanged_no_warning(self):
        from pystatistics.timeseries import arima, arima_batch
        Y = _healthy_rows(4)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            r = arima_batch(Y, order=(1, 0, 1), method="whittle",
                            backend="cpu", tol=1e-8)
        assert _no_contract_warning(rec)
        assert r.warnings == ()
        assert np.isfinite(r.ar).all() and np.isfinite(r.sigma2).all()
        assert r.converged.all()
        for k in range(4):
            rs = arima(Y[k], order=(1, 0, 1), method="whittle", tol=1e-8)
            np.testing.assert_array_equal(r.ar[k], rs.ar)
            assert r.sigma2[k] == rs.sigma2

    def test_single_series_arima_still_raises(self):
        """Constraint on the fix: the non-batch API keeps its raise
        behavior — it has no partial-failure case."""
        from pystatistics.timeseries import arima
        y = _failing_rows([0])[0]
        with pytest.raises(ConvergenceError, match="non-stationary"):
            arima(y, order=(1, 0, 1), method="whittle")

    def test_summary_reports_failed_count(self):
        from pystatistics.timeseries import arima_batch
        Y = np.vstack([_failing_rows([0]), _healthy_rows(2)])
        with pytest.warns(UserWarning):
            r = arima_batch(Y, order=(1, 0, 1), method="whittle",
                            backend="cpu")
        s = r.summary()
        assert "Failed: 1/3" in s
        # nan-aware aggregates — no NaN leaks into the statistics
        assert "=nan" not in s.lower()


# =====================================================================
# Integration — GPU backend (semantic parity with CPU)
# =====================================================================


@pytest.mark.skipif(not _gpu_available(), reason="no torch GPU (CUDA/MPS)")
class TestArimaBatchContractGPU:

    def test_partial_failure_semantics(self):
        from pystatistics.timeseries import arima_batch
        Y = np.vstack([_failing_rows([0, 1]), _healthy_rows(2)])
        with pytest.warns(UserWarning, match="2 of 4 series"):
            r = arima_batch(Y, order=(1, 0, 1), method="whittle",
                            backend="gpu")
        failed = np.isnan(r.sigma2)
        np.testing.assert_array_equal(failed, [True, True, False, False])
        assert np.isnan(r.ar[failed]).all()
        assert np.isnan(r.ma[failed]).all()
        assert np.isnan(r.mean[failed]).all()
        assert not r.converged[failed].any()
        assert np.isfinite(r.ar[~failed]).all()
        assert len(r.warnings) == 1 and CONTRACT_MSG in r.warnings[0]

    def test_surviving_rows_are_stationary(self):
        """Whatever the GPU returns as numbers must pass the fp64
        stationarity check — non-stationary estimates can only ever
        surface as NaN."""
        from pystatistics.timeseries import arima_batch
        from pystatistics.timeseries._arima_batch_contract import (
            nonstationary_rows,
        )
        Y = np.stack(
            [_arma_dgp(1500, 5000 + i, 0.97, 0.3) for i in range(16)]
        )
        with pytest.warns(UserWarning, match=CONTRACT_MSG):
            r = arima_batch(Y, order=(1, 0, 1), method="whittle",
                            backend="gpu")
        finite = ~np.isnan(r.sigma2)
        assert finite.any()             # measured: >= 3 of these survive
        assert not nonstationary_rows(r.ar[finite]).any()

    def test_all_fail_raises(self):
        from pystatistics.timeseries import arima_batch
        Y = _failing_rows([0, 1, 5])
        with pytest.raises(ConvergenceError, match="all 3 series failed"):
            arima_batch(Y, order=(1, 0, 1), method="whittle", backend="gpu")

    def test_all_good_no_warning(self):
        from pystatistics.timeseries import arima_batch
        Y = _healthy_rows(8)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            r = arima_batch(Y, order=(1, 0, 1), method="whittle",
                            backend="gpu")
        assert _no_contract_warning(rec)
        assert r.warnings == ()
        assert np.isfinite(r.ar).all() and np.isfinite(r.sigma2).all()
        # true AR is 0.6; loose sanity bound, not a tolerance-tier claim
        assert abs(float(r.ar[:, 0].mean()) - 0.6) < 0.05

    def test_contract_parity_with_cpu(self):
        """Same inputs → same failure semantics on both backends:
        partial fail warns on both, all-fail raises on both."""
        from pystatistics.timeseries import arima_batch
        Y_partial = np.vstack([_failing_rows([0, 1]), _healthy_rows(2)])
        Y_allfail = _failing_rows([0, 1, 5])
        for backend in ("cpu", "gpu"):
            with pytest.warns(UserWarning, match="2 of 4 series"):
                r = arima_batch(Y_partial, order=(1, 0, 1),
                                method="whittle", backend=backend)
            assert np.isnan(r.sigma2).sum() == 2
            with pytest.raises(ConvergenceError, match="all 3 series"):
                arima_batch(Y_allfail, order=(1, 0, 1),
                            method="whittle", backend=backend)
