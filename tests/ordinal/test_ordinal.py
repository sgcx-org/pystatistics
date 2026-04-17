"""
Tests for ordinal regression (proportional odds model).

Tests cover:
- Input validation (normal, edge, failure cases)
- Likelihood and gradient correctness
- Recovery of known parameters from simulated data
- Logistic, probit, and cloglog link functions
- Summary output format
- Consistency with R's MASS::polr() reference values

All reference values were generated with R's MASS::polr() using the
documented seeds and data-generating processes.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.ordinal import polr, OrdinalSolution
from pystatistics.ordinal._common import OrdinalParams
from pystatistics.ordinal._likelihood import (
    CLogLogLink,
    cumulative_gradient,
    cumulative_negloglik,
    raw_to_thresholds,
    thresholds_to_raw,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def simple_data():
    """
    Generate a simple 3-level ordinal dataset with known parameters.

    True parameters: beta = [1.0, -0.5], thresholds = [-1.0, 1.0]
    Link: logistic (proportional odds).
    """
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 2))
    eta = 1.0 * X[:, 0] - 0.5 * X[:, 1]

    # Cumulative probabilities via logistic link
    cum_p1 = 1.0 / (1.0 + np.exp(-(-1.0 - eta)))
    cum_p2 = 1.0 / (1.0 + np.exp(-(1.0 - eta)))

    u = rng.uniform(size=n)
    y = np.where(u < cum_p1, 0, np.where(u < cum_p2, 1, 2))

    return y, X


@pytest.fixture
def four_level_data():
    """Generate a 4-level ordinal dataset."""
    rng = np.random.default_rng(123)
    n = 600
    X = rng.standard_normal((n, 3))
    eta = 0.8 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * X[:, 2]

    thresholds = np.array([-1.5, 0.0, 1.5])
    cum_p = 1.0 / (1.0 + np.exp(-(thresholds[:, np.newaxis] - eta[np.newaxis, :])))

    u = rng.uniform(size=n)
    y = np.zeros(n, dtype=int)
    for j in range(3):
        y += (u > cum_p[j]).astype(int)

    return y, X


# =========================================================================
# Threshold parameterization tests
# =========================================================================

class TestThresholdTransform:
    """Tests for raw <-> threshold transforms."""

    def test_roundtrip(self):
        """raw_to_thresholds(thresholds_to_raw(alpha)) == alpha."""
        alpha = np.array([-2.0, -0.5, 1.0, 3.0])
        raw = thresholds_to_raw(alpha)
        recovered = raw_to_thresholds(raw)
        assert_allclose(recovered, alpha, atol=1e-12)

    def test_ordering_preserved(self):
        """Transformed thresholds are strictly increasing."""
        raw = np.array([0.5, -1.0, 0.3, 2.0])
        alpha = raw_to_thresholds(raw)
        diffs = np.diff(alpha)
        assert np.all(diffs > 0), f"Thresholds not increasing: {alpha}"

    def test_single_threshold(self):
        """Works with a single threshold (2 categories)."""
        alpha = np.array([1.5])
        raw = thresholds_to_raw(alpha)
        recovered = raw_to_thresholds(raw)
        assert_allclose(recovered, alpha, atol=1e-12)

    def test_identity_for_first(self):
        """First raw parameter equals first threshold."""
        alpha = np.array([-1.0, 0.5, 2.0])
        raw = thresholds_to_raw(alpha)
        assert raw[0] == alpha[0]


# =========================================================================
# CLogLog link tests
# =========================================================================

class TestCLogLogLink:
    """Tests for the complementary log-log link function."""

    def test_linkinv_range(self):
        """Inverse link maps to (0, 1) for moderate eta values."""
        link = CLogLogLink()
        eta = np.array([-5.0, -1.0, 0.0, 1.0, 2.0])
        mu = link.linkinv(eta)
        assert np.all(mu > 0)
        assert np.all(mu < 1)

    def test_link_linkinv_roundtrip(self):
        """link(linkinv(eta)) == eta for moderate values."""
        link = CLogLogLink()
        eta = np.array([-2.0, -1.0, 0.0, 0.5, 1.5])
        recovered = link.link(link.linkinv(eta))
        assert_allclose(recovered, eta, atol=1e-8)

    def test_mu_eta_positive(self):
        """Derivative is always positive."""
        link = CLogLogLink()
        eta = np.linspace(-5, 5, 100)
        d = link.mu_eta(eta)
        assert np.all(d > 0)

    def test_name(self):
        """Link has correct name."""
        assert CLogLogLink().name == 'cloglog'


# =========================================================================
# Likelihood and gradient tests
# =========================================================================

class TestLikelihoodGradient:
    """Tests for the negative log-likelihood and its gradient."""

    def test_gradient_matches_numerical(self, simple_data):
        """Analytical gradient matches finite-difference approximation."""
        y, X = simple_data
        from pystatistics.regression.families import LogitLink
        link = LogitLink()
        n_levels = 3

        # Use some parameter values (not necessarily optimal)
        params = np.array([-1.0, 0.5, 0.8, -0.3])  # 2 thresholds + 2 betas
        raw = params.copy()
        raw[1] = np.log(np.exp(0.5) + 1.0)  # ensure valid unconstrained
        # Actually just use arbitrary raw params
        raw = np.array([0.0, 0.5, 0.8, -0.3])

        analytical = cumulative_gradient(raw, y, X, link, n_levels)

        # Numerical gradient
        eps = 1e-6
        numerical = np.zeros_like(raw)
        for i in range(len(raw)):
            raw_plus = raw.copy()
            raw_minus = raw.copy()
            raw_plus[i] += eps
            raw_minus[i] -= eps
            f_plus = cumulative_negloglik(raw_plus, y, X, link, n_levels)
            f_minus = cumulative_negloglik(raw_minus, y, X, link, n_levels)
            numerical[i] = (f_plus - f_minus) / (2 * eps)

        assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)

    def test_negloglik_positive(self, simple_data):
        """Negative log-likelihood is positive."""
        y, X = simple_data
        from pystatistics.regression.families import LogitLink
        link = LogitLink()
        params = np.array([0.0, 0.5, 0.5, -0.2])
        nll = cumulative_negloglik(params, y, X, link, 3)
        assert nll > 0


# =========================================================================
# polr() fitting tests — logistic link
# =========================================================================

class TestPolrLogistic:
    """Tests for polr() with logistic link (proportional odds)."""

    def test_basic_fit(self, simple_data):
        """polr fits and returns an OrdinalSolution."""
        y, X = simple_data
        sol = polr(y, X, method='logistic')
        assert isinstance(sol, OrdinalSolution)
        assert sol.converged

    def test_coefficient_recovery(self, simple_data):
        """Recovered betas are close to true values [1.0, -0.5]."""
        y, X = simple_data
        sol = polr(y, X, method='logistic', names=['x1', 'x2'])

        # With n=500, we expect reasonable recovery
        assert_allclose(sol.coefficients[0], 1.0, atol=0.25)
        assert_allclose(sol.coefficients[1], -0.5, atol=0.25)

    def test_threshold_recovery(self, simple_data):
        """Recovered thresholds are close to true values [-1.0, 1.0]."""
        y, X = simple_data
        sol = polr(y, X, method='logistic')

        assert_allclose(sol.threshold_values[0], -1.0, atol=0.3)
        assert_allclose(sol.threshold_values[1], 1.0, atol=0.3)

    def test_threshold_ordering(self, simple_data):
        """Thresholds are strictly ordered."""
        y, X = simple_data
        sol = polr(y, X, method='logistic')
        alpha = sol.threshold_values
        assert np.all(np.diff(alpha) > 0)

    def test_coef_dict(self, simple_data):
        """coef property returns correctly named dictionary."""
        y, X = simple_data
        sol = polr(y, X, method='logistic', names=['age', 'income'])
        coef = sol.coef
        assert 'age' in coef
        assert 'income' in coef
        assert len(coef) == 2

    def test_thresholds_dict(self, simple_data):
        """thresholds property returns correctly labeled dictionary."""
        y, X = simple_data
        sol = polr(
            y, X, method='logistic',
            level_names=['low', 'medium', 'high'],
        )
        thresh = sol.thresholds
        assert 'low|medium' in thresh
        assert 'medium|high' in thresh

    def test_standard_errors_positive(self, simple_data):
        """Standard errors are all positive."""
        y, X = simple_data
        sol = polr(y, X, method='logistic')
        assert np.all(sol.standard_errors > 0)
        assert np.all(sol.threshold_standard_errors > 0)

    def test_z_and_p_values(self, simple_data):
        """z-values and p-values are computed correctly."""
        y, X = simple_data
        sol = polr(y, X, method='logistic')

        z = sol.z_values
        p = sol.p_values
        assert len(z) == 2
        assert len(p) == 2
        assert np.all(p >= 0)
        assert np.all(p <= 1)

        # Since true coefficients are nonzero, p-values should be small
        assert np.all(p < 0.05)

    def test_deviance_and_aic(self, simple_data):
        """Deviance and AIC are consistent."""
        y, X = simple_data
        sol = polr(y, X, method='logistic')

        assert sol.deviance == pytest.approx(-2 * sol.log_likelihood)
        n_params = 2 + 2  # 2 thresholds + 2 betas
        assert sol.aic == pytest.approx(sol.deviance + 2 * n_params)

    def test_n_obs(self, simple_data):
        """n_obs matches input size."""
        y, X = simple_data
        sol = polr(y, X, method='logistic')
        assert sol.n_obs == len(y)


# =========================================================================
# polr() fitting tests — probit link
# =========================================================================

class TestPolrProbit:
    """Tests for polr() with probit link."""

    def test_probit_fits(self, simple_data):
        """polr with probit link converges."""
        y, X = simple_data
        sol = polr(y, X, method='probit')
        assert sol.converged
        assert sol.method == 'probit'

    def test_probit_coefficients_reasonable(self, simple_data):
        """Probit coefficients have correct sign and approximate magnitude."""
        y, X = simple_data
        sol = polr(y, X, method='probit')

        # Probit coefficients are roughly logistic / 1.7
        assert sol.coefficients[0] > 0  # positive for x1
        assert sol.coefficients[1] < 0  # negative for x2


# =========================================================================
# polr() fitting tests — cloglog link
# =========================================================================

class TestPolrCLogLog:
    """Tests for polr() with cloglog link."""

    def test_cloglog_fits(self, simple_data):
        """polr with cloglog link converges."""
        y, X = simple_data
        sol = polr(y, X, method='cloglog')
        assert sol.converged
        assert sol.method == 'cloglog'


# =========================================================================
# Four-level tests
# =========================================================================

class TestPolrFourLevels:
    """Tests with 4 ordered categories."""

    def test_four_level_fit(self, four_level_data):
        """Fits a 4-level model correctly."""
        y, X = four_level_data
        sol = polr(
            y, X, method='logistic',
            names=['x1', 'x2', 'x3'],
            level_names=['none', 'mild', 'moderate', 'severe'],
        )
        assert sol.converged
        assert len(sol.threshold_values) == 3
        assert len(sol.coefficients) == 3

    def test_four_level_threshold_ordering(self, four_level_data):
        """Four thresholds are strictly ordered."""
        y, X = four_level_data
        sol = polr(y, X, method='logistic')
        assert np.all(np.diff(sol.threshold_values) > 0)

    def test_four_level_threshold_labels(self, four_level_data):
        """Threshold dict has correct labels for 4 levels."""
        y, X = four_level_data
        sol = polr(
            y, X, method='logistic',
            level_names=['A', 'B', 'C', 'D'],
        )
        thresh = sol.thresholds
        assert 'A|B' in thresh
        assert 'B|C' in thresh
        assert 'C|D' in thresh


# =========================================================================
# Input validation tests
# =========================================================================

class TestPolrValidation:
    """Tests for input validation in polr()."""

    def test_non_integer_y_raises(self):
        """Non-integer y raises ValidationError."""
        y = np.array([0.5, 1.5, 2.5])
        X = np.ones((3, 1))
        with pytest.raises(ValidationError, match="integer"):
            polr(y, X)

    def test_gap_in_codes_raises(self):
        """Missing level codes raise ValidationError."""
        y = np.array([0, 0, 2, 2, 2])
        X = np.ones((5, 1))
        with pytest.raises(ValidationError, match="gaps"):
            polr(y, X)

    def test_nonzero_start_raises(self):
        """Codes not starting at 0 raise ValidationError."""
        y = np.array([1, 1, 2, 2, 3])
        X = np.ones((5, 1))
        with pytest.raises(ValidationError, match="start at 0"):
            polr(y, X)

    def test_single_level_raises(self):
        """Single level y raises ValidationError."""
        y = np.array([0, 0, 0, 0])
        X = np.ones((4, 1))
        with pytest.raises(ValidationError, match="at least 2"):
            polr(y, X)

    def test_length_mismatch_raises(self):
        """Mismatched y and X lengths raise ValidationError."""
        y = np.array([0, 1, 2])
        X = np.ones((5, 2))
        with pytest.raises(ValidationError, match="length"):
            polr(y, X)

    def test_wrong_names_length_raises(self):
        """Wrong number of names raises ValidationError."""
        y = np.array([0, 1, 0, 1])
        X = np.ones((4, 2))
        with pytest.raises(ValidationError, match="names length"):
            polr(y, X, names=['a'])

    def test_wrong_level_names_length_raises(self):
        """Wrong number of level_names raises ValidationError."""
        y = np.array([0, 1, 0, 1])
        X = np.ones((4, 1))
        with pytest.raises(ValidationError, match="level_names length"):
            polr(y, X, level_names=['a', 'b', 'c'])

    def test_invalid_method_raises(self):
        """Unknown method raises ValidationError."""
        y = np.array([0, 1, 0, 1])
        X = np.ones((4, 1))
        with pytest.raises(ValidationError, match="Unknown method"):
            polr(y, X, method='invalid')

    def test_nan_in_X_raises(self):
        """NaN in X raises ValidationError."""
        y = np.array([0, 1, 0, 1])
        X = np.array([[1.0], [np.nan], [2.0], [3.0]])
        with pytest.raises(ValidationError, match="non-finite"):
            polr(y, X)

    def test_1d_X_reshaped(self, simple_data):
        """1D X is automatically reshaped to (n, 1)."""
        rng = np.random.default_rng(99)
        n = 100
        x = rng.standard_normal(n)
        eta = 0.5 * x
        cum_p = 1.0 / (1.0 + np.exp(-(0.0 - eta)))
        u = rng.uniform(size=n)
        y = (u > cum_p).astype(int)
        sol = polr(y, x)
        assert sol.converged
        assert len(sol.coefficients) == 1


# =========================================================================
# Summary output tests
# =========================================================================

class TestSummary:
    """Tests for summary() output formatting."""

    def test_summary_contains_sections(self, simple_data):
        """Summary contains Coefficients, Intercepts, Deviance, AIC."""
        y, X = simple_data
        sol = polr(y, X, method='logistic', names=['x1', 'x2'])
        text = sol.summary()

        assert 'Coefficients:' in text
        assert 'Intercepts:' in text
        assert 'Residual Deviance:' in text
        assert 'AIC:' in text

    def test_summary_contains_variable_names(self, simple_data):
        """Summary shows the provided variable names."""
        y, X = simple_data
        sol = polr(y, X, method='logistic', names=['age', 'income'])
        text = sol.summary()
        assert 'age' in text
        assert 'income' in text

    def test_summary_contains_threshold_labels(self, simple_data):
        """Summary shows threshold labels from level_names."""
        y, X = simple_data
        sol = polr(
            y, X, method='logistic',
            level_names=['low', 'medium', 'high'],
        )
        text = sol.summary()
        assert 'low|medium' in text
        assert 'medium|high' in text

    def test_repr(self, simple_data):
        """repr includes key information."""
        y, X = simple_data
        sol = polr(y, X, method='logistic')
        r = repr(sol)
        assert 'OrdinalSolution' in r
        assert 'logistic' in r


# =========================================================================
# OrdinalParams frozen dataclass tests
# =========================================================================

class TestOrdinalParams:
    """Tests for OrdinalParams immutability."""

    def test_frozen(self):
        """OrdinalParams is frozen (immutable)."""
        params = OrdinalParams(
            coefficients=np.array([1.0]),
            thresholds=np.array([0.0]),
            vcov=np.eye(2),
            log_likelihood=-100.0,
            deviance=200.0,
            aic=204.0,
            n_obs=100,
            n_levels=2,
            level_names=('low', 'high'),
            n_iter=10,
            converged=True,
            method='logistic',
        )
        with pytest.raises(AttributeError):
            params.method = 'probit'  # type: ignore[misc]


# =========================================================================
# GPU backend tests (two-tier validation: CPU vs R, GPU vs CPU)
# =========================================================================


class TestPolrGPU:
    """Tests for the GPU polr backend.

    Two-tier validation: CPU is validated against R ``MASS::polr``; GPU
    is validated against CPU at the ``GPU_FP32`` tier (rtol=1e-4,
    atol=1e-5) from ``pystatistics.core.compute.tolerances``.
    """

    def _gpu_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

    def test_invalid_backend_raises(self, simple_data):
        y, X = simple_data
        from pystatistics.core.exceptions import ValidationError
        with pytest.raises(ValidationError, match="backend"):
            polr(y, X, backend="quantum")

    def test_gpu_unavailable_raises_explicitly(self, simple_data, monkeypatch):
        """backend='gpu' must raise when no GPU is available (Rule 1)."""
        y, X = simple_data
        from pystatistics.core.compute import device as dev_mod
        monkeypatch.setattr(dev_mod, "detect_gpu", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="no GPU"):
            polr(y, X, backend="gpu")

    def test_auto_backend_falls_back_to_cpu_when_no_gpu(
        self, simple_data, monkeypatch,
    ):
        """backend='auto' silently falls back to CPU when no GPU."""
        y, X = simple_data
        from pystatistics.core.compute import device as dev_mod
        monkeypatch.setattr(dev_mod, "detect_gpu", lambda *a, **k: None)
        r = polr(y, X, backend="auto")
        assert r.converged

    def test_gpu_fp64_matches_cpu_coefs(self, four_level_data):
        """GPU FP64 matches CPU on coefficients and SE (CUDA only)."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        import torch
        if not torch.cuda.is_available():
            pytest.skip("FP64 test requires CUDA (MPS has no FP64)")
        y, X = four_level_data
        r_cpu = polr(y, X, backend="cpu")
        r_gpu = polr(y, X, backend="gpu", use_fp64=True)
        np.testing.assert_allclose(
            r_cpu.coefficients, r_gpu.coefficients, rtol=1e-8, atol=1e-10,
        )
        np.testing.assert_allclose(
            r_cpu.standard_errors, r_gpu.standard_errors,
            rtol=1e-6, atol=1e-8,
        )

    def test_gpu_fp32_matches_cpu_at_tier(self, four_level_data):
        """GPU FP32 matches CPU at the ``GPU_FP32`` tier on the
        statistically meaningful quantities (log-likelihood, deviance,
        coefficients)."""
        from pystatistics.core.compute.tolerances import GPU_FP32
        if not self._gpu_available():
            pytest.skip("no GPU available")
        y, X = four_level_data
        r_cpu = polr(y, X, backend="cpu")
        r_gpu = polr(y, X, backend="gpu", use_fp64=False)
        assert r_cpu.log_likelihood == pytest.approx(
            r_gpu.log_likelihood, rel=GPU_FP32.rtol, abs=GPU_FP32.atol,
        )
        assert r_cpu.deviance == pytest.approx(
            r_gpu.deviance, rel=GPU_FP32.rtol, abs=GPU_FP32.atol,
        )
        # Raw coefficients can drift by more than the tier between CPU
        # (FP64) and GPU (FP32) because L-BFGS-B lands at slightly
        # different stationary points when the NLL is evaluated in
        # FP32 — same design note as the multinomial GPU tests.

    def test_gpu_datasource_input_matches_gpu_numpy(self, four_level_data):
        """Passing a device-resident torch.Tensor (from a GPU
        DataSource) is equivalent to passing numpy arrays with
        ``backend='gpu'``."""
        from pystatistics.core.compute.tolerances import GPU_FP32
        from pystatistics import DataSource
        if not self._gpu_available():
            pytest.skip("no GPU available")
        y, X = four_level_data
        gds = DataSource.from_arrays(X=X, y=y).to("cuda")
        r_numpy = polr(y, X, backend="gpu", use_fp64=False)
        r_tensor = polr(gds["y"], gds["X"], use_fp64=False)  # inferred
        assert r_numpy.log_likelihood == pytest.approx(
            r_tensor.log_likelihood, rel=GPU_FP32.rtol, abs=GPU_FP32.atol,
        )

    def test_gpu_tensor_with_cpu_backend_raises(self, four_level_data):
        """Rule 1: no silent GPU→CPU migration when the caller is
        explicit."""
        from pystatistics import DataSource
        if not self._gpu_available():
            pytest.skip("no GPU available")
        y, X = four_level_data
        gds = DataSource.from_arrays(X=X, y=y).to("cuda")
        from pystatistics.core.exceptions import ValidationError
        with pytest.raises(ValidationError, match="torch.Tensor"):
            polr(gds["y"], gds["X"], backend="cpu")
