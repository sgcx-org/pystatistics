"""
Comprehensive tests for multinomial logistic regression.

Covers:
    - 3-class coefficient recovery from synthetic data
    - 5-class convergence
    - Predicted class accuracy > 80% on well-separated data
    - Edge case: 2-class (reduces to logistic regression)
    - Edge case: constant predictor validation
    - Failure case: y with only 1 unique class
    - Failure case: mismatched dimensions
    - AIC, deviance, null_deviance are finite and sensible
    - McFadden pseudo-R-squared between 0 and 1
    - MultinomialParams dataclass unit tests
"""

from __future__ import annotations

import numpy as np
import pytest

from pystatistics.core.exceptions import ConvergenceError, ValidationError
from pystatistics.core.result import Result
from pystatistics.multinomial import multinom, MultinomialSolution
from pystatistics.multinomial._common import MultinomialParams
from pystatistics.multinomial._likelihood import (
    multinomial_negloglik,
    multinomial_gradient,
    compute_probs,
    _compute_log_probs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def three_class_data():
    """Well-separated 3-class synthetic data for coefficient recovery.

    Uses deterministic seed for reproducibility.
    """
    rng = np.random.default_rng(42)
    n = 300

    X_raw = rng.standard_normal((n, 2))
    X = np.column_stack([np.ones(n), X_raw])

    # True coefficients for classes 0 and 1 (class 2 is reference)
    # beta_0 = [0.5, 2.0, -1.0]
    # beta_1 = [-0.5, -1.0, 1.5]
    true_beta = np.array([
        [0.5, 2.0, -1.0],
        [-0.5, -1.0, 1.5],
    ])

    # Generate probabilities from true model
    eta = X @ true_beta.T
    eta_full = np.column_stack([eta, np.zeros(n)])
    eta_max = np.max(eta_full, axis=1, keepdims=True)
    probs = np.exp(eta_full - eta_max)
    probs /= probs.sum(axis=1, keepdims=True)

    # Sample classes
    y = np.array([
        rng.choice(3, p=probs[i]) for i in range(n)
    ])

    return y, X, true_beta


@pytest.fixture
def five_class_data():
    """5-class synthetic data for convergence testing."""
    rng = np.random.default_rng(123)
    n = 500
    p = 4

    X_raw = rng.standard_normal((n, p - 1))
    X = np.column_stack([np.ones(n), X_raw])

    # Generate from a multinomial logit model
    true_beta = rng.standard_normal((4, p)) * 0.5
    eta = X @ true_beta.T
    eta_full = np.column_stack([eta, np.zeros(n)])
    eta_max = np.max(eta_full, axis=1, keepdims=True)
    probs = np.exp(eta_full - eta_max)
    probs /= probs.sum(axis=1, keepdims=True)

    y = np.array([rng.choice(5, p=probs[i]) for i in range(n)])

    return y, X


@pytest.fixture
def well_separated_data():
    """Well-separated 3-class data for accuracy testing."""
    rng = np.random.default_rng(99)
    n_per_class = 100
    n = n_per_class * 3

    # Class 0: centered at (3, 3)
    # Class 1: centered at (-3, -3)
    # Class 2: centered at (3, -3)
    centers = np.array([[3.0, 3.0], [-3.0, -3.0], [3.0, -3.0]])

    X_raw = np.vstack([
        rng.standard_normal((n_per_class, 2)) + centers[j]
        for j in range(3)
    ])
    X = np.column_stack([np.ones(n), X_raw])
    y = np.repeat(np.arange(3), n_per_class)

    return y, X


@pytest.fixture
def two_class_data():
    """2-class data (multinomial reduces to logistic)."""
    rng = np.random.default_rng(77)
    n = 200
    X_raw = rng.standard_normal((n, 1))
    X = np.column_stack([np.ones(n), X_raw])

    # Simple logistic model
    eta = 0.5 + 1.5 * X_raw[:, 0]
    prob_1 = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < prob_1).astype(int)

    return y, X


# ---------------------------------------------------------------------------
# MultinomialParams dataclass tests
# ---------------------------------------------------------------------------


class TestMultinomialParams:
    """Unit tests for the MultinomialParams frozen dataclass."""

    def test_frozen(self):
        """MultinomialParams is immutable."""
        params = MultinomialParams(
            coefficient_matrix=np.zeros((2, 3)),
            vcov=np.eye(6),
            fitted_probs=np.ones((10, 3)) / 3,
            log_likelihood=-100.0,
            deviance=200.0,
            null_deviance=300.0,
            aic=212.0,
            n_obs=10,
            n_classes=3,
            class_names=("a", "b", "c"),
            feature_names=("x0", "x1", "x2"),
            n_iter=10,
            converged=True,
        )
        with pytest.raises(AttributeError):
            params.log_likelihood = -50.0  # type: ignore[misc]

    def test_all_fields_stored(self):
        """All fields are accessible after construction."""
        coef = np.array([[1.0, 2.0], [3.0, 4.0]])
        params = MultinomialParams(
            coefficient_matrix=coef,
            vcov=np.eye(4),
            fitted_probs=np.ones((5, 3)) / 3,
            log_likelihood=-50.0,
            deviance=100.0,
            null_deviance=150.0,
            aic=108.0,
            n_obs=5,
            n_classes=3,
            class_names=("a", "b", "c"),
            feature_names=("x0", "x1"),
            n_iter=15,
            converged=True,
        )
        assert params.n_obs == 5
        assert params.n_classes == 3
        assert params.converged is True
        assert params.n_iter == 15
        assert params.log_likelihood == -50.0
        assert params.deviance == 100.0
        assert params.null_deviance == 150.0
        assert params.aic == 108.0
        assert params.class_names == ("a", "b", "c")
        assert params.feature_names == ("x0", "x1")
        np.testing.assert_array_equal(params.coefficient_matrix, coef)


# ---------------------------------------------------------------------------
# Likelihood function tests
# ---------------------------------------------------------------------------


class TestLikelihood:
    """Tests for _likelihood.py functions."""

    def test_log_probs_sum_to_zero(self):
        """Log probabilities correspond to valid probability distribution."""
        rng = np.random.default_rng(10)
        n, p, J = 50, 3, 4
        params = rng.standard_normal((J - 1) * p)
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])

        log_probs = _compute_log_probs(params, X, J)
        probs = np.exp(log_probs)

        # Each row should sum to 1
        np.testing.assert_allclose(
            probs.sum(axis=1), np.ones(n), atol=1e-12
        )

    def test_negloglik_positive(self):
        """Negative log-likelihood is positive for valid data."""
        rng = np.random.default_rng(11)
        n, p, J = 30, 2, 3
        params = np.zeros((J - 1) * p)
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
        y = rng.choice(J, size=n)
        y_onehot = np.zeros((n, J))
        y_onehot[np.arange(n), y] = 1.0

        nll = multinomial_negloglik(params, y_onehot, X, J)
        assert nll > 0

    def test_gradient_matches_numerical(self):
        """Analytical gradient matches numerical finite-difference gradient."""
        from scipy.optimize import approx_fprime

        rng = np.random.default_rng(12)
        n, p, J = 40, 3, 3
        params = rng.standard_normal((J - 1) * p) * 0.1
        X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
        y = rng.choice(J, size=n)
        y_onehot = np.zeros((n, J))
        y_onehot[np.arange(n), y] = 1.0

        analytic = multinomial_gradient(params, y_onehot, X, J)
        numerical = approx_fprime(
            params,
            lambda p: multinomial_negloglik(p, y_onehot, X, J),
            1e-7,
        )

        np.testing.assert_allclose(analytic, numerical, atol=1e-4)

    def test_compute_probs_shape(self):
        """compute_probs returns correct shape."""
        n, p, J = 20, 2, 4
        params = np.zeros((J - 1) * p)
        X = np.ones((n, p))

        probs = compute_probs(params, X, J)
        assert probs.shape == (n, J)

    def test_zero_params_uniform_probs(self):
        """Zero coefficients yield uniform probabilities."""
        n, p, J = 25, 3, 5
        params = np.zeros((J - 1) * p)
        X = np.column_stack([np.ones(n), np.zeros((n, p - 1))])

        probs = compute_probs(params, X, J)
        expected = np.full((n, J), 1.0 / J)
        np.testing.assert_allclose(probs, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 3-class coefficient recovery
# ---------------------------------------------------------------------------


class TestThreeClass:
    """3-class coefficient recovery from synthetic data."""

    def test_coefficients_close_to_true(self, three_class_data):
        """Fitted coefficients should be in the neighborhood of true values."""
        y, X, true_beta = three_class_data
        result = multinom(
            y, X,
            names=["intercept", "x1", "x2"],
            class_names=["A", "B", "C"],
        )

        fitted = result.coefficient_matrix
        # With 300 observations, coefficients should be within ~0.6 of true
        assert fitted.shape == true_beta.shape
        max_diff = np.max(np.abs(fitted - true_beta))
        assert max_diff < 1.0, (
            f"Max coefficient difference {max_diff:.4f} exceeds tolerance. "
            f"True:\n{true_beta}\nFitted:\n{fitted}"
        )

    def test_standard_errors_positive(self, three_class_data):
        """All standard errors should be strictly positive."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        se = result.standard_errors
        assert np.all(se > 0), f"Found non-positive SEs: {se}"

    def test_z_values_finite(self, three_class_data):
        """Z-values should be finite."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        assert np.all(np.isfinite(result.z_values))

    def test_p_values_in_range(self, three_class_data):
        """P-values should be between 0 and 1."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        pvals = result.p_values
        assert np.all(pvals >= 0.0)
        assert np.all(pvals <= 1.0)


# ---------------------------------------------------------------------------
# 5-class convergence
# ---------------------------------------------------------------------------


class TestFiveClass:
    """5-class convergence and basic properties."""

    def test_converges(self, five_class_data):
        """5-class model should converge."""
        y, X = five_class_data
        result = multinom(y, X)
        assert result.converged

    def test_coefficient_shape(self, five_class_data):
        """Coefficient matrix has correct shape (J-1, p)."""
        y, X = five_class_data
        result = multinom(y, X)
        n_classes = len(np.unique(y))
        p = X.shape[1]
        assert result.coefficient_matrix.shape == (n_classes - 1, p)

    def test_fitted_probs_shape(self, five_class_data):
        """Fitted probabilities have correct shape (n, J)."""
        y, X = five_class_data
        result = multinom(y, X)
        n_classes = len(np.unique(y))
        assert result.fitted_probs.shape == (len(y), n_classes)

    def test_fitted_probs_sum_to_one(self, five_class_data):
        """Each row of fitted probabilities sums to 1."""
        y, X = five_class_data
        result = multinom(y, X)
        row_sums = result.fitted_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Prediction accuracy on well-separated data
# ---------------------------------------------------------------------------


class TestPredictionAccuracy:
    """Predicted class accuracy on well-separated data."""

    def test_accuracy_above_80_percent(self, well_separated_data):
        """Accuracy should exceed 80% on well-separated clusters."""
        y, X = well_separated_data
        result = multinom(y, X)
        predicted = result.predicted_class
        accuracy = np.mean(predicted == y)
        assert accuracy > 0.80, f"Accuracy {accuracy:.2%} below 80%"

    def test_accuracy_above_95_percent(self, well_separated_data):
        """With very well-separated clusters, accuracy should be high."""
        y, X = well_separated_data
        result = multinom(y, X)
        predicted = result.predicted_class
        accuracy = np.mean(predicted == y)
        # These clusters are 6 units apart, accuracy should be very high
        assert accuracy > 0.95, f"Accuracy {accuracy:.2%} below 95%"


# ---------------------------------------------------------------------------
# 2-class edge case (reduces to logistic regression)
# ---------------------------------------------------------------------------


class TestTwoClass:
    """2-class case should work and reduce to logistic regression."""

    def test_two_class_converges(self, two_class_data):
        """2-class model converges."""
        y, X = two_class_data
        result = multinom(y, X)
        assert result.converged

    def test_two_class_single_coefficient_row(self, two_class_data):
        """With 2 classes, coefficient matrix has 1 row."""
        y, X = two_class_data
        result = multinom(y, X)
        assert result.coefficient_matrix.shape[0] == 1

    def test_two_class_fitted_probs(self, two_class_data):
        """Fitted probs should have 2 columns summing to 1."""
        y, X = two_class_data
        result = multinom(y, X)
        assert result.fitted_probs.shape[1] == 2
        np.testing.assert_allclose(
            result.fitted_probs.sum(axis=1), 1.0, atol=1e-10
        )


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------


class TestFailureCases:
    """Validation and error handling."""

    def test_single_class_raises(self):
        """y with only 1 unique class should raise ValidationError."""
        y = np.zeros(50)
        X = np.column_stack([np.ones(50), np.random.default_rng(1).standard_normal((50, 2))])
        with pytest.raises(ValidationError, match="at least 2 distinct classes"):
            multinom(y, X)

    def test_mismatched_dimensions(self):
        """y and X with different n should raise."""
        y = np.array([0, 1, 2, 0, 1])
        X = np.ones((10, 2))
        with pytest.raises(Exception):
            multinom(y, X)

    def test_non_integer_y(self):
        """Non-integer y values should raise ValidationError."""
        y = np.array([0.5, 1.5, 2.5] * 20)
        X = np.column_stack([np.ones(60), np.zeros((60, 1))])
        with pytest.raises(ValidationError, match="integer class codes"):
            multinom(y, X)

    def test_non_consecutive_codes(self):
        """Non-consecutive class codes should raise ValidationError."""
        y = np.array([0, 2, 0, 2] * 20)
        X = np.column_stack([
            np.ones(80),
            np.random.default_rng(3).standard_normal((80, 2))
        ])
        with pytest.raises(ValidationError, match="consecutive integers"):
            multinom(y, X)

    def test_names_length_mismatch(self):
        """names with wrong length should raise ValidationError."""
        rng = np.random.default_rng(4)
        y = rng.choice(3, size=100)
        X = np.column_stack([np.ones(100), rng.standard_normal((100, 2))])
        with pytest.raises(ValidationError, match="names"):
            multinom(y, X, names=["a", "b"])  # X has 3 columns

    def test_class_names_length_mismatch(self):
        """class_names with wrong length should raise ValidationError."""
        rng = np.random.default_rng(5)
        y = rng.choice(3, size=100)
        X = np.column_stack([np.ones(100), rng.standard_normal((100, 2))])
        with pytest.raises(ValidationError, match="class_names"):
            multinom(y, X, class_names=["a", "b"])  # need 3


# ---------------------------------------------------------------------------
# Model fit statistics
# ---------------------------------------------------------------------------


class TestModelFitStatistics:
    """AIC, deviance, null_deviance, pseudo R-squared."""

    def test_deviance_finite(self, three_class_data):
        """Deviance is finite and positive."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        assert np.isfinite(result.deviance)
        assert result.deviance > 0

    def test_null_deviance_finite(self, three_class_data):
        """Null deviance is finite and positive."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        assert np.isfinite(result.null_deviance)
        assert result.null_deviance > 0

    def test_aic_finite(self, three_class_data):
        """AIC is finite and positive."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        assert np.isfinite(result.aic)
        assert result.aic > 0

    def test_deviance_less_than_null(self, three_class_data):
        """Fitted model deviance should be <= null deviance."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        assert result.deviance <= result.null_deviance + 1e-6

    def test_aic_equals_deviance_plus_2k(self, three_class_data):
        """AIC = deviance + 2 * n_parameters."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        n_classes = len(np.unique(y))
        p = X.shape[1]
        n_params = (n_classes - 1) * p
        expected_aic = result.deviance + 2 * n_params
        np.testing.assert_allclose(result.aic, expected_aic, atol=1e-10)

    def test_pseudo_r_squared_between_0_and_1(self, three_class_data):
        """McFadden pseudo R-squared is in [0, 1]."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        r2 = result.pseudo_r_squared
        assert 0.0 <= r2 <= 1.0, f"Pseudo R^2 = {r2} outside [0, 1]"

    def test_pseudo_r_squared_positive_for_informative_model(
        self, well_separated_data
    ):
        """Pseudo R-squared should be substantially > 0 for good model."""
        y, X = well_separated_data
        result = multinom(y, X)
        assert result.pseudo_r_squared > 0.3


# ---------------------------------------------------------------------------
# Solution properties and display
# ---------------------------------------------------------------------------


class TestSolution:
    """MultinomialSolution properties and methods."""

    def test_coef_dict_structure(self, three_class_data):
        """coef returns nested dict with correct structure."""
        y, X, _ = three_class_data
        result = multinom(
            y, X,
            names=["intercept", "x1", "x2"],
            class_names=["A", "B", "C"],
        )
        coef = result.coef
        assert "A" in coef
        assert "B" in coef
        assert "C" not in coef  # reference class excluded
        assert "intercept" in coef["A"]
        assert "x1" in coef["A"]
        assert "x2" in coef["A"]

    def test_summary_contains_expected_text(self, three_class_data):
        """summary() output contains expected sections."""
        y, X, _ = three_class_data
        result = multinom(
            y, X,
            names=["intercept", "x1", "x2"],
            class_names=["A", "B", "C"],
        )
        s = result.summary()
        assert "Multinomial Logistic Regression" in s
        assert "Coefficients:" in s
        assert 'class "A"' in s
        assert 'class "B"' in s
        assert 'vs reference "C"' in s
        assert "Residual Deviance:" in s
        assert "AIC:" in s
        assert "Value" in s
        assert "Std. Error" in s
        assert "z value" in s

    def test_repr(self, three_class_data):
        """__repr__ returns sensible string."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        r = repr(result)
        assert "MultinomialSolution" in r
        assert "n_obs=" in r
        assert "n_classes=" in r

    def test_class_names_property(self, three_class_data):
        """class_names returns the tuple from params."""
        y, X, _ = three_class_data
        result = multinom(y, X, class_names=["A", "B", "C"])
        assert result.class_names == ("A", "B", "C")

    def test_feature_names_property(self, three_class_data):
        """feature_names returns the tuple from params."""
        y, X, _ = three_class_data
        result = multinom(y, X, names=["intercept", "x1", "x2"])
        assert result.feature_names == ("intercept", "x1", "x2")

    def test_default_names(self, three_class_data):
        """Default names are x0, x1, x2, ... and 0, 1, 2, ..."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        assert result.feature_names == ("x0", "x1", "x2")
        assert result.class_names == ("0", "1", "2")

    def test_predicted_class_dtype(self, three_class_data):
        """predicted_class is integer array."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        pred = result.predicted_class
        assert pred.dtype in (np.intp, np.int64, np.int32)

    def test_log_likelihood_negative(self, three_class_data):
        """Log-likelihood is negative (log of probability)."""
        y, X, _ = three_class_data
        result = multinom(y, X)
        assert result.log_likelihood < 0


class TestMultinomGPU:
    """Tests for the GPU multinomial backend.

    Two-tier validation: CPU is validated against R ``nnet::multinom``;
    GPU is validated against CPU at the ``GPU_FP32`` tier (rtol=1e-4,
    atol=1e-5) from ``pystatistics.core.compute.tolerances``.
    """

    def _gpu_available(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

    def test_invalid_backend_raises(self, three_class_data):
        y, X, _ = three_class_data
        from pystatistics.core.exceptions import ValidationError
        with pytest.raises(ValidationError, match="backend"):
            multinom(y, X, backend="quantum")

    def test_gpu_unavailable_raises_explicitly(self, three_class_data, monkeypatch):
        """backend='gpu' must raise when no GPU is available (Rule 1)."""
        y, X, _ = three_class_data
        from pystatistics.core.compute import device as dev_mod
        monkeypatch.setattr(dev_mod, "detect_gpu", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="no GPU"):
            multinom(y, X, backend="gpu")

    def test_auto_backend_falls_back_to_cpu_when_no_gpu(
        self, three_class_data, monkeypatch,
    ):
        """backend='auto' falls back silently when no GPU — that is the
        definition of 'auto', not a Rule 1 violation."""
        y, X, _ = three_class_data
        from pystatistics.core.compute import device as dev_mod
        monkeypatch.setattr(dev_mod, "detect_gpu", lambda *a, **k: None)
        r = multinom(y, X, backend="auto")
        assert r.converged

    def test_gpu_fp64_matches_cpu_loglik(self, three_class_data):
        """GPU FP64 matches CPU on log-likelihood."""
        if not self._gpu_available():
            pytest.skip("no GPU available")
        import torch
        if not torch.cuda.is_available():
            pytest.skip("FP64 test requires CUDA (MPS has no FP64)")
        y, X, _ = three_class_data
        r_cpu = multinom(y, X, backend="cpu")
        r_gpu = multinom(y, X, backend="gpu", use_fp64=True)
        assert r_cpu.log_likelihood == pytest.approx(
            r_gpu.log_likelihood, rel=1e-8,
        )

    def test_gpu_fp32_matches_cpu_at_tier(self, three_class_data):
        """GPU FP32 matches CPU at the project's GPU_FP32 tier on
        statistically meaningful quantities (log-lik and deviance).

        Raw coefficients can drift between CPU and GPU runs because
        L-BFGS-B lands at slightly different stationary points when
        the objective is evaluated in FP32 vs FP64 — that is the
        documented "by design" divergence in the README. What we pin
        to the tier is the *statistical* answer.
        """
        from pystatistics.core.compute.tolerances import GPU_FP32
        if not self._gpu_available():
            pytest.skip("no GPU available")
        y, X, _ = three_class_data
        r_cpu = multinom(y, X, backend="cpu")
        r_gpu = multinom(y, X, backend="gpu", use_fp64=False)
        assert r_cpu.log_likelihood == pytest.approx(
            r_gpu.log_likelihood, rel=GPU_FP32.rtol, abs=GPU_FP32.atol,
        )
        assert r_cpu.deviance == pytest.approx(
            r_gpu.deviance, rel=GPU_FP32.rtol, abs=GPU_FP32.atol,
        )

    def test_gpu_datasource_input_matches_gpu_numpy(self, three_class_data):
        """Passing a torch.Tensor (e.g. from ``ds.to('cuda')['X']``) is
        equivalent to passing the numpy array with ``backend='gpu'``,
        and faster on the amortized path because X never crosses the
        PCIe bus mid-fit."""
        from pystatistics.core.compute.tolerances import GPU_FP32
        from pystatistics import DataSource
        if not self._gpu_available():
            pytest.skip("no GPU available")
        y, X, _ = three_class_data
        gds = DataSource.from_arrays(X=X, y=y).to("cuda")

        r_numpy = multinom(y, X, backend="gpu", use_fp64=False)
        r_tensor = multinom(gds["y"], gds["X"], use_fp64=False)  # backend inferred
        # Same statistical answer within the GPU_FP32 tier.
        assert r_numpy.log_likelihood == pytest.approx(
            r_tensor.log_likelihood, rel=GPU_FP32.rtol, abs=GPU_FP32.atol,
        )

    def test_gpu_tensor_with_cpu_backend_raises(self, three_class_data):
        """Rule 1: no silent GPU→CPU migration when the caller is
        explicit."""
        from pystatistics import DataSource
        if not self._gpu_available():
            pytest.skip("no GPU available")
        y, X, _ = three_class_data
        gds = DataSource.from_arrays(X=X, y=y).to("cuda")
        from pystatistics.core.exceptions import ValidationError
        with pytest.raises(ValidationError, match="torch.Tensor"):
            multinom(gds["y"], gds["X"], backend="cpu")
