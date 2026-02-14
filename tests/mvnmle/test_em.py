"""
EM algorithm tests for MVN MLE.

Validates the EM implementation against:
1. R's norm::em.norm() reference values (fixtures)
2. The existing direct BFGS MLE (cross-validation)
3. Theoretical EM properties (monotone likelihood, convergence)
"""

import json
import numpy as np
import pytest
from pathlib import Path

from pystatistics.mvnmle import mlest, datasets, MVNDesign, MVNSolution

REFERENCES = Path(__file__).parent / "references"


@pytest.fixture
def apple_em_ref():
    with open(REFERENCES / "apple_em_reference.json") as f:
        return json.load(f)


@pytest.fixture
def missvals_em_ref():
    with open(REFERENCES / "missvals_em_reference.json") as f:
        return json.load(f)


# =====================================================================
# R-compatibility: EM vs R's norm::em.norm()
# =====================================================================

class TestEMAppleDataset:
    """Validate EM on apple dataset against R norm::em.norm() reference."""

    def test_convergence(self):
        result = mlest(datasets.apple, algorithm='em')
        assert result.converged

    def test_mean_estimates_match_r(self, apple_em_ref):
        result = mlest(datasets.apple, algorithm='em', tol=1e-8, max_iter=10000)
        expected_mu = np.array(apple_em_ref['muhat'])
        np.testing.assert_allclose(result.muhat, expected_mu, rtol=1e-4,
                                   err_msg="EM means differ from R norm::em.norm()")

    def test_covariance_estimates_match_r(self, apple_em_ref):
        result = mlest(datasets.apple, algorithm='em', tol=1e-8, max_iter=10000)
        expected_sigma = np.array(apple_em_ref['sigmahat'])
        np.testing.assert_allclose(result.sigmahat, expected_sigma, rtol=1e-4,
                                   err_msg="EM covariance differs from R norm::em.norm()")

    def test_covariance_symmetric(self):
        result = mlest(datasets.apple, algorithm='em')
        np.testing.assert_allclose(
            result.sigmahat, result.sigmahat.T,
            atol=1e-14,
            err_msg="EM covariance not symmetric"
        )

    def test_covariance_positive_definite(self):
        result = mlest(datasets.apple, algorithm='em')
        eigenvals = np.linalg.eigvalsh(result.sigmahat)
        assert np.all(eigenvals > 0), f"Not PD: min eigenvalue = {eigenvals.min()}"


class TestEMMissvalsDataset:
    """Validate EM on missvals dataset against R norm::em.norm() reference."""

    def test_mean_estimates_match_r(self, missvals_em_ref):
        result = mlest(datasets.missvals, algorithm='em', tol=1e-8, max_iter=100000)
        expected_mu = np.array(missvals_em_ref['muhat'])
        np.testing.assert_allclose(result.muhat, expected_mu, rtol=1e-3,
                                   err_msg="EM means differ from R norm::em.norm()")

    def test_covariance_estimates_match_r(self, missvals_em_ref):
        result = mlest(datasets.missvals, algorithm='em', tol=1e-8, max_iter=100000)
        expected_sigma = np.array(missvals_em_ref['sigmahat'])
        np.testing.assert_allclose(result.sigmahat, expected_sigma, rtol=1e-3,
                                   err_msg="EM covariance differs from R norm::em.norm()")

    def test_covariance_symmetric(self):
        result = mlest(datasets.missvals, algorithm='em')
        np.testing.assert_allclose(
            result.sigmahat, result.sigmahat.T,
            atol=1e-14,
            err_msg="EM covariance not symmetric"
        )

    def test_covariance_positive_definite(self):
        result = mlest(datasets.missvals, algorithm='em')
        eigenvals = np.linalg.eigvalsh(result.sigmahat)
        assert np.all(eigenvals > 0), f"Not PD: min eigenvalue = {eigenvals.min()}"

    def test_dimensions(self):
        result = mlest(datasets.missvals, algorithm='em')
        assert result.muhat.shape == (5,)
        assert result.sigmahat.shape == (5, 5)


# =====================================================================
# Cross-validation: EM vs direct BFGS
# =====================================================================

class TestEMMatchesDirect:
    """EM and direct BFGS should converge to the same MLE."""

    def test_apple_loglik_agrees(self):
        em = mlest(datasets.apple, algorithm='em', tol=1e-8, max_iter=10000)
        direct = mlest(datasets.apple, algorithm='direct')
        assert abs(em.loglik - direct.loglik) < 1e-6, (
            f"EM loglik {em.loglik} differs from direct {direct.loglik} "
            f"by {abs(em.loglik - direct.loglik)}"
        )

    def test_apple_means_agree(self):
        em = mlest(datasets.apple, algorithm='em', tol=1e-8, max_iter=10000)
        direct = mlest(datasets.apple, algorithm='direct')
        np.testing.assert_allclose(em.muhat, direct.muhat, rtol=1e-4)

    def test_apple_covariance_agrees(self):
        em = mlest(datasets.apple, algorithm='em', tol=1e-8, max_iter=10000)
        direct = mlest(datasets.apple, algorithm='direct')
        np.testing.assert_allclose(em.sigmahat, direct.sigmahat, rtol=1e-4)

    def test_missvals_loglik_agrees(self):
        em = mlest(datasets.missvals, algorithm='em', tol=1e-8, max_iter=100000)
        direct = mlest(datasets.missvals, algorithm='direct', max_iter=500)
        assert abs(em.loglik - direct.loglik) < 1e-4, (
            f"EM loglik {em.loglik} differs from direct {direct.loglik} "
            f"by {abs(em.loglik - direct.loglik)}"
        )

    def test_missvals_means_agree(self):
        em = mlest(datasets.missvals, algorithm='em', tol=1e-8, max_iter=100000)
        direct = mlest(datasets.missvals, algorithm='direct', max_iter=500)
        np.testing.assert_allclose(em.muhat, direct.muhat, rtol=5e-3)

    def test_missvals_covariance_agrees(self):
        em = mlest(datasets.missvals, algorithm='em', tol=1e-8, max_iter=100000)
        direct = mlest(datasets.missvals, algorithm='direct', max_iter=500)
        np.testing.assert_allclose(em.sigmahat, direct.sigmahat, rtol=5e-3)


# =====================================================================
# Edge cases
# =====================================================================

class TestEMEdgeCases:
    """Edge cases specific to EM."""

    def test_complete_data(self):
        """Complete data should converge very quickly."""
        rng = np.random.default_rng(42)
        data = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=50)
        result = mlest(data, algorithm='em')
        assert result.converged
        # With no missing data, sample statistics are exact — EM should
        # converge in very few iterations
        assert result.n_iter <= 5

    def test_high_missing_rate(self):
        """Should handle high missingness."""
        rng = np.random.default_rng(42)
        data = rng.multivariate_normal([5, 10], [[2, 1], [1, 3]], size=100)
        mask = rng.random(data.shape) < 0.4
        for i in range(data.shape[0]):
            if mask[i].all():
                mask[i, 0] = False
        for j in range(data.shape[1]):
            if mask[:, j].all():
                mask[0, j] = False
        data[mask] = np.nan
        result = mlest(data, algorithm='em', max_iter=5000)
        assert np.all(np.isfinite(result.muhat))
        assert np.all(np.isfinite(result.sigmahat))

    def test_monotone_pattern(self):
        """Should handle monotone missingness."""
        rng = np.random.default_rng(42)
        data = rng.multivariate_normal([0, 0, 0], np.eye(3), size=30)
        data[15:, 2] = np.nan
        data[20:, 1] = np.nan
        result = mlest(data, algorithm='em')
        assert np.all(np.isfinite(result.muhat))

    def test_two_variables(self):
        """Minimum viable: 2 variables."""
        rng = np.random.default_rng(42)
        data = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=20)
        data[15:, 1] = np.nan
        result = mlest(data, algorithm='em')
        assert result.muhat.shape == (2,)

    def test_reproducibility(self):
        """EM is deterministic — same input should give identical output."""
        r1 = mlest(datasets.apple, algorithm='em')
        r2 = mlest(datasets.apple, algorithm='em')
        np.testing.assert_allclose(r1.muhat, r2.muhat, atol=1e-14)
        np.testing.assert_allclose(r1.sigmahat, r2.sigmahat, atol=1e-14)
        assert abs(r1.loglik - r2.loglik) < 1e-14


# =====================================================================
# Solution interface
# =====================================================================

class TestEMSolutionInterface:
    """Test that EM results work with the full MVNSolution interface."""

    def test_aic_bic(self):
        result = mlest(datasets.apple, algorithm='em')
        p = 2
        k = p + p * (p + 1) // 2
        expected_aic = -2 * result.loglik + 2 * k
        assert abs(result.aic - expected_aic) < 1e-10

    def test_summary_string(self):
        result = mlest(datasets.apple, algorithm='em')
        summary = result.summary()
        assert "MVN MLE Results" in summary
        assert "Converged: True" in summary
        assert "Log-likelihood" in summary

    def test_to_dict(self):
        result = mlest(datasets.apple, algorithm='em')
        d = result.to_dict()
        assert 'muhat' in d
        assert 'sigmahat' in d
        assert 'loglik' in d
        assert 'converged' in d

    def test_timing(self):
        result = mlest(datasets.apple, algorithm='em')
        assert result.timing is not None
        assert 'total_seconds' in result.timing
        assert result.timing['total_seconds'] > 0

    def test_backend_name(self):
        result = mlest(datasets.apple, algorithm='em')
        assert result.backend_name == 'cpu_em'

    def test_gradient_norm_is_none(self):
        result = mlest(datasets.apple, algorithm='em')
        assert result.gradient_norm is None


# =====================================================================
# Algorithm parameter dispatch
# =====================================================================

class TestAlgorithmParameter:
    """Test the algorithm parameter dispatch in mlest()."""

    def test_default_is_direct(self):
        result = mlest(datasets.apple)
        assert 'bfgs' in result.backend_name.lower() or 'cpu' in result.backend_name.lower()
        assert result.backend_name != 'cpu_em'

    def test_explicit_direct(self):
        result = mlest(datasets.apple, algorithm='direct')
        assert result.converged
        assert result.backend_name != 'cpu_em'

    def test_explicit_em(self):
        result = mlest(datasets.apple, algorithm='em')
        assert result.backend_name == 'cpu_em'

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            mlest(datasets.apple, algorithm='bogus')

    def test_em_default_tolerance(self):
        # EM with default tol (1e-4) should converge on apple
        result = mlest(datasets.apple, algorithm='em')
        assert result.converged

    def test_custom_tol_and_max_iter(self):
        result = mlest(datasets.apple, algorithm='em', tol=1e-8, max_iter=5000)
        assert result.converged

    def test_em_with_design(self):
        """EM should accept MVNDesign objects."""
        design = MVNDesign.from_array(datasets.apple)
        result = mlest(design, algorithm='em')
        assert isinstance(result, MVNSolution)
        assert result.converged
