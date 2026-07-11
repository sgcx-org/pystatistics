"""Joint penalty-group determinant: singleton equivalence + FD gradient.

Verifies the ``_penalty_group`` math in isolation, before it is wired into
the REML criterion:

* a SINGLETON group reproduces the block-orthogonal shortcut exactly
  (``log|lam S|_+ = rank log lam + logdet_pos``, gradient ``= rank``);
* an OVERLAPPING tensor group's analytic ``d log|S_lambda|_+/d rho`` matches
  central finite differences of the log pseudo-determinant;
* the joint rank is the lambda-invariant ``k - null_a * null_b``.
"""

from __future__ import annotations

import numpy as np

from pystatistics.gam._penalty_group import penalty_logdet, penalty_logdet_grad
from pystatistics.gam._pirls import make_penalty_roots


def _diff_penalty(k: int) -> np.ndarray:
    """Second-difference penalty ``D'D`` (k x k), rank ``k-2`` (null {1, x})."""
    d = np.zeros((k - 2, k))
    for i in range(k - 2):
        d[i, i:i + 3] = (1.0, -2.0, 1.0)
    return d.T @ d


def _singleton_roots():
    # Two ordinary smooths, disjoint blocks, each its own group.
    s_a = _diff_penalty(6)          # rank 4
    s_b = _diff_penalty(5)          # rank 3
    blocks = [(0, 6), (6, 11)]
    return make_penalty_roots([s_a, s_b], blocks, groups=[0, 1])


def _tensor_roots():
    """One tensor smooth: two overlapping Kronecker-embedded margin penalties."""
    ka, kb = 5, 4
    s_a = _diff_penalty(ka)                     # null dim 2
    s_b = _diff_penalty(kb)                     # null dim 2
    s1 = np.kron(s_a, np.eye(kb))              # margin-1 penalty on the block
    s2 = np.kron(np.eye(ka), s_b)             # margin-2 penalty on the block
    k = ka * kb
    blocks = [(0, k), (0, k)]                   # SHARED block
    return make_penalty_roots([s1, s2], blocks, groups=[0, 0]), ka, kb


def test_singleton_matches_shortcut():
    roots = _singleton_roots()
    lambdas = np.array([2.5, 0.3])
    logdet, rank = penalty_logdet(roots, lambdas)
    expect_logdet = sum(
        r.rank * np.log(lam) + r.logdet_pos
        for r, lam in zip(roots, lambdas)
    )
    expect_rank = sum(r.rank for r in roots)
    assert abs(logdet - expect_logdet) < 1e-10
    assert rank == expect_rank

    grad = penalty_logdet_grad(roots, lambdas)
    np.testing.assert_allclose(grad, [r.rank for r in roots], atol=1e-10)


def test_tensor_rank_is_lambda_invariant():
    roots, ka, kb = _tensor_roots()
    # null(S_lambda) = null_a (x) null_b = 2 * 2 = 4 for any positive lambdas.
    for lam in ([1.0, 1.0], [1e6, 1e-4], [3e-3, 7e2]):
        _, rank = penalty_logdet(roots, np.array(lam))
        assert rank == ka * kb - 4


def test_tensor_gradient_matches_finite_difference():
    roots, _ka, _kb = _tensor_roots()
    rho = np.array([0.7, -1.3])          # log lambda
    analytic = penalty_logdet_grad(roots, np.exp(rho))

    def logdet_at(rho_vec):
        return penalty_logdet(roots, np.exp(rho_vec))[0]

    h = 1e-6
    fd = np.empty(2)
    for j in range(2):
        rp, rm = rho.copy(), rho.copy()
        rp[j] += h
        rm[j] -= h
        fd[j] = (logdet_at(rp) - logdet_at(rm)) / (2 * h)
    np.testing.assert_allclose(analytic, fd, rtol=1e-6, atol=1e-7)
