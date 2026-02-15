"""Tests for random effects specification and Z matrix construction."""

import numpy as np
import pytest

from pystatistics.mixed._random_effects import (
    parse_random_effects,
    build_z_matrix,
    build_lambda,
    theta_lower_bounds,
    theta_start,
    RandomEffectSpec,
)


class TestParseRandomEffects:
    """Tests for parse_random_effects."""

    def test_intercept_only_default(self):
        """When random_effects is None, default to intercept for each group."""
        groups = {'subject': np.array([0, 0, 1, 1, 2, 2])}
        specs = parse_random_effects(groups, None, None, 6)
        assert len(specs) == 1
        assert specs[0].group_name == 'subject'
        assert specs[0].terms == ('1',)
        assert specs[0].n_groups == 3
        assert specs[0].n_terms == 1
        assert specs[0].theta_size == 1

    def test_intercept_and_slope(self):
        """Test parsing intercept + slope specification."""
        n = 6
        groups = {'subject': np.array([0, 0, 1, 1, 2, 2])}
        re = {'subject': ['1', 'time']}
        rd = {'time': np.array([0., 1., 0., 1., 0., 1.])}

        specs = parse_random_effects(groups, re, rd, n)
        assert specs[0].n_terms == 2
        assert specs[0].terms == ('1', 'time')
        assert specs[0].theta_size == 3  # 2*(2+1)/2

    def test_crossed_random_effects(self):
        """Two grouping factors produce two specs."""
        n = 6
        groups = {
            'subject': np.array([0, 0, 1, 1, 2, 2]),
            'item': np.array([0, 1, 0, 1, 0, 1]),
        }
        specs = parse_random_effects(groups, None, None, n)
        assert len(specs) == 2
        names = {s.group_name for s in specs}
        assert names == {'subject', 'item'}

    def test_group_ids_are_consecutive(self):
        """Group labels are mapped to 0-indexed consecutive integers."""
        groups = {'subject': np.array(['B', 'B', 'A', 'A', 'C', 'C'])}
        specs = parse_random_effects(groups, None, None, 6)
        # 'A'=0, 'B'=1, 'C'=2 after sorting
        assert specs[0].n_groups == 3
        expected_ids = np.array([1, 1, 0, 0, 2, 2])
        np.testing.assert_array_equal(specs[0].group_ids, expected_ids)

    def test_missing_random_data_raises(self):
        """Missing slope data raises ValueError."""
        groups = {'subject': np.array([0, 0, 1, 1])}
        re = {'subject': ['1', 'time']}
        with pytest.raises(ValueError, match="requires data"):
            parse_random_effects(groups, re, None, 4)

    def test_length_mismatch_raises(self):
        """Mismatched group length raises ValueError."""
        groups = {'subject': np.array([0, 0, 1])}  # 3 != 4
        with pytest.raises(ValueError, match="expected 4"):
            parse_random_effects(groups, None, None, 4)


class TestZBlock:
    """Tests for Z block construction within parse_random_effects."""

    def test_intercept_z_is_indicator(self):
        """Random intercept Z block is an indicator matrix."""
        groups = {'g': np.array([0, 0, 1, 1, 2, 2])}
        specs = parse_random_effects(groups, None, None, 6)
        Z = specs[0].Z_block

        assert Z.shape == (6, 3)
        expected = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=float)
        np.testing.assert_array_equal(Z, expected)

    def test_slope_z_values(self):
        """Random slope Z block has slope values in correct positions."""
        n = 4
        groups = {'g': np.array([0, 0, 1, 1])}
        re = {'g': ['1', 'time']}
        time = np.array([0., 1., 0., 1.])
        rd = {'time': time}

        specs = parse_random_effects(groups, re, rd, n)
        Z = specs[0].Z_block

        # Shape: (4, 2*2) = (4, 4) — 2 groups × 2 terms
        assert Z.shape == (4, 4)

        # Intercept columns (first 2): indicator
        np.testing.assert_array_equal(Z[:, 0], [1, 1, 0, 0])
        np.testing.assert_array_equal(Z[:, 1], [0, 0, 1, 1])

        # Slope columns (next 2): indicator × time
        np.testing.assert_array_equal(Z[:, 2], [0, 1, 0, 0])
        np.testing.assert_array_equal(Z[:, 3], [0, 0, 0, 1])


class TestBuildZMatrix:
    """Tests for build_z_matrix (horizontal concatenation)."""

    def test_single_group(self):
        """Single group: build_z_matrix returns the Z_block directly."""
        groups = {'g': np.array([0, 0, 1, 1])}
        specs = parse_random_effects(groups, None, None, 4)
        Z = build_z_matrix(specs)
        np.testing.assert_array_equal(Z, specs[0].Z_block)

    def test_crossed_concatenation(self):
        """Crossed groups: Z is horizontal concatenation of blocks."""
        n = 4
        groups = {
            'subject': np.array([0, 0, 1, 1]),
            'item': np.array([0, 1, 0, 1]),
        }
        specs = parse_random_effects(groups, None, None, n)
        Z = build_z_matrix(specs)

        # Should be (4, 2 + 2) = (4, 4)
        total_cols = sum(s.n_groups * s.n_terms for s in specs)
        assert Z.shape == (n, total_cols)


class TestBuildLambda:
    """Tests for Λ_θ construction."""

    def test_single_intercept_scalar(self):
        """Single intercept: Λ is θ[0] × I_J."""
        specs = [RandomEffectSpec(
            group_name='g', group_ids=np.array([0, 0, 1, 1]),
            terms=('1',), Z_block=np.zeros((4, 2)),
            n_groups=2, n_terms=1, theta_size=1,
        )]
        theta = np.array([2.5])
        Lambda = build_lambda(theta, specs)

        assert Lambda.shape == (2, 2)
        expected = np.diag([2.5, 2.5])
        np.testing.assert_array_almost_equal(Lambda, expected)

    def test_intercept_slope_cholesky(self):
        """Intercept + slope: Λ uses T⊗I_J layout (term-major ordering).

        For T = [[3, 0], [0.5, 2]] and J=2 groups, T⊗I_2 gives:
            [[3*I_2,   0  ],   =  [[3, 0, 0, 0],
             [0.5*I_2, 2*I_2]]      [0, 3, 0, 0],
                                     [0.5, 0, 2, 0],
                                     [0, 0.5, 0, 2]]

        Column ordering: [int_g0, int_g1, slope_g0, slope_g1]
        """
        specs = [RandomEffectSpec(
            group_name='g', group_ids=np.array([0, 0, 1, 1]),
            terms=('1', 'time'), Z_block=np.zeros((4, 4)),
            n_groups=2, n_terms=2, theta_size=3,
        )]
        # θ = [T[0,0], T[1,0], T[1,1]]
        theta = np.array([3.0, 0.5, 2.0])
        Lambda = build_lambda(theta, specs)

        # T⊗I_2: term-major Kronecker product
        assert Lambda.shape == (4, 4)
        expected = np.array([
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.5, 0.0, 2.0, 0.0],
            [0.0, 0.5, 0.0, 2.0],
        ])
        np.testing.assert_array_almost_equal(Lambda, expected)

    def test_crossed_block_diagonal(self):
        """Crossed groups: Λ is block-diagonal with separate blocks."""
        specs = [
            RandomEffectSpec(
                group_name='s', group_ids=np.zeros(4, dtype=int),
                terms=('1',), Z_block=np.zeros((4, 2)),
                n_groups=2, n_terms=1, theta_size=1,
            ),
            RandomEffectSpec(
                group_name='i', group_ids=np.zeros(4, dtype=int),
                terms=('1',), Z_block=np.zeros((4, 3)),
                n_groups=3, n_terms=1, theta_size=1,
            ),
        ]
        theta = np.array([2.0, 1.5])
        Lambda = build_lambda(theta, specs)

        assert Lambda.shape == (5, 5)
        # First block: 2×2 with diag 2.0
        np.testing.assert_array_almost_equal(Lambda[0:2, 0:2], np.diag([2.0, 2.0]))
        # Second block: 3×3 with diag 1.5
        np.testing.assert_array_almost_equal(Lambda[2:5, 2:5], np.diag([1.5, 1.5, 1.5]))
        # Off-diagonal blocks are zero
        np.testing.assert_array_almost_equal(Lambda[0:2, 2:5], np.zeros((2, 3)))


class TestThetaBounds:
    """Tests for theta bounds and starting values."""

    def test_intercept_only_bound(self):
        """Single intercept: lower bound is [0]."""
        specs = [RandomEffectSpec(
            group_name='g', group_ids=np.zeros(4, dtype=int),
            terms=('1',), Z_block=np.zeros((4, 2)),
            n_groups=2, n_terms=1, theta_size=1,
        )]
        lb = theta_lower_bounds(specs)
        assert len(lb) == 1
        assert lb[0] == 0.0

    def test_intercept_slope_bounds(self):
        """Intercept + slope: diagonal ≥ 0, off-diagonal unbounded."""
        specs = [RandomEffectSpec(
            group_name='g', group_ids=np.zeros(4, dtype=int),
            terms=('1', 'time'), Z_block=np.zeros((4, 4)),
            n_groups=2, n_terms=2, theta_size=3,
        )]
        lb = theta_lower_bounds(specs)
        assert len(lb) == 3
        assert lb[0] == 0.0       # T[0,0] diagonal
        assert lb[1] == -np.inf   # T[1,0] off-diagonal
        assert lb[2] == 0.0       # T[1,1] diagonal

    def test_start_values(self):
        """Starting values: 1 on diagonal, 0 off-diagonal."""
        specs = [RandomEffectSpec(
            group_name='g', group_ids=np.zeros(4, dtype=int),
            terms=('1', 'time'), Z_block=np.zeros((4, 4)),
            n_groups=2, n_terms=2, theta_size=3,
        )]
        t0 = theta_start(specs)
        np.testing.assert_array_equal(t0, [1.0, 0.0, 1.0])
