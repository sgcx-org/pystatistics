"""Tests for the deterministic RNG management in `mice._rng`."""

import numpy as np
import pytest

from pystatistics.core.exceptions import ValidationError
from pystatistics.mice._rng import make_rng, spawn_streams


class TestMakeRng:
    def test_returns_generator(self):
        rng = make_rng(0)
        assert isinstance(rng, np.random.Generator)

    def test_same_seed_same_stream(self):
        a = make_rng(42).standard_normal(10)
        b = make_rng(42).standard_normal(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_stream(self):
        a = make_rng(1).standard_normal(10)
        b = make_rng(2).standard_normal(10)
        assert not np.array_equal(a, b)


class TestSpawnStreams:
    def test_count(self):
        streams = spawn_streams(0, 5)
        assert len(streams) == 5
        assert all(isinstance(s, np.random.Generator) for s in streams)

    def test_streams_are_independent(self):
        streams = spawn_streams(0, 3)
        draws = [s.standard_normal(20) for s in streams]
        # No two substreams produce identical draws.
        for i in range(len(draws)):
            for j in range(i + 1, len(draws)):
                assert not np.array_equal(draws[i], draws[j])

    def test_reproducible_given_seed(self):
        a = [s.standard_normal(5) for s in spawn_streams(7, 4)]
        b = [s.standard_normal(5) for s in spawn_streams(7, 4)]
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_substream_independent_of_consumption_order(self):
        # Drawing from stream 2 first must not change stream 0's draws:
        # substreams depend only on (seed, index), not on consumption order.
        s_forward = spawn_streams(123, 3)
        first = s_forward[0].standard_normal(8)

        s_reverse = spawn_streams(123, 3)
        _ = s_reverse[2].standard_normal(8)  # consume a later stream first
        first_again = s_reverse[0].standard_normal(8)

        np.testing.assert_array_equal(first, first_again)


class TestRngFailures:
    @pytest.mark.parametrize("bad", [-1, -100])
    def test_negative_seed_rejected(self, bad):
        with pytest.raises(ValidationError):
            make_rng(bad)

    @pytest.mark.parametrize("bad", [1.5, "0", None, True])
    def test_non_integer_seed_rejected(self, bad):
        with pytest.raises(ValidationError):
            make_rng(bad)

    def test_spawn_zero_streams_rejected(self):
        with pytest.raises(ValidationError):
            spawn_streams(0, 0)

    def test_spawn_negative_streams_rejected(self):
        with pytest.raises(ValidationError):
            spawn_streams(0, -3)
