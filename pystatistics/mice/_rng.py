"""
Deterministic random-number management for MICE.

Multiple imputation is *intrinsically* stochastic: each completed dataset is a
draw from the posterior predictive distribution of the missing data. That
randomness is the method, not a bug. What we owe the user (CLAUDE.md Rule 6) is
that the randomness be fully controlled by an explicit, injectable seed, and
isolated behind this module so the rest of the system stays deterministic and
testable.

# NON-DETERMINISTIC: every function here produces or wraps a pseudo-random
# stream. The non-determinism is real but is *seeded* — given the same seed,
# every downstream draw is bit-for-bit reproducible across runs and machines.
# No other module in `mice` is permitted to call into numpy.random directly;
# all randomness must flow through the Generators handed out here.

We deliberately do NOT try to reproduce R's Mersenne-Twister stream. MICE is
validated against R *distributionally* (same defaults, same algorithm, matching
statistical behaviour across seeds), not by RNG-stream parity — see
``tests/mice/references/generate_mice_fixtures.R``.
"""

from __future__ import annotations

import numpy as np

from pystatistics.core.exceptions import ValidationError


def make_rng(seed: int) -> np.random.Generator:
    """Construct a NumPy Generator from an integer seed.

    Parameters
    ----------
    seed : int
        Non-negative integer seed. Required (not optional): reproducibility is
        a first-class guarantee of this module, so there is no "unseeded" path.

    Returns
    -------
    numpy.random.Generator
    """
    _validate_seed(seed)
    return np.random.default_rng(seed)


def spawn_streams(seed: int, m: int) -> list[np.random.Generator]:
    """Spawn ``m`` independent, reproducible Generators from one master seed.

    Each of the ``m`` imputation chains needs its own random stream so the
    chains are statistically independent, yet the whole set must be reproducible
    from the single user-facing ``seed``. ``numpy.random.SeedSequence`` provides
    exactly this: deterministic, high-quality spawning of independent substreams.

    The substreams do not depend on the order in which they are *consumed*, only
    on ``(seed, m)`` and their index — so a future GPU backend that runs the
    chains concurrently gets identical draws to the sequential CPU backend.

    Parameters
    ----------
    seed : int
        Master seed.
    m : int
        Number of independent streams (one per imputation). Must be >= 1.

    Returns
    -------
    list of numpy.random.Generator
        Length ``m``.
    """
    _validate_seed(seed)
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool):
        raise ValidationError(f"m must be an integer, got {type(m).__name__}")
    if m < 1:
        raise ValidationError(f"m must be >= 1, got {m}")

    child_seeds = np.random.SeedSequence(int(seed)).spawn(int(m))
    return [np.random.default_rng(cs) for cs in child_seeds]


def _validate_seed(seed: int) -> None:
    """Fail loud on an invalid seed (Rule 1)."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValidationError(
            f"seed must be a non-negative integer, got {type(seed).__name__}"
        )
    if seed < 0:
        raise ValidationError(f"seed must be non-negative, got {seed}")
