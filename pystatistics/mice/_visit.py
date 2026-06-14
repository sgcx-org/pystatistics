"""
Visit-sequence construction for the chained-equations sweep.

The visit sequence is the order in which incomplete columns are imputed within
each iteration. R mice's default ("roman") visits the columns that contain
missing data in left-to-right column order, which is what Stage 1 implements.

Isolated in its own module because R supports several sequence policies
(monotone, reverse, custom orderings); keeping the policy here means adding one
later does not touch the sweep loop.
"""

from __future__ import annotations

from pystatistics.core.exceptions import ValidationError


def default_visit_sequence(incomplete_columns: tuple[int, ...]) -> tuple[int, ...]:
    """R-default ("roman"): incomplete columns in ascending index order."""
    return tuple(sorted(incomplete_columns))


def resolve_visit_sequence(
    incomplete_columns: tuple[int, ...],
    override,
) -> tuple[int, ...]:
    """Resolve the visit sequence, validating any user override.

    Parameters
    ----------
    incomplete_columns : tuple of int
        Column indices that actually contain missing values.
    override : sequence of int or None
        Explicit visit order. If None, uses the R default. If given, every
        entry must be an incomplete column index; columns may be repeated
        (legal in MICE — a variable can be visited more than once per
        iteration) but every incomplete column must appear at least once so no
        missing values are left un-imputed.
    """
    if override is None:
        return default_visit_sequence(incomplete_columns)

    incomplete_set = set(incomplete_columns)
    seq = tuple(int(j) for j in override)

    unknown = [j for j in seq if j not in incomplete_set]
    if unknown:
        raise ValidationError(
            f"visit_sequence references columns {unknown} that have no missing "
            f"values. Incomplete columns are {sorted(incomplete_set)}."
        )
    missing_from_seq = incomplete_set - set(seq)
    if missing_from_seq:
        raise ValidationError(
            f"visit_sequence omits incomplete columns {sorted(missing_from_seq)}; "
            f"their missing values would never be imputed."
        )
    return seq
