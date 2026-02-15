"""
Conditional test collection for Monte Carlo tests.

Prevents NOTSET parametrize errors when R fixtures are not generated.
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _has_mc_r_fixtures() -> bool:
    """Check if any mc_*_r_results.json files exist."""
    return any(FIXTURES_DIR.glob("mc_*_r_results.json"))


# Exclude R validation tests if fixtures are not available
collect_ignore_glob: list[str] = []

if not _has_mc_r_fixtures():
    collect_ignore_glob.append("test_r_validation.py")
