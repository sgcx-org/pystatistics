"""
Regression test configuration.

Prevents collection of R validation test modules when no R fixtures exist,
avoiding pytest's NOTSET parametrize entries on CI without R.
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _has_lm_r_fixtures():
    """Check if any LM R result fixtures exist."""
    return any(
        f for f in FIXTURES_DIR.glob("*_r_results.json")
        if not f.name.startswith("glm_")
    )


def _has_glm_r_fixtures():
    """Check if any GLM R result fixtures exist."""
    return any(FIXTURES_DIR.glob("glm_*_r_results.json"))


collect_ignore_glob = []
if not _has_lm_r_fixtures():
    collect_ignore_glob.append("test_r_validation.py")
if not _has_glm_r_fixtures():
    collect_ignore_glob.append("test_glm_r_validation.py")
