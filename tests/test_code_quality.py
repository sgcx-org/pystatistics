"""
CLAUDE.md Rule 4: LOC limits.

Scans all source files and asserts none exceed 500 lines of code
(comments and docstrings excluded).
"""

import re
from pathlib import Path

import pytest


PYSTAT_ROOT = Path(__file__).parent.parent / "pystatistics"


def _count_code_lines(filepath: Path) -> int:
    """Count lines of code excluding comments and docstrings."""
    content = filepath.read_text()
    content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
    content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)
    lines = content.split('\n')
    return len([l for l in lines if l.strip() and not l.strip().startswith('#')])


def test_no_file_exceeds_500_code_lines():
    """Rule 4: hard limit of 500 code lines per file."""
    violations = []
    for py_file in sorted(PYSTAT_ROOT.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        count = _count_code_lines(py_file)
        if count > 500:
            violations.append((str(py_file.relative_to(PYSTAT_ROOT.parent)), count))

    assert not violations, (
        "Files exceeding 500 code lines:\n"
        + "\n".join(f"  {p}: {c} lines" for p, c in violations)
    )
