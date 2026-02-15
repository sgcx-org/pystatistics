"""
Conditional collection for survival tests.

Skips test_r_validation.py if no R fixture results are available.
"""

import json
from pathlib import Path


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def pytest_collect_file(parent, file_path):
    """Skip R validation test file if no R results are available."""
    if file_path.name == "test_r_validation.py":
        # Check if at least one surv_*_r_results.json exists
        results = list(FIXTURES_DIR.glob("surv_*_r_results.json"))
        if len(results) == 0:
            return None  # skip collection
    return None  # fall through to default collection
