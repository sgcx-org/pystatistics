#!/usr/bin/env python3
"""
Generate hypothesis test fixtures for R validation.

Each fixture is a JSON file defining a test scenario (test type, data, params)
and placeholder for R reference results.

Usage:
    python tests/fixtures/generate_hypothesis_fixtures.py
    Rscript tests/fixtures/run_r_hypothesis_validation.R
    pytest tests/hypothesis/test_r_validation.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

FIXTURES_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)


def _save(name: str, scenario: dict) -> None:
    """Save a hypothesis test scenario to JSON."""
    path = FIXTURES_DIR / f"{name}_meta.json"
    with open(path, "w") as f:
        json.dump(scenario, f, indent=2)
    print(f"  Saved {path.name}")


def make_t_test_fixtures():
    """Generate t-test scenarios."""
    # 1. One-sample t-test
    x = RNG.normal(5.0, 2.0, 20).tolist()
    _save("htest_t_onesample", {
        "test": "t.test",
        "data": {"x": x},
        "params": {"mu": 5.0, "alternative": "two.sided", "conf.level": 0.95},
        "description": "One-sample t-test, mu=5",
    })

    # 2. Two-sample Welch t-test
    x = RNG.normal(5.0, 1.5, 25).tolist()
    y = RNG.normal(6.0, 2.0, 30).tolist()
    _save("htest_t_welch", {
        "test": "t.test",
        "data": {"x": x, "y": y},
        "params": {"mu": 0, "var.equal": False, "alternative": "two.sided",
                   "conf.level": 0.95},
        "description": "Two-sample Welch t-test",
    })

    # 3. Paired t-test
    x = RNG.normal(10, 3, 15).tolist()
    y = (np.array(x) + RNG.normal(1.5, 1.0, 15)).tolist()
    _save("htest_t_paired", {
        "test": "t.test",
        "data": {"x": x, "y": y},
        "params": {"mu": 0, "paired": True, "alternative": "two.sided",
                   "conf.level": 0.95},
        "description": "Paired t-test",
    })

    # 4. Two-sample pooled (Student's) t-test
    x = RNG.normal(0, 1, 20).tolist()
    y = RNG.normal(0.5, 1, 20).tolist()
    _save("htest_t_pooled", {
        "test": "t.test",
        "data": {"x": x, "y": y},
        "params": {"mu": 0, "var.equal": True, "alternative": "two.sided",
                   "conf.level": 0.95},
        "description": "Two-sample pooled t-test (var.equal=TRUE)",
    })


def make_chisq_fixtures():
    """Generate chi-squared test scenarios."""
    # 5. Independence test - 2x2 with Yates
    _save("htest_chisq_2x2_yates", {
        "test": "chisq.test",
        "data": {"table": [[10, 20], [30, 40]]},
        "params": {"correct": True},
        "description": "2x2 independence test with Yates correction",
    })

    # 6. Independence test - 3x3
    _save("htest_chisq_3x3", {
        "test": "chisq.test",
        "data": {"table": [[10, 20, 30], [40, 50, 60], [70, 80, 90]]},
        "params": {"correct": False},
        "description": "3x3 independence test without correction",
    })

    # 7. GOF test
    _save("htest_chisq_gof", {
        "test": "chisq.test",
        "data": {"x": [16, 18, 16, 14, 12, 12]},
        "params": {"p": [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]},
        "description": "Goodness-of-fit test (uniform die)",
    })

    # 8. GOF test with unequal p
    _save("htest_chisq_gof_unequal", {
        "test": "chisq.test",
        "data": {"x": [30, 20, 10, 40]},
        "params": {"p": [0.3, 0.2, 0.1, 0.4]},
        "description": "GOF test with unequal expected proportions",
    })


def make_fisher_fixtures():
    """Generate Fisher's exact test scenarios."""
    # 9. 2x2 table
    _save("htest_fisher_2x2", {
        "test": "fisher.test",
        "data": {"table": [[1, 9], [11, 3]]},
        "params": {"alternative": "two.sided", "conf.level": 0.95},
        "description": "Fisher 2x2 test (Lady tea-tasting)",
    })

    # 10. 2x2 one-sided
    _save("htest_fisher_2x2_less", {
        "test": "fisher.test",
        "data": {"table": [[1, 9], [11, 3]]},
        "params": {"alternative": "less", "conf.level": 0.95},
        "description": "Fisher 2x2 test (alternative=less)",
    })

    # 11. 3x3 table (uses Monte Carlo)
    _save("htest_fisher_3x3", {
        "test": "fisher.test",
        "data": {"table": [[5, 10, 15], [10, 5, 20], [15, 20, 5]]},
        "params": {"simulate.p.value": True, "B": 10000},
        "description": "Fisher 3x3 test with Monte Carlo",
    })


def make_wilcox_fixtures():
    """Generate Wilcoxon test scenarios."""
    # 12. Signed-rank test
    x = [1.5, 2.3, 3.1, 4.0, 5.2, 3.8, 2.9, 4.5, 5.0, 3.5]
    _save("htest_wilcox_signed", {
        "test": "wilcox.test",
        "data": {"x": x},
        "params": {"mu": 3.0, "alternative": "two.sided", "conf.int": True,
                   "conf.level": 0.95},
        "description": "Wilcoxon signed-rank test, mu=3",
    })

    # 13. Rank-sum test
    x = [1.2, 2.5, 3.1, 4.0, 5.3]
    y = [3.0, 4.5, 5.8, 6.2, 7.1, 8.0]
    _save("htest_wilcox_ranksum", {
        "test": "wilcox.test",
        "data": {"x": x, "y": y},
        "params": {"mu": 0, "alternative": "two.sided", "conf.int": True,
                   "conf.level": 0.95, "exact": True},
        "description": "Wilcoxon rank-sum test (exact)",
    })


def make_ks_fixtures():
    """Generate KS test scenarios."""
    # 14. Two-sample KS test
    x = RNG.normal(0, 1, 30).tolist()
    y = RNG.normal(0.5, 1.5, 25).tolist()
    _save("htest_ks_twosample", {
        "test": "ks.test",
        "data": {"x": x, "y": y},
        "params": {"alternative": "two.sided"},
        "description": "Two-sample KS test (shifted normal)",
    })

    # 15. One-sample KS test against normal
    x = RNG.normal(3.0, 1.5, 20).tolist()
    _save("htest_ks_onesample_norm", {
        "test": "ks.test",
        "data": {"x": x},
        "params": {"distribution": "pnorm", "mean": 3.0, "sd": 1.5},
        "description": "One-sample KS test against N(3, 1.5)",
    })


def make_prop_fixtures():
    """Generate proportion test scenarios."""
    # 16. One-sample proportion test
    _save("htest_prop_onesample", {
        "test": "prop.test",
        "data": {"x": [45], "n": [100]},
        "params": {"p": 0.5, "alternative": "two.sided", "conf.level": 0.95,
                   "correct": True},
        "description": "One-sample proportion test (45/100 vs 0.5)",
    })

    # 17. Two-sample proportion test
    _save("htest_prop_twosample", {
        "test": "prop.test",
        "data": {"x": [30, 50], "n": [100, 120]},
        "params": {"alternative": "two.sided", "conf.level": 0.95,
                   "correct": True},
        "description": "Two-sample proportion test (equality)",
    })


def make_var_fixtures():
    """Generate F-test (var.test) scenarios."""
    # 18. Basic var.test
    x = RNG.normal(0, 2, 20).tolist()
    y = RNG.normal(0, 3, 25).tolist()
    _save("htest_var_basic", {
        "test": "var.test",
        "data": {"x": x, "y": y},
        "params": {"ratio": 1.0, "alternative": "two.sided",
                   "conf.level": 0.95},
        "description": "F-test for equality of variances",
    })


def main():
    print("Generating hypothesis test fixtures...")
    make_t_test_fixtures()
    make_chisq_fixtures()
    make_fisher_fixtures()
    make_wilcox_fixtures()
    make_ks_fixtures()
    make_prop_fixtures()
    make_var_fixtures()
    print(f"\nDone! Generated 18 hypothesis test fixtures in {FIXTURES_DIR}")
    print("Next step: Rscript tests/fixtures/run_r_hypothesis_validation.R")


if __name__ == "__main__":
    main()
