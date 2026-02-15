"""
CPU reference backend for hypothesis tests.

Dispatches to test-specific submodules based on design.test_type.
Validated against R to rtol=1e-10.
"""

from __future__ import annotations

from pystatistics.core.result import Result
from pystatistics.core.compute.timing import Timer
from pystatistics.hypothesis._common import HTestParams
from pystatistics.hypothesis.design import HypothesisDesign


class CPUHypothesisBackend:
    """CPU reference backend for hypothesis tests."""

    @property
    def name(self) -> str:
        return 'cpu_hypothesis'

    def solve(self, design: HypothesisDesign) -> Result[HTestParams]:
        """Dispatch to test-specific implementation based on design.test_type."""
        timer = Timer()
        timer.start()

        test_type = design.test_type

        with timer.section(test_type):
            if test_type == "t_one_sample":
                from pystatistics.hypothesis.backends._t_test import t_one_sample
                params, warnings_list = t_one_sample(design)
            elif test_type == "t_two_sample":
                from pystatistics.hypothesis.backends._t_test import t_two_sample
                params, warnings_list = t_two_sample(design)
            elif test_type == "t_paired":
                from pystatistics.hypothesis.backends._t_test import t_paired
                params, warnings_list = t_paired(design)
            elif test_type == "chisq_independence":
                from pystatistics.hypothesis.backends._chisq_test import chisq_independence
                params, warnings_list = chisq_independence(design)
            elif test_type == "chisq_gof":
                from pystatistics.hypothesis.backends._chisq_test import chisq_gof
                params, warnings_list = chisq_gof(design)
            elif test_type == "prop_test":
                from pystatistics.hypothesis.backends._prop_test import prop_test
                params, warnings_list = prop_test(design)
            elif test_type == "fisher_test":
                from pystatistics.hypothesis.backends._fisher_test import fisher_test
                params, warnings_list = fisher_test(design)
            elif test_type == "wilcox_signed_rank":
                from pystatistics.hypothesis.backends._wilcox_test import wilcox_signed_rank
                params, warnings_list = wilcox_signed_rank(design)
            elif test_type == "wilcox_rank_sum":
                from pystatistics.hypothesis.backends._wilcox_test import wilcox_rank_sum
                params, warnings_list = wilcox_rank_sum(design)
            elif test_type == "ks_two_sample":
                from pystatistics.hypothesis.backends._ks_test import ks_test
                params, warnings_list = ks_test(design)
            elif test_type == "ks_one_sample":
                from pystatistics.hypothesis.backends._ks_test import ks_test
                params, warnings_list = ks_test(design)
            elif test_type == "var_test":
                from pystatistics.hypothesis.backends._var_test import var_test
                params, warnings_list = var_test(design)
            else:
                raise ValueError(f"Unknown test_type: {test_type!r}")

        timer.stop()

        return Result(
            params=params,
            info={'test_type': test_type},
            timing=timer.result(),
            backend_name=self.name,
            warnings=tuple(warnings_list),
        )
