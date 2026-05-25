"""
GPU tests for the structured term spec (categorical predictors + interactions).

A term-built Design is backend-agnostic: build_terms_design produces a plain
float64 design matrix, and the GPU backends consume it exactly like any other
Design. These tests confirm that fit(..., backend='gpu') on a term spec is
statistically equivalent to the CPU path (GPU uses FP32) and preserves the
expanded factor labels.

Skipped automatically when no CUDA GPU is available.
"""

import numpy as np
import pytest

from pystatistics import DataSource
from pystatistics.regression import Design, fit, C


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _cuda_available(), reason="CUDA GPU required"
)


@pytest.fixture
def ds():
    rng = np.random.default_rng(0)
    n = 4000
    age = rng.normal(50, 10, n)
    sex = rng.choice(["F", "M"], n)
    treatment = rng.choice(["A", "B", "C"], n)
    y = (
        1.0 + 0.05 * age + 1.5 * (sex == "M")
        + 2.0 * (treatment == "B") - 1.0 * (treatment == "C")
        + 0.8 * ((sex == "M") & (treatment == "B"))
        + rng.normal(0, 0.5, n)
    )
    yb = (y > np.median(y)).astype(float)
    df_dict = {"age": age, "sex": sex, "treatment": treatment, "y": y, "yb": yb}
    import pandas as pd
    return DataSource.from_dataframe(pd.DataFrame(df_dict))


TERMS = [
    "age", C("sex", ref="F"), C("treatment", ref="A"),
    (C("treatment", ref="A"), C("sex", ref="F")),
]


def test_ols_terms_gpu_matches_cpu(ds):
    design = Design.from_datasource(ds, y="y", terms=TERMS)
    cpu = fit(design, backend="cpu")
    gpu = fit(design, backend="gpu")
    # Same labels, and FP32 GPU within rtol of CPU on a well-conditioned fit.
    assert list(gpu.coef) == list(cpu.coef)
    np.testing.assert_allclose(
        gpu.coefficients, cpu.coefficients, rtol=1e-3, atol=1e-4
    )
    corr = np.corrcoef(gpu.coefficients, cpu.coefficients)[0, 1]
    assert corr > 0.9999


def test_glm_binomial_terms_gpu_matches_cpu(ds):
    design = Design.from_datasource(ds, y="yb", terms=TERMS)
    cpu = fit(design, family="binomial", backend="cpu")
    gpu = fit(design, family="binomial", backend="gpu")
    assert list(gpu.coef) == list(cpu.coef)
    np.testing.assert_allclose(
        gpu.coefficients, cpu.coefficients, rtol=2e-3, atol=1e-3
    )


def test_auto_backend_terms_runs(ds):
    # 'auto' prefers GPU when present; just confirm the term path flows through.
    design = Design.from_datasource(ds, y="y", terms=TERMS)
    res = fit(design, backend="auto")
    assert list(res.coef)[0] == "(Intercept)"
    assert "treatment[B]:sex[M]" in res.coef
