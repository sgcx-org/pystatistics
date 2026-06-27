"""Smoke tests for the shared ``SolutionReprMixin`` HTML repr.

Every public ``*Solution`` mixes in
:class:`pystatistics.core.result.SolutionReprMixin`, which renders the
object's ``summary()`` inside a ``<pre>`` block for uniform Jupyter
display. These tests construct a few representative solutions through the
public API and assert the mixin is wired up — i.e. ``_repr_html_`` returns
a string containing ``<pre>``.
"""

import numpy as np

import pystatistics as ps


def _design_matrix() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 3))
    y = x @ np.array([1.0, 2.0, 3.0]) + rng.standard_normal(50)
    return x, y


def test_linear_solution_repr_html() -> None:
    x, y = _design_matrix()
    result = ps.regression.fit(x, y)
    assert "<pre>" in result._repr_html_()


def test_pca_solution_repr_html() -> None:
    x, _ = _design_matrix()
    result = ps.multivariate.pca(x)
    assert "<pre>" in result._repr_html_()


def test_arima_solution_repr_html() -> None:
    rng = np.random.default_rng(1)
    series = rng.standard_normal(80)
    result = ps.timeseries.arima(series, order=(1, 0, 0))
    assert "<pre>" in result._repr_html_()


def test_t_test_solution_repr_html() -> None:
    rng = np.random.default_rng(2)
    result = ps.hypothesis.t_test(rng.standard_normal(30))
    assert "<pre>" in result._repr_html_()
