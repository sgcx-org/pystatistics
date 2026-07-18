# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **New public API `regression.simple_ols(x, y)`** — a lean, 1-D–friendly front
  door for with-intercept univariate OLS (`pystatistics/regression/_simple_ols.py`,
  new module, Rule 3). Returns a frozen `SimpleOLSResult` dataclass with
  `slope, intercept, r_squared, adjusted_r_squared, slope_se, n` — deliberately
  **not** the `Result[LinearParams]` / `LinearSolution` wrapper (Rule 3 note in
  the module): the lean path skips the heavy lazy-inference object to stay
  allocation-light in hot loops, which is the feature's reason to exist. Math is
  the standard centered-sum OLS; `RSS` is computed from residuals directly (not
  `Syy − Sxy²/Sxx`) so `slope_se` doesn't cancel catastrophically on
  near-collinear data. Both `simple_ols` and `SimpleOLSResult` are exported from
  `regression/__init__.py` and added to `__all__`.
  - **Why:** the engine already *can* do ordinary regression via `fit(X, y)`, but
    `fit()` wants a 2-D design and rebuilds a `Design` + backend per call, so a
    univariate caller in a tight loop pays real setup tax. That friction is enough
    that a domain module reached past the engine for `scipy.stats.linregress`
    instead — concretely, `pystatsbio`'s PK/NCA terminal-slope (λz) estimation,
    which fits 3–8 candidate windows per profile across many profiles scored on
    adjusted R². `simple_ols` is the lean primitive that closes that gap so domain
    modules delegate to the house engine rather than to scipy. Per the issue, this
    is a separate primitive — `fit()` is **not** given a 1-D mode or a `lite=` flag.
  - **Failure behavior (Rule 1 / Rule 2):** raises
    `core.exceptions.ValidationError` (not bare `ValueError`) on length mismatch,
    `n < 3`, non-`1-D` input, any non-finite value in `x` or `y` (never silently
    dropped — the caller owns masking), zero-variance `x` (`Sxx == 0`), and
    zero-variance `y` (`Syy == 0`), each with a descriptive message.
  - **Validation:** the reference case matches R `lm(y ~ x)` — coefficients,
    `summary()$r.squared` / `$adj.r.squared`, and the slope's `Std. Error` — to
    `rtol=1e-10`, cross-checked against `scipy.stats.linregress`; 19 tests in
    `tests/regression/test_simple_ols.py` cover normal / edge / failure cases.
  - **Downstream (separate, dependent task, NOT in this change):** re-wire
    `pystatsbio`'s NCA λz estimator to call `simple_ols` and re-peg `pystatsbio`
    to `pystatistics>=5.1`. That work lives in the `pystatsbio` repo (Rule 8).
