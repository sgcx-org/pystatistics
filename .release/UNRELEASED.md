# Unreleased Changes

> This file tracks all changes since the last stable release.
> Updated by whoever makes a change, on whatever machine.
> Synced via git so all sessions (Mac, Linux, etc.) see the same state.
>
> When ready to release, run: `python .release/release.py --status`
> and follow the manual release flow in the script docstring.

## Changes

- **timeseries: `arima` / `auto_arima` now support regression with ARIMA errors
  (`xreg`), drift (`include_drift`), and parameter masking (`fixed=`)** (VA-4 /
  VA-4b). Previously these R capabilities were absent (documented only as a TODO).
  - `arima(y, order, xreg=X)` fits `y = X @ beta + eta` where `eta` follows the
    ARIMA process, matching `stats::arima(xreg=)` / `forecast::Arima(xreg=)`. The
    regression coefficients are reported on the solution as `xreg_coef` (with a
    `'drift'` / `'intercept'` / `xreg1..xregk` naming in `xreg_names`), with joint
    standard errors in the trailing block of `vcov`. Validated vs `stats::arima`:
    coefficients ~5e-6, log-likelihood ~1e-9, AIC/sigma2 machine-precision,
    standard errors ~4e-5 on the reference fits.
  - `include_drift=True` adds a linear time-trend regressor (the models R reports
    "with drift"); it reproduces R's drift/`d` interaction exactly (an
    ARIMA(p,1,q) "with drift" is an ARMA-with-mean on the differenced series).
    Fails loud when total differencing `d + D >= 2` (the trend is unidentifiable).
  - `fixed=` holds coefficients fixed during estimation, as a Pythonic
    `{name: value}` mapping (e.g. `fixed={'ma1': 0}`) or R's positional nan-vector.
    Fixed coefficients carry zero variance in `vcov` and are excluded from the
    information criteria. Matches `stats::arima(fixed=)`.
  - `auto_arima(..., allowdrift=True)` (default) now SELECTS drift models when the
    total differencing order is 1 and drift lowers the information criterion —
    matching `forecast::auto.arima`'s "with drift". Non-drift and seasonal
    selections (e.g. AirPassengers `(0,1,1)(0,1,1)[12]`) are unchanged.
  - `forecast_arima(..., newxreg=)` forecasts regression-with-ARIMA-errors models:
    future regressor values are required for `xreg` models; drift/intercept future
    columns are synthesized. Point forecasts and prediction SEs match
    `predict.Arima` to ~1e-6. (Prediction intervals use the MLE `sigma2` that
    matches `stats::arima`/`predict.Arima`; `forecast::Arima` reports a
    df-adjusted `sigma2 = SSR/(n - ncoef)` and hence slightly wider intervals — a
    documented convention difference, consistent with the rest of the module.)
  - The exact-ML/CSS regression path is CPU-only (the Whittle / `arima_batch` GPU
    kernels do not carry a regression term); `xreg`/`include_drift`/`fixed` fail
    loud with `method='Whittle'`.
  - New module `pystatistics/timeseries/_arima_xreg.py`; the general numerical
    Hessian moved to `_arima_likelihood.compute_numerical_hessian`. The plain
    (no-regressor) ARIMA fit path is unchanged.
