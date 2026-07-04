# Generates stationarity_r_reference.json — R tseries::adf.test and
# tseries::kpss.test references for the RIGOR R18 stationarity fixes
# (ADF default regression 'ct', MacKinnon p-values, KPSS lshort
# bandwidth).
#
# The simulated series are embedded in the fixture so the tests do not
# need R (or this seed) at run time.
#
# Run from the repo root:
#   Rscript tests/fixtures/generate_stationarity_r_reference.R
# R 4.5.2 / tseries 0.10-58 was used for the committed fixture.

library(jsonlite)
library(tseries)
suppressMessages(library(forecast))

set.seed(42)
series <- list(
  near_unit_root = as.numeric(arima.sim(n = 200, list(ar = 0.98))),
  stationary_ar1 = as.numeric(arima.sim(n = 200, list(ar = 0.3))),
  nile = as.numeric(Nile),
  log_lynx = as.numeric(log(lynx)),
  # WWWusage is a borderline KPSS/ndiffs case: forecast::ndiffs uses
  # use.lag = trunc(3*sqrt(n)/13) = 2 lags (reject, d=1) where the
  # tseries lshort default (4 lags) fails to reject (d=0). Guards the
  # ndiffs bandwidth pin.
  wwwusage = as.numeric(WWWusage)
)

ref <- list()
for (nm in names(series)) {
  x <- series[[nm]]
  a <- suppressWarnings(adf.test(x))          # always constant + trend
  kl <- suppressWarnings(kpss.test(x, null = "Level", lshort = TRUE))
  kt <- suppressWarnings(kpss.test(x, null = "Trend", lshort = TRUE))
  ref[[nm]] <- list(
    adf_stat = unname(a$statistic),
    adf_lag = unname(a$parameter),
    adf_pvalue_tseries = unname(a$p.value),
    kpss_level_stat = unname(kl$statistic),
    kpss_level_lag = unname(kl$parameter),
    kpss_level_p = unname(kl$p.value),
    kpss_trend_stat = unname(kt$statistic),
    kpss_trend_lag = unname(kt$parameter),
    kpss_trend_p = unname(kt$p.value)
  )
}

# forecast::ndiffs references (default KPSS path only; the ADF path
# has documented convention differences and is not pinned here).
ndiffs_ref <- list()
for (nm in names(series)) {
  ndiffs_ref[[nm]] <- forecast::ndiffs(series[[nm]])
}

out <- list(
  r_version = as.character(getRversion()),
  tseries_version = as.character(packageVersion("tseries")),
  forecast_version = as.character(packageVersion("forecast")),
  series = series,
  reference = ref,
  ndiffs_kpss = ndiffs_ref
)

writeLines(
  toJSON(out, digits = 12, auto_unbox = TRUE, pretty = TRUE),
  file.path("tests", "fixtures", "stationarity_r_reference.json")
)
cat("wrote stationarity_r_reference.json\n")
