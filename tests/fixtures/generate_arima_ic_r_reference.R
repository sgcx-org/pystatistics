# Generates arima_ic_r_reference.json — R stats::arima information-criterion
# references for the seasonal-ARIMA free-parameter-count fix (RIGOR R18).
#
# For each model, records the free parameter count k = length(coef) + 1
# (R's AIC convention: coefficients + sigma2), the log-likelihood, and
# AIC/AICc/BIC computed exactly as forecast::auto.arima does:
#   AICc = AIC + 2k(k+1)/(nobs-k-1),  BIC = AIC + (log(nobs)-2)k
#
# Run from the repo root:  Rscript tests/fixtures/generate_arima_ic_r_reference.R
# R 4.5.2 was used for the committed fixture.

library(jsonlite)
suppressMessages(library(forecast))

ap <- as.numeric(AirPassengers)

fit_ref <- function(order, seasonal = NULL, label) {
  if (is.null(seasonal)) {
    f <- arima(ap, order = order, method = "CSS-ML")
    seas <- NULL
  } else {
    f <- arima(ap, order = order,
               seasonal = list(order = seasonal, period = 12),
               method = "CSS-ML")
    seas <- c(seasonal, 12)
  }
  k <- length(f$coef) + 1
  n <- f$nobs
  aic <- f$aic
  list(
    label = label,
    order = order,
    seasonal = seas,
    k = k,
    nobs = n,
    loglik = f$loglik,
    aic = aic,
    aicc = aic + 2 * k * (k + 1) / (n - k - 1),
    bic = aic + (log(n) - 2) * k,
    coef_names = names(f$coef)
  )
}

models <- list(
  fit_ref(c(0, 1, 1), c(0, 1, 1), "airline_011_011_12"),
  fit_ref(c(2, 1, 1), c(0, 1, 0), "s211_010_12"),
  fit_ref(c(2, 1, 1), NULL, "ns211"),
  fit_ref(c(1, 0, 1), NULL, "ns101_with_mean"),
  fit_ref(c(0, 1, 0), NULL, "ns010_degenerate"),
  fit_ref(c(2, 1, 3), NULL, "ns213"),
  fit_ref(c(0, 1, 0), c(0, 1, 0), "s010_010_12_degenerate")
)
names(models) <- vapply(models, function(m) m$label, character(1))

aa <- auto.arima(AirPassengers, ic = "aicc", stepwise = TRUE,
                 approximation = FALSE)
aa_order <- unname(arimaorder(aa))  # p d q P D Q m

out <- list(
  r_version = as.character(getRversion()),
  forecast_version = as.character(packageVersion("forecast")),
  series = list(airpassengers = ap),
  models = models,
  auto_arima_seasonal = list(
    order = aa_order[1:3],
    seasonal = c(aa_order[4:6], aa_order[7]),
    aicc = aa$aicc
  )
)

writeLines(
  toJSON(out, digits = 12, auto_unbox = TRUE, pretty = TRUE),
  file.path("tests", "fixtures", "arima_ic_r_reference.json")
)
cat("wrote arima_ic_r_reference.json\n")
