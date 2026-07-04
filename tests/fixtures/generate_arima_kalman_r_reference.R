# Generates arima_kalman_r_reference.json — R references for the
# Kalman-path fixes (RIGOR R18 follow-up, 4.6.4):
#   * seasonal-AR exact-ML log-likelihood (the stationary-init defect
#     silently fell back to a diffuse init and shifted loglik ~80 units
#     on (1,1,1)(1,1,0)[12] models),
#   * predict.Arima forecast parity (point forecasts from the filtered
#     state; SEs from the exact forecast-error covariance aggregated
#     through the differencing operators),
#   * coefficient standard errors (vcov in the factored
#     parameterization) and the Kalman profile sigma2.
#
# Run from the repo root:
#   Rscript tests/fixtures/generate_arima_kalman_r_reference.R
# R 4.5.2 was used for the committed fixture.

library(jsonlite)

series <- list(
  airpassengers = as.numeric(AirPassengers),
  log_airpassengers = as.numeric(log(AirPassengers)),
  nottem = as.numeric(nottem),
  co2 = as.numeric(co2)
)

fit_ref <- function(x, order, seasonal, label) {
  if (is.null(seasonal)) {
    f <- arima(x, order = order, method = "CSS-ML")
    seas <- NULL
  } else {
    f <- arima(x, order = order,
               seasonal = list(order = seasonal, period = 12),
               method = "CSS-ML")
    seas <- c(seasonal, 12)
  }
  k <- length(f$coef) + 1
  n <- f$nobs
  pr <- predict(f, n.ahead = 12)
  list(
    label = label,
    order = order,
    seasonal = seas,
    k = k,
    nobs = n,
    loglik = f$loglik,
    aic = f$aic,
    aicc = f$aic + 2 * k * (k + 1) / (n - k - 1),
    sigma2 = f$sigma2,
    coef = as.list(coef(f)),
    se = as.list(sqrt(diag(f$var.coef))),
    fc_mean = as.numeric(pr$pred),
    fc_se = as.numeric(pr$se),
    # R's residuals() = standardized Kalman innovations v_t/sqrt(F_t);
    # length is the ORIGINAL series length (the first d + D*m entries
    # belong to observations consumed by differencing).
    resid = as.numeric(residuals(f))
  )
}

fits <- list(
  # The two models that exposed the stationary-init defect.
  logap_111_110 = fit_ref(log(AirPassengers), c(1, 1, 1), c(1, 1, 0),
                          "logap_111_110"),
  nottem_111_110 = fit_ref(nottem, c(1, 1, 1), c(1, 1, 0),
                           "nottem_111_110"),
  # Forecast-parity battery on AirPassengers: seasonal, near-unit MA,
  # plain differenced, and pure random walk (analytic se).
  airline = fit_ref(AirPassengers, c(0, 1, 1), c(0, 1, 1), "airline"),
  s211_010 = fit_ref(AirPassengers, c(2, 1, 1), c(0, 1, 0), "s211_010"),
  ns211 = fit_ref(AirPassengers, c(2, 1, 1), NULL, "ns211"),
  ns010 = fit_ref(AirPassengers, c(0, 1, 0), NULL, "ns010"),
  # Higher differencing orders: guards the _undifference integration
  # order (d >= 2 / D >= 2 forecasts diverged before the fix).
  d2_121 = fit_ref(AirPassengers, c(1, 2, 1), NULL, "d2_121"),
  d2_020 = fit_ref(AirPassengers, c(0, 2, 0), NULL, "d2_020"),
  D2_011_021 = fit_ref(AirPassengers, c(0, 1, 1), c(0, 2, 1),
                       "D2_011_021"),
  # co2 (2,1,1): the CSS stage aborts (NaN objective) while the ML
  # refinement converges to R's optimum — guards the converged-flag
  # bookkeeping.
  co2_211 = fit_ref(co2, c(2, 1, 1), NULL, "co2_211"),
  # MA-mirror cases: the exact-ML optimizer can land on the
  # non-invertible reflection (theta -> 1/theta, identical likelihood);
  # these guard the invertibility normalization (coefficients, sigma2).
  ap_111_111 = fit_ref(AirPassengers, c(1, 1, 1), c(1, 1, 1),
                       "ap_111_111"),
  co2_101m = fit_ref(co2, c(1, 0, 1), NULL, "co2_101m")
)

# init= parity. Note R's OWN init handling errors on non-invertible MA
# inits (the CSS/optim stage diverges before maInvert applies) and on
# non-stationary AR inits ('non-stationary AR part from CSS'); those
# are unit-tested directly. The R-parity case pinned here is the
# warm start near the optimum, which must land at the standard fit.
init_fits <- list(
  airline_warm = local({
    f <- arima(AirPassengers, c(0, 1, 1),
               seasonal = list(order = c(0, 1, 1), period = 12),
               init = c(-0.3, -0.1), method = "CSS-ML")
    list(label = "airline_warm", init = c(-0.3, -0.1),
         coef = as.list(coef(f)), loglik = f$loglik, sigma2 = f$sigma2)
  })
)

out <- list(
  r_version = as.character(getRversion()),
  series = series,
  fits = fits,
  init_fits = init_fits
)

writeLines(
  toJSON(out, digits = 12, auto_unbox = TRUE, pretty = TRUE),
  file.path("tests", "fixtures", "arima_kalman_r_reference.json")
)
cat("wrote arima_kalman_r_reference.json\n")
