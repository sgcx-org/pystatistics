# Generates arima_xreg_r_reference.json — R stats::arima references for
# regression with ARIMA errors (xreg), drift (include.drift as xreg = 1:n),
# and fixed= parameter masking. Covers the VA-4 / VA-4b feature and the
# RIGOR R10 hard cases (xreg under differencing, collinear xreg, all-but-one
# fixed, drift under d=1, seasonal + xreg).
#
# stats::arima is the exact-ML reference pystatistics targets: it reports the
# MLE sigma2 (SSR/n) and predict.Arima prediction SEs use that sigma2, so the
# estimator-invariant quantities (coef, loglik, AIC, SE, forecasts) match
# tightly. (forecast::Arima reports a df-adjusted sigma2 = SSR/(n-ncoef) and
# hence slightly wider intervals — a documented convention difference.)
#
# Run from the repo root:  Rscript tests/fixtures/generate_arima_xreg_r_reference.R
# R 4.5.2 was used for the committed fixture.

library(jsonlite)

set.seed(20260711)
n <- 140
x1 <- rnorm(n)
x2 <- rnorm(n)
tt <- 1:n
e_ar   <- as.numeric(arima.sim(list(ar = 0.6), n = n))
e_arma <- as.numeric(arima.sim(list(ar = 0.5, ma = 0.3), n = n))
# genuine seasonal AR(1)[4] error (identified sar1)
e_s <- numeric(n); es_innov <- rnorm(n)
for (i in 5:n) e_s[i] <- 0.6 * e_s[i - 4] + es_innov[i]

# --- Series ---
yA  <- 2.0 + 1.5 * x1 - 0.8 * x2 + e_ar          # (1,0,0) + xreg, d=0
yMA <- -1.0 + 0.9 * x1 + e_arma                  # (1,0,1) + xreg, d=0
yD  <- cumsum(c(30, 0.6 + rnorm(n - 1)))         # random walk + drift, d=1
yXd <- cumsum(0.7 * x1 + e_ar)                    # I(1) driven by xreg, d=1
xc1 <- rnorm(n); xc2 <- xc1 + rnorm(n, sd = 0.02) # near-collinear pair
yC  <- 1.0 + 0.5 * xc1 + 0.5 * xc2 + e_ar         # collinear xreg, d=0
yS  <- 3.0 + 1.2 * x1 + e_s                        # seasonal + xreg, d=0

fit_xreg <- function(y, order, xreg, seasonal = NULL, fixed = NULL,
                     newxreg = NULL, h = 0, label = "") {
  if (!is.null(xreg)) {
    xreg <- as.matrix(xreg)
    if (is.null(colnames(xreg)))
      colnames(xreg) <- paste0("xreg", seq_len(ncol(xreg)))
  }
  args <- list(x = y, order = order, xreg = xreg, method = "CSS-ML")
  if (!is.null(seasonal)) args$seasonal <- list(order = seasonal[1:3],
                                                period = seasonal[4])
  if (!is.null(fixed)) args$fixed <- fixed
  f <- do.call(stats::arima, args)
  se <- sqrt(diag(f$var.coef))
  out <- list(
    label = label, order = order,
    seasonal = if (is.null(seasonal)) NULL else seasonal,
    coef_names = names(f$coef),
    coef = as.numeric(f$coef),
    se = as.numeric(se),
    loglik = f$loglik, aic = f$aic, sigma2 = f$sigma2, nobs = f$nobs
  )
  if (h > 0) {
    p <- predict(f, n.ahead = h, newxreg = newxreg)
    out$fc <- as.numeric(p$pred)
    out$fcse <- as.numeric(p$se)
    out$h <- h
  }
  out
}

# Deterministic future regressors for forecast cases.
set.seed(7)
newxA <- cbind(rnorm(6), rnorm(6))
newxXd <- matrix(rnorm(6), ncol = 1)

models <- list(
  A = fit_xreg(yA, c(1, 0, 0), cbind(x1, x2), newxreg = newxA, h = 6,
               label = "ar1_xreg2_d0"),
  MA = fit_xreg(yMA, c(1, 0, 1), matrix(x1, ncol = 1), label = "arma11_xreg1_d0"),
  D = fit_xreg(yD, c(0, 1, 1), matrix(tt, ncol = 1),
               newxreg = matrix((n + 1):(n + 6), ncol = 1), h = 6,
               label = "drift_011_d1"),
  Xd = fit_xreg(yXd, c(2, 1, 0), matrix(x1, ncol = 1),
                newxreg = newxXd, h = 6, label = "ar2_xreg1_d1"),
  FixedMA = fit_xreg(e_arma, c(1, 0, 1), NULL, fixed = c(NA, 0, NA),
                     label = "fixed_ma1_zero"),
  FixedAR = fit_xreg(e_arma, c(2, 0, 0), NULL, fixed = c(NA, 0.2, NA),
                     label = "fixed_ar2_nonzero"),
  Collinear = fit_xreg(yC, c(1, 0, 0), cbind(xc1, xc2),
                       label = "collinear_xreg_d0"),
  SeasonalXreg = fit_xreg(yS, c(0, 0, 0), matrix(x1, ncol = 1),
                          seasonal = c(1, 0, 0, 4), label = "seasonal_xreg_d0")
)

series <- list(
  yA = yA, yMA = yMA, yD = yD, yXd = yXd, yC = yC, yS = yS,
  x1 = x1, x2 = x2, tt = tt, xc1 = xc1, xc2 = xc2,
  e_arma = e_arma, newxA = newxA, newxXd = as.numeric(newxXd)
)

out <- list(
  r_version = as.character(getRversion()),
  note = "stats::arima CSS-ML references for xreg / drift / fixed (VA-4).",
  n = n,
  series = series,
  models = models
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 12, pretty = TRUE),
           "tests/fixtures/arima_xreg_r_reference.json")
cat("wrote tests/fixtures/arima_xreg_r_reference.json\n")
