# Reference fixtures for VA-1 gam tensor-product / multivariate smooths.
#
# Emits, for te()/ti()/isotropic s(x,z):
#   * unconstrained basis + penalty list (absorb.cons=FALSE) for the
#     coordinate-exact te() check;
#   * full gam() fits (EDF, scale, fitted, smoothing parameters) for the
#     function-space fit checks.
#
# Run: Rscript tests/fixtures/generate_tensor_fixtures.R
suppressMessages(library(mgcv))
suppressMessages(library(jsonlite))

set.seed(11)
n <- 200
x <- round(runif(n), 5)
z <- round(runif(n), 5)
# A smooth surface + noise so the fits are non-degenerate.
f <- 0.6 * sin(2 * pi * x) + cos(2 * pi * z) + 1.5 * x * z
y <- f + rnorm(n, sd = 0.3)
dat <- data.frame(x = x, z = z, y = y)

out <- list(x = x, z = z, y = y)

# --- te() unconstrained basis (coordinate-exact target) --------------------
sm <- smoothCon(te(x, z, bs = "cr", k = c(5, 4)),
                data = dat, absorb.cons = FALSE)[[1]]
out$te_basis <- list(
  k = c(5, 4), bs = c("cr", "cr"),
  X = sm$X, S1 = sm$S[[1]], S2 = sm$S[[2]]
)

fit_info <- function(formula) {
  g <- gam(formula, data = dat, method = "REML")
  list(
    edf_total = sum(g$edf),
    scale = g$sig2,  # scale VALUE (g$scale partial-matches a logical flag)
    sp = as.numeric(g$sp),
    fitted = as.numeric(fitted(g)),
    reml = as.numeric(g$gcv.ubre)
  )
}

out$te_fit  <- fit_info(y ~ te(x, z, bs = "cr", k = c(5, 4)))
out$te_gcv  <- {
  g <- gam(y ~ te(x, z, bs = "cr", k = c(5, 4)), data = dat, method = "GCV.Cp")
  list(edf_total = sum(g$edf), scale = g$sig2, sp = as.numeric(g$sp),
       fitted = as.numeric(fitted(g)))
}
out$ti_fit  <- fit_info(y ~ ti(x, k = 5) + ti(z, k = 5) + ti(x, z, k = c(5, 5)))
out$iso_fit <- fit_info(y ~ s(x, z, k = 20))
# Mixed-margin hard case: cyclic x cubic.
out$te_mixed <- fit_info(y ~ te(x, z, bs = c("cc", "cr"), k = c(6, 5)))

# GLM-family tensor smooth (exercises the joint multi-lambda GLM REML
# gradient). Poisson counts from the same smooth surface.
set.seed(23)
ycount <- rpois(n, lambda = exp(0.4 * f))
datc <- data.frame(x = x, z = z, y = ycount)
gp <- gam(y ~ te(x, z, bs = "cr", k = c(5, 4)), data = datc,
          family = poisson(), method = "REML")
out$ycount <- ycount
out$te_poisson <- list(
  edf_total = sum(gp$edf), scale = gp$sig2, sp = as.numeric(gp$sp),
  fitted = as.numeric(fitted(gp)), reml = as.numeric(gp$gcv.ubre)
)

writeLines(toJSON(out, digits = 12, auto_unbox = TRUE),
           "tests/fixtures/gam_tensor_mgcv.json")
cat("wrote gam_tensor_mgcv.json\n")
