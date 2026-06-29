#!/usr/bin/env Rscript
#
# R reference for PyStatistics prior-weights / offset validation.
#
# Reads weights_offset_cases.json (inputs) and fits each case with glm.fit /
# lm.wfit using the matrix interface (intercept=FALSE — X already carries the
# intercept column, matching PyStatistics' IRLS). Writes the reference outputs
# to weights_offset_r_results.json.
#
# Run from /path/to/pystatistics:
#   Rscript tests/fixtures/run_r_weights_offset_validation.R
#
library(jsonlite)
options(digits = 22)

fixtures_dir <- "tests/fixtures"
cases <- fromJSON(file.path(fixtures_dir, "weights_offset_cases.json"),
                  simplifyVector = FALSE)

make_family <- function(fam, link, theta) {
    if (fam == "gaussian") gaussian(link = link)
    else if (fam == "binomial") binomial(link = link)
    else if (fam == "poisson") poisson(link = link)
    else if (fam == "Gamma") Gamma(link = link)
    else if (fam == "negative.binomial") MASS::negative.binomial(theta = theta, link = link)
    else stop(paste("unknown family", fam))
}

out <- list()
for (name in names(cases)) {
    cs <- cases[[name]]
    X <- do.call(rbind, lapply(cs$X, function(r) as.numeric(unlist(r))))
    y <- as.numeric(unlist(cs$y))
    n <- nrow(X); p <- ncol(X)
    w <- if (is.null(cs$weights)) rep(1, n) else as.numeric(unlist(cs$weights))
    off <- if (is.null(cs$offset)) rep(0, n) else as.numeric(unlist(cs$offset))
    theta <- if (is.null(cs$theta)) NA else as.numeric(cs$theta)
    fam <- make_family(cs$family, cs$link, theta)

    fit <- glm.fit(X, y, weights = w, offset = off, family = fam,
                   intercept = FALSE,
                   control = glm.control(epsilon = 1e-12, maxit = 200))

    coefs <- as.numeric(fit$coefficients)
    mu <- as.numeric(fit$fitted.values)
    eta <- as.numeric(fit$linear.predictors)
    dev <- fit$deviance
    null_dev <- fit$null.deviance
    rank <- fit$rank
    df_resid <- n - rank
    # Fixed-dispersion families: binomial, poisson, and negative.binomial with
    # known theta (MASS::summary.negbin uses dispersion = 1). Gaussian/Gamma
    # estimate the dispersion as dev/df.
    fixed_disp <- cs$family %in% c("binomial", "poisson", "negative.binomial")
    disp <- if (fixed_disp) 1.0 else dev / df_resid

    # Standard errors from the final-iteration QR (R's summary path).
    R <- qr.R(fit$qr)
    Rinv <- backsolve(R, diag(p))
    XtWXinv <- Rinv %*% t(Rinv)
    se <- sqrt(disp * diag(XtWXinv))
    piv <- fit$qr$pivot
    se_un <- rep(NA_real_, p); se_un[piv] <- se
    coef_un <- rep(NA_real_, p); coef_un[piv] <- coefs

    # For gaussian, also capture R's weighted R² from lm() — an independent
    # reference for the OLS/WLS path's r_squared (the GLM null deviance is a
    # different quantity, mu=0 under intercept=FALSE, so it is not compared).
    r_squared <- NA_real_
    if (cs$family == "gaussian") {
        df <- as.data.frame(X[, -1, drop = FALSE])
        names(df) <- paste0("x", seq_len(ncol(X) - 1))
        df$y <- y
        form <- as.formula(paste("y ~", paste(names(df)[names(df) != "y"], collapse = " + ")))
        lmfit <- lm(form, data = df, weights = w, offset = off)
        r_squared <- summary(lmfit)$r.squared
    }

    out[[name]] <- list(
        family = cs$family, link = cs$link,
        coefficients = as.numeric(coef_un),
        r_squared = r_squared,
        standard_errors = as.numeric(se_un),
        fitted_values = mu,
        linear_predictor = eta,
        deviance = dev,
        null_deviance = null_dev,
        aic = fit$aic,
        dispersion = disp,
        rank = rank,
        df_residual = df_resid,
        n_iter = fit$iter,
        converged = fit$converged
    )
    cat(sprintf("%-26s dev=%.8g null=%.8g aic=%.8g iter=%d\n",
                name, dev, null_dev, fit$aic, fit$iter))
}

writeLines(toJSON(out, auto_unbox = TRUE, digits = 17, pretty = TRUE),
           file.path(fixtures_dir, "weights_offset_r_results.json"))
cat("Wrote weights_offset_r_results.json\n")
