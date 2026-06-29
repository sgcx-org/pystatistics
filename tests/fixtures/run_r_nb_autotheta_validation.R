#!/usr/bin/env Rscript
#
# R reference for PyStatistics negative-binomial auto-θ validation.
#
# Fits each case in nb_autotheta_cases.json with MASS::glm.nb (estimates θ by
# profile likelihood) and writes nb_autotheta_r_results.json. glm.nb uses a
# formula (intercept in the model), so the design matches PyStatistics' X =
# [1, x1, ...]. The null deviance is NOT recorded: glm.nb's null model uses the
# intercept-true convention (weighted-mean μ), which differs from PyStatistics'
# glm.fit(intercept=FALSE) convention; null-deviance parity is covered by the
# fixed-θ fixtures instead.
#
# Run from /path/to/pystatistics:
#   Rscript tests/fixtures/run_r_nb_autotheta_validation.R
#
suppressMessages(library(MASS))
library(jsonlite)
options(digits = 22)

fixtures_dir <- "tests/fixtures"
cases <- fromJSON(file.path(fixtures_dir, "nb_autotheta_cases.json"),
                  simplifyVector = FALSE)

out <- list()
for (name in names(cases)) {
    cs <- cases[[name]]
    X <- do.call(rbind, lapply(cs$X, function(r) as.numeric(unlist(r))))
    y <- as.numeric(unlist(cs$y))
    n <- nrow(X); p <- ncol(X)
    w <- if (is.null(cs$weights)) rep(1, n) else as.numeric(unlist(cs$weights))
    off <- if (is.null(cs$offset)) rep(0, n) else as.numeric(unlist(cs$offset))

    # Build a formula with an intercept; X[,1] is the ones column, so use the
    # remaining columns as named predictors.
    df <- as.data.frame(X[, -1, drop = FALSE])
    names(df) <- paste0("x", seq_len(p - 1))
    df$y <- y
    rhs <- paste(names(df)[names(df) != "y"], collapse = " + ")
    form <- as.formula(paste("y ~", rhs, "+ offset(off)"))

    fit <- glm.nb(form, data = df, weights = w,
                  control = glm.control(epsilon = 1e-12, maxit = 200))
    sm <- summary(fit)

    out[[name]] <- list(
        theta = fit$theta,
        se_theta = fit$SE.theta,
        coefficients = as.numeric(coef(fit)),
        standard_errors = as.numeric(sm$coefficients[, 2]),
        deviance = fit$deviance,
        aic = fit$aic,
        bic = as.numeric(BIC(fit)),
        n_iter = fit$iter,
        converged = fit$converged
    )
    cat(sprintf("%-24s theta=%.6f dev=%.6f aic=%.6f bic=%.6f\n",
                name, fit$theta, fit$deviance, fit$aic, BIC(fit)))
}

writeLines(toJSON(out, auto_unbox = TRUE, digits = 17, pretty = TRUE),
           file.path(fixtures_dir, "nb_autotheta_r_results.json"))
cat("Wrote nb_autotheta_r_results.json\n")
