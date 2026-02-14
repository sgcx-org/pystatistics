#!/usr/bin/env Rscript
#
# R Reference Validation for PyStatistics GLM
#
# Runs glm.fit() on all GLM fixture CSVs and saves results with maximum precision.
#
# IMPORTANT: We use glm.fit() (raw matrix interface) to match PyStatistics exactly.
# Since glm.fit() returns a list (not a glm S3 object), we compute residual types
# manually using the same formulas as R's residuals.glm().
#
# Run from /path/to/pystatistics:
#   Rscript tests/fixtures/run_r_glm_validation.R
#

library(jsonlite)

options(digits = 22)

fixtures_dir <- "tests/fixtures"
meta_files <- list.files(fixtures_dir, pattern = "^glm_.*_meta\\.json$", full.names = TRUE)

cat("Running R GLM validation on", length(meta_files), "fixtures...\n\n")

for (meta_file in meta_files) {
    # Load metadata
    meta <- fromJSON(meta_file)
    fixture_name <- meta[["name"]]

    cat("Processing:", fixture_name, "\n")
    cat("  Family:", meta[["family"]], "\n")

    # Read data
    csv_file <- file.path(fixtures_dir, paste0(fixture_name, ".csv"))
    data <- read.csv(csv_file)

    # Extract X and y
    y <- as.numeric(data[["y"]])
    X <- as.matrix(data[, setdiff(names(data), "y")])

    n <- nrow(X)
    p <- ncol(X)

    # Select family
    if (meta[["family"]] == "gaussian") {
        fam <- gaussian()
    } else if (meta[["family"]] == "binomial") {
        fam <- binomial()
    } else if (meta[["family"]] == "poisson") {
        fam <- poisson()
    } else {
        cat("  SKIPPING: unknown family", meta[["family"]], "\n\n")
        next
    }

    # Fit using glm.fit (raw matrix interface, no formula)
    # intercept=FALSE because X already contains the intercept column
    fit <- glm.fit(X, y, family = fam, intercept = FALSE,
                   control = glm.control(epsilon = 1e-8, maxit = 25))

    # Extract results
    coefficients <- as.numeric(fit[["coefficients"]])
    mu <- as.numeric(fit[["fitted.values"]])
    eta <- as.numeric(fit[["linear.predictors"]])

    deviance <- fit[["deviance"]]
    null_deviance <- fit[["null.deviance"]]
    aic <- fit[["aic"]]

    rank <- fit[["rank"]]
    df_residual <- n - rank
    df_null <- n - 1
    n_iter <- fit[["iter"]]
    converged <- fit[["converged"]]

    # Dispersion
    if (meta[["family"]] %in% c("binomial", "poisson")) {
        dispersion <- 1.0
    } else {
        dispersion <- deviance / df_residual
    }

    # ====================================================================
    # Compute residual types manually (glm.fit returns a list, not S3 obj)
    # These formulas match R's residuals.glm()
    # ====================================================================

    # Response residuals: y - mu
    resid_response <- y - mu

    # Pearson residuals: (y - mu) / sqrt(V(mu))
    var_mu <- fam[["variance"]](mu)
    resid_pearson <- (y - mu) / sqrt(var_mu)

    # Deviance residuals: sign(y - mu) * sqrt(d_i)
    # where d_i is the unit deviance contribution
    # Use family$dev.resids to get wt * unit_deviance
    wt <- rep(1, n)
    dev_resids_raw <- fam[["dev.resids"]](y, mu, wt)  # wt * d_i
    resid_deviance <- sign(y - mu) * sqrt(pmax(dev_resids_raw, 0))

    # Working residuals: (y - mu) / (dmu/deta)
    # dmu/deta = fam$mu.eta(eta)
    dmu_deta <- fam[["mu.eta"]](eta)
    resid_working <- (y - mu) / dmu_deta

    # ====================================================================
    # Standard errors via QR decomposition from the final IRLS iteration
    # ====================================================================
    qr_fit <- fit[["qr"]]
    R <- qr.R(qr_fit)
    R_inv <- backsolve(R, diag(p))
    XtWX_inv <- R_inv %*% t(R_inv)
    se <- sqrt(dispersion * diag(XtWX_inv))

    # Unpivot: glm.fit uses pivoting
    pivot <- qr_fit[["pivot"]]
    se_unpivoted <- rep(NA_real_, p)
    se_unpivoted[pivot] <- se

    coef_unpivoted <- rep(NA_real_, p)
    coef_unpivoted[pivot] <- coefficients

    # Test statistics (z for binomial/poisson, t for gaussian)
    if (meta[["family"]] %in% c("binomial", "poisson")) {
        test_stats <- coef_unpivoted / se_unpivoted
        p_values <- 2 * pnorm(abs(test_stats), lower.tail = FALSE)
    } else {
        test_stats <- coef_unpivoted / se_unpivoted
        p_values <- 2 * pt(abs(test_stats), df = df_residual, lower.tail = FALSE)
    }

    results <- list(
        fixture = fixture_name,
        family = meta[["family"]],
        link = fam[["link"]],
        method = "glm.fit (raw matrix, matching PyStatistics IRLS)",

        coefficients = coef_unpivoted,
        standard_errors = se_unpivoted,
        test_statistics = as.numeric(test_stats),
        p_values = as.numeric(p_values),

        fitted_values = mu,
        linear_predictor = eta,

        residuals_deviance = as.numeric(resid_deviance),
        residuals_pearson = as.numeric(resid_pearson),
        residuals_working = as.numeric(resid_working),
        residuals_response = as.numeric(resid_response),

        deviance = deviance,
        null_deviance = null_deviance,
        aic = aic,
        dispersion = dispersion,

        rank = rank,
        df_residual = df_residual,
        df_null = df_null,
        n_iter = n_iter,
        converged = converged
    )

    output_file <- file.path(fixtures_dir, paste0(fixture_name, "_r_results.json"))
    json_str <- toJSON(results, auto_unbox = TRUE, digits = 17, pretty = TRUE)
    writeLines(json_str, output_file)

    cat("  Deviance:", format(deviance, digits = 10), "\n")
    cat("  Null deviance:", format(null_deviance, digits = 10), "\n")
    cat("  AIC:", format(aic, digits = 10), "\n")
    cat("  Converged:", converged, "(", n_iter, "iterations)\n")
    cat("  Saved to:", output_file, "\n\n")
}

cat("R GLM validation complete!\n")
