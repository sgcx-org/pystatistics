#!/usr/bin/env Rscript
#
# Compute R reference values for descriptive statistics fixtures.
#
# For each desc_*_meta.json, loads the CSV data and computes:
# - mean, var, sd (column-wise)
# - cov (everything, complete.obs, pairwise.complete.obs)
# - cor Pearson/Spearman/Kendall (everything, complete.obs, pairwise.complete.obs)
# - quantiles types 1-9
# - summary
# - skewness and kurtosis (e1071 type 2, bias-adjusted)
#
# Run from pystatistics root:
#   Rscript tests/fixtures/run_r_descriptive_validation.R

library(jsonlite)
library(e1071)

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
    fixtures_dir <- paste0(dirname(script_path), "/")
} else {
    fixtures_dir <- "tests/fixtures/"
}

# Discover desc_ fixtures
meta_files <- Sys.glob(paste0(fixtures_dir, "desc_*_meta.json"))

if (length(meta_files) == 0) {
    cat("No desc_*_meta.json files found in", fixtures_dir, "\n")
    cat("Run: python tests/fixtures/generate_descriptive_fixtures.py first\n")
    quit(status = 1)
}

for (meta_file in meta_files) {
    meta <- fromJSON(meta_file)
    name <- meta$name
    cat("Processing:", name, "\n")

    # Load CSV
    csv_file <- sub("_meta\\.json$", ".csv", meta_file)
    data <- as.matrix(read.csv(csv_file))
    n <- nrow(data)
    p <- ncol(data)
    has_nan <- any(is.na(data))

    results <- list(
        name = name,
        n = n,
        p = p,
        has_nan = has_nan
    )

    # ---- Column-wise statistics ----
    # With NaN propagation (use='everything')
    results$mean_everything <- as.numeric(colMeans(data))
    results$var_everything <- as.numeric(apply(data, 2, var))
    results$sd_everything <- as.numeric(apply(data, 2, sd))

    # With complete.obs (listwise deletion)
    complete_rows <- complete.cases(data)
    if (sum(complete_rows) >= 1) {
        clean <- data[complete_rows, , drop = FALSE]
        results$mean_complete <- as.numeric(colMeans(clean))
        results$var_complete <- as.numeric(apply(clean, 2, var))
        results$sd_complete <- as.numeric(apply(clean, 2, sd))
        results$n_complete <- sum(complete_rows)
    }

    # ---- Covariance ----
    results$cov_everything <- as.numeric(cov(data, use = "everything"))
    if (sum(complete_rows) >= 2) {
        results$cov_complete <- as.numeric(cov(data, use = "complete.obs"))
    }
    results$cov_pairwise <- as.numeric(cov(data, use = "pairwise.complete.obs"))

    # ---- Pearson correlation ----
    results$cor_pearson_everything <- as.numeric(cor(data, method = "pearson", use = "everything"))
    if (sum(complete_rows) >= 2) {
        results$cor_pearson_complete <- as.numeric(cor(data, method = "pearson", use = "complete.obs"))
    }
    results$cor_pearson_pairwise <- as.numeric(cor(data, method = "pearson", use = "pairwise.complete.obs"))

    # ---- Spearman correlation ----
    tryCatch({
        results$cor_spearman_everything <- as.numeric(cor(data, method = "spearman", use = "everything"))
    }, error = function(e) {
        results$cor_spearman_everything <<- NULL
    })
    if (sum(complete_rows) >= 2) {
        tryCatch({
            results$cor_spearman_complete <- as.numeric(cor(data, method = "spearman", use = "complete.obs"))
        }, error = function(e) {
            results$cor_spearman_complete <<- NULL
        })
    }
    tryCatch({
        results$cor_spearman_pairwise <- as.numeric(cor(data, method = "spearman", use = "pairwise.complete.obs"))
    }, error = function(e) {
        results$cor_spearman_pairwise <<- NULL
    })

    # ---- Kendall correlation ----
    tryCatch({
        results$cor_kendall_everything <- as.numeric(cor(data, method = "kendall", use = "everything"))
    }, error = function(e) {
        results$cor_kendall_everything <<- NULL
    })
    if (sum(complete_rows) >= 2) {
        tryCatch({
            results$cor_kendall_complete <- as.numeric(cor(data, method = "kendall", use = "complete.obs"))
        }, error = function(e) {
            results$cor_kendall_complete <<- NULL
        })
    }
    tryCatch({
        results$cor_kendall_pairwise <- as.numeric(cor(data, method = "kendall", use = "pairwise.complete.obs"))
    }, error = function(e) {
        results$cor_kendall_pairwise <<- NULL
    })

    # ---- Quantiles (all 9 types) ----
    probs <- c(0, 0.25, 0.5, 0.75, 1.0)
    results$quantile_probs <- probs

    for (t in 1:9) {
        key <- paste0("quantiles_type", t)
        if (has_nan) {
            # Use complete.obs for quantile computation with NaN data
            qvals <- apply(clean, 2, function(col) quantile(col, probs, type = t))
        } else {
            qvals <- apply(data, 2, function(col) quantile(col, probs, type = t))
        }
        results[[key]] <- as.numeric(qvals)
    }

    # ---- Summary ----
    # R summary() gives Min, 1st Qu, Median, Mean, 3rd Qu, Max
    if (has_nan) {
        sum_vals <- apply(clean, 2, summary)
    } else {
        sum_vals <- apply(data, 2, summary)
    }
    results$summary <- as.numeric(sum_vals)

    # ---- Skewness and Kurtosis (e1071 type 2, bias-adjusted) ----
    if (has_nan) {
        results$skewness <- as.numeric(apply(clean, 2, function(col) skewness(col, type = 2)))
        results$kurtosis <- as.numeric(apply(clean, 2, function(col) kurtosis(col, type = 2)))
    } else {
        results$skewness <- as.numeric(apply(data, 2, function(col) skewness(col, type = 2)))
        results$kurtosis <- as.numeric(apply(data, 2, function(col) kurtosis(col, type = 2)))
    }

    # Save results
    results_file <- sub("_meta\\.json$", "_r_results.json", meta_file)
    write_json(results, results_file, digits = 17, auto_unbox = TRUE, pretty = TRUE)
    cat("  Saved:", results_file, "\n")
}

cat("\nDone! All R reference values computed.\n")
