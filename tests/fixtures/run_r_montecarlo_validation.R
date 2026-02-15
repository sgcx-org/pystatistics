#!/usr/bin/env Rscript
#
# R validation script for Monte Carlo module.
#
# Reads mc_*_meta.json fixtures, runs R's boot package, writes mc_*_r_results.json.
#
# Requirements:
#   install.packages("boot")
#   install.packages("jsonlite")
#
# Usage:
#   Rscript tests/fixtures/run_r_montecarlo_validation.R
#
# Note on seed correspondence:
#   R and NumPy use different RNG algorithms. We cannot get the same bootstrap
#   samples. Instead we validate:
#   - t0 (observed statistic) — must match exactly (deterministic)
#   - bias, SE — moderate tolerance (stochastic, but converges)
#   - CI endpoints — moderate tolerance (stochastic)
#   - permutation p-values — wide tolerance (stochastic)

suppressPackageStartupMessages({
  library(boot)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
  fixtures_dir <- dirname(script_path)
} else {
  fixtures_dir <- "tests/fixtures"
}

cat("Monte Carlo R validation\n")
cat("Fixtures dir:", fixtures_dir, "\n\n")

# Find all mc_*_meta.json files
meta_files <- sort(Sys.glob(file.path(fixtures_dir, "mc_*_meta.json")))

if (length(meta_files) == 0) {
  stop("No mc_*_meta.json files found. Run generate_montecarlo_fixtures.py first.")
}

for (meta_file in meta_files) {
  meta <- fromJSON(meta_file, simplifyVector = TRUE)
  name <- sub("_meta\\.json$", "", basename(meta_file))
  cat("Processing:", name, "\n")

  fixture_type <- meta$type

  if (fixture_type == "boot" || fixture_type == "boot_ci") {
    # -----------------------------------------------------------------------
    # Bootstrap
    # -----------------------------------------------------------------------
    data_vec <- as.numeric(meta$data)
    R <- as.integer(meta$R)
    sim_type <- meta$sim
    seed <- as.integer(meta$seed)

    # Determine which statistic to use from fixture description
    if (grepl("variance", tolower(meta$description))) {
      # Bootstrap for variance
      stat_fn <- function(d, i) var(d[i])
      stat_name <- "variance"
    } else if (grepl("median", tolower(meta$description))) {
      # Bootstrap for median
      stat_fn <- function(d, i) median(d[i])
      stat_name <- "median"
    } else {
      # Default: bootstrap for mean
      stat_fn <- function(d, i) mean(d[i])
      stat_name <- "mean"
    }

    # Run boot with R's seed
    set.seed(seed)
    boot_result <- boot(data_vec, stat_fn, R = R, sim = sim_type)

    # Compute reference values
    t0 <- as.numeric(boot_result$t0)
    t <- as.numeric(boot_result$t)  # vector of R replicates
    bias <- mean(t) - t0
    se <- sd(t)

    result <- list(
      fixture = name,
      statistic = stat_name,
      t0 = t0,
      bias = bias,
      se = se,
      R = R,
      sim = sim_type
    )

    # If this is a boot_ci fixture, also compute CIs
    if (fixture_type == "boot_ci") {
      conf_level <- meta$conf_level
      ci_types <- meta$ci_types

      ci_results <- list()

      # Map our type names to R's boot.ci type names
      r_type_map <- list(
        "normal" = "norm",
        "basic"  = "basic",
        "perc"   = "perc",
        "bca"    = "bca",
        "stud"   = "stud"
      )

      for (ci_type in ci_types) {
        tryCatch({
          r_ci_type <- r_type_map[[ci_type]]
          ci <- boot.ci(boot_result, conf = conf_level, type = r_ci_type)

          # Extract CI bounds depending on type.
          # R's boot.ci returns different structures per type:
          #   $normal is a 1x3 matrix: [conf, lower, upper]
          #   $basic, $percent, $bca, $student are 1x5 matrices:
          #     [conf, unused, unused, lower, upper]
          if (ci_type == "normal") {
            ci_bounds <- as.numeric(ci$normal[c(2, 3)])
          } else if (ci_type == "basic") {
            ci_bounds <- as.numeric(ci$basic[c(4, 5)])
          } else if (ci_type == "perc") {
            ci_bounds <- as.numeric(ci$percent[c(4, 5)])
          } else if (ci_type == "bca") {
            ci_bounds <- as.numeric(ci$bca[c(4, 5)])
          } else if (ci_type == "stud") {
            ci_bounds <- as.numeric(ci$student[c(4, 5)])
          }

          ci_results[[ci_type]] <- ci_bounds
        }, error = function(e) {
          cat("  Warning: CI type", ci_type, "failed:", e$message, "\n")
          ci_results[[ci_type]] <<- NULL
        })
      }

      result$conf_level <- conf_level
      result$ci <- ci_results
    }

    # Write results
    out_path <- file.path(fixtures_dir, paste0(name, "_r_results.json"))
    write_json(result, out_path, digits = 17, auto_unbox = TRUE, pretty = TRUE)
    cat("  Wrote", basename(out_path), "\n")

  } else if (fixture_type == "permutation") {
    # -----------------------------------------------------------------------
    # Permutation test
    # -----------------------------------------------------------------------
    x <- as.numeric(meta$x)
    y <- as.numeric(meta$y)
    R <- as.integer(meta$R)
    alternative <- meta$alternative
    seed <- as.integer(meta$seed)

    # Observed statistic: mean difference
    observed <- mean(x) - mean(y)

    # Run permutation test manually (matching our Phipson-Smyth correction)
    combined <- c(x, y)
    n1 <- length(x)
    n_total <- length(combined)
    set.seed(seed)

    perm_stats <- numeric(R)
    for (b in seq_len(R)) {
      shuffled <- sample(combined)
      perm_stats[b] <- mean(shuffled[1:n1]) - mean(shuffled[(n1+1):n_total])
    }

    # Phipson-Smyth p-value
    if (alternative == "two.sided") {
      count <- sum(abs(perm_stats) >= abs(observed))
    } else if (alternative == "greater") {
      count <- sum(perm_stats >= observed)
    } else if (alternative == "less") {
      count <- sum(perm_stats <= observed)
    }
    p_value <- (count + 1) / (R + 1)

    result <- list(
      fixture = name,
      observed_stat = observed,
      p_value = p_value,
      R = R,
      alternative = alternative,
      perm_stats_mean = mean(perm_stats),
      perm_stats_sd = sd(perm_stats)
    )

    out_path <- file.path(fixtures_dir, paste0(name, "_r_results.json"))
    write_json(result, out_path, digits = 17, auto_unbox = TRUE, pretty = TRUE)
    cat("  Wrote", basename(out_path), "\n")

  } else {
    cat("  Unknown fixture type:", fixture_type, "- skipping\n")
  }
}

cat("\nDone.\n")
