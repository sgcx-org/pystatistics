#!/usr/bin/env Rscript
#
# Compute R reference values for hypothesis test fixtures.
#
# Usage:
#   Rscript tests/fixtures/run_r_hypothesis_validation.R
#
# Reads htest_*_meta.json files, runs the corresponding R test,
# and writes htest_*_r_results.json files with full precision.

suppressPackageStartupMessages({
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
  fixtures_dir <- dirname(script_path)
} else {
  fixtures_dir <- "tests/fixtures"
}

cat("Looking for fixtures in:", fixtures_dir, "\n")

meta_files <- sort(Sys.glob(file.path(fixtures_dir, "htest_*_meta.json")))
if (length(meta_files) == 0) {
  stop("No htest_*_meta.json files found. Run generate_hypothesis_fixtures.py first.")
}

cat(sprintf("Found %d hypothesis test fixtures\n\n", length(meta_files)))

for (meta_file in meta_files) {
  name <- sub("_meta\\.json$", "", basename(meta_file))
  cat(sprintf("Processing %s...\n", name))

  meta <- fromJSON(meta_file)
  test_type <- meta$test
  data <- meta$data
  params <- meta$params

  result <- tryCatch({
    if (test_type == "t.test") {
      x <- data$x
      y <- if (!is.null(data$y)) data$y else NULL
      args <- list(x = x)
      if (!is.null(y)) args$y <- y
      if (!is.null(params$mu)) args$mu <- params$mu
      if (!is.null(params$paired)) args$paired <- params$paired
      if (!is.null(params[["var.equal"]])) args$var.equal <- params[["var.equal"]]
      if (!is.null(params$alternative)) args$alternative <- params$alternative
      if (!is.null(params[["conf.level"]])) args$conf.level <- params[["conf.level"]]
      do.call(t.test, args)

    } else if (test_type == "chisq.test") {
      if (!is.null(data$table)) {
        tbl <- as.matrix(data$table)
        args <- list(x = tbl)
      } else {
        args <- list(x = data$x)
      }
      if (!is.null(params$correct)) args$correct <- params$correct
      if (!is.null(params$p)) args$p <- params$p
      if (!is.null(params[["simulate.p.value"]])) args$simulate.p.value <- params[["simulate.p.value"]]
      if (!is.null(params$B)) args$B <- params$B
      if (!is.null(params[["rescale.p"]])) args$rescale.p <- params[["rescale.p"]]
      do.call(chisq.test, args)

    } else if (test_type == "fisher.test") {
      tbl <- as.matrix(data$table)
      args <- list(x = tbl)
      if (!is.null(params$alternative)) args$alternative <- params$alternative
      if (!is.null(params[["conf.level"]])) args$conf.level <- params[["conf.level"]]
      if (!is.null(params[["conf.int"]])) args$conf.int <- params[["conf.int"]]
      if (!is.null(params[["simulate.p.value"]])) args$simulate.p.value <- params[["simulate.p.value"]]
      if (!is.null(params$B)) args$B <- params$B
      do.call(fisher.test, args)

    } else if (test_type == "wilcox.test") {
      x <- data$x
      y <- if (!is.null(data$y)) data$y else NULL
      args <- list(x = x)
      if (!is.null(y)) args$y <- y
      if (!is.null(params$mu)) args$mu <- params$mu
      if (!is.null(params$alternative)) args$alternative <- params$alternative
      if (!is.null(params$paired)) args$paired <- params$paired
      if (!is.null(params$exact)) args$exact <- params$exact
      if (!is.null(params$correct)) args$correct <- params$correct
      if (!is.null(params[["conf.int"]])) args$conf.int <- params[["conf.int"]]
      if (!is.null(params[["conf.level"]])) args$conf.level <- params[["conf.level"]]
      suppressWarnings(do.call(wilcox.test, args))

    } else if (test_type == "ks.test") {
      x <- data$x
      if (!is.null(data$y)) {
        y <- data$y
        args <- list(x = x, y = y)
        if (!is.null(params$alternative)) args$alternative <- params$alternative
        suppressWarnings(do.call(ks.test, args))
      } else {
        dist <- params$distribution
        args <- list(x = x, y = dist)
        # Add distribution params (mean, sd, rate, etc.)
        dist_params <- params[!names(params) %in% c("distribution", "alternative")]
        for (pn in names(dist_params)) {
          args[[pn]] <- dist_params[[pn]]
        }
        if (!is.null(params$alternative)) args$alternative <- params$alternative
        suppressWarnings(do.call(ks.test, args))
      }

    } else if (test_type == "prop.test") {
      x <- data$x
      n <- data$n
      args <- list(x = x, n = n)
      if (!is.null(params$p)) args$p <- params$p
      if (!is.null(params$alternative)) args$alternative <- params$alternative
      if (!is.null(params[["conf.level"]])) args$conf.level <- params[["conf.level"]]
      if (!is.null(params$correct)) args$correct <- params$correct
      do.call(prop.test, args)

    } else if (test_type == "var.test") {
      x <- data$x
      y <- data$y
      args <- list(x = x, y = y)
      if (!is.null(params$ratio)) args$ratio <- params$ratio
      if (!is.null(params$alternative)) args$alternative <- params$alternative
      if (!is.null(params[["conf.level"]])) args$conf.level <- params[["conf.level"]]
      do.call(var.test, args)

    } else {
      stop(sprintf("Unknown test type: %s", test_type))
    }
  }, error = function(e) {
    cat(sprintf("  ERROR: %s\n", e$message))
    NULL
  })

  if (is.null(result)) next

  # Extract htest components
  out <- list()

  # Statistic
  if (!is.null(result$statistic)) {
    out$statistic <- as.numeric(result$statistic)
    out$statistic_name <- names(result$statistic)
  } else {
    out$statistic <- NULL
    out$statistic_name <- NULL
  }

  # Parameter (df, etc.)
  if (!is.null(result$parameter)) {
    out$parameter <- as.list(result$parameter)
    names(out$parameter) <- names(result$parameter)
  } else {
    out$parameter <- NULL
  }

  # p-value
  out$p_value <- result$p.value

  # Confidence interval
  if (!is.null(result$conf.int)) {
    out$conf_int <- as.numeric(result$conf.int)
    out$conf_level <- attr(result$conf.int, "conf.level")
  } else {
    out$conf_int <- NULL
    out$conf_level <- NULL
  }

  # Estimate
  if (!is.null(result$estimate)) {
    out$estimate <- as.list(result$estimate)
    names(out$estimate) <- names(result$estimate)
  } else {
    out$estimate <- NULL
  }

  # Null value
  if (!is.null(result$null.value)) {
    out$null_value <- as.list(result$null.value)
    names(out$null_value) <- names(result$null.value)
  } else {
    out$null_value <- NULL
  }

  # Alternative and method
  out$alternative <- result$alternative
  out$method <- result$method
  out$data_name <- result$data.name

  # Write results
  out_file <- file.path(fixtures_dir, sprintf("%s_r_results.json", name))
  write(toJSON(out, auto_unbox = TRUE, digits = 17, na = "string",
               null = "null", pretty = TRUE),
        file = out_file)
  cat(sprintf("  -> %s\n", basename(out_file)))
}

cat("\nDone!\n")
