#!/usr/bin/env Rscript
# =============================================================================
# Generate R `mice` reference fixtures for distributional validation.
#
# This script imputes a FIXED incomplete dataset (mice_validation_data.csv,
# produced by pystatistics.mice.datasets so both languages see identical input)
# with R's mice package and writes summary fixtures to JSON. The Python test
# suite (test_r_validation.py) then checks that pystatistics' imputations match
# R's *distributionally* — same defaults, same algorithm, matching statistical
# behaviour across many imputations — NOT by reproducing R's RNG stream.
#
# Pinned for reproducibility: regenerate only with the version below.
#   - R:    4.3.3
#   - mice: 3.19.0
#
# Run from the repo root:
#   Rscript tests/mice/references/generate_mice_fixtures.R
# =============================================================================

suppressPackageStartupMessages({
  library(mice)
  library(jsonlite)
})

PINNED_MICE_VERSION <- "3.19.0"
actual <- as.character(packageVersion("mice"))
if (actual != PINNED_MICE_VERSION) {
  stop(sprintf(
    "mice version mismatch: fixtures are pinned to %s but %s is installed.
     Install the pinned version or update PINNED_MICE_VERSION (and the Python
     tolerance assumptions) deliberately.",
    PINNED_MICE_VERSION, actual))
}

ref_dir <- dirname(sub("--file=", "",
                       grep("--file=", commandArgs(FALSE), value = TRUE)[1]))
if (length(ref_dir) == 0 || is.na(ref_dir) || ref_dir == "") {
  ref_dir <- "tests/mice/references"
}

data_path <- file.path(ref_dir, "mice_validation_data.csv")
incomplete <- read.csv(data_path)
stopifnot(all(colnames(incomplete) == c("x0", "x1", "x2")))

M <- 50L
MAXIT <- 20L
SEED <- 12345L

# Columns that carry missing values, in column order (matches our visit seq).
incomplete_cols <- names(incomplete)[sapply(incomplete, function(v) any(is.na(v)))]

# ---- helper: collect all imputed values for a column across the m imputations
collect_imputed <- function(mids, col) {
  # mids$imp[[col]] is a (n_missing x m) data frame of imputed values.
  vals <- as.numeric(as.matrix(mids$imp[[col]]))
  vals[is.finite(vals)]
}

summarise_method <- function(method_name) {
  mids <- mice(incomplete, m = M, maxit = MAXIT, method = method_name,
               seed = SEED, printFlag = FALSE)

  per_col <- list()
  for (col in incomplete_cols) {
    vals <- collect_imputed(mids, col)
    observed <- incomplete[[col]][!is.na(incomplete[[col]])]
    per_col[[col]] <- list(
      n_missing   = sum(is.na(incomplete[[col]])),
      imputed     = vals,             # all m*n_missing imputed values, flattened
      imputed_mean = mean(vals),
      imputed_sd   = sd(vals),
      observed_mean = mean(observed),
      observed_sd   = sd(observed)
    )
  }

  # A pooled analysis: regress x0 on x1 + x2 on each completed dataset and pool
  # with Rubin's rules. Gives Python a citable target for pool().
  fit <- with(mids, lm(x0 ~ x1 + x2))
  pooled <- summary(pool(fit))
  pool_out <- list(
    term     = as.character(pooled$term),
    estimate = pooled$estimate,
    std.error = pooled$std.error,
    df       = pooled$df
  )

  list(per_col = per_col, pooled_lm_x0 = pool_out)
}

result <- list(
  meta = list(
    mice_version = actual,
    R_version    = as.character(getRversion()),
    m = M, maxit = MAXIT, seed = SEED,
    incomplete_columns = incomplete_cols
  ),
  pmm  = summarise_method("pmm"),
  norm = summarise_method("norm")
)

out_path <- file.path(ref_dir, "mice_reference.json")
write_json(result, out_path, auto_unbox = TRUE, digits = 12, pretty = TRUE)
cat("Wrote", out_path, "\n")
