#!/usr/bin/env Rscript
# =============================================================================
# Generate R `mice` reference fixtures for CATEGORICAL imputation validation.
#
# Imputes a fixed mixed-type dataset (mice_categorical_data.csv) with R mice
# using logreg (binary), polyreg (unordered factor) and polr (ordered factor),
# and records the marginal distribution of imputed category codes per column.
# The Python suite (test_r_validation_categorical.py) checks that pystatistics'
# imputed category proportions match R's distributionally.
#
# Pinned: R 4.3.3, mice 3.19.0.
# Run from repo root:
#   Rscript tests/mice/references/generate_categorical_fixtures.R
# =============================================================================

suppressPackageStartupMessages({
  library(mice)
  library(jsonlite)
})

PINNED_MICE_VERSION <- "3.19.0"
actual <- as.character(packageVersion("mice"))
if (actual != PINNED_MICE_VERSION) {
  stop(sprintf("mice version mismatch: pinned %s, installed %s",
               PINNED_MICE_VERSION, actual))
}

ref_dir <- dirname(sub("--file=", "",
                       grep("--file=", commandArgs(FALSE), value = TRUE)[1]))
if (length(ref_dir) == 0 || is.na(ref_dir) || ref_dir == "") {
  ref_dir <- "tests/mice/references"
}

df <- read.csv(file.path(ref_dir, "mice_categorical_data.csv"))
# Codes are 0-based integers in the CSV; convert to factors. Keeping the code
# strings as factor labels lets us map imputed labels straight back to codes.
df$bin <- factor(df$bin)
df$nom <- factor(df$nom)
df$ord <- ordered(df$ord)

M <- 50L
MAXIT <- 20L
SEED <- 2024L

meth <- make.method(df)
meth["bin"] <- "logreg"
meth["nom"] <- "polyreg"
meth["ord"] <- "polr"

mids <- mice(df, m = M, maxit = MAXIT, method = meth, seed = SEED,
             printFlag = FALSE)

# Marginal proportion of each category among the imputed cells (pooled over m).
prop_table <- function(col, levels_chr) {
  vals <- as.character(unlist(mids$imp[[col]]))
  tab <- table(factor(vals, levels = levels_chr))
  as.numeric(tab) / sum(tab)
}

result <- list(
  meta = list(
    mice_version = actual, m = M, maxit = MAXIT, seed = SEED,
    columns = c("bin", "nom", "ord")
  ),
  bin = list(levels = c("0", "1"),
             proportions = prop_table("bin", c("0", "1"))),
  nom = list(levels = c("0", "1", "2"),
             proportions = prop_table("nom", c("0", "1", "2"))),
  ord = list(levels = c("0", "1", "2", "3"),
             proportions = prop_table("ord", c("0", "1", "2", "3")))
)

out_path <- file.path(ref_dir, "mice_categorical_reference.json")
write_json(result, out_path, auto_unbox = TRUE, digits = 10, pretty = TRUE)
cat("Wrote", out_path, "\n")
