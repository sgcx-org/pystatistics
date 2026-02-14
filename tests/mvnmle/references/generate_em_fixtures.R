#!/usr/bin/env Rscript
# Generate EM reference fixtures for pystatistics/mvnmle validation.
# Uses R's norm::em.norm() as the reference implementation.
#
# Usage: Rscript generate_em_fixtures.R

library(norm)
library(jsonlite)

cat("Generating EM fixtures...\n")

# =====================================================================
# Apple dataset (18 x 2, from mvnmle package)
# =====================================================================
apple <- matrix(c(
   8, 59,
   6, 58,
  11, 56,
  22, 53,
  14, 50,
  17, 45,
  18, 43,
  24, 42,
  19, 39,
  23, 38,
  26, 30,
  40, 27,
   4, NA,
   4, NA,
   5, NA,
   6, NA,
   8, NA,
  10, NA
), ncol = 2, byrow = TRUE)

s_apple <- prelim.norm(apple)
# Use tight criterion for high-precision reference
th_apple <- em.norm(s_apple, criterion = 1e-8, maxits = 100000)
p_apple <- getparam.norm(s_apple, th_apple)

# Compute log-likelihood at converged parameters
# norm doesn't directly return loglik, so we compute it manually
# using the observed-data log-likelihood formula
compute_loglik <- function(data, mu, sigma) {
  n <- nrow(data)
  p <- ncol(data)
  loglik <- 0

  # Group by missingness pattern
  patterns <- apply(!is.na(data), 1, function(x) paste(as.integer(x), collapse=""))
  unique_patterns <- unique(patterns)

  for (pat in unique_patterns) {
    rows <- which(patterns == pat)
    n_k <- length(rows)
    obs_mask <- !is.na(data[rows[1], ])
    obs_idx <- which(obs_mask)
    p_k <- length(obs_idx)

    if (p_k == 0) next

    mu_o <- mu[obs_idx]
    sigma_oo <- sigma[obs_idx, obs_idx, drop = FALSE]

    # Log-likelihood contribution
    logdet <- determinant(sigma_oo, logarithm = TRUE)$modulus[1]
    sigma_oo_inv <- solve(sigma_oo)

    for (i in rows) {
      x_o <- data[i, obs_idx]
      diff <- x_o - mu_o
      quad <- as.numeric(t(diff) %*% sigma_oo_inv %*% diff)
      loglik <- loglik - 0.5 * (p_k * log(2 * pi) + logdet + quad)
    }
  }

  return(loglik)
}

loglik_apple <- compute_loglik(apple, p_apple$mu, p_apple$sigma)

apple_result <- list(
  muhat = as.vector(p_apple$mu),
  sigmahat = p_apple$sigma,
  loglik = loglik_apple
)

write_json(apple_result, "apple_em_reference.json",
           digits = 17, auto_unbox = TRUE, matrix = "rowmajor")
cat(sprintf("Apple: mu = [%.6f, %.6f], loglik = %.10f\n",
            p_apple$mu[1], p_apple$mu[2], loglik_apple))

# =====================================================================
# Missvals dataset (13 x 5, from mvnmle package)
# =====================================================================
missvals <- matrix(c(
   6,  56,  11,  20,  97,
   4,  47,   5,  29,  99,
   5,  57,   2,  NA,  97,
  10,  NA,   4,  NA, 101,
  NA,  NA,  13,  NA,  92,
   3,  60,  17,  23,  95,
   7,  NA,  15,  25,  91,
   5,  NA,  16,  33,  88,
  NA,  51,  NA,  27,  89,
  12,  52,  11,  NA,  NA,
  NA,  37,  NA,  NA, 102,
  NA,  NA,  NA,  NA,  95,
  NA,  40,  NA,  NA,  NA
), ncol = 5, byrow = TRUE)

s_missvals <- prelim.norm(missvals)
th_missvals <- em.norm(s_missvals, criterion = 1e-8, maxits = 100000)
p_missvals <- getparam.norm(s_missvals, th_missvals)

loglik_missvals <- compute_loglik(missvals, p_missvals$mu, p_missvals$sigma)

missvals_result <- list(
  muhat = as.vector(p_missvals$mu),
  sigmahat = p_missvals$sigma,
  loglik = loglik_missvals
)

write_json(missvals_result, "missvals_em_reference.json",
           digits = 17, auto_unbox = TRUE, matrix = "rowmajor")
cat(sprintf("Missvals: loglik = %.10f\n", loglik_missvals))

cat("Done! Fixture files written.\n")
