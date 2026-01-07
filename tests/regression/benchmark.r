# bench_multiple_sizes.R
bench_r <- function(n, p, trials=5) {
  X <- matrix(rnorm(n*p), n, p)
  y <- rnorm(n)
  
  # Warmup
  fit <- lm.fit(X, y)
  
  times <- numeric(trials)
  for (i in 1:trials) {
    times[i] <- system.time(fit <- lm.fit(X, y))["elapsed"]
  }
  
  median(times)
}

cat("R lm.fit() Benchmark\n")
cat(paste0(rep("=", 60), collapse=""), "\n")

for (config in list(c(1000, 10), c(10000, 50), c(100000, 100), 
                     c(500000, 200), c(1000000, 500))) {
  n <- config[1]
  p <- config[2]
  t <- bench_r(n, p)
  cat(sprintf("n=%7d, p=%4d: %8.2f ms\n", n, p, t * 1000))
}