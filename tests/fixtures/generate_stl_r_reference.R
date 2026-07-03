# Provenance: generates tests/fixtures/stl_r_reference.json.
# Generated with R 4.5.2 on 2026-07-03. Run from the repo root:
#   Rscript tests/fixtures/generate_stl_r_reference.R
# Generate R stats::stl reference fixtures for pystatistics STL parity tests.
# Every case records the input series, the exact parameters fed to stl(),
# the windows/degrees/jumps R actually used, and the output components.
suppressMessages(library(jsonlite))

run_case <- function(name, x, freq, args) {
  xt <- ts(as.numeric(x), frequency = freq)
  fit <- do.call(stats::stl, c(list(x = xt), args))
  ts_mat <- fit$time.series
  list(
    name = name,
    x = as.numeric(xt),
    period = as.integer(freq),
    args = args,
    win = as.list(fit$win),
    deg = as.list(fit$deg),
    jump = as.list(fit$jump),
    inner = fit$inner,
    outer = fit$outer,
    seasonal = as.numeric(ts_mat[, "seasonal"]),
    trend = as.numeric(ts_mat[, "trend"]),
    remainder = as.numeric(ts_mat[, "remainder"]),
    weights = as.numeric(fit$weights)
  )
}

cases <- list(
  run_case("co2_periodic", co2, 12, list(s.window = "periodic")),
  run_case("co2_s13_d0_t23", co2, 12,
           list(s.window = 13, s.degree = 0, t.window = 23)),
  run_case("co2_s13_d1_t23", co2, 12,
           list(s.window = 13, s.degree = 1, t.window = 23)),
  run_case("co2_s35_jump1", co2, 12,
           list(s.window = 35, s.degree = 0,
                s.jump = 1, t.jump = 1, l.jump = 1)),
  run_case("co2_periodic_robust", co2, 12,
           list(s.window = "periodic", robust = TRUE)),
  run_case("co2_n461_s13", co2[1:461], 12,
           list(s.window = 13, s.degree = 0)),
  run_case("airpassengers_s11", AirPassengers, 12, list(s.window = 11)),
  run_case("airpassengers_s11_robust", AirPassengers, 12,
           list(s.window = 11, robust = TRUE)),
  run_case("sunspots_periodic", sunspots, 12, list(s.window = "periodic")),
  run_case("sunspots_s25", sunspots, 12,
           list(s.window = 25, s.degree = 0)),
  run_case("lynx_f10_s7", lynx, 10, list(s.window = 7, s.degree = 0)),
  run_case("nile_f5_periodic", Nile, 5, list(s.window = "periodic")),
  run_case("nile_f5_s7", Nile, 5, list(s.window = 7, s.degree = 1)),
  run_case("co2_n467_periodic_robust", co2[1:467], 12,
           list(s.window = "periodic", robust = TRUE)),
  run_case("airpassengers_n143_s11_robust", AirPassengers[1:143], 12,
           list(s.window = 11, robust = TRUE))
)
names(cases) <- vapply(cases, function(z) z$name, character(1))

out <- toJSON(cases, digits = NA, auto_unbox = TRUE, pretty = FALSE)
writeLines(out, "tests/fixtures/stl_r_reference.json")
cat("Wrote", length(cases), "STL cases\n")
for (z in cases) {
  cat(sprintf("  %-26s n=%4d win=(%s,%s,%s) jump=(%s,%s,%s) inner=%d outer=%d\n",
              z$name, length(z$x), z$win$s, z$win$t, z$win$l,
              z$jump$s, z$jump$t, z$jump$l, z$inner, z$outer))
}
