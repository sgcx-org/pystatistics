# Provenance: generates tests/fixtures/ets_r_reference.json.
# Generated with R 4.5.2 + forecast 9.0.0 on 2026-07-03. Run from the repo root:
#   Rscript tests/fixtures/generate_ets_r_reference.R
# Generate forecast::ets reference fixtures:
#  (a) ZZZ auto-selection results per dataset (selected components + ICs)
#  (b) fixed-spec fits for the log-likelihood convention constant check
suppressMessages(library(jsonlite))
suppressMessages(library(forecast))

sel_case <- function(name, x, freq, model = "ZZZ") {
  xt <- ts(as.numeric(x), frequency = freq)
  fit <- forecast::ets(xt, model = model)
  list(
    name = name,
    x = as.numeric(xt),
    period = as.integer(freq),
    model_arg = model,
    method = fit$method,
    components = as.list(fit$components),  # error, trend, season, damped
    loglik = fit$loglik,
    aic = unname(fit$aic),
    aicc = unname(fit$aicc),
    bic = unname(fit$bic),
    n = length(xt)
  )
}

fixed_case <- function(name, x, freq, model, damped = FALSE) {
  xt <- ts(as.numeric(x), frequency = freq)
  fit <- forecast::ets(xt, model = model, damped = damped)
  list(
    name = name, x = as.numeric(xt), period = as.integer(freq),
    model_arg = model, damped = damped,
    method = fit$method, loglik = fit$loglik,
    aic = unname(fit$aic), aicc = unname(fit$aicc), bic = unname(fit$bic),
    alpha = unname(fit$par["alpha"]), n = length(xt)
  )
}

selection <- list(
  sel_case("airpassengers", AirPassengers, 12),
  sel_case("usaccdeaths", USAccDeaths, 12),
  sel_case("co2", co2, 12),
  sel_case("nile", Nile, 1),
  sel_case("wwwusage", WWWusage, 1),
  sel_case("lynx", lynx, 1),
  sel_case("diff_nile", diff(Nile), 1),          # has negative values
  sel_case("airpassengers_zzn", AirPassengers, 12, model = "ZZN"),
  sel_case("airpassengers_azz", AirPassengers, 12, model = "AZZ"),
  sel_case("airpassengers_mzz", AirPassengers, 12, model = "MZZ")
)
names(selection) <- vapply(selection, function(z) z$name, character(1))

fixed <- list(
  fixed_case("nile_ann", Nile, 1, "ANN"),
  fixed_case("nile_aan", Nile, 1, "AAN"),
  fixed_case("airpassengers_aaa", AirPassengers, 12, "AAA"),
  fixed_case("airpassengers_mam", AirPassengers, 12, "MAM"),
  fixed_case("usaccdeaths_ana", USAccDeaths, 12, "ANA")
)
names(fixed) <- vapply(fixed, function(z) z$name, character(1))

out <- toJSON(list(selection = selection, fixed = fixed),
              digits = NA, auto_unbox = TRUE, pretty = FALSE)
writeLines(out, "tests/fixtures/ets_r_reference.json")
cat("ZZZ selection results:\n")
for (z in selection) cat(sprintf("  %-22s -> %-14s aicc=%.3f\n", z$name, z$method, z$aicc))
cat("Fixed-spec loglik (for convention constant):\n")
for (z in fixed) {
  const <- 0.5 * z$n * (log(z$n / (2 * pi)) - 1)
  cat(sprintf("  %-22s loglik_R=%.4f  const(n=%d)=%.4f\n", z$name, z$loglik, z$n, const))
}
