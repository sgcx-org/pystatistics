# Provenance: generates tests/fixtures/ets_r_reference.json.
# Generated with R 4.5.2 + forecast 9.0.0 on 2026-07-03; cross-scoring
# section added 2026-07-04. Run from the repo root:
#   KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=$PWD \
#       python tests/fixtures/generate_ets_py_params.py   # stage 1
#   Rscript tests/fixtures/generate_ets_r_reference.R     # stage 2
# Generates forecast::ets reference fixtures:
#  (a) ZZZ auto-selection results per dataset (selected components + ICs)
#  (b) fixed-spec fits for the log-likelihood convention constant check
#  (c) PyStatistics' selected fits (from ets_py_params.json) scored under
#      R's own likelihood via forecast:::pegelsresid.C, plus R's
#      admissible() verdict — the honest cross-engine comparison.  The
#      transplant harness is self-validating: it first feeds R's own
#      fitted states back through pegelsresid.C and stops unless that
#      reproduces R's reported -2*loglik (1e-8 relative tolerance).
suppressMessages(library(jsonlite))
suppressMessages(library(forecast))

pegelsresid <- get("pegelsresid.C", envir = getNamespace("forecast"))
admissible_fn <- get("admissible", envir = getNamespace("forecast"))

# Score a parameter set under R's own likelihood. init_state must be in
# R's ordering: c(l0, b0?, s_0, s_{-1}, ..., s_{1-m}) — most-recent
# seasonal FIRST (PyStatistics stores oldest first, so callers reverse).
r_aicc_for_params <- function(y, m, init_state, error, trend, season,
                              damped, alpha, beta, gamma, phi) {
  e <- pegelsresid(as.numeric(y), m, init_state, error, trend, season,
                   damped, alpha, beta, gamma, phi, 3)
  # Parameter count matches ets(): smoothing + FREE initial states +
  # sigma^2.  init_state carries all m seasonal states but only m-1 are
  # free (the last is fixed by the normalisation), hence the -1.
  np <- sum(!sapply(list(alpha, beta, gamma, phi), is.null)) +
    length(init_state) - (season != "N") + 1
  n <- length(y)
  aic <- e$lik + 2 * np
  list(lik = e$lik, np = np, aicc = aic + 2 * np * (np + 1) / (n - np - 1))
}

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

# ---------------------------------------------------------------------------
# (c) Cross-score PyStatistics' selected fits under R's own likelihood
# ---------------------------------------------------------------------------

# Harness self-validation: R's own fitted states fed back through
# pegelsresid.C must reproduce R's own -2*loglik (to 1e-8 relative
# tolerance), or the transplant cannot be trusted on foreign parameters.
for (z in selection) {
  xt <- ts(z$x, frequency = z$period)
  f <- forecast::ets(xt, model = z$model_arg)
  g <- function(nm) { v <- f$par[nm]; if (is.na(v)) NULL else unname(v) }
  cmp <- f$components
  # pegelsresid.C expects m = 1 for non-seasonal models (as etsmodel does)
  e <- pegelsresid(as.numeric(xt), if (cmp[3] == "N") 1 else z$period,
                   as.numeric(f$states[1, ]), cmp[1], cmp[2], cmp[3],
                   cmp[4] == "TRUE", g("alpha"), g("beta"), g("gamma"),
                   g("phi"), 3)
  stopifnot(isTRUE(all.equal(e$lik, -2 * f$loglik, tolerance = 1e-8)))
}
cat("pegelsresid.C harness self-validation: OK on all selection datasets\n")

py_params_path <- "tests/fixtures/ets_py_params.json"
if (!file.exists(py_params_path)) {
  stop(py_params_path, " missing — run generate_ets_py_params.py first ",
       "(stage 1); regenerating without it would drop the py_pick_* ",
       "fields 10 committed tests require.")
}
py <- fromJSON(py_params_path, simplifyVector = FALSE)
for (name in names(py)) {
  p <- py[[name]]
  ist <- c(p$init_level)
  if (p$trend != "N") ist <- c(ist, p$init_trend)
  # PyStatistics stores seasonals oldest-first; R wants newest-first.
  if (p$season != "N") ist <- c(ist, rev(unlist(p$init_season)))
  beta <- if (p$trend != "N") p$beta else NULL
  gamma <- if (p$season != "N") p$gamma else NULL
  phi <- if (isTRUE(p$damped)) p$phi else NULL
  sc <- r_aicc_for_params(
    selection[[name]]$x, if (p$season == "N") 1 else p$period, ist,
    p$error, sub("d$", "", p$trend), p$season, isTRUE(p$damped),
    p$alpha, beta, gamma, phi)
  selection[[name]]$py_pick <- p$picked
  selection[[name]]$py_pick_aicc_in_r <- sc$aicc
  selection[[name]]$py_pick_admissible <- admissible_fn(
    p$alpha, beta, gamma, if (isTRUE(p$damped)) p$phi else 1,
    max(p$period, 1))
}

out <- toJSON(list(selection = selection, fixed = fixed),
              digits = NA, auto_unbox = TRUE, pretty = FALSE)
writeLines(out, "tests/fixtures/ets_r_reference.json")
cat("ZZZ selection results:\n")
for (z in selection) {
  extra <- if (!is.null(z$py_pick)) {
    sprintf("  py=%s pyAiccInR=%.3f adm=%s", z$py_pick,
            z$py_pick_aicc_in_r, z$py_pick_admissible)
  } else ""
  cat(sprintf("  %-22s -> %-14s aicc=%.3f%s\n", z$name, z$method, z$aicc, extra))
}
cat("Fixed-spec loglik (for convention constant):\n")
for (z in fixed) {
  const <- 0.5 * z$n * (log(z$n / (2 * pi)) - 1)
  cat(sprintf("  %-22s loglik_R=%.4f  const(n=%d)=%.4f\n", z$name, z$loglik, z$n, const))
}
