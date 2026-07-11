# Stratified Cox PH reference fixtures from survival::coxph
# R 4.5.2, survival 3.8.3 — anchors for pystatistics stratified Cox (A1).
# Emits JSON fixtures (15 significant digits) + the exact data as CSV so the
# Python side fits on bit-identical inputs.

suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
})

OUT <- Sys.getenv("FIXTURE_OUT", "fixtures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

emit <- function(name, data, fits) {
  write.csv(data, file.path(OUT, paste0(name, "_data.csv")), row.names = FALSE)
  write_json(fits, file.path(OUT, paste0(name, "_ref.json")),
             digits = 15, auto_unbox = TRUE, pretty = TRUE)
  cat("wrote", name, "\n")
}

fit_ref <- function(formula, data, ties) {
  f <- coxph(formula, data = data, ties = ties,
             control = coxph.control(eps = 1e-9, iter.max = 50))
  s <- summary(f)
  conc <- f$concordance  # named vector: concordant discordant tied.x tied.y tied.xy concordance std
  list(
    ties = ties,
    coefficients = as.numeric(coef(f)),
    se = as.numeric(sqrt(diag(vcov(f)))),
    loglik_null = f$loglik[1],
    loglik_model = f$loglik[2],
    concordance = as.numeric(conc["concordance"]),
    concordance_counts = as.list(conc[c("concordant", "discordant")]),
    n = f$n, nevent = f$nevent, iter = f$iter,
    z = as.numeric(s$coefficients[, "z"]),
    p = as.numeric(s$coefficients[, "Pr(>|z|)"])
  )
}

both_ties <- function(formula, data) {
  list(efron = fit_ref(formula, data, "efron"),
       breslow = fit_ref(formula, data, "breslow"))
}

## ---- S1: basic 2-strata, 2 covariates, a few ties, n=30 -------------------
set.seed(101)
n <- 30
s1 <- data.frame(
  time  = round(rexp(n, rate = 0.1), 1),           # 1-decimal → some ties
  event = rbinom(n, 1, 0.7),
  x1    = round(rnorm(n), 6),
  x2    = round(rbinom(n, 1, 0.5), 0),
  g     = rep(c(1, 2), length.out = n)
)
emit("s1_basic", s1, both_ties(Surv(time, event) ~ x1 + x2 + strata(g), s1))

## ---- S2: singleton stratum + stratum with zero events, n=26 ---------------
set.seed(202)
s2 <- data.frame(
  time  = round(rexp(26, 0.2), 2),
  event = c(rbinom(20, 1, 0.7), rep(0, 5), 1),   # stratum 3 = all censored; stratum 4 = singleton (event)
  x1    = round(rnorm(26), 6),
  g     = c(rep(1, 10), rep(2, 10), rep(3, 5), 4)
)
emit("s2_degenerate", s2, both_ties(Surv(time, event) ~ x1 + strata(g), s2))

## ---- S3: heavy ties (integer times 1..5), 3 strata, n=45 ------------------
set.seed(303)
s3 <- data.frame(
  time  = sample(1:5, 45, replace = TRUE),
  event = rbinom(45, 1, 0.75),
  x1    = round(rnorm(45), 6),
  x2    = round(runif(45), 6),
  g     = sample(1:3, 45, replace = TRUE)
)
emit("s3_heavyties", s3, both_ties(Surv(time, event) ~ x1 + x2 + strata(g), s3))

## ---- S4: monotone likelihood (separation) in BOTH strata -------------------
# beta is shared, so divergence requires the covariate to perfectly order events
# in every stratum: x1=1 subjects all die strictly before any x1=0 subject in
# each stratum. R stops when the loglik plateaus and warns "coefficient may be
# infinite". Capture the warning + where R stops.
s4 <- data.frame(
  time  = c(1, 2, 3, 4, 5, 6,   1.5, 2.5, 3.5, 4.5, 5.5, 6.5),
  event = rep(1, 12),
  x1    = rep(c(1, 1, 1, 0, 0, 0), 2),
  g     = rep(c(1, 2), each = 6)
)
s4_warnings <- character(0)
s4_fit <- withCallingHandlers({
  f <- coxph(Surv(time, event) ~ x1 + strata(g), data = s4, ties = "efron",
             control = coxph.control(eps = 1e-9, iter.max = 50))
  list(coefficients = as.numeric(coef(f)),
       se = as.numeric(sqrt(diag(vcov(f)))),
       loglik_null = f$loglik[1], loglik_model = f$loglik[2],
       iter = f$iter)
}, warning = function(w) {
  s4_warnings <<- c(s4_warnings, conditionMessage(w))
  invokeRestart("muffleWarning")
})
s4_fit$warnings <- as.list(s4_warnings)
emit("s4_nearsep", s4, list(efron = s4_fit))

## ---- S5: flchain real data — age + strata(sex), also chapter strata -------
data(flchain)
fl <- flchain[!is.na(flchain$creatinine), ]
fl$time <- fl$futime + 0.0
# R note: futime==0 rows exist; keep them, R handles zero-length spells by dropping
# them from risk sets naturally (time 0 events). Match whatever R does.
f5 <- both_ties(Surv(futime, death) ~ age + kappa + lambda + strata(sex), fl)
emit("s5_flchain",
     fl[, c("futime", "death", "age", "kappa", "lambda", "sex")],
     f5)

## ---- S6: no-strata regression guard — strata() with ONE level == unstratified
set.seed(606)
s6 <- data.frame(
  time = round(rexp(20, 0.1), 2), event = rbinom(20, 1, 0.8),
  x1 = round(rnorm(20), 6), g = rep(1, 20)
)
emit("s6_onestratum", s6, list(
  efron_strata = fit_ref(Surv(time, event) ~ x1 + strata(g), s6, "efron"),
  efron_plain  = fit_ref(Surv(time, event) ~ x1, s6, "efron")
))

cat("all fixtures written to", normalizePath(OUT), "\n")
