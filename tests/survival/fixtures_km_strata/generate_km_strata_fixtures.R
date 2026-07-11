# Stratified Kaplan-Meier reference fixtures from survival::survfit(Surv(t,e)~g)
# R 4.5.2, survival 3.8.3. Emits one JSON per fixture: per-stratum curves with
# time/surv/n.risk/n.event/n.censor/std.err/lower/upper, plus the data CSV.

suppressPackageStartupMessages({library(survival); library(jsonlite)})
OUT <- Sys.getenv("FIXTURE_OUT", "km_fixtures")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# Per-stratum extraction. pystatistics KM emits ONE row per distinct EVENT time
# (not censor-only times), matching summary(survfit) — so extract from
# summary(f), whose std.err is already se(S). Split by summary's stratum factor.
km_ref <- function(formula, data, conf.type = "log") {
  f <- survfit(formula, data = data, conf.type = conf.type)
  sm <- summary(f)
  strata_fac <- sm$strata
  labels <- sub("^[^=]*=", "", levels(strata_fac))   # "g=1" -> "1"
  curves <- list()
  for (lab_lvl in levels(strata_fac)) {
    idx <- which(strata_fac == lab_lvl)
    lab <- sub("^[^=]*=", "", lab_lvl)
    lower <- if (is.null(sm$lower)) rep(-1, length(idx)) else sm$lower[idx]
    upper <- if (is.null(sm$upper)) rep(-1, length(idx)) else sm$upper[idx]
    curves[[lab]] <- list(
      time = sm$time[idx], surv = sm$surv[idx],
      n_risk = sm$n.risk[idx], n_event = sm$n.event[idx],
      std_err = ifelse(is.finite(sm$std.err[idx]), sm$std.err[idx], -1),
      lower = ifelse(is.na(lower), -1, lower),
      upper = ifelse(is.na(upper), -1, upper)
    )
  }
  list(strata = labels, conf_type = conf.type, curves = curves)
}

emit <- function(name, data, ref) {
  write.csv(data, file.path(OUT, paste0(name, "_data.csv")), row.names = FALSE)
  write_json(ref, file.path(OUT, paste0(name, "_ref.json")),
             digits = 15, auto_unbox = TRUE, pretty = TRUE)
  cat("wrote", name, "\n")
}

## K1: two strata, ties + censoring, n=40
set.seed(11)
k1 <- data.frame(time = round(rexp(40, 0.1), 1),
                 event = rbinom(40, 1, 0.7),
                 g = rep(c("A", "B"), length.out = 40))
emit("k1_basic", k1, km_ref(Surv(time, event) ~ g, k1))

## K2: three strata incl. one all-censored + log-log CI
set.seed(22)
k2 <- data.frame(time = round(rexp(36, 0.15), 2),
                 event = c(rbinom(24, 1, 0.7), rep(0, 12)),
                 g = c(rep(1, 12), rep(2, 12), rep(3, 12)))  # stratum 3 all censored
emit("k2_censored", k2, km_ref(Surv(time, event) ~ g, k2, conf.type = "log-log"))

## K3: heavy integer-time ties, 2 strata
set.seed(33)
k3 <- data.frame(time = sample(1:5, 50, replace = TRUE),
                 event = rbinom(50, 1, 0.8),
                 g = sample(c("x", "y"), 50, replace = TRUE))
emit("k3_heavyties", k3, km_ref(Surv(time, event) ~ g, k3))

## K5: single-level strata must equal the unstratified survfit(~1)
set.seed(55)
k5 <- data.frame(time = round(rexp(20, 0.1), 1), event = rbinom(20, 1, 0.8),
                 g = rep(1, 20))
sp <- summary(survfit(Surv(time, event) ~ 1, k5))
emit("k5_onelevel", k5, list(
  stratified = km_ref(Surv(time, event) ~ g, k5),
  plain = list(time = sp$time, surv = sp$surv,
               n_risk = sp$n.risk, std_err = sp$std.err)
))

cat("done ->", normalizePath(OUT), "\n")
