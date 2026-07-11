# Entry-time (left-truncation / counting-process) reference fixtures.
# R 4.5.2 / survival 3.8.3.
#   E1: left-truncated KM, survfit(Surv(entry, time, event) ~ 1)
#   E2: left-truncated STRATIFIED KM, ~ g
#   E3: (start, stop] Cox with a time-varying covariate (subject spells)
#   E4: (start, stop] Cox + strata + heavy ties, efron + breslow
#   E5: R's behavior when entry == stop (degenerate interval) — capture message
#   E6: simple left-truncated Cox (one row per subject, delayed entry)

suppressPackageStartupMessages({library(survival); library(jsonlite)})
OUT <- "entry_fixtures"; dir.create(OUT, showWarnings = FALSE)

emit <- function(name, data, ref) {
  write.csv(data, file.path(OUT, paste0(name, "_data.csv")), row.names = FALSE)
  write_json(ref, file.path(OUT, paste0(name, "_ref.json")), digits = 15,
             auto_unbox = TRUE, pretty = TRUE)
  cat("wrote", name, "\n")
}

km_summary <- function(f) {
  sm <- summary(f)
  if (is.null(sm$strata)) {
    list(time = sm$time, surv = sm$surv, n_risk = sm$n.risk,
         n_event = sm$n.event,
         std_err = ifelse(is.finite(sm$std.err), sm$std.err, -1),
         lower = ifelse(is.na(sm$lower), -1, sm$lower),
         upper = ifelse(is.na(sm$upper), -1, sm$upper))
  } else {
    labels <- sub("^[^=]*=", "", levels(sm$strata))
    curves <- list()
    for (lvl in levels(sm$strata)) {
      idx <- which(sm$strata == lvl)
      lab <- sub("^[^=]*=", "", lvl)
      curves[[lab]] <- list(
        time = sm$time[idx], surv = sm$surv[idx], n_risk = sm$n.risk[idx],
        n_event = sm$n.event[idx],
        std_err = ifelse(is.finite(sm$std.err[idx]), sm$std.err[idx], -1),
        lower = ifelse(is.na(sm$lower[idx]), -1, sm$lower[idx]),
        upper = ifelse(is.na(sm$upper[idx]), -1, sm$upper[idx]))
    }
    list(strata = labels, curves = curves)
  }
}

cox_ref <- function(formula, data, ties) {
  f <- coxph(formula, data = data, ties = ties,
             control = coxph.control(eps = 1e-9, iter.max = 50))
  list(ties = ties, coefficients = as.numeric(coef(f)),
       se = as.numeric(sqrt(diag(vcov(f)))),
       loglik_null = f$loglik[1], loglik_model = f$loglik[2],
       concordance = as.numeric(f$concordance["concordance"]),
       n = f$n, nevent = f$nevent, iter = f$iter)
}

## E1 — left-truncated KM (delayed entry), n=40
set.seed(101)
n <- 40
stop_t <- round(rexp(n, 0.08), 1) + 1
entry <- round(stop_t * runif(n, 0, 0.6), 1)   # entry strictly before stop
e1 <- data.frame(entry = entry, time = stop_t, event = rbinom(n, 1, 0.75))
emit("e1_lt_km", e1,
     list(km = km_summary(survfit(Surv(entry, time, event) ~ 1, e1))))

## E2 — left-truncated stratified KM
set.seed(202)
n <- 50
stop_t <- round(rexp(n, 0.1), 1) + 0.5
e2 <- data.frame(entry = round(stop_t * runif(n, 0, 0.5), 1),
                 time = stop_t, event = rbinom(n, 1, 0.7),
                 g = rep(c("A", "B"), length.out = n))
emit("e2_lt_km_strata", e2,
     list(km = km_summary(survfit(Surv(entry, time, event) ~ g, e2))))

## E3 — (start, stop] Cox with a time-varying covariate:
## each subject has 1-2 spells; x2 switches value at a change point.
set.seed(303)
ns <- 30
rows <- list()
for (i in 1:ns) {
  total <- round(rexp(1, 0.08), 1) + 2
  ev <- rbinom(1, 1, 0.75)
  x1 <- round(rnorm(1), 5)
  if (runif(1) < 0.6 && total > 3) {           # two spells
    cut <- round(total * runif(1, 0.25, 0.75), 1)
    rows[[length(rows)+1]] <- data.frame(id=i, start=0,   stop=cut,   event=0,  x1=x1, x2=0)
    rows[[length(rows)+1]] <- data.frame(id=i, start=cut, stop=total, event=ev, x1=x1, x2=1)
  } else {
    rows[[length(rows)+1]] <- data.frame(id=i, start=0, stop=total, event=ev, x1=x1, x2=0)
  }
}
e3 <- do.call(rbind, rows)
emit("e3_tvc_cox", e3, list(
  efron   = cox_ref(Surv(start, stop, event) ~ x1 + x2, e3, "efron"),
  breslow = cox_ref(Surv(start, stop, event) ~ x1 + x2, e3, "breslow")))

## E4 — (start, stop] Cox + strata + heavy integer ties
set.seed(404)
n <- 60
stop_t <- sample(2:8, n, replace = TRUE)
entry <- pmax(0, stop_t - sample(1:6, n, replace = TRUE))
keep <- entry < stop_t
e4 <- data.frame(start = entry[keep], stop = stop_t[keep],
                 event = rbinom(sum(keep), 1, 0.8),
                 x1 = round(rnorm(sum(keep)), 5),
                 g = sample(1:2, sum(keep), replace = TRUE))
emit("e4_tvc_strata", e4, list(
  efron   = cox_ref(Surv(start, stop, event) ~ x1 + strata(g), e4, "efron"),
  breslow = cox_ref(Surv(start, stop, event) ~ x1 + strata(g), e4, "breslow")))

## E5 — degenerate interval entry == stop: what does R do?
msg <- tryCatch({
  s <- Surv(c(0, 2), c(1, 2), c(1, 1))
  f <- coxph(s ~ c(0.5, -0.5))
  "no error"
}, warning = function(w) paste("WARNING:", conditionMessage(w)),
   error = function(e) paste("ERROR:", conditionMessage(e)))
cat("E5 (entry==stop):", msg, "\n")
write_json(list(behavior = msg), file.path(OUT, "e5_degenerate_ref.json"),
           auto_unbox = TRUE)

## E6 — simple left-truncated Cox (one row per subject)
set.seed(606)
n <- 45
stop_t <- round(rexp(n, 0.07), 1) + 1
e6 <- data.frame(start = round(stop_t * runif(n, 0, 0.5), 1),
                 stop = stop_t, event = rbinom(n, 1, 0.7),
                 x1 = round(rnorm(n), 5), x2 = rbinom(n, 1, 0.5))
emit("e6_lt_cox", e6, list(
  efron = cox_ref(Surv(start, stop, event) ~ x1 + x2, e6, "efron")))

## Also: cox.zph on a counting-process fit (for later wiring of zph+entry)
f3 <- coxph(Surv(start, stop, event) ~ x1 + x2, e3, ties = "efron",
            control = coxph.control(eps = 1e-9, iter.max = 50))
z3 <- cox.zph(f3, transform = "km")
write_json(list(rows = rownames(z3$table),
                chisq = as.numeric(z3$table[, "chisq"]),
                df = as.numeric(z3$table[, "df"]),
                p = as.numeric(z3$table[, "p"])),
           file.path(OUT, "e3_zph_ref.json"), digits = 15, auto_unbox = TRUE)
cat("done\n")
