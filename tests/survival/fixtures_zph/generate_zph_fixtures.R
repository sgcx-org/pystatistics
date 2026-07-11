# cox.zph reference fixtures (R 4.5.2 / survival 3.8.3).
# Z1: plain fit, no heavy ties — table for all 4 transforms + y/x for km.
# Z2: stratified fit with ties — table (km, rank) + schoenfeld residuals.
# Z3: heavy ties — efron vs breslow zph tables + raw schoenfeld residuals
#     (pins down the tie convention for both the kernel and the residuals).

suppressPackageStartupMessages({library(survival); library(jsonlite)})
OUT <- "zph_fixtures"; dir.create(OUT, showWarnings = FALSE)

tab_list <- function(z) {
  tab <- z$table
  list(rows = rownames(tab), chisq = as.numeric(tab[, "chisq"]),
       df = as.numeric(tab[, "df"]), p = as.numeric(tab[, "p"]))
}

## Z1 — plain fit
set.seed(7)
n <- 60
d1 <- data.frame(time = round(rexp(n, 0.1), 2), event = rbinom(n, 1, 0.75),
                 x1 = round(rnorm(n), 5), x2 = round(rnorm(n), 5))
write.csv(d1, file.path(OUT, "z1_data.csv"), row.names = FALSE)
f1 <- coxph(Surv(time, event) ~ x1 + x2, d1, ties = "efron",
            control = coxph.control(eps = 1e-9, iter.max = 50))
zkm <- cox.zph(f1, transform = "km")
out1 <- list(
  coef = as.numeric(coef(f1)),
  km = tab_list(zkm), identity = tab_list(cox.zph(f1, "identity")),
  rank = tab_list(cox.zph(f1, "rank")), log = tab_list(cox.zph(f1, "log")),
  km_x = as.numeric(zkm$x), km_time = as.numeric(zkm$time),
  km_y = unname(as.matrix(zkm$y)), km_var = unname(as.matrix(zkm$var))
)
write_json(out1, file.path(OUT, "z1_ref.json"), digits = 15,
           auto_unbox = TRUE, pretty = TRUE)

## Z2 — stratified fit with some ties
set.seed(21)
n <- 80
d2 <- data.frame(time = round(rexp(n, 0.12), 1), event = rbinom(n, 1, 0.7),
                 x1 = round(rnorm(n), 5), x2 = rbinom(n, 1, 0.4),
                 g = rep(c(1, 2, 3), length.out = n))
write.csv(d2, file.path(OUT, "z2_data.csv"), row.names = FALSE)
f2 <- coxph(Surv(time, event) ~ x1 + x2 + strata(g), d2, ties = "efron",
            control = coxph.control(eps = 1e-9, iter.max = 50))
z2km <- cox.zph(f2, transform = "km")
out2 <- list(
  coef = as.numeric(coef(f2)),
  km = tab_list(z2km), rank = tab_list(cox.zph(f2, "rank")),
  km_x = as.numeric(z2km$x), km_y = unname(as.matrix(z2km$y))
)
write_json(out2, file.path(OUT, "z2_ref.json"), digits = 15,
           auto_unbox = TRUE, pretty = TRUE)

## Z3 — heavy ties: efron vs breslow + raw schoenfeld residuals
set.seed(33)
n <- 50
d3 <- data.frame(time = sample(1:6, n, replace = TRUE),
                 event = rbinom(n, 1, 0.8),
                 x1 = round(rnorm(n), 5), x2 = round(runif(n), 5))
write.csv(d3, file.path(OUT, "z3_data.csv"), row.names = FALSE)
res <- list()
for (tie in c("efron", "breslow")) {
  f3 <- coxph(Surv(time, event) ~ x1 + x2, d3, ties = tie,
              control = coxph.control(eps = 1e-9, iter.max = 50))
  sr <- residuals(f3, type = "schoenfeld")   # (n_events, p), event-time order
  res[[tie]] <- list(
    coef = as.numeric(coef(f3)),
    km = tab_list(cox.zph(f3, "km")),
    identity = tab_list(cox.zph(f3, "identity")),
    schoenfeld = unname(as.matrix(sr)),
    schoen_times = as.numeric(rownames(sr))
  )
}
write_json(res, file.path(OUT, "z3_ref.json"), digits = 15,
           auto_unbox = TRUE, pretty = TRUE)

cat("zph fixtures written\n")
