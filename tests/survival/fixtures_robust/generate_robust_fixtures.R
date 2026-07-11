# Robust / cluster-robust Cox SE reference (survival 3.8.3).
# R1: robust=TRUE (independent obs, no cluster) — LWA sandwich.
# R2: cluster(id) grouped-robust SE (repeated subjects).
# R3: robust + strata.
# R4: robust on counting-process (start,stop] data with cluster(id).
# Also emit residuals(fit, type="dfbeta") to pin the residual convention.

suppressPackageStartupMessages({library(survival); library(jsonlite)})
OUT <- "robust_fixtures"; dir.create(OUT, showWarnings = FALSE)

emit <- function(name, data, ref) {
  write.csv(data, file.path(OUT, paste0(name, "_data.csv")), row.names = FALSE)
  write_json(ref, file.path(OUT, paste0(name, "_ref.json")), digits = 15,
             auto_unbox = TRUE, pretty = TRUE)
  cat("wrote", name, "\n")
}

ref_of <- function(f) {
  s <- summary(f)
  list(coefficients = as.numeric(coef(f)),
       robust_se = as.numeric(sqrt(diag(vcov(f)))),   # vcov IS robust when robust=TRUE
       naive_se = as.numeric(sqrt(diag(f$naive.var))),
       robust_z = as.numeric(s$coefficients[, "z"]),
       robust_p = as.numeric(s$coefficients[, ncol(s$coefficients)]),
       loglik_model = f$loglik[2])
}

## R1 — robust, no cluster
set.seed(1)
n <- 50
d1 <- data.frame(time = round(rexp(n, 0.1), 2), event = rbinom(n, 1, 0.75),
                 x1 = round(rnorm(n), 5), x2 = round(rnorm(n), 5))
f1 <- coxph(Surv(time, event) ~ x1 + x2, d1, ties = "efron", robust = TRUE)
db1 <- residuals(f1, type = "dfbeta")
f1b <- coxph(Surv(time, event) ~ x1 + x2, d1, ties = "breslow", robust = TRUE)
db1b <- residuals(f1b, type = "dfbeta")
emit("r1_robust", d1, c(ref_of(f1), list(
  dfbeta = unname(as.matrix(db1)),
  breslow = list(robust_se = as.numeric(sqrt(diag(vcov(f1b)))),
                 coefficients = as.numeric(coef(f1b)),
                 dfbeta = unname(as.matrix(db1b))))))

## R1c — heavy ties, efron + breslow dfbeta (isolates the tie correction)
set.seed(11)
n <- 45
d1c <- data.frame(time = sample(1:5, n, replace = TRUE), event = rbinom(n, 1, 0.8),
                  x1 = round(rnorm(n), 5), x2 = round(runif(n), 5))
mk <- function(tie) {
  f <- coxph(Surv(time, event) ~ x1 + x2, d1c, ties = tie, robust = TRUE)
  list(coefficients = as.numeric(coef(f)),
       robust_se = as.numeric(sqrt(diag(vcov(f)))),
       dfbeta = unname(as.matrix(residuals(f, type = "dfbeta"))))
}
emit("r1c_ties", d1c, list(efron = mk("efron"), breslow = mk("breslow")))

## R2 — cluster(id): 30 subjects, some with 2 correlated records
set.seed(2)
ns <- 30
d2 <- do.call(rbind, lapply(1:ns, function(i) {
  k <- if (runif(1) < 0.5) 2 else 1
  data.frame(id = i, time = round(rexp(k, 0.1), 2), event = rbinom(k, 1, 0.7),
             x1 = round(rnorm(k), 5), x2 = round(rnorm(1), 5))
}))
f2 <- coxph(Surv(time, event) ~ x1 + x2 + cluster(id), d2, ties = "efron")
db2 <- residuals(f2, type = "dfbeta", collapse = d2$id)  # per-cluster dfbeta
emit("r2_cluster", d2, c(ref_of(f2),
     list(dfbeta_cluster = unname(as.matrix(db2)),
          cluster_ids = sort(unique(d2$id)))))

## R3 — robust + strata
set.seed(3)
n <- 60
d3 <- data.frame(time = round(rexp(n, 0.12), 1), event = rbinom(n, 1, 0.7),
                 x1 = round(rnorm(n), 5), g = rep(c(1, 2), length.out = n))
f3 <- coxph(Surv(time, event) ~ x1 + strata(g), d3, ties = "efron",
            robust = TRUE)
emit("r3_robust_strata", d3, ref_of(f3))

## R4 — counting process + cluster
set.seed(4)
ns <- 25
rows <- list()
for (i in 1:ns) {
  total <- round(rexp(1, 0.1), 1) + 2
  x1 <- round(rnorm(1), 5)
  if (runif(1) < 0.6) {
    cut <- round(total * runif(1, 0.3, 0.7), 1)
    rows[[length(rows)+1]] <- data.frame(id=i, start=0, stop=cut, event=0, x1=x1, x2=0)
    rows[[length(rows)+1]] <- data.frame(id=i, start=cut, stop=total, event=rbinom(1,1,0.7), x1=x1, x2=1)
  } else {
    rows[[length(rows)+1]] <- data.frame(id=i, start=0, stop=total, event=rbinom(1,1,0.7), x1=x1, x2=0)
  }
}
d4 <- do.call(rbind, rows)
f4 <- coxph(Surv(start, stop, event) ~ x1 + x2 + cluster(id), d4, ties = "efron")
emit("r4_cp_cluster", d4, c(ref_of(f4), list(cluster_ids = sort(unique(d4$id)))))

cat("done\n")
