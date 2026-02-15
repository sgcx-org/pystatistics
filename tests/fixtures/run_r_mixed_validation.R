#!/usr/bin/env Rscript
# R validation script for mixed models.
# Reads fixture CSV/JSON, runs lme4/lmerTest, writes reference JSON.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(jsonlite)
})

fixture_dir <- file.path(dirname(sys.frame(1)$ofile), "mixed")
meta_path <- file.path(fixture_dir, "mixed_meta.json")

if (!file.exists(meta_path)) {
  cat("No mixed_meta.json found. Run generate_mixed_fixtures.py first.\n")
  quit(status = 1)
}

meta <- fromJSON(meta_path)
results <- list()

for (name in names(meta)) {
  info <- meta[[name]]
  csv_path <- file.path(fixture_dir, paste0(name, ".csv"))

  if (!file.exists(csv_path)) {
    cat("Skipping", name, "- CSV not found\n")
    next
  }

  dat <- read.csv(csv_path)
  cat("Processing:", name, "\n")

  tryCatch({
    if (info$type == "lmm") {
      res <- list()

      if (name == "lmm_intercept") {
        # Convert group to factor
        dat$group <- factor(dat$group)

        # REML fit
        fit_reml <- lmer(y ~ x + (1|group), data = dat, REML = TRUE)
        s_reml <- summary(fit_reml)

        res$reml <- list(
          fixef = as.list(fixef(fit_reml)),
          se = as.numeric(s_reml$coefficients[, "Std. Error"]),
          df = as.numeric(s_reml$coefficients[, "df"]),
          t_value = as.numeric(s_reml$coefficients[, "t value"]),
          p_value = as.numeric(s_reml$coefficients[, "Pr(>|t|)"]),
          var_group = as.numeric(VarCorr(fit_reml)$group[1,1]),
          var_resid = sigma(fit_reml)^2,
          logLik = as.numeric(logLik(fit_reml)),
          AIC = AIC(fit_reml),
          BIC = BIC(fit_reml),
          ranef = as.numeric(ranef(fit_reml)$group[,1])
        )

        # ML fit
        fit_ml <- lmer(y ~ x + (1|group), data = dat, REML = FALSE)
        s_ml <- summary(fit_ml)

        res$ml <- list(
          fixef = as.list(fixef(fit_ml)),
          var_group = as.numeric(VarCorr(fit_ml)$group[1,1]),
          var_resid = sigma(fit_ml)^2,
          logLik = as.numeric(logLik(fit_ml)),
          AIC = AIC(fit_ml),
          BIC = BIC(fit_ml)
        )

      } else if (name == "lmm_slope") {
        dat$subject <- factor(dat$subject)

        fit <- lmer(y ~ days + (1 + days|subject), data = dat, REML = TRUE)
        s <- summary(fit)
        vc <- VarCorr(fit)

        res$reml <- list(
          fixef = as.list(fixef(fit)),
          se = as.numeric(s$coefficients[, "Std. Error"]),
          df = as.numeric(s$coefficients[, "df"]),
          t_value = as.numeric(s$coefficients[, "t value"]),
          p_value = as.numeric(s$coefficients[, "Pr(>|t|)"]),
          var_intercept = as.numeric(vc$subject[1,1]),
          var_slope = as.numeric(vc$subject[2,2]),
          corr = as.numeric(attr(vc$subject, "correlation")[2,1]),
          var_resid = sigma(fit)^2,
          logLik = as.numeric(logLik(fit)),
          AIC = AIC(fit),
          BIC = BIC(fit)
        )

      } else if (name == "lmm_crossed") {
        dat$subject <- factor(dat$subject)
        dat$item <- factor(dat$item)

        fit <- lmer(y ~ x + (1|subject) + (1|item), data = dat, REML = TRUE)
        s <- summary(fit)
        vc <- VarCorr(fit)

        res$reml <- list(
          fixef = as.list(fixef(fit)),
          se = as.numeric(s$coefficients[, "Std. Error"]),
          df = as.numeric(s$coefficients[, "df"]),
          var_subject = as.numeric(vc$subject[1,1]),
          var_item = as.numeric(vc$item[1,1]),
          var_resid = sigma(fit)^2,
          logLik = as.numeric(logLik(fit)),
          AIC = AIC(fit),
          BIC = BIC(fit)
        )

      } else if (name == "lmm_ml") {
        dat$group <- factor(dat$group)

        fit <- lmer(y ~ x + (1|group), data = dat, REML = FALSE)
        s <- summary(fit)

        res$ml <- list(
          fixef = as.list(fixef(fit)),
          var_group = as.numeric(VarCorr(fit)$group[1,1]),
          var_resid = sigma(fit)^2,
          logLik = as.numeric(logLik(fit)),
          AIC = AIC(fit),
          BIC = BIC(fit)
        )

      } else if (name == "lmm_no_effect") {
        dat$group <- factor(dat$group)

        fit <- lmer(y ~ x + (1|group), data = dat, REML = TRUE)
        s <- summary(fit)

        res$reml <- list(
          fixef = as.list(fixef(fit)),
          se = as.numeric(s$coefficients[, "Std. Error"]),
          var_group = as.numeric(VarCorr(fit)$group[1,1]),
          var_resid = sigma(fit)^2,
          logLik = as.numeric(logLik(fit))
        )
      }

      results[[name]] <- res

    } else if (info$type == "glmm") {
      res <- list()

      dat$group <- factor(dat$group)
      family_name <- info$family

      fit <- glmer(y ~ x + (1|group), data = dat, family = family_name)
      s <- summary(fit)
      vc <- VarCorr(fit)

      res$fit <- list(
        fixef = as.list(fixef(fit)),
        se = as.numeric(s$coefficients[, "Std. Error"]),
        z_value = as.numeric(s$coefficients[, "z value"]),
        p_value = as.numeric(s$coefficients[, "Pr(>|z|)"]),
        var_group = as.numeric(vc$group[1,1]),
        deviance = deviance(fit),
        logLik = as.numeric(logLik(fit)),
        AIC = AIC(fit),
        BIC = BIC(fit)
      )

      results[[name]] <- res
    }

    cat("  OK\n")

  }, error = function(e) {
    cat("  ERROR:", conditionMessage(e), "\n")
    results[[name]] <<- list(error = conditionMessage(e))
  })
}

# Write results
out_path <- file.path(fixture_dir, "mixed_r_results.json")
write(toJSON(results, auto_unbox = TRUE, digits = 17), out_path)
cat("\nWrote results to", out_path, "\n")
