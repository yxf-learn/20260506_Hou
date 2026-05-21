suppressPackageStartupMessages({
  library(rdrobust)
  library(rddensity)
  library(data.table)
})
args <- commandArgs(trailingOnly = TRUE)
panel <- fread(args[1])
out  <- list()
for (yk in c("y_match_6m","y_quality_6m","y_stability_12m")) {
  r1 <- rdrobust(y = panel[[yk]], x = panel$risk_score, c = 75, fuzzy = panel$enrolled_d)
  r2 <- rddensity(X = panel$risk_score, c = 75)
  out[[yk]] <- list(coef = r1$coef, se = r1$se, ci = r1$ci, density_p = r2$test$p_jk)
}
saveRDS(out, file = args[2])
